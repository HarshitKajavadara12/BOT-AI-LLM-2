"""
Historical Data Management System for QUANTUM-FORGE
Comprehensive historical data storage, retrieval, and management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import sqlite3
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, DateTime, Index
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import os
import logging
import asyncio
import aiofiles
from typing import Dict, List, Any, Optional, Union, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import json
import warnings
from pathlib import Path
import zipfile
import shutil
warnings.filterwarnings('ignore')

class HistoricalDataManager:
    """Comprehensive historical data management system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize historical data manager."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.config = config or self._default_config()
        self.storage_engines = {}
        self.data_cache = {}
        self.metadata_cache = {}
        
        # Initialize storage systems
        self._init_storage_engines()
        
        # Create directory structure
        self._create_directories()
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("[OK] Historical Data Manager initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'database_url': 'sqlite:///quantum_forge_historical.db',
            'data_directory': './data/',
            'parquet_directory': './data/parquet/',
            'hdf5_directory': './data/hdf5/',
            'archive_directory': './data/archive/',
            'cache_size': 1000,
            'compression': 'gzip',
            'batch_size': 10000,
            'archive_after_days': 365,
            'supported_frequencies': ['1min', '5min', '15min', '30min', '1H', '4H', '1D'],
            'data_sources': {
                'yahoo': True,
                'alpha_vantage': True,
                'iex': True,
                'custom': True
            }
        }
    
    def _init_storage_engines(self):
        """Initialize storage engines."""
        try:
            # SQLAlchemy for metadata and small datasets
            self.engine = create_engine(self.config['database_url'])
            self.Session = sessionmaker(bind=self.engine)
            
            # Create metadata tables
            self._create_metadata_tables()
            
            self.logger.info("[OK] Storage engines initialized")
            
        except Exception as e:
            self.logger.error(f"[ERR] Storage engine initialization failed: {e}")
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.config['data_directory'],
            self.config['parquet_directory'],
            self.config['hdf5_directory'],
            self.config['archive_directory']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _create_metadata_tables(self):
        """Create metadata tables."""
        metadata = MetaData()
        
        # Data catalog table
        self.data_catalog = Table(
            'data_catalog', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(20)),
            Column('data_type', String(50)),
            Column('frequency', String(10)),
            Column('start_date', DateTime),
            Column('end_date', DateTime),
            Column('record_count', Integer),
            Column('file_path', String(500)),
            Column('file_format', String(20)),
            Column('file_size', Integer),
            Column('checksum', String(64)),
            Column('created_at', DateTime),
            Column('updated_at', DateTime),
            Column('source', String(50)),
            Column('quality_score', Float)
        )
        
        # Data quality table
        self.data_quality = Table(
            'data_quality', metadata,
            Column('id', Integer, primary_key=True),
            Column('catalog_id', Integer),
            Column('missing_data_pct', Float),
            Column('duplicate_records', Integer),
            Column('outliers_count', Integer),
            Column('gaps_count', Integer),
            Column('validation_errors', String(1000)),
            Column('last_validated', DateTime)
        )
        
        # Create indexes
        Index('idx_symbol_type_freq', self.data_catalog.c.symbol, 
              self.data_catalog.c.data_type, self.data_catalog.c.frequency)
        Index('idx_date_range', self.data_catalog.c.start_date, self.data_catalog.c.end_date)
        
        if self.engine:
            metadata.create_all(self.engine)
    
    def store_historical_data(self, symbol: str, data: pd.DataFrame, 
                            data_type: str = 'ohlcv', frequency: str = '1D',
                            source: str = 'unknown', format: str = 'parquet') -> bool:
        """Store historical data."""
        try:
            if data.empty:
                self.logger.warning(f" ️ Empty dataset for {symbol}")
                return False
            
            # Validate data
            quality_metrics = self._validate_data(data)
            
            # Generate file path
            file_path = self._generate_file_path(symbol, data_type, frequency, format)
            
            # Store data based on format
            if format == 'parquet':
                success = self._store_parquet(data, file_path)
            elif format == 'hdf5':
                success = self._store_hdf5(data, file_path, symbol)
            else:
                self.logger.error(f"  Unsupported format: {format}")
                return False
            
            if success:
                # Update catalog
                self._update_catalog(
                    symbol, data_type, frequency, data, 
                    file_path, format, source, quality_metrics
                )
                
                # Update cache
                cache_key = f"{symbol}_{data_type}_{frequency}"
                self.data_cache[cache_key] = data.tail(1000)  # Keep recent data in cache
                
                self.logger.info(f"  Stored {len(data)} records for {symbol}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"  Error storing data for {symbol}: {e}")
            return False
    
    def _store_parquet(self, data: pd.DataFrame, file_path: str) -> bool:
        """Store data in Parquet format."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert to PyArrow table for better performance
            table = pa.Table.from_pandas(data)
            
            # Write with compression
            pq.write_table(
                table, file_path, 
                compression=self.config['compression'],
                use_dictionary=True
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"  Parquet storage error: {e}")
            return False
    
    def _store_hdf5(self, data: pd.DataFrame, file_path: str, symbol: str) -> bool:
        """Store data in HDF5 format."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Store with compression
            with h5py.File(file_path, 'w') as f:
                # Convert DataFrame to HDF5 format
                for column in data.columns:
                    f.create_dataset(
                        column, 
                        data=data[column].values,
                        compression='gzip',
                        compression_opts=9
                    )
                
                # Store metadata
                f.attrs['symbol'] = symbol
                f.attrs['start_date'] = str(data.index.min())
                f.attrs['end_date'] = str(data.index.max())
                f.attrs['record_count'] = len(data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"  HDF5 storage error: {e}")
            return False
    
    def retrieve_historical_data(self, symbol: str, data_type: str = 'ohlcv',
                               frequency: str = '1D', start_date: datetime = None,
                               end_date: datetime = None) -> pd.DataFrame:
        """Retrieve historical data."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{data_type}_{frequency}"
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                if self._data_covers_range(cached_data, start_date, end_date):
                    return self._filter_date_range(cached_data, start_date, end_date)
            
            # Query catalog for file information
            file_info = self._get_file_info(symbol, data_type, frequency)
            
            if not file_info:
                self.logger.warning(f" ️ No data found for {symbol} {data_type} {frequency}")
                return pd.DataFrame()
            
            # Load data from file
            data = self._load_from_file(file_info)
            
            if not data.empty:
                # Update cache
                self.data_cache[cache_key] = data.tail(1000)
                
                # Filter by date range
                filtered_data = self._filter_date_range(data, start_date, end_date)
                
                self.logger.info(f"  Retrieved {len(filtered_data)} records for {symbol}")
                return filtered_data
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"  Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_file_info(self, symbol: str, data_type: str, frequency: str) -> Dict[str, Any]:
        """Get file information from catalog."""
        try:
            session = self.Session()
            
            result = session.execute(
                "SELECT file_path, file_format, start_date, end_date, record_count "
                "FROM data_catalog "
                "WHERE symbol = :symbol AND data_type = :data_type AND frequency = :frequency "
                "ORDER BY updated_at DESC LIMIT 1",
                {
                    'symbol': symbol,
                    'data_type': data_type,
                    'frequency': frequency
                }
            ).fetchone()
            
            session.close()
            
            if result:
                return {
                    'file_path': result[0],
                    'file_format': result[1],
                    'start_date': result[2],
                    'end_date': result[3],
                    'record_count': result[4]
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"  Error querying catalog: {e}")
            return {}
    
    def _load_from_file(self, file_info: Dict[str, Any]) -> pd.DataFrame:
        """Load data from file."""
        try:
            file_path = file_info['file_path']
            file_format = file_info['file_format']
            
            if not os.path.exists(file_path):
                self.logger.error(f"  File not found: {file_path}")
                return pd.DataFrame()
            
            if file_format == 'parquet':
                return pd.read_parquet(file_path)
            elif file_format == 'hdf5':
                return self._load_hdf5(file_path)
            else:
                self.logger.error(f"  Unsupported file format: {file_format}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"  Error loading file: {e}")
            return pd.DataFrame()
    
    def _load_hdf5(self, file_path: str) -> pd.DataFrame:
        """Load data from HDF5 file."""
        try:
            data_dict = {}
            
            with h5py.File(file_path, 'r') as f:
                # Load all datasets
                for key in f.keys():
                    data_dict[key] = f[key][:]
            
            return pd.DataFrame(data_dict)
            
        except Exception as e:
            self.logger.error(f"  HDF5 loading error: {e}")
            return pd.DataFrame()
    
    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality."""
        try:
            total_records = len(data)
            
            # Check for missing data
            missing_data_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            
            # Check for duplicates
            duplicate_records = data.duplicated().sum()
            
            # Check for outliers (simple z-score method)
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            outliers_count = 0
            
            for column in numeric_columns:
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outliers_count += (z_scores > 3).sum()
            
            # Check for gaps in time series (if datetime index)
            gaps_count = 0
            if isinstance(data.index, pd.DatetimeIndex):
                expected_freq = pd.infer_freq(data.index)
                if expected_freq:
                    expected_range = pd.date_range(
                        start=data.index.min(),
                        end=data.index.max(),
                        freq=expected_freq
                    )
                    gaps_count = len(expected_range) - len(data.index)
            
            # Calculate quality score
            quality_score = 100.0
            quality_score -= min(missing_data_pct, 50)  # Max 50 point deduction
            quality_score -= min((duplicate_records / total_records) * 100, 30)  # Max 30 point deduction
            quality_score -= min((outliers_count / total_records) * 100, 20)  # Max 20 point deduction
            
            return {
                'missing_data_pct': missing_data_pct,
                'duplicate_records': duplicate_records,
                'outliers_count': outliers_count,
                'gaps_count': gaps_count,
                'quality_score': max(quality_score, 0)
            }
            
        except Exception as e:
            self.logger.error(f"  Data validation error: {e}")
            return {
                'missing_data_pct': 0,
                'duplicate_records': 0,
                'outliers_count': 0,
                'gaps_count': 0,
                'quality_score': 50  # Default moderate score
            }
    
    def _update_catalog(self, symbol: str, data_type: str, frequency: str,
                       data: pd.DataFrame, file_path: str, file_format: str,
                       source: str, quality_metrics: Dict[str, Any]):
        """Update data catalog."""
        try:
            session = self.Session()
            
            # Calculate file size
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Calculate checksum (simplified)
            checksum = str(hash(str(data.values.tobytes())))[:16]
            
            # Insert or update catalog entry
            session.execute(
                "INSERT OR REPLACE INTO data_catalog "
                "(symbol, data_type, frequency, start_date, end_date, record_count, "
                "file_path, file_format, file_size, checksum, created_at, updated_at, "
                "source, quality_score) VALUES "
                "(:symbol, :data_type, :frequency, :start_date, :end_date, :record_count, "
                ":file_path, :file_format, :file_size, :checksum, :created_at, :updated_at, "
                ":source, :quality_score)",
                {
                    'symbol': symbol,
                    'data_type': data_type,
                    'frequency': frequency,
                    'start_date': data.index.min(),
                    'end_date': data.index.max(),
                    'record_count': len(data),
                    'file_path': file_path,
                    'file_format': file_format,
                    'file_size': file_size,
                    'checksum': checksum,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'source': source,
                    'quality_score': quality_metrics['quality_score']
                }
            )
            
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"  Catalog update error: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
    
    def _generate_file_path(self, symbol: str, data_type: str, 
                          frequency: str, format: str) -> str:
        """Generate file path for data storage."""
        if format == 'parquet':
            base_dir = self.config['parquet_directory']
            extension = '.parquet'
        elif format == 'hdf5':
            base_dir = self.config['hdf5_directory']
            extension = '.h5'
        else:
            base_dir = self.config['data_directory']
            extension = '.data'
        
        filename = f"{symbol}_{data_type}_{frequency}{extension}"
        return os.path.join(base_dir, filename)
    
    def _data_covers_range(self, data: pd.DataFrame, start_date: datetime = None,
                          end_date: datetime = None) -> bool:
        """Check if cached data covers the requested range."""
        if data.empty:
            return False
        
        data_start = data.index.min()
        data_end = data.index.max()
        
        if start_date and data_start > start_date:
            return False
        
        if end_date and data_end < end_date:
            return False
        
        return True
    
    def _filter_date_range(self, data: pd.DataFrame, start_date: datetime = None,
                          end_date: datetime = None) -> pd.DataFrame:
        """Filter data by date range."""
        if data.empty:
            return data
        
        if start_date:
            data = data[data.index >= start_date]
        
        if end_date:
            data = data[data.index <= end_date]
        
        return data
    
    def get_available_data(self) -> pd.DataFrame:
        """Get catalog of available data."""
        try:
            session = self.Session()
            
            result = session.execute(
                "SELECT symbol, data_type, frequency, start_date, end_date, "
                "record_count, source, quality_score, updated_at "
                "FROM data_catalog ORDER BY symbol, data_type, frequency"
            ).fetchall()
            
            session.close()
            
            if result:
                return pd.DataFrame(result, columns=[
                    'Symbol', 'Data Type', 'Frequency', 'Start Date', 'End Date',
                    'Record Count', 'Source', 'Quality Score', 'Updated'
                ])
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"  Error getting catalog: {e}")
            return pd.DataFrame()
    
    def archive_old_data(self, days_old: int = None):
        """Archive old data files."""
        try:
            days_old = days_old or self.config['archive_after_days']
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            session = self.Session()
            
            # Find old data files
            old_files = session.execute(
                "SELECT file_path FROM data_catalog WHERE updated_at < :cutoff",
                {'cutoff': cutoff_date}
            ).fetchall()
            
            archived_count = 0
            
            for (file_path,) in old_files:
                if os.path.exists(file_path):
                    # Create archive path
                    archive_path = os.path.join(
                        self.config['archive_directory'],
                        os.path.basename(file_path) + '.zip'
                    )
                    
                    # Compress and move to archive
                    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        zipf.write(file_path, os.path.basename(file_path))
                    
                    # Remove original file
                    os.remove(file_path)
                    
                    # Update catalog with archive path
                    session.execute(
                        "UPDATE data_catalog SET file_path = :archive_path "
                        "WHERE file_path = :original_path",
                        {'archive_path': archive_path, 'original_path': file_path}
                    )
                    
                    archived_count += 1
            
            session.commit()
            session.close()
            
            self.logger.info(f"  Archived {archived_count} old data files")
            
        except Exception as e:
            self.logger.error(f"  Archiving error: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
    
    def cleanup_cache(self):
        """Clean up in-memory cache."""
        cache_size = len(self.data_cache)
        
        if cache_size > self.config['cache_size']:
            # Remove oldest entries
            items_to_remove = cache_size - self.config['cache_size']
            keys_to_remove = list(self.data_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self.data_cache[key]
            
            self.logger.info(f"  Cleaned {items_to_remove} items from cache")

# Example usage and testing
if __name__ == "__main__":
    # Initialize historical data manager
    manager = HistoricalDataManager()
    
    # Generate sample OHLCV data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='1D')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, len(dates)),
        'high': np.random.uniform(150, 250, len(dates)),
        'low': np.random.uniform(50, 150, len(dates)),
        'close': np.random.uniform(100, 200, len(dates)),
        'volume': np.random.randint(10000, 1000000, len(dates))
    }, index=dates)
    
    # Store sample data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        print(f"  Storing data for {symbol}...")
        success = manager.store_historical_data(
            symbol=symbol,
            data=sample_data,
            data_type='ohlcv',
            frequency='1D',
            source='demo',
            format='parquet'
        )
        print(f"  Storage success: {success}")
    
    # Retrieve data
    print("\n  Retrieving AAPL data...")
    retrieved_data = manager.retrieve_historical_data(
        symbol='AAPL',
        data_type='ohlcv',
        frequency='1D',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 30)
    )
    
    print(f"  Retrieved {len(retrieved_data)} records")
    print(retrieved_data.head())
    
    # Show available data catalog
    print("\n  Available data catalog:")
    catalog = manager.get_available_data()
    print(catalog)
    
    # Archive old data (demo with 0 days for testing)
    print("\n ️ Archiving demo...")
    manager.archive_old_data(days_old=0)