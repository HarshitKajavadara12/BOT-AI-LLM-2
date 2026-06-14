#!/usr/bin/env python3
"""
QUANTUM-FORGE Benchmark Suite
Comprehensive performance testing and system validation
"""

import asyncio
import time
import statistics
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis
import psycopg2
import requests
import json
import logging
from datetime import datetime, timedelta
import subprocess
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    test_name: str
    duration: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    error_rate: float
    memory_usage: float
    cpu_usage: float
    metadata: Dict[str, Any]

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_io': [],
            'network_io': []
        }
        self.monitoring = False
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
    
    def _monitor_loop(self):
        """Monitoring loop"""
        try:
            import psutil
            while self.monitoring:
                self.metrics['cpu_percent'].append(psutil.cpu_percent())
                self.metrics['memory_percent'].append(psutil.virtual_memory().percent)
                time.sleep(0.1)
        except ImportError:
            logger.warning("psutil not available, skipping system monitoring")
    
    def get_stats(self) -> Dict[str, float]:
        """Get monitoring statistics"""
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[f"{metric}_avg"] = statistics.mean(values)
                stats[f"{metric}_max"] = max(values)
                stats[f"{metric}_min"] = min(values)
        return stats

class DatabaseBenchmark:
    """Database performance benchmarks"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    async def benchmark_write_performance(self, num_records: int = 10000) -> BenchmarkResult:
        """Benchmark database write performance"""
        logger.info(f"Benchmarking database writes with {num_records} records")
        
        latencies = []
        errors = 0
        
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Prepare test data
            test_data = [
                (
                    datetime.now() - timedelta(seconds=i),
                    f"TEST{i % 100}",
                    100.0 + (i % 1000) * 0.01,
                    1000 + (i % 10000)
                )
                for i in range(num_records)
            ]
            
            start_time = time.time()
            
            for timestamp, symbol, price, volume in test_data:
                record_start = time.perf_counter()
                try:
                    cursor.execute(
                        "INSERT INTO market_data (timestamp, symbol, price, volume) VALUES (%s, %s, %s, %s)",
                        (timestamp, symbol, price, volume)
                    )
                    latencies.append((time.perf_counter() - record_start) * 1_000_000)  # microseconds
                except Exception as e:
                    errors += 1
                    logger.debug(f"Write error: {e}")
            
            conn.commit()
            duration = time.time() - start_time
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database benchmark failed: {e}")
            return BenchmarkResult(
                test_name="database_write",
                duration=0,
                throughput=0,
                latency_p50=0,
                latency_p95=0,
                latency_p99=0,
                error_rate=1.0,
                memory_usage=0,
                cpu_usage=0,
                metadata={"error": str(e)}
            )
        
        return BenchmarkResult(
            test_name="database_write",
            duration=duration,
            throughput=num_records / duration,
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            error_rate=errors / num_records,
            memory_usage=0,  # Would need psutil
            cpu_usage=0,     # Would need psutil
            metadata={
                "num_records": num_records,
                "total_errors": errors
            }
        )
    
    async def benchmark_read_performance(self, num_queries: int = 1000) -> BenchmarkResult:
        """Benchmark database read performance"""
        logger.info(f"Benchmarking database reads with {num_queries} queries")
        
        latencies = []
        errors = 0
        
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            start_time = time.time()
            
            for i in range(num_queries):
                query_start = time.perf_counter()
                try:
                    cursor.execute(
                        "SELECT * FROM market_data WHERE symbol = %s ORDER BY timestamp DESC LIMIT 100",
                        (f"TEST{i % 100}",)
                    )
                    cursor.fetchall()
                    latencies.append((time.perf_counter() - query_start) * 1_000_000)  # microseconds
                except Exception as e:
                    errors += 1
                    logger.debug(f"Read error: {e}")
            
            duration = time.time() - start_time
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database read benchmark failed: {e}")
            return BenchmarkResult(
                test_name="database_read",
                duration=0,
                throughput=0,
                latency_p50=0,
                latency_p95=0,
                latency_p99=0,
                error_rate=1.0,
                memory_usage=0,
                cpu_usage=0,
                metadata={"error": str(e)}
            )
        
        return BenchmarkResult(
            test_name="database_read",
            duration=duration,
            throughput=num_queries / duration,
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            error_rate=errors / num_queries,
            memory_usage=0,
            cpu_usage=0,
            metadata={
                "num_queries": num_queries,
                "total_errors": errors
            }
        )

class CacheBenchmark:
    """Cache (Redis) performance benchmarks"""
    
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
    
    async def benchmark_cache_operations(self, num_operations: int = 10000) -> BenchmarkResult:
        """Benchmark cache read/write operations"""
        logger.info(f"Benchmarking cache operations with {num_operations} operations")
        
        latencies = []
        errors = 0
        
        start_time = time.time()
        
        # Mixed read/write operations
        for i in range(num_operations):
            op_start = time.perf_counter()
            try:
                if i % 4 == 0:  # 25% writes
                    key = f"benchmark:write:{i}"
                    value = {
                        'timestamp': time.time(),
                        'symbol': f'TEST{i % 100}',
                        'price': 100.0 + (i % 1000) * 0.01,
                        'volume': 1000 + (i % 10000)
                    }
                    self.redis_client.hset(key, mapping=value)
                else:  # 75% reads
                    key = f"benchmark:write:{max(0, i - (i % 4))}"
                    self.redis_client.hgetall(key)
                
                latencies.append((time.perf_counter() - op_start) * 1_000_000)  # microseconds
            except Exception as e:
                errors += 1
                logger.debug(f"Cache operation error: {e}")
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            test_name="cache_operations",
            duration=duration,
            throughput=num_operations / duration,
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            error_rate=errors / num_operations,
            memory_usage=0,
            cpu_usage=0,
            metadata={
                "num_operations": num_operations,
                "total_errors": errors,
                "read_write_ratio": "3:1"
            }
        )

class ComputeBenchmark:
    """Computational performance benchmarks"""
    
    async def benchmark_numpy_operations(self, matrix_size: int = 1000, num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark NumPy matrix operations"""
        logger.info(f"Benchmarking NumPy operations with {matrix_size}x{matrix_size} matrices")
        
        latencies = []
        
        start_time = time.time()
        
        for i in range(num_iterations):
            op_start = time.perf_counter()
            
            # Generate deterministic matrices for reproducible benchmarking
            i = np.arange(matrix_size).reshape(-1, 1)
            j = np.arange(matrix_size).reshape(1, -1)
            A = np.sin((i + 1) * 0.001 + (j + 1) * 0.001)
            B = np.cos((i + 1) * 0.001 - (j + 1) * 0.001)
            
            # Matrix multiplication
            C = np.dot(A, B)
            
            # Eigenvalue decomposition (smaller matrix for speed)
            if matrix_size <= 100:
                eigenvals = np.linalg.eigvals(A[:100, :100])
            
            # Statistical operations
            mean_val = np.mean(C)
            std_val = np.std(C)
            
            latencies.append((time.perf_counter() - op_start) * 1000)  # milliseconds
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            test_name="numpy_operations",
            duration=duration,
            throughput=num_iterations / duration,
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            error_rate=0.0,
            memory_usage=0,
            cpu_usage=0,
            metadata={
                "matrix_size": matrix_size,
                "num_iterations": num_iterations,
                "operations": ["matrix_mult", "eigenvals", "statistics"]
            }
        )
    
    async def benchmark_pandas_operations(self, num_rows: int = 100000, num_iterations: int = 50) -> BenchmarkResult:
        """Benchmark Pandas DataFrame operations"""
        logger.info(f"Benchmarking Pandas operations with {num_rows} rows")
        
        latencies = []
        
        # Generate test data
        # Deterministic test data (reproducible)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=num_rows, freq='1s'),
            'symbol': [f'STOCK{i % 100}' for i in range(num_rows)],
            'price': np.linspace(50, 200, num_rows),
            'volume': (np.arange(num_rows) % 1000) + 100,
            'returns': np.sin(np.linspace(0, 2 * np.pi, num_rows)) * 0.0005
        })
        
        start_time = time.time()
        
        for i in range(num_iterations):
            op_start = time.perf_counter()
            
            # Complex operations
            grouped = df.groupby('symbol').agg({
                'price': ['mean', 'std', 'count'],
                'volume': 'sum',
                'returns': lambda x: x.rolling(20).mean().iloc[-1]
            })
            
            # Filtering and sorting
            filtered = df[df['volume'] > df['volume'].quantile(0.8)]
            sorted_df = filtered.sort_values(['symbol', 'timestamp'])
            
            # Rolling operations
            df['price_ma20'] = df.groupby('symbol')['price'].rolling(20).mean().reset_index(0, drop=True)
            
            latencies.append((time.perf_counter() - op_start) * 1000)  # milliseconds
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            test_name="pandas_operations",
            duration=duration,
            throughput=num_iterations / duration,
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            error_rate=0.0,
            memory_usage=0,
            cpu_usage=0,
            metadata={
                "num_rows": num_rows,
                "num_iterations": num_iterations,
                "operations": ["groupby", "agg", "filter", "sort", "rolling"]
            }
        )

class NetworkBenchmark:
    """Network and API performance benchmarks"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def benchmark_api_endpoints(self, num_requests: int = 1000) -> BenchmarkResult:
        """Benchmark API endpoint performance"""
        logger.info(f"Benchmarking API endpoints with {num_requests} requests")
        
        latencies = []
        errors = 0
        
        endpoints = [
            "/health",
            "/api/v1/positions",
            "/api/v1/market-data/BTCUSDT",
            "/api/v1/portfolio/summary"
        ]
        
        start_time = time.time()
        
        for i in range(num_requests):
            endpoint = endpoints[i % len(endpoints)]
            request_start = time.perf_counter()
            
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code != 200:
                    errors += 1
                latencies.append((time.perf_counter() - request_start) * 1000)  # milliseconds
            except Exception as e:
                errors += 1
                logger.debug(f"API request error: {e}")
                latencies.append(5000)  # 5 second timeout
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            test_name="api_endpoints",
            duration=duration,
            throughput=num_requests / duration,
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            error_rate=errors / num_requests,
            memory_usage=0,
            cpu_usage=0,
            metadata={
                "num_requests": num_requests,
                "total_errors": errors,
                "endpoints": endpoints
            }
        )

class TradingBenchmark:
    """Trading-specific performance benchmarks"""
    
    async def benchmark_order_processing(self, num_orders: int = 10000) -> BenchmarkResult:
        """Benchmark order processing latency"""
        logger.info(f"Benchmarking order processing with {num_orders} orders")
        
        latencies = []
        
        start_time = time.time()
        
        for i in range(num_orders):
            order_start = time.perf_counter()
            
            # Simulate order processing
            order = {
                'symbol': f'STOCK{i % 100}',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': 100 + (i % 1000),
                'price': 100.0 + (i % 100) * 0.01,
                'timestamp': time.time()
            }
            
            # Simulate validation, risk checks, etc.
            # In real implementation, this would call actual order processing
            # Deterministic validation time (milliseconds)
            validation_time = 0.5
            time.sleep(validation_time / 1000)
            
            latencies.append((time.perf_counter() - order_start) * 1_000_000)  # microseconds
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            test_name="order_processing",
            duration=duration,
            throughput=num_orders / duration,
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            error_rate=0.0,
            memory_usage=0,
            cpu_usage=0,
            metadata={
                "num_orders": num_orders,
                "target_latency_us": 100,  # 100 microseconds target
                "orders_per_second": num_orders / duration
            }
        )
    
    async def benchmark_portfolio_calculation(self, num_positions: int = 1000, num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark portfolio calculations"""
        logger.info(f"Benchmarking portfolio calculations with {num_positions} positions")
        
        latencies = []
        
        # Generate test portfolio
        # Deterministic test portfolio
        positions = pd.DataFrame({
            'symbol': [f'STOCK{i}' for i in range(num_positions)],
            'quantity': (np.arange(num_positions) % 200) - 100,
            'price': np.linspace(50, 200, num_positions),
            'sector': [f'SECTOR{i % 11}' for i in range(num_positions)],  # 11 sectors
        })

        # Deterministic correlation matrix (small constant correlation with identity on diagonal)
        correlation_matrix = np.full((num_positions, num_positions), 0.15)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        start_time = time.time()
        
        for i in range(num_iterations):
            calc_start = time.perf_counter()
            
            # Portfolio value calculation
            positions['market_value'] = positions['quantity'] * positions['price']
            total_value = positions['market_value'].sum()
            
            # Risk calculations
            sector_exposure = positions.groupby('sector')['market_value'].sum()
            max_sector_exposure = sector_exposure.abs().max()
            
            # VaR calculation (simplified)
            # Deterministic returns vector for reproducible VaR calculation
            returns = np.linspace(-0.001, 0.001, num_positions)
            portfolio_return = np.sum(positions['market_value'] * returns) / total_value
            var_95 = np.percentile([portfolio_return] * 1000, 5)  # Simplified VaR
            
            latencies.append((time.perf_counter() - calc_start) * 1000)  # milliseconds
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            test_name="portfolio_calculation",
            duration=duration,
            throughput=num_iterations / duration,
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            error_rate=0.0,
            memory_usage=0,
            cpu_usage=0,
            metadata={
                "num_positions": num_positions,
                "num_iterations": num_iterations,
                "calculations": ["market_value", "sector_exposure", "var"]
            }
        )

class BenchmarkSuite:
    """Main benchmark suite orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.monitor = PerformanceMonitor()
    
    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark suites"""
        logger.info("Starting QUANTUM-FORGE benchmark suite...")
        
        self.monitor.start_monitoring()
        
        try:
            # Database benchmarks
            if self.config.get('database', {}).get('enabled', True):
                db_bench = DatabaseBenchmark(self.config['database']['connection_string'])
                self.results.append(await db_bench.benchmark_write_performance())
                self.results.append(await db_bench.benchmark_read_performance())
            
            # Cache benchmarks
            if self.config.get('cache', {}).get('enabled', True):
                cache_bench = CacheBenchmark(self.config['cache']['url'])
                self.results.append(await cache_bench.benchmark_cache_operations())
            
            # Compute benchmarks
            compute_bench = ComputeBenchmark()
            self.results.append(await compute_bench.benchmark_numpy_operations())
            self.results.append(await compute_bench.benchmark_pandas_operations())
            
            # Network benchmarks
            if self.config.get('api', {}).get('enabled', True):
                network_bench = NetworkBenchmark(self.config['api']['base_url'])
                self.results.append(await network_bench.benchmark_api_endpoints())
            
            # Trading benchmarks
            trading_bench = TradingBenchmark()
            self.results.append(await trading_bench.benchmark_order_processing())
            self.results.append(await trading_bench.benchmark_portfolio_calculation())
            
        finally:
            self.monitor.stop_monitoring()
        
        logger.info(f"Completed {len(self.results)} benchmark tests")
        return self.results
    
    def generate_report(self, output_path: str = "benchmark_report.html"):
        """Generate comprehensive benchmark report"""
        logger.info(f"Generating benchmark report: {output_path}")
        
        # Create DataFrame from results
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Generate visualizations
        self._create_visualizations(df)
        
        # Generate HTML report
        html_content = self._generate_html_report(df)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Benchmark report saved to: {output_path}")
    
    def _create_visualizations(self, df: pd.DataFrame):
        """Create benchmark visualizations"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('QUANTUM-FORGE Benchmark Results', fontsize=16, fontweight='bold')
        
        # Throughput comparison
        axes[0, 0].bar(df['test_name'], df['throughput'])
        axes[0, 0].set_title('Throughput (ops/sec)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Latency percentiles
        latency_data = df[['test_name', 'latency_p50', 'latency_p95', 'latency_p99']]
        latency_data.set_index('test_name').plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Latency Percentiles')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Error rates
        axes[0, 2].bar(df['test_name'], df['error_rate'] * 100)
        axes[0, 2].set_title('Error Rate (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Duration comparison
        axes[1, 0].bar(df['test_name'], df['duration'])
        axes[1, 0].set_title('Test Duration (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Performance heatmap
        performance_metrics = df[['test_name', 'throughput', 'latency_p99', 'error_rate']].set_index('test_name')
        # Normalize for heatmap
        performance_normalized = (performance_metrics - performance_metrics.min()) / (performance_metrics.max() - performance_metrics.min())
        sns.heatmap(performance_normalized.T, annot=True, cmap='RdYlGn_r', ax=axes[1, 1])
        axes[1, 1].set_title('Performance Heatmap (Normalized)')
        
        # Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""
        BENCHMARK SUMMARY
        
        Total Tests: {len(df)}
        Average Throughput: {df['throughput'].mean():.2f} ops/sec
        Average P99 Latency: {df['latency_p99'].mean():.2f}
        Total Error Rate: {(df['error_rate'].mean() * 100):.2f}%
        Total Duration: {df['duration'].sum():.2f} seconds
        
        PERFORMANCE GRADES:
        Throughput: {'A' if df['throughput'].mean() > 1000 else 'B' if df['throughput'].mean() > 500 else 'C'}
        Latency: {'A' if df['latency_p99'].mean() < 100 else 'B' if df['latency_p99'].mean() < 1000 else 'C'}
        Reliability: {'A' if df['error_rate'].mean() < 0.01 else 'B' if df['error_rate'].mean() < 0.05 else 'C'}
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, df: pd.DataFrame) -> str:
        """Generate HTML benchmark report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>QUANTUM-FORGE Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; text-align: center; }
                .section { background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #667eea; color: white; }
                .metric-good { color: #27ae60; font-weight: bold; }
                .metric-warning { color: #f39c12; font-weight: bold; }
                .metric-bad { color: #e74c3c; font-weight: bold; }
                .chart { text-align: center; margin: 20px 0; }
                .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
                .summary-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               color: white; padding: 20px; border-radius: 10px; text-align: center; }
                .summary-card h3 { margin: 0 0 10px 0; }
                .summary-card .value { font-size: 2em; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>QUANTUM-FORGE Benchmark Report</h1>
                <p>Performance Analysis & System Validation</p>
                <p>Generated: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary">
                    <div class="summary-card">
                        <h3>Total Tests</h3>
                        <div class="value">{total_tests}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Avg Throughput</h3>
                        <div class="value">{avg_throughput:.0f}</div>
                        <div>ops/sec</div>
                    </div>
                    <div class="summary-card">
                        <h3>P99 Latency</h3>
                        <div class="value">{avg_p99_latency:.1f}</div>
                        <div>ms</div>
                    </div>
                    <div class="summary-card">
                        <h3>Error Rate</h3>
                        <div class="value">{avg_error_rate:.2f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Visualization</h2>
                <div class="chart">
                    <img src="benchmark_results.png" alt="Benchmark Results" style="max-width: 100%; height: auto;">
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                {detailed_table}
            </div>
            
            <div class="section">
                <h2>Performance Analysis</h2>
                {performance_analysis}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {recommendations}
            </div>
        </body>
        </html>
        """
        
        # Generate detailed table
        detailed_table = df.to_html(
            classes='benchmark-table',
            table_id='results-table',
            escape=False,
            formatters={
                'duration': lambda x: f'{x:.3f}s',
                'throughput': lambda x: f'{x:.2f}',
                'latency_p50': lambda x: f'{x:.2f}',
                'latency_p95': lambda x: f'{x:.2f}',
                'latency_p99': lambda x: f'{x:.2f}',
                'error_rate': lambda x: f'{x*100:.2f}%'
            }
        )
        
        # Generate performance analysis
        performance_analysis = self._generate_performance_analysis(df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df)
        
        return html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_tests=len(df),
            avg_throughput=df['throughput'].mean(),
            avg_p99_latency=df['latency_p99'].mean(),
            avg_error_rate=df['error_rate'].mean() * 100,
            detailed_table=detailed_table,
            performance_analysis=performance_analysis,
            recommendations=recommendations
        )
    
    def _generate_performance_analysis(self, df: pd.DataFrame) -> str:
        """Generate performance analysis text"""
        analysis_points = []
        
        # Throughput analysis
        max_throughput = df.loc[df['throughput'].idxmax()]
        min_throughput = df.loc[df['throughput'].idxmin()]
        analysis_points.append(f"<li><strong>Highest throughput:</strong> {max_throughput['test_name']} at {max_throughput['throughput']:.2f} ops/sec</li>")
        analysis_points.append(f"<li><strong>Lowest throughput:</strong> {min_throughput['test_name']} at {min_throughput['throughput']:.2f} ops/sec</li>")
        
        # Latency analysis
        high_latency_tests = df[df['latency_p99'] > df['latency_p99'].quantile(0.75)]
        if not high_latency_tests.empty:
            analysis_points.append(f"<li><strong>High latency tests:</strong> {', '.join(high_latency_tests['test_name'].tolist())}</li>")
        
        # Error analysis
        error_tests = df[df['error_rate'] > 0]
        if not error_tests.empty:
            analysis_points.append(f"<li><strong>Tests with errors:</strong> {', '.join(error_tests['test_name'].tolist())}</li>")
        else:
            analysis_points.append("<li><strong>All tests completed without errors</strong>  </li>")
        
        return f"<ul>{''.join(analysis_points)}</ul>"
    
    def _generate_recommendations(self, df: pd.DataFrame) -> str:
        """Generate performance recommendations"""
        recommendations = []
        
        # Database recommendations
        db_tests = df[df['test_name'].str.contains('database')]
        if not db_tests.empty:
            avg_db_latency = db_tests['latency_p99'].mean()
            if avg_db_latency > 1000:  # > 1ms
                recommendations.append("<li>Consider database query optimization and indexing improvements</li>")
            if db_tests['throughput'].mean() < 1000:
                recommendations.append("<li>Database connection pooling and batch operations recommended</li>")
        
        # Cache recommendations
        cache_tests = df[df['test_name'].str.contains('cache')]
        if not cache_tests.empty:
            if cache_tests['latency_p99'].iloc[0] > 100:  # > 0.1ms
                recommendations.append("<li>Redis configuration tuning may improve cache performance</li>")
        
        # API recommendations
        api_tests = df[df['test_name'].str.contains('api')]
        if not api_tests.empty:
            if api_tests['error_rate'].iloc[0] > 0.01:  # > 1%
                recommendations.append("<li>API error handling and timeout configurations need review</li>")
        
        # Trading recommendations
        trading_tests = df[df['test_name'].str.contains('order')]
        if not trading_tests.empty:
            if trading_tests['latency_p99'].iloc[0] > 1000:  # > 1ms
                recommendations.append("<li>Order processing pipeline optimization critical for HFT performance</li>")
        
        if not recommendations:
            recommendations.append("<li>System performance is within acceptable parameters  </li>")
            recommendations.append("<li>Continue monitoring in production environment</li>")
        
        return f"<ul>{''.join(recommendations)}</ul>"

async def main():
    """Main benchmark execution"""
    
    # Configuration
    config = {
        'database': {
            'enabled': True,
            'connection_string': 'postgresql://quantum_user:quantum_secure_password_2024@localhost:5432/quantum_forge'
        },
        'cache': {
            'enabled': True,
            'url': 'redis://localhost:6379/0'
        },
        'api': {
            'enabled': True,
            'base_url': 'http://localhost:8000'
        }
    }
    
    # Create benchmark suite
    benchmark_suite = BenchmarkSuite(config)
    
    # Run all benchmarks
    try:
        results = await benchmark_suite.run_all_benchmarks()
        
        # Generate report
        benchmark_suite.generate_report()
        
        # Print summary
        print("\n" + "="*80)
        print("QUANTUM-FORGE BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        for result in results:
            print(f"\n{result.test_name.upper()}:")
            print(f"  Throughput: {result.throughput:.2f} ops/sec")
            print(f"  Latency P99: {result.latency_p99:.2f} ms")
            print(f"  Error Rate: {result.error_rate*100:.2f}%")
            print(f"  Duration: {result.duration:.3f}s")
        
        print(f"\nDetailed report saved to: benchmark_report.html")
        print(f"Visualizations saved to: benchmark_results.png")
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())