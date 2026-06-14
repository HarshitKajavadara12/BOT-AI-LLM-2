"""
Alternative Data Loader
======================

Comprehensive alternative data ingestion system for non-traditional market data sources.
Handles social media sentiment, satellite imagery, news feeds, economic indicators,
and other alternative data sources for quantitative analysis.

Features:
- Multi-source alternative data ingestion
- Social media sentiment analysis (Twitter, Reddit, Discord)
- News feed processing with NLP sentiment extraction
- Economic indicator and macro data integration
- Satellite and geospatial data processing
- ESG (Environmental, Social, Governance) data feeds
- Real-time and historical data synchronization
- Data normalization and standardization
- Quality scoring and reliability metrics

Author: Quantum Forge Data Team
Date: November 2025
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import json
import requests
import tweepy
import praw  # Reddit API
import feedparser
import yfinance as yf
from textblob import TextBlob
import re
from urllib.parse import urlparse
import schedule
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Types of alternative data sources."""
    SOCIAL_MEDIA = "social_media"
    NEWS_FEED = "news_feed"
    ECONOMIC_INDICATOR = "economic_indicator"
    SATELLITE = "satellite"
    ESG = "esg"
    CRYPTO_METRICS = "crypto_metrics"
    WEB_SCRAPING = "web_scraping"
    API_FEED = "api_feed"

class SentimentPolarity(Enum):
    """Sentiment polarity classifications."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

@dataclass
class AlternativeDataPoint:
    """Standardized alternative data point."""
    source: str
    source_type: DataSourceType
    symbol: Optional[str]
    timestamp: datetime
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processed fields
    sentiment_score: Optional[float] = None
    sentiment_polarity: Optional[SentimentPolarity] = None
    confidence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    # Technical fields
    data_quality_score: Optional[float] = None
    processing_timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source': self.source,
            'source_type': self.source_type.value,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'metadata': self.metadata,
            'sentiment_score': self.sentiment_score,
            'sentiment_polarity': self.sentiment_polarity.value if self.sentiment_polarity else None,
            'confidence_score': self.confidence_score,
            'relevance_score': self.relevance_score,
            'data_quality_score': self.data_quality_score,
            'processing_timestamp': self.processing_timestamp.isoformat(),
            'raw_data': self.raw_data
        }

@dataclass
class DataSourceConfig:
    """Configuration for alternative data sources."""
    name: str
    source_type: DataSourceType
    enabled: bool = True
    
    # API Configuration
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_endpoint: Optional[str] = None
    
    # Processing Configuration
    symbols: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    update_frequency: int = 300  # seconds
    max_items_per_fetch: int = 100
    
    # Quality filters
    min_confidence: float = 0.3
    min_relevance: float = 0.5
    language_filter: List[str] = field(default_factory=lambda: ['en'])

class SentimentAnalyzer:
    """Analyze sentiment from text content."""
    
    def __init__(self):
        # Market-specific sentiment lexicon
        self.market_lexicon = {
            # Positive terms
            'bullish': 0.8, 'rally': 0.7, 'surge': 0.8, 'moon': 0.9, 'pump': 0.6,
            'breakout': 0.7, 'uptrend': 0.6, 'support': 0.4, 'buy': 0.5, 'long': 0.4,
            'hodl': 0.6, 'diamond hands': 0.8, 'to the moon': 0.9, 'green': 0.5,
            
            # Negative terms
            'bearish': -0.8, 'crash': -0.9, 'dump': -0.7, 'rekt': -0.8, 'fud': -0.6,
            'breakdown': -0.7, 'downtrend': -0.6, 'resistance': -0.4, 'sell': -0.5,
            'short': -0.4, 'panic': -0.8, 'red': -0.5, 'liquidation': -0.9,
            
            # Neutral/Uncertainty
            'sideways': 0.0, 'consolidation': 0.0, 'range': 0.0, 'dyor': 0.0
        }
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.
        Returns: (sentiment_score, confidence_score)
        """
        if not text:
            return 0.0, 0.0
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Use TextBlob for basic sentiment
        blob = TextBlob(cleaned_text)
        textblob_sentiment = blob.sentiment.polarity
        textblob_confidence = abs(blob.sentiment.polarity)
        
        # Apply market-specific lexicon
        market_sentiment = self._calculate_market_sentiment(cleaned_text)
        
        # Combine scores with weights
        combined_sentiment = (textblob_sentiment * 0.6) + (market_sentiment * 0.4)
        combined_confidence = max(textblob_confidence, abs(market_sentiment))
        
        # Normalize to [-1, 1] range
        combined_sentiment = max(-1.0, min(1.0, combined_sentiment))
        combined_confidence = max(0.0, min(1.0, combined_confidence))
        
        return combined_sentiment, combined_confidence
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (but keep the word)
        text = re.sub(r'[@#](\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _calculate_market_sentiment(self, text: str) -> float:
        """Calculate sentiment using market-specific lexicon."""
        words = text.split()
        sentiment_scores = []
        
        for phrase_length in range(3, 0, -1):  # Check 3-word, 2-word, then 1-word phrases
            for i in range(len(words) - phrase_length + 1):
                phrase = ' '.join(words[i:i + phrase_length])
                if phrase in self.market_lexicon:
                    sentiment_scores.append(self.market_lexicon[phrase])
                    # Remove processed words to avoid double counting
                    for j in range(phrase_length):
                        words[i + j] = ''
        
        if sentiment_scores:
            return np.mean(sentiment_scores)
        return 0.0
    
    def classify_polarity(self, sentiment_score: float) -> SentimentPolarity:
        """Classify sentiment score into polarity categories."""
        if sentiment_score >= 0.6:
            return SentimentPolarity.VERY_POSITIVE
        elif sentiment_score >= 0.2:
            return SentimentPolarity.POSITIVE
        elif sentiment_score <= -0.6:
            return SentimentPolarity.VERY_NEGATIVE
        elif sentiment_score <= -0.2:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.NEUTRAL

class TwitterDataSource:
    """Twitter data source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.api = None
        self._setup_api()
    
    def _setup_api(self):
        """Setup Twitter API client."""
        try:
            if not all([self.config.api_key, self.config.api_secret]):
                logger.error("Twitter API credentials not provided")
                return
            
            # Twitter API v2 client
            self.api = tweepy.Client(
                consumer_key=self.config.api_key,
                consumer_secret=self.config.api_secret,
                wait_on_rate_limit=True
            )
            
            logger.info("Twitter API client initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup Twitter API: {str(e)}")
    
    def fetch_data(self, symbol: str) -> List[AlternativeDataPoint]:
        """Fetch Twitter data for a symbol."""
        if not self.api:
            return []
        
        data_points = []
        
        try:
            # Search for tweets containing the symbol
            query = f"{symbol} OR ${symbol} -is:retweet lang:en"
            
            tweets = tweepy.Paginator(
                self.api.search_recent_tweets,
                query=query,
                max_results=min(100, self.config.max_items_per_fetch),
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations']
            ).flatten(limit=self.config.max_items_per_fetch)
            
            for tweet in tweets:
                data_point = AlternativeDataPoint(
                    source="twitter",
                    source_type=DataSourceType.SOCIAL_MEDIA,
                    symbol=symbol,
                    timestamp=tweet.created_at,
                    content=tweet.text,
                    metadata={
                        'author_id': tweet.author_id,
                        'tweet_id': tweet.id,
                        'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                        'like_count': tweet.public_metrics.get('like_count', 0),
                        'reply_count': tweet.public_metrics.get('reply_count', 0)
                    },
                    raw_data={'tweet': tweet.data}
                )
                
                data_points.append(data_point)
        
        except Exception as e:
            logger.error(f"Error fetching Twitter data for {symbol}: {str(e)}")
        
        return data_points

class RedditDataSource:
    """Reddit data source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.reddit = None
        self._setup_api()
    
    def _setup_api(self):
        """Setup Reddit API client."""
        try:
            if not self.config.api_key:
                logger.error("Reddit API credentials not provided")
                return
            
            self.reddit = praw.Reddit(
                client_id=self.config.api_key,
                client_secret=self.config.api_secret,
                user_agent="QuantumForge/1.0"
            )
            
            logger.info("Reddit API client initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup Reddit API: {str(e)}")
    
    def fetch_data(self, symbol: str) -> List[AlternativeDataPoint]:
        """Fetch Reddit data for a symbol."""
        if not self.reddit:
            return []
        
        data_points = []
        
        try:
            # Search relevant subreddits
            subreddits = ['stocks', 'investing', 'wallstreetbets', 'cryptocurrency', 'bitcoin']
            
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts mentioning the symbol
                search_results = subreddit.search(
                    symbol, 
                    sort='new', 
                    time_filter='day',
                    limit=50
                )
                
                for submission in search_results:
                    # Process submission
                    data_point = AlternativeDataPoint(
                        source=f"reddit_{subreddit_name}",
                        source_type=DataSourceType.SOCIAL_MEDIA,
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(submission.created_utc),
                        content=f"{submission.title} {submission.selftext}",
                        metadata={
                            'subreddit': subreddit_name,
                            'post_id': submission.id,
                            'score': submission.score,
                            'upvote_ratio': submission.upvote_ratio,
                            'num_comments': submission.num_comments,
                            'author': str(submission.author) if submission.author else 'deleted'
                        },
                        raw_data={'submission': submission}
                    )
                    
                    data_points.append(data_point)
                    
                    if len(data_points) >= self.config.max_items_per_fetch:
                        break
                
                if len(data_points) >= self.config.max_items_per_fetch:
                    break
        
        except Exception as e:
            logger.error(f"Error fetching Reddit data for {symbol}: {str(e)}")
        
        return data_points

class NewsDataSource:
    """News feed data source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.news_sources = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://feeds.reuters.com/reuters/businessNews',
            'https://www.marketwatch.com/rss/realtimeheadlines'
        ]
    
    def fetch_data(self, symbol: str) -> List[AlternativeDataPoint]:
        """Fetch news data for a symbol."""
        data_points = []
        
        for news_url in self.news_sources:
            try:
                feed = feedparser.parse(news_url)
                
                for entry in feed.entries[:self.config.max_items_per_fetch // len(self.news_sources)]:
                    # Check if entry mentions the symbol
                    content = f"{entry.title} {entry.get('summary', '')}"
                    if self._is_relevant(content, symbol):
                        
                        # Parse timestamp
                        timestamp = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            timestamp = datetime(*entry.published_parsed[:6])
                        
                        data_point = AlternativeDataPoint(
                            source=urlparse(news_url).netloc,
                            source_type=DataSourceType.NEWS_FEED,
                            symbol=symbol,
                            timestamp=timestamp,
                            content=content,
                            metadata={
                                'title': entry.title,
                                'link': entry.get('link', ''),
                                'author': entry.get('author', ''),
                                'tags': entry.get('tags', [])
                            },
                            raw_data={'entry': entry}
                        )
                        
                        data_points.append(data_point)
            
            except Exception as e:
                logger.error(f"Error fetching news from {news_url}: {str(e)}")
        
        return data_points
    
    def _is_relevant(self, content: str, symbol: str) -> bool:
        """Check if content is relevant to symbol."""
        content_lower = content.lower()
        symbol_lower = symbol.lower()
        
        # Direct symbol mention
        if symbol_lower in content_lower:
            return True
        
        # Company name mapping (simplified)
        company_names = {
            'AAPL': ['apple', 'iphone', 'tim cook'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle'],
            'AMZN': ['amazon', 'aws', 'jeff bezos'],
            'GOOGL': ['google', 'alphabet', 'youtube'],
            'MSFT': ['microsoft', 'azure', 'windows'],
            'BTC': ['bitcoin', 'btc', 'cryptocurrency'],
            'ETH': ['ethereum', 'eth', 'smart contract']
        }
        
        if symbol in company_names:
            for name in company_names[symbol]:
                if name in content_lower:
                    return True
        
        return False

class EconomicDataSource:
    """Economic indicator data source."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.indicators = {
            'GDP': 'Gross Domestic Product',
            'CPI': 'Consumer Price Index',
            'UNEMPLOYMENT': 'Unemployment Rate',
            'INTEREST_RATE': 'Federal Funds Rate',
            'VIX': 'Volatility Index'
        }
    
    def fetch_data(self, symbol: str = None) -> List[AlternativeDataPoint]:
        """Fetch economic indicator data."""
        data_points = []
        
        try:
            # Use yfinance for basic economic data (VIX, etc.)
            if symbol:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                data_point = AlternativeDataPoint(
                    source="yahoo_finance",
                    source_type=DataSourceType.ECONOMIC_INDICATOR,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    content=f"Economic data for {symbol}",
                    metadata=info,
                    raw_data={'info': info}
                )
                
                data_points.append(data_point)
        
        except Exception as e:
            logger.error(f"Error fetching economic data: {str(e)}")
        
        return data_points

class AlternativeDataLoader:
    """
    Main alternative data loader orchestrating multiple data sources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize alternative data loader."""
        self.config = self._load_config(config_path)
        self.data_sources = {}
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Processing
        self.is_running = False
        self.scheduler_thread = None
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Storage
        self.db_path = self.config.get('database_path', 'alternative_data.db')
        self._setup_database()
        
        # Statistics
        self.stats = {
            'data_points_processed': 0,
            'sentiment_analyses': 0,
            'failed_fetches': 0,
            'last_update': datetime.now()
        }
        
        # Callbacks
        self.data_callbacks = []
        
        # Initialize data sources
        self._initialize_data_sources()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration."""
        # Default configuration
        default_config = {
            'database_path': 'alternative_data.db',
            'update_frequency': 300,
            'max_workers': 8,
            'data_sources': [
                {
                    'name': 'twitter',
                    'type': 'social_media',
                    'enabled': False,  # Requires API keys
                    'api_key': None,
                    'api_secret': None
                },
                {
                    'name': 'reddit',
                    'type': 'social_media',
                    'enabled': False,  # Requires API keys
                    'api_key': None,
                    'api_secret': None
                },
                {
                    'name': 'news',
                    'type': 'news_feed',
                    'enabled': True
                },
                {
                    'name': 'economic',
                    'type': 'economic_indicator',
                    'enabled': True
                }
            ],
            'symbols': ['BTC', 'ETH', 'AAPL', 'TSLA', 'AMZN']
        }
        
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def _setup_database(self):
        """Setup SQLite database for storing alternative data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alternative_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    source_type TEXT,
                    symbol TEXT,
                    timestamp DATETIME,
                    content TEXT,
                    sentiment_score REAL,
                    sentiment_polarity INTEGER,
                    confidence_score REAL,
                    relevance_score REAL,
                    data_quality_score REAL,
                    metadata TEXT,
                    processing_timestamp DATETIME,
                    raw_data TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON alternative_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_timestamp ON alternative_data(source, timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database setup complete: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database setup error: {str(e)}")
    
    def _initialize_data_sources(self):
        """Initialize configured data sources."""
        for source_config in self.config.get('data_sources', []):
            if not source_config.get('enabled', False):
                continue
            
            try:
                config = DataSourceConfig(
                    name=source_config['name'],
                    source_type=DataSourceType(source_config['type']),
                    api_key=source_config.get('api_key'),
                    api_secret=source_config.get('api_secret'),
                    symbols=self.config.get('symbols', []),
                    update_frequency=source_config.get('update_frequency', 300)
                )
                
                # Create appropriate data source
                if config.source_type == DataSourceType.SOCIAL_MEDIA:
                    if config.name == 'twitter':
                        self.data_sources[config.name] = TwitterDataSource(config)
                    elif config.name == 'reddit':
                        self.data_sources[config.name] = RedditDataSource(config)
                elif config.source_type == DataSourceType.NEWS_FEED:
                    self.data_sources[config.name] = NewsDataSource(config)
                elif config.source_type == DataSourceType.ECONOMIC_INDICATOR:
                    self.data_sources[config.name] = EconomicDataSource(config)
                
                logger.info(f"Initialized data source: {config.name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize data source {source_config['name']}: {str(e)}")
    
    def add_callback(self, callback: Callable[[List[AlternativeDataPoint]], None]):
        """Add callback for processed data."""
        self.data_callbacks.append(callback)
    
    def start(self):
        """Start alternative data collection."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Schedule data collection
        for name, source in self.data_sources.items():
            schedule.every(source.config.update_frequency).seconds.do(
                self._fetch_and_process_data, source
            )
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info(f"Started alternative data loader with {len(self.data_sources)} sources")
    
    def stop(self):
        """Stop alternative data collection."""
        self.is_running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        
        logger.info("Alternative data loader stopped")
    
    def _run_scheduler(self):
        """Run scheduled tasks."""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def _fetch_and_process_data(self, source):
        """Fetch and process data from a source."""
        try:
            symbols = source.config.symbols or self.config.get('symbols', [])
            
            # Submit fetch tasks for each symbol
            futures = []
            for symbol in symbols:
                future = self.executor.submit(self._process_symbol_data, source, symbol)
                futures.append(future)
            
            # Collect results
            all_data_points = []
            for future in as_completed(futures):
                try:
                    data_points = future.result()
                    all_data_points.extend(data_points)
                except Exception as e:
                    logger.error(f"Error processing symbol data: {str(e)}")
                    self.stats['failed_fetches'] += 1
            
            # Process and store data
            if all_data_points:
                processed_data = self._process_data_points(all_data_points)
                self._store_data_points(processed_data)
                
                # Notify callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(processed_data)
                    except Exception as e:
                        logger.error(f"Error in data callback: {str(e)}")
                
                self.stats['data_points_processed'] += len(processed_data)
                self.stats['last_update'] = datetime.now()
        
        except Exception as e:
            logger.error(f"Error in fetch and process: {str(e)}")
    
    def _process_symbol_data(self, source, symbol: str) -> List[AlternativeDataPoint]:
        """Process data for a single symbol."""
        try:
            data_points = source.fetch_data(symbol)
            return data_points
        except Exception as e:
            logger.error(f"Error fetching data from {source.config.name} for {symbol}: {str(e)}")
            return []
    
    def _process_data_points(self, data_points: List[AlternativeDataPoint]) -> List[AlternativeDataPoint]:
        """Process raw data points (sentiment analysis, scoring, etc.)."""
        processed_points = []
        
        for point in data_points:
            try:
                # Sentiment analysis
                sentiment_score, confidence = self.sentiment_analyzer.analyze_sentiment(point.content)
                point.sentiment_score = sentiment_score
                point.confidence_score = confidence
                point.sentiment_polarity = self.sentiment_analyzer.classify_polarity(sentiment_score)
                
                # Relevance scoring (simplified)
                point.relevance_score = self._calculate_relevance_score(point)
                
                # Data quality scoring
                point.data_quality_score = self._calculate_quality_score(point)
                
                # Filter by quality thresholds
                if (point.confidence_score >= 0.3 and 
                    point.relevance_score >= 0.3 and 
                    point.data_quality_score >= 0.5):
                    processed_points.append(point)
                
                self.stats['sentiment_analyses'] += 1
                
            except Exception as e:
                logger.error(f"Error processing data point: {str(e)}")
        
        return processed_points
    
    def _calculate_relevance_score(self, point: AlternativeDataPoint) -> float:
        """Calculate relevance score for data point."""
        score = 0.5  # Base score
        
        # Symbol mention bonus
        if point.symbol and point.symbol.lower() in point.content.lower():
            score += 0.3
        
        # Source credibility
        if point.source in ['bloomberg.com', 'reuters.com', 'cnbc.com']:
            score += 0.2
        
        # Engagement metrics (for social media)
        if point.source_type == DataSourceType.SOCIAL_MEDIA:
            if 'like_count' in point.metadata:
                likes = point.metadata.get('like_count', 0)
                score += min(0.2, likes / 1000)  # Up to 0.2 bonus for likes
        
        return min(1.0, score)
    
    def _calculate_quality_score(self, point: AlternativeDataPoint) -> float:
        """Calculate data quality score."""
        score = 0.5  # Base score
        
        # Content length (not too short or long)
        content_length = len(point.content)
        if 20 <= content_length <= 1000:
            score += 0.2
        
        # Timestamp freshness
        age_hours = (datetime.now() - point.timestamp).total_seconds() / 3600
        if age_hours <= 24:
            score += 0.2
        elif age_hours <= 72:
            score += 0.1
        
        # Metadata completeness
        if point.metadata and len(point.metadata) > 2:
            score += 0.1
        
        return min(1.0, score)
    
    def _store_data_points(self, data_points: List[AlternativeDataPoint]):
        """Store data points in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for point in data_points:
                cursor.execute('''
                    INSERT INTO alternative_data (
                        source, source_type, symbol, timestamp, content,
                        sentiment_score, sentiment_polarity, confidence_score,
                        relevance_score, data_quality_score, metadata,
                        processing_timestamp, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    point.source,
                    point.source_type.value,
                    point.symbol,
                    point.timestamp,
                    point.content,
                    point.sentiment_score,
                    point.sentiment_polarity.value if point.sentiment_polarity else None,
                    point.confidence_score,
                    point.relevance_score,
                    point.data_quality_score,
                    json.dumps(point.metadata),
                    point.processing_timestamp,
                    json.dumps(point.raw_data)
                ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored {len(data_points)} data points")
            
        except Exception as e:
            logger.error(f"Error storing data points: {str(e)}")
    
    def query_data(self, symbol: str = None, start_date: datetime = None, 
                   end_date: datetime = None, source: str = None,
                   min_sentiment: float = None) -> pd.DataFrame:
        """Query stored alternative data."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM alternative_data WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            if source:
                query += " AND source = ?"
                params.append(source)
            
            if min_sentiment is not None:
                query += " AND sentiment_score >= ?"
                params.append(min_sentiment)
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error querying data: {str(e)}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loader statistics."""
        stats = self.stats.copy()
        
        # Add source status
        stats['active_sources'] = list(self.data_sources.keys())
        stats['total_sources'] = len(self.data_sources)
        
        # Database stats
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM alternative_data")
            stats['total_records'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM alternative_data")
            stats['unique_symbols'] = cursor.fetchone()[0]
            
            conn.close()
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
        
        return stats

def example_data_callback(data_points: List[AlternativeDataPoint]):
    """Example callback for processed data."""
    if not data_points:
        return
    
    # Group by sentiment
    positive = [p for p in data_points if p.sentiment_score and p.sentiment_score > 0.2]
    negative = [p for p in data_points if p.sentiment_score and p.sentiment_score < -0.2]
    
    logger.info(f"Processed {len(data_points)} data points: {len(positive)} positive, {len(negative)} negative")
    
    # Log high-confidence insights
    high_confidence = [p for p in data_points if p.confidence_score and p.confidence_score > 0.8]
    for point in high_confidence[:3]:  # Top 3
        logger.info(f"High confidence insight: {point.symbol} - {point.sentiment_score:.2f} - {point.content[:100]}...")

def main():
    """Example usage of AlternativeDataLoader."""
    # Create loader
    loader = AlternativeDataLoader()
    
    # Add callback
    loader.add_callback(example_data_callback)
    
    try:
        # Start loader
        loader.start()
        
        # Run for some time
        time.sleep(60)  # 1 minute
        
        # Query some data
        df = loader.query_data(symbol='BTC', start_date=datetime.now() - timedelta(hours=1))
        print(f"Retrieved {len(df)} records for BTC")
        
        if not df.empty:
            avg_sentiment = df['sentiment_score'].mean()
            print(f"Average sentiment for BTC: {avg_sentiment:.3f}")
        
        # Print statistics
        stats = loader.get_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2, default=str)}")
        
    finally:
        loader.stop()

if __name__ == "__main__":
    main()