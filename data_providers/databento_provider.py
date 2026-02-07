"""
Databento Data Provider
Fetches historical OHLCV data from Databento API
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path


class DatabentoProvider:
    """
    Provider for fetching historical market data from Databento.

    Supports multiple data sources and timeframes.
    """

    # Databento dataset mappings
    DATASETS = {
        'US_EQUITIES': 'XNAS.ITCH',  # NASDAQ
        'NASDAQ': 'XNAS.ITCH',
        'NYSE': 'XNYS.PILLAR',
        'CME': 'GLBX.MDP3',
    }

    TIMEFRAME_MAP = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1H': 3600,
        '4H': 14400,
        '1D': 86400,
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Databento provider.

        Args:
            api_key: Databento API key. If not provided, looks for DATABENTO_API_KEY env var
        """
        self.api_key = api_key or os.environ.get('DATABENTO_API_KEY', '')
        self._client = None

    @property
    def client(self):
        """Lazy load Databento client."""
        if self._client is None:
            try:
                import databento as db
                self._client = db.Historical(self.api_key)
            except ImportError:
                raise ImportError("databento package not installed. Run: pip install databento")
        return self._client

    def set_api_key(self, api_key: str):
        """Update API key and reset client."""
        self.api_key = api_key
        self._client = None

    def validate_api_key(self) -> tuple[bool, str]:
        """
        Validate the API key by making a test request.

        Returns:
            Tuple of (is_valid, message)
        """
        if not self.api_key:
            return False, "No API key provided"

        try:
            import databento as db
            client = db.Historical(self.api_key)
            # Try to get metadata to validate key
            client.metadata.list_datasets()
            return True, "API key is valid"
        except ImportError:
            return False, "databento package not installed"
        except Exception as e:
            return False, f"Invalid API key: {str(e)}"

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1H',
        dataset: str = 'NASDAQ',
        rth_only: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Databento.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'NDAQ')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Candle timeframe ('1m', '5m', '15m', '1H', '4H', '1D')
            dataset: Exchange dataset ('NASDAQ', 'NYSE', 'CME')
            rth_only: If True, filter to Regular Trading Hours only

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        try:
            import databento as db
        except ImportError:
            raise ImportError("databento package not installed. Run: pip install databento")

        # Get dataset
        ds = self.DATASETS.get(dataset.upper(), self.DATASETS['NASDAQ'])

        # Get timeframe in seconds
        tf_seconds = self.TIMEFRAME_MAP.get(timeframe, 3600)

        # Fetch data
        client = db.Historical(self.api_key)

        data = client.timeseries.get_range(
            dataset=ds,
            symbols=[symbol],
            schema='ohlcv-1s' if tf_seconds < 60 else f'ohlcv-{tf_seconds // 60}m' if tf_seconds < 3600 else f'ohlcv-{tf_seconds // 3600}h' if tf_seconds < 86400 else 'ohlcv-1d',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
        )

        # Convert to DataFrame
        df = data.to_df()

        if df.empty:
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

        # Standardize column names
        df = df.reset_index()
        column_map = {
            'ts_event': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        }

        # Handle different column names from databento
        if 'ts_event' in df.columns:
            df['time'] = pd.to_datetime(df['ts_event'], unit='ns')
        elif 'index' in df.columns:
            df['time'] = pd.to_datetime(df['index'])

        # Ensure we have required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0.0

        # Select and order columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Filter to RTH if requested
        if rth_only:
            df = self._filter_rth(df)

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        return df

    def _filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to Regular Trading Hours (09:30 - 16:00 ET).

        Args:
            df: DataFrame with 'time' column

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df

        # Convert to ET
        df = df.copy()
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute

        # RTH is 09:30 - 16:00 ET
        # For simplicity, we use UTC offsets
        # ET is UTC-5 (EST) or UTC-4 (EDT)
        # RTH in UTC: roughly 14:30 - 21:00 (EST) or 13:30 - 20:00 (EDT)

        # Simple filter based on typical ETH window
        df['total_minutes'] = df['hour'] * 60 + df['minute']

        # Filter to 09:30 - 16:00 ET (assuming times are in ET)
        # If UTC, adjust: 14:30-21:00 = 870-1260 minutes
        rth_mask = (df['total_minutes'] >= 570) & (df['total_minutes'] < 960)  # 09:30 - 16:00

        df = df[rth_mask].drop(columns=['hour', 'minute', 'total_minutes'])

        return df.reset_index(drop=True)

    def get_available_symbols(self, dataset: str = 'NASDAQ') -> List[str]:
        """
        Get list of available symbols for a dataset.

        Args:
            dataset: Exchange dataset name

        Returns:
            List of available symbols
        """
        # Common symbols for quick access
        common_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'NDAQ', 'SPY', 'QQQ', 'IWM', 'DIA',
            'AMD', 'INTC', 'CRM', 'ORCL', 'IBM',
            'JPM', 'BAC', 'GS', 'MS', 'WFC',
        ]
        return common_symbols


class CSVDataProvider:
    """
    Provider for loading OHLCV data from CSV files.

    Fallback option when Databento API is not available.
    """

    def __init__(self):
        pass

    def load_csv(
        self,
        filepath: str,
        time_column: str = None,
        rth_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.

        Args:
            filepath: Path to CSV file
            time_column: Name of time/date column (auto-detected if None)
            rth_only: If True, filter to Regular Trading Hours

        Returns:
            DataFrame with standardized columns
        """
        df = pd.read_csv(filepath)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Find time column
        time_cols = ['time', 'date', 'datetime', 'timestamp', 'ts_event']
        time_col = time_column.lower() if time_column else None

        if time_col is None:
            for col in time_cols:
                if col in df.columns:
                    time_col = col
                    break

        if time_col is None:
            # Use first column as time
            time_col = df.columns[0]

        # Rename to standard
        df = df.rename(columns={time_col: 'time'})

        # Parse datetime
        df['time'] = pd.to_datetime(df['time'])

        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0

        # Select columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Filter RTH if requested
        if rth_only:
            df = self._filter_rth(df)

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        return df

    def _filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to Regular Trading Hours."""
        if df.empty:
            return df

        df = df.copy()
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['total_minutes'] = df['hour'] * 60 + df['minute']

        # 09:30 - 16:00 = 570 - 960 minutes from midnight
        rth_mask = (df['total_minutes'] >= 570) & (df['total_minutes'] < 960)

        df = df[rth_mask].drop(columns=['hour', 'minute', 'total_minutes'])

        return df.reset_index(drop=True)
