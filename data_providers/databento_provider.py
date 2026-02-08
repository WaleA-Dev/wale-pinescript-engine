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
        'AMEX': 'XNYS.PILLAR',       # AMEX/NYSE ARCA uses same feed
        'NYSE_ARCA': 'XNYS.PILLAR',
        'CME': 'GLBX.MDP3',
    }

    # Earliest data availability per dataset
    DATASET_START_DATES = {
        'XNAS.ITCH': datetime(2018, 5, 1),
        'XNYS.PILLAR': datetime(2018, 5, 1),
        'GLBX.MDP3': datetime(2010, 1, 1),
    }

    # Common ticker → exchange mapping for auto-detection
    # NYSE/AMEX tickers (ETFs, large-cap NYSE stocks)
    NYSE_TICKERS = {
        'SCHD', 'SPY', 'DIA', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND',
        'GLD', 'SLV', 'XLF', 'XLE', 'XLV', 'XLK', 'XLI', 'XLU', 'XLP',
        'XLY', 'XLB', 'XLRE', 'VNQ', 'TLT', 'IEF', 'HYG', 'LQD', 'AGG',
        'EFA', 'EEM', 'VIG', 'ARKK', 'ARKG', 'ARKW', 'ARKF',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
        'JNJ', 'UNH', 'PFE', 'MRK', 'ABT', 'TMO', 'ABBV',
        'KO', 'PEP', 'PG', 'WMT', 'HD', 'MCD', 'DIS', 'NKE',
        'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX',
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'VLO', 'MPC',
        'V', 'MA', 'AXP', 'BRK.B', 'T', 'VZ',
    }

    # NASDAQ tickers
    NASDAQ_TICKERS = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
        'NDAQ', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NFLX', 'PYPL',
        'AVGO', 'QCOM', 'TXN', 'CSCO', 'CMCSA', 'COST', 'AMGN', 'SBUX',
        'ISRG', 'MDLZ', 'GILD', 'ADP', 'BKNG', 'REGN', 'LRCX', 'KLAC',
        'SNPS', 'CDNS', 'MRVL', 'FTNT', 'PANW', 'CRWD', 'DDOG', 'ZS',
        'QQQ', 'TQQQ', 'SQQQ',
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

    @classmethod
    def detect_exchange(cls, symbol: str) -> str:
        """Auto-detect exchange for a given ticker symbol."""
        sym = symbol.upper().strip()
        if sym in cls.NASDAQ_TICKERS:
            return 'NASDAQ'
        if sym in cls.NYSE_TICKERS:
            return 'NYSE'
        # Default: try NASDAQ first (most common for tech stocks)
        return 'NASDAQ'

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
        Validate the API key by making a test request to Databento metadata API.

        Returns:
            Tuple of (is_valid, message)
        """
        if not self.api_key or not self.api_key.strip():
            return False, "No API key provided."

        try:
            import databento as db
        except ImportError:
            return False, "databento package not installed. Run: pip install databento"

        try:
            client = db.Historical(self.api_key.strip())
            # Validate key by listing datasets - lightweight metadata call
            # If key is invalid, this will raise an AuthenticationError
            client.metadata.list_datasets()
            return True, "API key is valid. Connection successful."
        except Exception as e:
            # Databento raises specific errors, we catch generic here to be safe
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "401" in error_msg:
                return False, "Invalid API Key (Authentication Failed)"
            return False, f"Connection Failed: {error_msg}"

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1H',
        dataset: str = 'AUTO',
        rth_only: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Databento.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'NDAQ')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Candle timeframe ('1m', '5m', '15m', '1H', '4H', '1D')
            dataset: Exchange dataset ('NASDAQ', 'NYSE', 'AMEX', 'CME', 'AUTO')
            rth_only: If True, filter to Regular Trading Hours only

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        try:
            import databento as db
        except ImportError:
            raise ImportError("databento package not installed. Run: pip install databento")

        # Auto-detect exchange if needed
        if dataset.upper() == 'AUTO':
            dataset = self.detect_exchange(symbol)

        # Get dataset identifier
        ds = self.DATASETS.get(dataset.upper(), self.DATASETS['NASDAQ'])

        # Clamp start date to dataset availability
        min_date = self.DATASET_START_DATES.get(ds, datetime(2018, 5, 1))
        if start_date < min_date:
            print(f"Note: {ds} data starts {min_date.strftime('%Y-%m-%d')}. Clamping start date.")
            start_date = min_date

        # Clamp end date to today (Databento data is typically 1 day behind)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if end_date > today:
            end_date = today

        # Ensure end_date is after start_date
        if end_date <= start_date:
            end_date = start_date + timedelta(days=30)

        # Get timeframe in seconds
        tf_seconds = self.TIMEFRAME_MAP.get(timeframe, 3600)

        # Build schema string
        if tf_seconds < 60:
            schema = 'ohlcv-1s'
        elif tf_seconds < 3600:
            schema = f'ohlcv-{tf_seconds // 60}m'
        elif tf_seconds < 86400:
            schema = f'ohlcv-{tf_seconds // 3600}h'
        else:
            schema = 'ohlcv-1d'

        # Fetch data
        client = db.Historical(self.api_key)

        data = client.timeseries.get_range(
            dataset=ds,
            symbols=[symbol],
            schema=schema,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
        )

        # Convert to DataFrame
        df = data.to_df()

        if df.empty:
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

        # Standardize column names
        df = df.reset_index()

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

        Handles both UTC and ET timestamps. Databento returns UTC timestamps,
        so we filter to 13:30 - 21:00 UTC to cover both EST and EDT.

        Args:
            df: DataFrame with 'time' column

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df

        df = df.copy()
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['total_minutes'] = df['hour'] * 60 + df['minute']

        # Detect if timestamps are in UTC (have timezone info or hours > 12)
        # Databento always returns UTC timestamps
        is_utc = True
        if hasattr(df['time'].dtype, 'tz') and df['time'].dtype.tz is not None:
            is_utc = True
        elif df['hour'].median() > 12:
            is_utc = True
        else:
            is_utc = False

        if is_utc:
            # UTC: US RTH is roughly 13:30-21:00 (covers both EST and EDT)
            rth_start = 13 * 60 + 30  # 13:30 UTC
            rth_end = 21 * 60         # 21:00 UTC
        else:
            # ET: 09:30 - 16:00
            rth_start = 9 * 60 + 30   # 09:30 ET
            rth_end = 16 * 60         # 16:00 ET

        rth_mask = (df['total_minutes'] >= rth_start) & (df['total_minutes'] < rth_end)
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
