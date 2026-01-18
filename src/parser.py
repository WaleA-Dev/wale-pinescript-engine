"""
PineScript Parser - Extract Strategy Parameters from .pine Files

This module parses PineScript strategy files to extract:
1. Input parameters (input.float, input.bool, input.int, input.string)
2. Strategy settings (initial_capital, commission, etc.)
3. Indicator configurations
4. Entry/exit logic patterns

The parser uses regex patterns to handle PineScript's syntax variations.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
from pathlib import Path


@dataclass
class StrategyParams:
    """
    Strategy parameters extracted from PineScript.
    
    This dataclass holds all configurable parameters that affect
    backtesting behavior. Default values match common strategy patterns.
    """
    # Risk Management
    stop_loss_pct: float = 2.0
    trailing_pct: float = 1.5
    profit_target_pct: float = 4.0
    use_trailing_stop: bool = True
    use_profit_target: bool = True
    
    # Exit Conditions
    use_ob_exit: bool = False  # Overbought exit
    ob_threshold: float = 70.0
    use_os_exit: bool = False  # Oversold exit
    os_threshold: float = 30.0
    
    # EMA Filter
    use_ema_filter: bool = False
    ema_length: int = 200
    ema_slope_lookback: int = 5
    ema_slope_threshold: float = 0.0
    
    # Entry Conditions
    entry_threshold: float = 0.0
    
    # Consolidation Filter
    use_consolidation_filter: bool = False
    range_lookback: int = 20
    range_threshold: float = 0.5
    
    # Momentum Confirmation
    use_momentum_confirm: bool = False
    momentum_ema_fast: int = 12
    momentum_ema_slow: int = 26
    
    # ADX Filter
    use_adx_filter: bool = False
    adx_length: int = 14
    adx_threshold: float = 25.0
    
    # RSI Settings
    use_rsi_filter: bool = False
    rsi_length: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # ATR Settings
    atr_length: int = 14
    atr_multiplier: float = 1.5
    
    # Position Sizing
    order_size_pct: float = 100.0
    pyramiding: int = 0
    
    # Strategy Direction
    long_only: bool = True
    short_only: bool = False
    
    # Custom parameters (for strategy-specific inputs)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySettings:
    """
    Strategy-level settings from strategy() declaration.
    """
    title: str = "Unnamed Strategy"
    shorttitle: str = ""
    overlay: bool = True
    initial_capital: float = 100000.0
    default_qty_type: str = "percent_of_equity"
    default_qty_value: float = 100.0
    commission_type: str = "percent"
    commission_value: float = 0.1
    slippage: int = 0
    process_orders_on_close: bool = False
    calc_on_every_tick: bool = False
    pyramiding: int = 0


class PineScriptParser:
    """
    Parser for PineScript strategy files.
    
    Extracts parameters, settings, and logic patterns from .pine files
    using regex-based parsing. Handles common PineScript idioms and
    provides sensible defaults when parsing fails.
    """
    
    def __init__(self, pine_path: Optional[str] = None, pine_content: Optional[str] = None):
        """
        Initialize parser with either a file path or content string.
        
        Args:
            pine_path: Path to .pine file
            pine_content: Raw PineScript content
        """
        if pine_path:
            self.content = Path(pine_path).read_text(encoding='utf-8')
        elif pine_content:
            self.content = pine_content
        else:
            raise ValueError("Must provide either pine_path or pine_content")
        
        # Remove comments for cleaner parsing
        self._clean_content = self._remove_comments(self.content)
    
    def _remove_comments(self, text: str) -> str:
        """Remove single-line and multi-line comments."""
        # Remove single-line comments
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        # Remove multi-line comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    def parse_strategy_settings(self) -> StrategySettings:
        """
        Extract strategy() declaration settings.
        
        Returns:
            StrategySettings with parsed values
        """
        settings = StrategySettings()
        
        # Find strategy() call
        strategy_match = re.search(
            r'strategy\s*\((.*?)\)',
            self._clean_content,
            re.DOTALL
        )
        
        if not strategy_match:
            return settings
        
        args = strategy_match.group(1)
        
        # Parse title
        title_match = re.search(r'["\']([^"\']+)["\']', args)
        if title_match:
            settings.title = title_match.group(1)
        
        # Parse shorttitle
        shorttitle_match = re.search(r'shorttitle\s*=\s*["\']([^"\']+)["\']', args)
        if shorttitle_match:
            settings.shorttitle = shorttitle_match.group(1)
        
        # Parse overlay
        overlay_match = re.search(r'overlay\s*=\s*(true|false)', args, re.IGNORECASE)
        if overlay_match:
            settings.overlay = overlay_match.group(1).lower() == 'true'
        
        # Parse initial_capital
        capital_match = re.search(r'initial_capital\s*=\s*(\d+(?:\.\d+)?)', args)
        if capital_match:
            settings.initial_capital = float(capital_match.group(1))
        
        # Parse default_qty_type
        qty_type_match = re.search(r'default_qty_type\s*=\s*strategy\.(\w+)', args)
        if qty_type_match:
            settings.default_qty_type = qty_type_match.group(1)
        
        # Parse default_qty_value
        qty_value_match = re.search(r'default_qty_value\s*=\s*(\d+(?:\.\d+)?)', args)
        if qty_value_match:
            settings.default_qty_value = float(qty_value_match.group(1))
        
        # Parse commission
        comm_type_match = re.search(r'commission_type\s*=\s*strategy\.commission\.(\w+)', args)
        if comm_type_match:
            settings.commission_type = comm_type_match.group(1)
        
        comm_value_match = re.search(r'commission_value\s*=\s*(\d+(?:\.\d+)?)', args)
        if comm_value_match:
            settings.commission_value = float(comm_value_match.group(1))
        
        # Parse slippage
        slippage_match = re.search(r'slippage\s*=\s*(\d+)', args)
        if slippage_match:
            settings.slippage = int(slippage_match.group(1))
        
        # Parse process_orders_on_close
        process_match = re.search(r'process_orders_on_close\s*=\s*(true|false)', args, re.IGNORECASE)
        if process_match:
            settings.process_orders_on_close = process_match.group(1).lower() == 'true'
        
        # Parse pyramiding
        pyramid_match = re.search(r'pyramiding\s*=\s*(\d+)', args)
        if pyramid_match:
            settings.pyramiding = int(pyramid_match.group(1))
        
        return settings
    
    def parse_params(self) -> StrategyParams:
        """
        Extract all input parameters from PineScript.
        
        Returns:
            StrategyParams with parsed values
        """
        params = StrategyParams()
        
        # Parse all input types
        self._parse_float_inputs(params)
        self._parse_int_inputs(params)
        self._parse_bool_inputs(params)
        self._parse_string_inputs(params)
        
        return params
    
    def _parse_float_inputs(self, params: StrategyParams) -> None:
        """Parse input.float() declarations."""
        # Pattern: varname = input.float(defval, ...)
        pattern = r'(\w+)\s*=\s*input\.float\s*\(\s*([^,\)]+)'
        
        for match in re.finditer(pattern, self._clean_content):
            name = match.group(1).lower()
            try:
                value = float(match.group(2).strip())
            except ValueError:
                continue
            
            self._assign_param(params, name, value)
    
    def _parse_int_inputs(self, params: StrategyParams) -> None:
        """Parse input.int() declarations."""
        pattern = r'(\w+)\s*=\s*input\.int\s*\(\s*([^,\)]+)'
        
        for match in re.finditer(pattern, self._clean_content):
            name = match.group(1).lower()
            try:
                value = int(match.group(2).strip())
            except ValueError:
                continue
            
            self._assign_param(params, name, value)
    
    def _parse_bool_inputs(self, params: StrategyParams) -> None:
        """Parse input.bool() declarations."""
        pattern = r'(\w+)\s*=\s*input\.bool\s*\(\s*(true|false)'
        
        for match in re.finditer(pattern, self._clean_content, re.IGNORECASE):
            name = match.group(1).lower()
            value = match.group(2).lower() == 'true'
            self._assign_param(params, name, value)
    
    def _parse_string_inputs(self, params: StrategyParams) -> None:
        """Parse input.string() declarations."""
        pattern = r'(\w+)\s*=\s*input\.string\s*\(\s*["\']([^"\']+)["\']'
        
        for match in re.finditer(pattern, self._clean_content):
            name = match.group(1).lower()
            value = match.group(2)
            params.custom_params[name] = value
    
    def _assign_param(self, params: StrategyParams, name: str, value: Any) -> None:
        """
        Assign parsed value to appropriate parameter field.
        
        Uses fuzzy matching to handle naming variations.
        """
        # Normalize name
        name = name.lower().replace('_', '').replace('-', '')
        
        # Stop loss variations
        if any(x in name for x in ['stoploss', 'sl', 'stoplossp']):
            if isinstance(value, (int, float)):
                params.stop_loss_pct = float(value)
        
        # Trailing stop variations
        elif any(x in name for x in ['trailing', 'trail', 'trailp']):
            if isinstance(value, (int, float)):
                params.trailing_pct = float(value)
            elif isinstance(value, bool):
                params.use_trailing_stop = value
        
        # Profit target variations
        elif any(x in name for x in ['profittarget', 'takeprofit', 'tp', 'pt']):
            if isinstance(value, (int, float)):
                params.profit_target_pct = float(value)
            elif isinstance(value, bool):
                params.use_profit_target = value
        
        # EMA filter variations
        elif any(x in name for x in ['emafilter', 'useema', 'ematrend']):
            if isinstance(value, bool):
                params.use_ema_filter = value
        elif any(x in name for x in ['emalen', 'emalength', 'emaperiod']):
            if isinstance(value, int):
                params.ema_length = value
        elif 'emaslopelookback' in name:
            if isinstance(value, int):
                params.ema_slope_lookback = value
        elif 'emaslopethreshold' in name:
            if isinstance(value, (int, float)):
                params.ema_slope_threshold = float(value)
        
        # ADX filter variations
        elif any(x in name for x in ['adxfilter', 'useadx']):
            if isinstance(value, bool):
                params.use_adx_filter = value
        elif any(x in name for x in ['adxlen', 'adxlength', 'adxperiod']):
            if isinstance(value, int):
                params.adx_length = value
        elif any(x in name for x in ['adxthreshold', 'adxmin']):
            if isinstance(value, (int, float)):
                params.adx_threshold = float(value)
        
        # RSI variations
        elif any(x in name for x in ['rsifilter', 'usersi']):
            if isinstance(value, bool):
                params.use_rsi_filter = value
        elif any(x in name for x in ['rsilen', 'rsilength', 'rsiperiod']):
            if isinstance(value, int):
                params.rsi_length = value
        elif any(x in name for x in ['rsioverbought', 'rsiob']):
            if isinstance(value, (int, float)):
                params.rsi_overbought = float(value)
        elif any(x in name for x in ['rsioversold', 'rsios']):
            if isinstance(value, (int, float)):
                params.rsi_oversold = float(value)
        
        # ATR variations
        elif any(x in name for x in ['atrlen', 'atrlength', 'atrperiod']):
            if isinstance(value, int):
                params.atr_length = value
        elif any(x in name for x in ['atrmult', 'atrmultiplier']):
            if isinstance(value, (int, float)):
                params.atr_multiplier = float(value)
        
        # Momentum variations
        elif any(x in name for x in ['momentumconfirm', 'usemomentum']):
            if isinstance(value, bool):
                params.use_momentum_confirm = value
        elif any(x in name for x in ['momentumfast', 'fastema']):
            if isinstance(value, int):
                params.momentum_ema_fast = value
        elif any(x in name for x in ['momentumslow', 'slowema']):
            if isinstance(value, int):
                params.momentum_ema_slow = value
        
        # Consolidation filter
        elif any(x in name for x in ['consolidation', 'rangefilter']):
            if isinstance(value, bool):
                params.use_consolidation_filter = value
        elif 'rangelookback' in name:
            if isinstance(value, int):
                params.range_lookback = value
        elif 'rangethreshold' in name:
            if isinstance(value, (int, float)):
                params.range_threshold = float(value)
        
        # OB/OS exit
        elif any(x in name for x in ['obexit', 'overboughtexit']):
            if isinstance(value, bool):
                params.use_ob_exit = value
        elif any(x in name for x in ['obthreshold', 'overboughtlevel']):
            if isinstance(value, (int, float)):
                params.ob_threshold = float(value)
        
        # Order sizing
        elif any(x in name for x in ['ordersize', 'positionsize', 'qtyp']):
            if isinstance(value, (int, float)):
                params.order_size_pct = float(value)
        
        # Direction
        elif 'longonly' in name:
            if isinstance(value, bool):
                params.long_only = value
        elif 'shortonly' in name:
            if isinstance(value, bool):
                params.short_only = value
        
        # Entry threshold
        elif 'entrythreshold' in name:
            if isinstance(value, (int, float)):
                params.entry_threshold = float(value)
        
        # Store unmatched parameters
        else:
            params.custom_params[name] = value
    
    def extract_entry_conditions(self) -> List[str]:
        """
        Extract entry condition patterns from PineScript.
        
        Returns:
            List of condition strings found
        """
        conditions = []
        
        # Look for strategy.entry() calls
        entry_pattern = r'strategy\.entry\s*\([^)]*when\s*=\s*([^,\)]+)'
        for match in re.finditer(entry_pattern, self._clean_content):
            conditions.append(match.group(1).strip())
        
        # Look for if statements before strategy.entry
        if_entry_pattern = r'if\s+([^\n]+)\n[^\n]*strategy\.entry'
        for match in re.finditer(if_entry_pattern, self._clean_content):
            conditions.append(match.group(1).strip())
        
        return conditions
    
    def extract_exit_conditions(self) -> List[str]:
        """
        Extract exit condition patterns from PineScript.
        
        Returns:
            List of condition strings found
        """
        conditions = []
        
        # Look for strategy.exit() calls
        exit_pattern = r'strategy\.exit\s*\([^)]*'
        for match in re.finditer(exit_pattern, self._clean_content):
            conditions.append(match.group(0).strip())
        
        # Look for strategy.close() calls
        close_pattern = r'strategy\.close\s*\([^)]*when\s*=\s*([^,\)]+)'
        for match in re.finditer(close_pattern, self._clean_content):
            conditions.append(match.group(1).strip())
        
        return conditions
    
    def get_indicator_calls(self) -> Dict[str, List[str]]:
        """
        Extract indicator function calls.
        
        Returns:
            Dict mapping indicator names to their call patterns
        """
        indicators = {}
        
        # Common ta.* functions
        ta_pattern = r'ta\.(\w+)\s*\(([^)]+)\)'
        for match in re.finditer(ta_pattern, self._clean_content):
            func_name = match.group(1)
            args = match.group(2)
            if func_name not in indicators:
                indicators[func_name] = []
            indicators[func_name].append(args)
        
        return indicators
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export all parsed data as a dictionary.
        
        Returns:
            Dict with settings, params, conditions, and indicators
        """
        return {
            'settings': self.parse_strategy_settings().__dict__,
            'params': self.parse_params().__dict__,
            'entry_conditions': self.extract_entry_conditions(),
            'exit_conditions': self.extract_exit_conditions(),
            'indicators': self.get_indicator_calls(),
        }


def parse_pine_file(path: str) -> tuple:
    """
    Convenience function to parse a PineScript file.
    
    Args:
        path: Path to .pine file
        
    Returns:
        Tuple of (StrategyParams, StrategySettings)
    """
    parser = PineScriptParser(pine_path=path)
    return parser.parse_params(), parser.parse_strategy_settings()
