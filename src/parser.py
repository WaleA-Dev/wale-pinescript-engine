"""
PineScript Parser - Extract Strategy Parameters from .pine Files

This module parses PineScript strategy files to extract:
1. Input parameters (input.float, input.bool, input.int, input.string)
2. Strategy settings (initial_capital, commission, etc.)
3. Indicator configurations
4. Entry/exit logic patterns
5. Preset-based ternary assignments
6. Custom oscillator patterns

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
    trailing_pct: float = 1.5       # Trail step (offset from highest)
    profit_target_pct: float = 4.0  # Also used as trailing activation threshold
    use_trailing_stop: bool = False  # Only enabled when actual trailing exit code found
    use_profit_target: bool = True
    trail_activation_pct: float = 0.0  # Separate activation threshold (0 = use profit_target_pct)

    # ATR-based Stop/TP (from strategy.exit with stop/limit)
    sl_atr_mult: float = 0.0  # 0 = not using ATR-based SL
    tp_atr_mult: float = 0.0  # 0 = not using ATR-based TP
    dynamic_exits: bool = False  # True = recalculate stop/TP every bar

    # Exit Conditions
    use_ob_exit: bool = False  # Overbought exit
    ob_threshold: float = 70.0
    use_os_exit: bool = False  # Oversold exit
    os_threshold: float = 30.0
    use_stale_recycle: bool = False
    stale_recycle_bars: int = 380
    stale_recycle_min_pnl_pct: float = 0.0
    stale_recycle_max_pnl_pct: float = 1.5
    use_emergency_exit: bool = False
    max_hold_bars: int = 10000

    # EMA Filter
    use_ema_filter: bool = False
    ema_length: int = 200
    ema_slope_lookback: int = 5
    ema_slope_threshold: float = 0.0

    # Two-EMA Trend System (emaFast > emaSlow for uptrend)
    use_ema_crossover: bool = False
    ema_fast_length: int = 20
    ema_slow_length: int = 50
    use_rsi_pullback_entry: bool = False
    rsi_pullback_long: float = 45.0
    rsi_pullback_short: float = 55.0

    # Custom Oscillator Entry (Saty Phase pattern)
    use_oscillator_entry: bool = False
    oscillator_ema_len: int = 21
    oscillator_atr_len: int = 14
    oscillator_smooth_len: int = 3
    oscillator_atr_mult: float = 3.0
    oscillator_scale: float = 100.0
    entry_threshold: float = -50.0
    use_secondary_osc_entry: bool = False
    secondary_prev_threshold: float = -50.0
    secondary_curr_threshold: float = -110.0
    use_extreme_entry: bool = False
    extreme_threshold: float = -110.0

    # Secondary oscillator entry: osc[1] <= prev_threshold and osc > curr_threshold
    # with different thresholds (e.g. "leaving extreme zone" entries)
    use_secondary_osc_entry: bool = False
    secondary_prev_threshold: float = -50.0
    secondary_curr_threshold: float = -110.0

    # Consolidation Filter
    use_consolidation_filter: bool = False
    consolidation_lookback: int = 400
    consolidation_threshold: float = 8.0
    ema_slope_check_lookback: int = 200
    ema_slope_check_threshold: float = 2.0
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
    rsi_min: float = 0.0   # Custom RSI min for entry (rsiMin)
    rsi_max: float = 100.0  # Custom RSI max for entry (rsiMax)

    # ATR Settings
    atr_length: int = 14
    atr_multiplier: float = 1.5

    # Position Sizing
    order_size_pct: float = 100.0
    pyramiding: int = 0

    # Strategy Direction
    long_only: bool = True
    short_only: bool = False
    enable_shorts: bool = False
    allow_reversal: bool = False

    # Exit type: "strategy_exit" (stop/limit orders) or "strategy_close" (market close next bar)
    exit_type: str = "strategy_exit"
    trailing_activation_on_close: bool = False

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
    calc_on_order_fills: bool = False
    process_orders_on_close: bool = False
    calc_on_every_tick: bool = False
    use_bar_magnifier: bool = False
    pyramiding: int = 0


class PineScriptParser:
    """
    Parser for PineScript strategy files.

    Extracts parameters, settings, and logic patterns from .pine files
    using regex-based parsing. Handles common PineScript idioms and
    provides sensible defaults when parsing fails.
    """

    def __init__(self, pine_path: Optional[str] = None, pine_content: Optional[str] = None):
        if pine_path:
            self.content = Path(pine_path).read_text(encoding='utf-8')
        elif pine_content:
            self.content = pine_content
        else:
            raise ValueError("Must provide either pine_path or pine_content")

        # Remove comments for cleaner parsing
        self._clean_content = self._remove_comments(self.content)

        # Fields explicitly set from inputs/ternaries. Pattern detectors must
        # not override these (e.g. a preset that turns an exit off).
        self._explicit_fields: set = set()

    def _remove_comments(self, text: str) -> str:
        """Remove single-line and multi-line comments."""
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text

    def _extract_function_call_args(self, func_name: str) -> Optional[str]:
        """
        Extract top-level function call arguments with balanced parentheses.

        This avoids regex breakage for calls containing nested parentheses
        inside quoted strings, e.g. strategy("Name (v1)", ...).
        """
        text = self._clean_content
        match = re.search(rf'\b{re.escape(func_name)}\s*\(', text)
        if not match:
            return None

        depth = 0
        start = None
        in_single = False
        in_double = False
        escape = False

        for i in range(match.end() - 1, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            if ch == '\\':
                escape = True
                continue

            if in_single:
                if ch == "'":
                    in_single = False
                continue

            if in_double:
                if ch == '"':
                    in_double = False
                continue

            if ch == "'":
                in_single = True
                continue

            if ch == '"':
                in_double = True
                continue

            if ch == '(':
                depth += 1
                if depth == 1:
                    start = i + 1
                continue

            if ch == ')':
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i]

        return None

    def parse_strategy_settings(self) -> StrategySettings:
        """Extract strategy() declaration settings."""
        settings = StrategySettings()

        args = self._extract_function_call_args('strategy')
        if not args:
            return settings

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

        # Parse calc_on_order_fills
        fills_match = re.search(r'calc_on_order_fills\s*=\s*(true|false)', args, re.IGNORECASE)
        if fills_match:
            settings.calc_on_order_fills = fills_match.group(1).lower() == 'true'

        # Parse calc_on_every_tick
        cot_match = re.search(r'calc_on_every_tick\s*=\s*(true|false)', args, re.IGNORECASE)
        if cot_match:
            settings.calc_on_every_tick = cot_match.group(1).lower() == 'true'

        # Parse use_bar_magnifier
        bm_match = re.search(r'use_bar_magnifier\s*=\s*(true|false)', args, re.IGNORECASE)
        if bm_match:
            settings.use_bar_magnifier = bm_match.group(1).lower() == 'true'

        # Parse pyramiding
        pyramid_match = re.search(r'pyramiding\s*=\s*(\d+)', args)
        if pyramid_match:
            settings.pyramiding = int(pyramid_match.group(1))

        return settings

    def parse_params(self) -> StrategyParams:
        """Extract all input parameters from PineScript."""
        params = StrategyParams()

        # Parse simple assignments FIRST (lowest priority)
        # e.g., emaFastLen = 20, slMult = 1.8, useShorts = false
        self._parse_simple_assignments(params)

        # Parse generic input() calls (medium priority, overrides simple assignments)
        # e.g., emaFastLen = input(20, "EMA Fast Length")
        self._parse_generic_inputs(params)

        # Parse typed input calls (highest priority, overrides all above)
        self._parse_float_inputs(params)
        self._parse_int_inputs(params)
        self._parse_bool_inputs(params)
        self._parse_string_inputs(params)

        # Evaluate preset-based ternary assignments
        self._evaluate_preset_assignments(params)

        # Detect strategy patterns
        self._detect_oscillator_pattern(params)
        self._detect_consolidation_filter(params)
        self._detect_rsi_pullback_pattern(params)
        self._detect_directional_entries(params)
        self.detect_exit_pattern(params)

        return params

    def _parse_float_inputs(self, params: StrategyParams) -> None:
        """Parse input.float() declarations."""
        pattern = r'(\w+)\s*=\s*input\.float\s*\(\s*([^,\)]+)'
        for match in re.finditer(pattern, self._clean_content):
            name = match.group(1)
            try:
                value = float(match.group(2).strip())
            except ValueError:
                continue
            self._assign_param(params, name, value)

    def _parse_int_inputs(self, params: StrategyParams) -> None:
        """Parse input.int() declarations."""
        pattern = r'(\w+)\s*=\s*input\.int\s*\(\s*([^,\)]+)'
        for match in re.finditer(pattern, self._clean_content):
            name = match.group(1)
            try:
                value = int(match.group(2).strip())
            except ValueError:
                continue
            self._assign_param(params, name, value)

    def _parse_bool_inputs(self, params: StrategyParams) -> None:
        """Parse input.bool() declarations."""
        pattern = r'(\w+)\s*=\s*input\.bool\s*\(\s*(true|false)'
        for match in re.finditer(pattern, self._clean_content, re.IGNORECASE):
            name = match.group(1)
            value = match.group(2).lower() == 'true'
            self._assign_param(params, name, value)

    def _parse_string_inputs(self, params: StrategyParams) -> None:
        """Parse input.string() declarations."""
        pattern = r'(\w+)\s*=\s*input\.string\s*\(\s*["\']([^"\']+)["\']'
        for match in re.finditer(pattern, self._clean_content):
            name = match.group(1)
            value = match.group(2)
            params.custom_params[name] = value

    def _parse_generic_inputs(self, params: StrategyParams) -> None:
        """
        Parse generic input() declarations (without type suffix).
        e.g., emaFastLen = input(20, "EMA Fast Length")
        """
        # Match: name = input(value, ...) but NOT input.int/float/bool/string
        pattern = r'(\w+)\s*=\s*input\s*\(\s*([^,\)]+)'
        for match in re.finditer(pattern, self._clean_content):
            # Skip if this is actually input.int/float/bool/string (already handled)
            full_match = match.group(0)
            if re.search(r'input\.(int|float|bool|string|source|timeframe)', full_match):
                continue

            name = match.group(1)
            raw_value = match.group(2).strip().strip('"\'')

            # Try to parse as number
            try:
                if '.' in raw_value:
                    value = float(raw_value)
                else:
                    value = int(raw_value)
                self._assign_param(params, name, value)
                continue
            except ValueError:
                pass

            # Try to parse as boolean
            if raw_value.lower() in ('true', 'false'):
                self._assign_param(params, name, raw_value.lower() == 'true')
                continue

            # Store as custom string param
            params.custom_params[name] = raw_value

    def _parse_simple_assignments(self, params: StrategyParams) -> None:
        """
        Parse simple variable assignments (not input calls).

        Matches patterns like:
            emaFastLen = 20
            slMult = 1.8
            useShorts = false

        Only matches when the right-hand side is a bare literal (number or boolean).
        Skips reassignments (:=) and lines with function calls or operators on RHS.
        """
        content = self._clean_content

        # Match: name = number (integer or float)
        # Negative lookbehind for ':' prevents matching ':=' reassignments
        # Ensure RHS is just a number, not part of a larger expression
        num_pattern = r'(?:^|\n)\s*(?:var\s+)?(?:int\s+|float\s+)?(\w+)\s*(?<!:)=\s*(-?\d+(?:\.\d+)?)\s*(?:\n|$|//)'
        for match in re.finditer(num_pattern, content, re.MULTILINE):
            name = match.group(1)
            raw = match.group(2).strip()

            # Skip if this line also has input(), ta.*, strategy.*, or other function calls
            line_start = match.start()
            line_end = match.end()
            line = content[line_start:line_end]
            if 'input' in line or 'ta.' in line or 'strategy.' in line:
                continue
            # Skip common non-parameter names
            if name in ('i', 'j', 'k', 'n', 'x', 'y', 'bar_index', 'time', 'close',
                        'open', 'high', 'low', 'volume', 'na', 'color', 'label',
                        'line', 'box', 'table', 'array', 'matrix', 'map'):
                continue

            try:
                if '.' in raw:
                    value = float(raw)
                else:
                    value = int(raw)
                self._assign_param(params, name, value)
            except ValueError:
                pass

        # Match: name = true/false
        bool_pattern = r'(?:^|\n)\s*(?:var\s+)?(?:bool\s+)?(\w+)\s*(?<!:)=\s*(true|false)\s*(?:\n|$|//)'
        for match in re.finditer(bool_pattern, content, re.MULTILINE | re.IGNORECASE):
            name = match.group(1)
            line_start = match.start()
            line_end = match.end()
            line = content[line_start:line_end]
            if 'input' in line:
                continue
            if name in ('i', 'j', 'k', 'n', 'x', 'y', 'bar_index'):
                continue

            value = match.group(2).lower() == 'true'
            self._assign_param(params, name, value)

    def _resolve_token_value(self, token: str, params: StrategyParams) -> Optional[Any]:
        """
        Resolve a Pine token into a concrete value when possible.

        Supports:
        - numeric literals
        - booleans
        - quoted strings
        - already parsed params/custom params references
        """
        t = token.strip().rstrip(',')

        if not t:
            return None

        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            return t[1:-1]

        tl = t.lower()
        if tl == 'true':
            return True
        if tl == 'false':
            return False

        try:
            if '.' in t:
                return float(t)
            return int(t)
        except ValueError:
            pass

        # Resolve identifiers already parsed into StrategyParams fields.
        if hasattr(params, t):
            return getattr(params, t)

        # Resolve from custom params (exact or case-insensitive key match).
        if t in params.custom_params:
            return params.custom_params[t]
        for k, v in params.custom_params.items():
            if str(k).lower() == tl:
                return v

        return None

    def _resolve_numeric_token(self, token: str, params: StrategyParams) -> Optional[float]:
        """Resolve a token to float, if possible."""
        v = self._resolve_token_value(token, params)
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _evaluate_preset_assignments(self, params: StrategyParams) -> None:
        """
        Evaluate ternary assignments that depend on preset string values.

        Pattern: varname = preset == "X" ? val1 : preset == "Y" ? val2 : val3
        """
        # Find preset variable/value from parsed inputs.
        preset_var = None
        preset_val = None
        for key, val in params.custom_params.items():
            if 'preset' in str(key).lower():
                preset_var = str(key)
                preset_val = val
                break

        if not preset_var or preset_val is None:
            return

<<<<<<< HEAD
        preset_val_str = str(preset_val).strip()
=======
        # Find all ternary assignments based on preset
        # Pattern: var = preset == "X" ? value : preset == "Y" ? value : default
        # Values may be numbers (incl. negative), true/false, or identifiers
        # that refer to previously parsed inputs (e.g. custom_sl).
        value_token = r'-?[0-9.]+|true|false|\w+'
        ternary_pattern = (
            r'(\w+)\s*=\s*'  # variable name
            rf'(?:preset\s*==\s*["\']([^"\']+)["\']\s*\?\s*({value_token})\s*:\s*)'  # first branch
            rf'(?:preset\s*==\s*["\']([^"\']+)["\']\s*\?\s*({value_token})\s*:\s*)?'  # optional second branch
            rf'({value_token})'  # default
        )
>>>>>>> origin/main

        # Evaluate line-by-line to keep regex simple and robust.
        for raw_line in self._clean_content.splitlines():
            line = raw_line.strip()
            if not line or '=' not in line or '?' not in line or ':' not in line:
                continue
            if preset_var not in line:
                continue

<<<<<<< HEAD
            # 2-branch ternary:
            # var = preset == "A" ? a : preset == "B" ? b : c
            two_branch = re.match(
                rf'^(?:var\s+)?(?:float|int|bool|string)?\s*(\w+)\s*=\s*{re.escape(preset_var)}\s*==\s*["\']([^"\']+)["\']\s*\?\s*([^:]+?)\s*:\s*'
                rf'{re.escape(preset_var)}\s*==\s*["\']([^"\']+)["\']\s*\?\s*([^:]+?)\s*:\s*(.+)$',
                line
            )
            if two_branch:
                var_name = two_branch.group(1)
                p1 = two_branch.group(2).strip()
                v1 = two_branch.group(3).strip()
                p2 = two_branch.group(4).strip()
                v2 = two_branch.group(5).strip()
                v_default = two_branch.group(6).strip()

                if preset_val_str == p1:
                    chosen = self._resolve_token_value(v1, params)
                elif preset_val_str == p2:
                    chosen = self._resolve_token_value(v2, params)
                else:
                    chosen = self._resolve_token_value(v_default, params)

                if chosen is not None:
                    self._assign_param(params, var_name, chosen)
                continue

            # 1-branch ternary:
            # var = preset == "A" ? a : b
            one_branch = re.match(
                rf'^(?:var\s+)?(?:float|int|bool|string)?\s*(\w+)\s*=\s*{re.escape(preset_var)}\s*==\s*["\']([^"\']+)["\']\s*\?\s*([^:]+?)\s*:\s*(.+)$',
                line
            )
            if one_branch:
                var_name = one_branch.group(1)
                p1 = one_branch.group(2).strip()
                v1 = one_branch.group(3).strip()
                v_default = one_branch.group(4).strip()

                chosen = self._resolve_token_value(v1, params) if preset_val_str == p1 else self._resolve_token_value(v_default, params)
                if chosen is not None:
                    self._assign_param(params, var_name, chosen)
=======
            # Pick the branch token matching the preset value
            if match.group(2) and preset_val == match.group(2):
                token = match.group(3)
            elif match.group(4) and preset_val == match.group(4):
                token = match.group(5)
            else:
                token = match.group(6)

            value = self._resolve_value_token(token, params)
            if value is not None:
                self._assign_param(params, var_name, value)
>>>>>>> origin/main

    def _resolve_value_token(self, token: Optional[str], params: StrategyParams) -> Optional[Any]:
        """Resolve a Pine literal or identifier to a Python value."""
        if token is None:
            return None
        t = token.strip()
        if t.lower() == 'true':
            return True
        if t.lower() == 'false':
            return False
        try:
            return float(t)
        except ValueError:
            pass
        # Identifier: look up a previously parsed input value
        return params.custom_params.get(t)

    def _detect_oscillator_pattern(self, params: StrategyParams) -> None:
        """
        Detect custom oscillator patterns like Saty Phase:
        oscillator = ((close - ema) / (mult * atr)) * scale
        """
        content = self._clean_content

        # Look for oscillator-like formula: (close - ema) / (N * atr)
        # N and the scale may be ints or floats (e.g. 3.0 * atr14)
        osc_pattern = re.search(
            r'(\w+)\s*=\s*\(\s*\(\s*close\s*-\s*(\w+)\s*\)\s*/\s*\(\s*(\d+(?:\.\d+)?)\s*\*\s*(\w+)\s*\)\s*\)\s*\*\s*(\d+(?:\.\d+)?)',
            content
        )
        if osc_pattern:
            params.use_oscillator_entry = True
            params.oscillator_atr_mult = float(osc_pattern.group(3))
            params.oscillator_scale = float(osc_pattern.group(5))

        # Also detect if the oscillator is smoothed by EMA
        # smoothed = ta.ema(raw_osc, smooth_len)
        if params.use_oscillator_entry:
            smooth_pattern = re.search(
                r'(\w+)\s*=\s*ta\.ema\s*\(\s*\w+\s*,\s*(\d+)\s*\)',
                content
            )
            if smooth_pattern:
                try:
                    params.oscillator_smooth_len = int(smooth_pattern.group(2))
                except ValueError:
                    pass

        # Detect oscillator crossover entry
        # ta.crossover(oscillator, threshold) or ta.crossover(osc, -50)
        cross_entry = re.search(
            r'ta\.crossover\s*\(\s*\w+\s*,\s*(-?\d+(?:\.\d+)?)\s*\)',
            content
        )
        if cross_entry:
            try:
                threshold = float(cross_entry.group(1))
                if threshold < 0:  # Likely an oscillator threshold
                    params.use_oscillator_entry = True
                    params.entry_threshold = threshold
            except ValueError:
                pass

        # Detect OB exit: ta.crossunder(oscillator, threshold)
        cross_exit = re.search(
            r'ta\.crossunder\s*\(\s*\w+\s*,\s*(\d+(?:\.\d+)?)\s*\)',
            content
        )
        if cross_exit:
            try:
                threshold = float(cross_exit.group(1))
                if threshold > 50:  # Likely an overbought threshold
<<<<<<< HEAD
=======
                    if 'use_ob_exit' not in self._explicit_fields:
                        params.use_ob_exit = True
>>>>>>> origin/main
                    params.ob_threshold = threshold
            except ValueError:
                pass

<<<<<<< HEAD
        # Detect expression-style oscillator crossings:
        # e.g. oscillator[1] <= entry_threshold and oscillator > entry_threshold
        # or   oscillator[1] <= -50 and oscillator > -110
        expr_patterns = re.finditer(
            r'(\w+)\s*\[\s*1\s*\]\s*<=\s*([-\w\.]+)\s*and\s*\1\s*>\s*([-\w\.]+)',
            content,
            re.IGNORECASE
        )
        for match in expr_patterns:
            series_name = match.group(1)
            if 'osc' not in series_name.lower() and not params.use_oscillator_entry:
                continue

            prev_thr = self._resolve_numeric_token(match.group(2), params)
            curr_thr = self._resolve_numeric_token(match.group(3), params)
            if prev_thr is None or curr_thr is None:
                continue

            params.use_oscillator_entry = True
            if abs(prev_thr - curr_thr) < 1e-9:
                params.entry_threshold = prev_thr
            else:
                params.use_secondary_osc_entry = True
                params.secondary_prev_threshold = prev_thr
                params.secondary_curr_threshold = curr_thr

        # Detect expression-style OB exits:
        # e.g. oscillator[1] >= 100 and oscillator < 100
        ob_expr = re.finditer(
            r'(\w+)\s*\[\s*1\s*\]\s*>=\s*([-\w\.]+)\s*and\s*\1\s*<\s*([-\w\.]+)',
            content,
            re.IGNORECASE
        )
        ob_is_guarded = bool(re.search(r'\bif\s+use_ob_exit\b', content))
        for match in ob_expr:
            series_name = match.group(1)
            if 'osc' not in series_name.lower() and not params.use_oscillator_entry:
                continue
            prev_thr = self._resolve_numeric_token(match.group(2), params)
            curr_thr = self._resolve_numeric_token(match.group(3), params)
            if prev_thr is None or curr_thr is None:
                continue
            if prev_thr > 50 and abs(prev_thr - curr_thr) < 1e-9:
                params.ob_threshold = prev_thr
                if not ob_is_guarded:
                    params.use_ob_exit = True
=======
        self._detect_manual_crossovers(params)

    def _detect_manual_crossovers(self, params: StrategyParams) -> None:
        """
        Detect crossovers written without ta.crossover/crossunder:

            osc[1] <= A and osc > B   (entry, A == B: plain threshold cross;
                                       A != B: secondary/extreme-zone entry)
            osc[1] >= A and osc < A   (overbought exit)

        Thresholds may be numeric literals or identifiers bound to inputs.
        """
        content = self._clean_content
        token = r'-?\d+(?:\.\d+)?|\w+'

        def resolve(tok: str) -> Optional[float]:
            value = self._resolve_value_token(tok, params)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            # Identifier routed to a known params field during input parsing
            norm = tok.strip().lower().replace('_', '')
            if norm in ('entrythreshold', 'entrylevel', 'entrythresh'):
                return params.entry_threshold
            if norm in ('extremethreshold', 'extremelevel'):
                return params.extreme_threshold
            return None

        # --- Entry-style crossover: osc[1] <= A and osc > B ---
        entry_pattern = re.finditer(
            rf'(\w+)\s*\[\s*1\s*\]\s*<=\s*({token})\s+and\s+\1\s*>\s*({token})',
            content
        )
        for match in entry_pattern:
            prev_tok, curr_tok = match.group(2).strip(), match.group(3).strip()

            if prev_tok == curr_tok:
                # Plain threshold cross: osc leaves the zone below A
                params.use_oscillator_entry = True
                threshold = resolve(prev_tok)
                if threshold is not None:
                    if 'entry_threshold' in self._explicit_fields:
                        # Input already fixed the primary threshold; a different
                        # numeric level is an extreme-zone variant
                        if threshold != params.entry_threshold:
                            params.use_extreme_entry = True
                            params.extreme_threshold = threshold
                    else:
                        params.entry_threshold = threshold
            else:
                prev_val, curr_val = resolve(prev_tok), resolve(curr_tok)
                if prev_val is not None and curr_val is not None:
                    params.use_oscillator_entry = True
                    params.use_secondary_osc_entry = True
                    params.secondary_prev_threshold = prev_val
                    params.secondary_curr_threshold = curr_val

        # --- Exit-style crossunder: osc[1] >= A and osc < A ---
        exit_pattern = re.finditer(
            rf'(\w+)\s*\[\s*1\s*\]\s*>=\s*({token})\s+and\s+\1\s*<\s*({token})',
            content
        )
        for match in exit_pattern:
            prev_tok, curr_tok = match.group(2).strip(), match.group(3).strip()
            if prev_tok != curr_tok:
                continue
            threshold = resolve(prev_tok)
            if threshold is not None and threshold > 50:
                if 'use_ob_exit' not in self._explicit_fields:
                    params.use_ob_exit = True
                params.ob_threshold = threshold
>>>>>>> origin/main

    def _detect_consolidation_filter(self, params: StrategyParams) -> None:
        """Detect consolidation filter patterns."""
        content = self._clean_content

        # Check for consolidation-related variable names
        has_consolidation = bool(re.search(
            r'(is_?consolidat|in_?consolidat|consolidation_?filter)',
            content, re.IGNORECASE
        ))

        if has_consolidation:
            params.use_consolidation_filter = True

            # Look for EMA slope check
            slope_pattern = re.search(
                r'ema_?slope.*?(\d+(?:\.\d+)?)\s*[/%]',
                content, re.IGNORECASE | re.DOTALL
            )

            # Look for price range compression check
            range_pattern = re.search(
                r'(?:range|compression).*?(\d+(?:\.\d+)?)\s*[/%]',
                content, re.IGNORECASE | re.DOTALL
            )

    def _detect_rsi_pullback_pattern(self, params: StrategyParams) -> None:
        """
        Detect RSI pullback crossover systems.

        Typical shape:
          longSignal  = trendUp and ta.crossover(rsiVal, rsiPull)
          shortSignal = trendDn and ta.crossunder(rsiVal, rsiPullS)
        """
        content = self._clean_content

        long_cross = re.search(
            r'ta\.crossover\s*\(\s*(\w*rsi\w*)\s*,\s*([A-Za-z_]\w*|-?\d+(?:\.\d+)?)\s*\)',
            content,
            re.IGNORECASE,
        )
        short_cross = re.search(
            r'ta\.crossunder\s*\(\s*(\w*rsi\w*)\s*,\s*([A-Za-z_]\w*|-?\d+(?:\.\d+)?)\s*\)',
            content,
            re.IGNORECASE,
        )

        if not long_cross and not short_cross:
            return

        params.use_rsi_pullback_entry = True
        params.use_rsi_filter = True

        if long_cross:
            resolved_long = self._resolve_numeric_token(long_cross.group(2), params)
            if resolved_long is not None:
                params.rsi_pullback_long = float(resolved_long)

        if short_cross:
            resolved_short = self._resolve_numeric_token(short_cross.group(2), params)
            if resolved_short is not None:
                params.rsi_pullback_short = float(resolved_short)

        # Trend checks based on dual EMA are typical in this pattern.
        if re.search(r'ema\w*\s*[<>]\s*ema\w*', content, re.IGNORECASE):
            params.use_ema_crossover = True

    def _detect_directional_entries(self, params: StrategyParams) -> None:
        """Detect long/short strategy.entry usage and reversal intent."""
        content = self._clean_content

        has_long_entry = bool(re.search(
            r'strategy\.entry\s*\([^)]*strategy\.long',
            content,
            re.IGNORECASE | re.DOTALL,
        ))
        has_short_entry = bool(re.search(
            r'strategy\.entry\s*\([^)]*strategy\.short',
            content,
            re.IGNORECASE | re.DOTALL,
        ))

        if has_short_entry:
            params.enable_shorts = True
            params.long_only = False

        if has_long_entry and not has_short_entry:
            params.long_only = True
            params.short_only = False

        if has_long_entry and has_short_entry:
            params.allow_reversal = True
        if re.search(r'strategy\.position_size\s*[<>]=?\s*0', content):
            params.allow_reversal = True

    def _assign_param(self, params: StrategyParams, name: str, value: Any) -> None:
        """Assign parsed value to appropriate parameter field."""
        original_name = name
        # Normalize name
        norm = name.lower().replace('_', '').replace('-', '')

<<<<<<< HEAD
        # --- Momentum (must be checked before EMA crossover aliases) ---
        if any(x in norm for x in ['momentumconfirm', 'usemomentum']):
            if isinstance(value, bool):
                params.use_momentum_confirm = value
                return
        if any(x in norm for x in ['momentumfastlen', 'momentumfast', 'momfastlen', 'momentumemafast']):
            if isinstance(value, int):
                params.momentum_ema_fast = value
                return
        if any(x in norm for x in ['momentumslowlen', 'momentumslow', 'momslowlen', 'momentumemaslow']):
=======
        # --- Momentum EMAs (checked before the two-EMA system because names
        # like momentum_ema_fast contain "emafast" and must not enable it) ---
        if any(x in norm for x in ['momentumemafast', 'momentumfastlen', 'momentumfast', 'momfastlen']):
            if isinstance(value, int):
                params.momentum_ema_fast = value
                return
        if any(x in norm for x in ['momentumemaslow', 'momentumslowlen', 'momentumslow', 'momslowlen']):
>>>>>>> origin/main
            if isinstance(value, int):
                params.momentum_ema_slow = value
                return

        # --- Two-EMA system (emaFastLen / emaSlowLen) ---
        # Avoid matching momentum aliases like momentum_ema_fast.
        if ('momentum' not in norm and 'mom' not in norm):
            if any(x in norm for x in ['emafastlen', 'emafastlength', 'fastema', 'emafast', 'fastlen']):
                if isinstance(value, int):
                    params.ema_fast_length = value
                    params.use_ema_crossover = True
                    return
            if any(x in norm for x in ['emaslowlen', 'emaslowlength', 'slowema', 'emaslow', 'slowlen']):
                if isinstance(value, int):
                    params.ema_slow_length = value
                    params.use_ema_crossover = True
                    return

        # --- Oscillator entry threshold ---
        if norm in ('entrythreshold', 'entrylevel', 'entrythresh'):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                params.entry_threshold = float(value)
                self._explicit_fields.add('entry_threshold')
                return

        # --- Overbought / oversold exit toggles and levels ---
        if norm in ('useobexit', 'obexit', 'useoverboughtexit'):
            if isinstance(value, bool):
                params.use_ob_exit = value
                self._explicit_fields.add('use_ob_exit')
                return
        if norm in ('obthreshold', 'oblevel', 'overboughtthreshold', 'overboughtlevel'):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                params.ob_threshold = float(value)
                return
        if norm in ('useosexit', 'osexit', 'useoversoldexit'):
            if isinstance(value, bool):
                params.use_os_exit = value
                self._explicit_fields.add('use_os_exit')
                return
        if norm in ('osthreshold', 'oslevel', 'oversoldthreshold', 'oversoldlevel'):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                params.os_threshold = float(value)
                return

        # --- RSI min/max range filter ---
        if norm in ('rsimin', 'rsiminimum', 'rsilower'):
            if isinstance(value, (int, float)):
                params.rsi_min = float(value)
                params.use_rsi_filter = True
                return
        if norm in ('rsimax', 'rsimaximum', 'rsiupper'):
            if isinstance(value, (int, float)):
                params.rsi_max = float(value)
                params.use_rsi_filter = True
                return
        if norm in ('rsipull', 'rsipullback', 'rsipullbacklong', 'rsipulllong'):
            if isinstance(value, (int, float)):
                params.rsi_pullback_long = float(value)
                params.use_rsi_pullback_entry = True
                params.use_rsi_filter = True
                return
        if norm in ('rsipulls', 'rsipullshort', 'rsipullbacks', 'rsipullbackshort'):
            if isinstance(value, (int, float)):
                params.rsi_pullback_short = float(value)
                params.use_rsi_pullback_entry = True
                params.use_rsi_filter = True
                params.enable_shorts = True
                params.long_only = False
                return

        # --- ATR-based stop/TP multipliers ---
        if any(x in norm for x in ['slmult', 'stopatrmult', 'slatrmult', 'stopmult', 'slatr']):
            if isinstance(value, (int, float)):
                params.sl_atr_mult = float(value)
                params.use_trailing_stop = False
                return
        if any(x in norm for x in ['tpmult', 'targetatrmult', 'tpatrmult', 'targetmult', 'takeprofitmult', 'tpatr']):
            if isinstance(value, (int, float)):
                params.tp_atr_mult = float(value)
                params.use_profit_target = True
                return

        # --- Enable shorts ---
        if norm in ('useshorts', 'enableshorts', 'allowshorts'):
            if isinstance(value, bool):
                params.enable_shorts = value
                if value:
                    params.long_only = False
                return

        # --- Stop loss percentage ---
        if any(x in norm for x in ['stoplosspct', 'stoploss', 'stoplossp', 'slpct']):
            if isinstance(value, (int, float)):
                params.stop_loss_pct = float(value)
                return

        # --- Trailing stop step/offset ---
        # Note: Only store the value here. use_trailing_stop is set by detect_exit_pattern
        # when actual trailing exit code is found (strategy.exit with trail params)
        if any(x in norm for x in ['trailingsteppct', 'trailstep', 'trailingoffset', 'trailpct']):
            if isinstance(value, (int, float)):
                params.trailing_pct = float(value)
                return
        if any(x in norm for x in ['trailing', 'trail', 'trailp']):
            if isinstance(value, (int, float)):
                params.trailing_pct = float(value)
            elif isinstance(value, bool):
                params.use_trailing_stop = value
            return

        # --- Profit target / trail activation ---
        if any(x in norm for x in ['profittargetpct', 'profittarget', 'takeprofitpct', 'takeprofit']):
            if isinstance(value, (int, float)):
                params.profit_target_pct = float(value)
                params.use_profit_target = True
            elif isinstance(value, bool):
                params.use_profit_target = value
            return

        # --- Exit toggles ---
        if norm in ('useobexit', 'obexit', 'enableobexit'):
            if isinstance(value, bool):
                params.use_ob_exit = value
                return
        if norm in ('useosexit', 'osexit', 'enableosexit'):
            if isinstance(value, bool):
                params.use_os_exit = value
                return
        if norm in ('usestalerecycle', 'stalerecycle', 'enablestalerecycleexit'):
            if isinstance(value, bool):
                params.use_stale_recycle = value
                return
        if norm in ('stalerecyclebars', 'stalebars'):
            if isinstance(value, int):
                params.stale_recycle_bars = value
                return
        if norm in ('stalerecyclemin', 'stalerecycleminpnl', 'stalerecycleminpnlpct', 'staleminpnlpct'):
            if isinstance(value, (int, float)):
                params.stale_recycle_min_pnl_pct = float(value)
                return
        if norm in ('stalerecyclemax', 'stalerecyclemaxpnl', 'stalerecyclemaxpnlpct', 'stalemaxpnlpct'):
            if isinstance(value, (int, float)):
                params.stale_recycle_max_pnl_pct = float(value)
                return
        if norm in ('useemergencyexit', 'emergencyexit', 'enablemaxholdexit', 'usetimeexit'):
            if isinstance(value, bool):
                params.use_emergency_exit = value
                return
        if norm in ('maxholdbars', 'maxhold', 'maxtradebars'):
            if isinstance(value, int):
                params.max_hold_bars = value
                return

        # --- Oscillator parameters ---
        if norm in ('oscemalen', 'oscillatoremalen', 'emalen21'):
            if isinstance(value, int):
                params.oscillator_ema_len = value
                return
        if norm in ('oscsmoothlen', 'oscillatorsmoothlen', 'smoothlen'):
            if isinstance(value, int):
                params.oscillator_smooth_len = value
                return

        # --- Extreme entry ---
        if norm in ('useextremeentry', 'extremeentry'):
            if isinstance(value, bool):
                params.use_extreme_entry = value
                return
        if norm in ('extremethreshold', 'extremelevel'):
            if isinstance(value, (int, float)):
                params.extreme_threshold = float(value)
                return
        if norm in ('entrythreshold', 'entrylevel', 'threshold'):
            if isinstance(value, (int, float)):
                params.entry_threshold = float(value)
                return
        if norm in ('obthreshold', 'overboughtthreshold'):
            if isinstance(value, (int, float)):
                params.ob_threshold = float(value)
                params.use_ob_exit = True
                return

        # --- EMA filter ---
        if any(x in norm for x in ['emafilterenabled', 'emafilter', 'useema', 'ematrend']):
            if isinstance(value, bool):
                params.use_ema_filter = value
                return
        if any(x in norm for x in ['emalen', 'emalength', 'emaperiod']):
            if isinstance(value, int):
                params.ema_length = value
                return

        # --- EMA slope ---
        if any(x in norm for x in ['slopelookback', 'slopeperiod']):
            if isinstance(value, int):
                params.ema_slope_check_lookback = value
                params.ema_slope_lookback = value
                return
        if any(x in norm for x in ['slopethreshold', 'slopepct']):
            if isinstance(value, (int, float)):
                params.ema_slope_check_threshold = float(value)
                params.ema_slope_threshold = float(value)
                return

        # --- Consolidation filter ---
        if any(x in norm for x in ['useconsolidationfilter', 'consolidationfilter', 'anticonsolidationfilter']):
            if isinstance(value, bool):
                params.use_consolidation_filter = value
                return
        if any(x in norm for x in ['consolidationlookback', 'rangelookback']):
            if isinstance(value, int):
                params.consolidation_lookback = value
                params.range_lookback = value
                return
        if any(x in norm for x in ['consolidationthreshold', 'rangethreshold']):
            if isinstance(value, (int, float)):
                params.consolidation_threshold = float(value)
                params.range_threshold = float(value)
                return

        # --- ADX filter ---
        if any(x in norm for x in ['adxfilter', 'useadx']):
            if isinstance(value, bool):
                params.use_adx_filter = value
                return
        if any(x in norm for x in ['adxlen', 'adxlength', 'adxperiod']):
            if isinstance(value, int):
                params.adx_length = value
                return
        if any(x in norm for x in ['adxthreshold', 'adxmin']):
            if isinstance(value, (int, float)):
                params.adx_threshold = float(value)
                return

        # --- RSI ---
        if any(x in norm for x in ['rsifilter', 'usersi']):
            if isinstance(value, bool):
                params.use_rsi_filter = value
                return
        if any(x in norm for x in ['rsilen', 'rsilength', 'rsiperiod']):
            if isinstance(value, int):
                params.rsi_length = value
                params.use_rsi_filter = True
                return
        if any(x in norm for x in ['rsioverbought', 'rsiob']):
            if isinstance(value, (int, float)):
                params.rsi_overbought = float(value)
                return
        if any(x in norm for x in ['rsioversold', 'rsios']):
            if isinstance(value, (int, float)):
                params.rsi_oversold = float(value)
                return

        # --- ATR ---
        if any(x in norm for x in ['atrlen', 'atrlength', 'atrperiod']):
            if isinstance(value, int):
                params.atr_length = value
                return
        if any(x in norm for x in ['atrmult', 'atrmultiplier']):
            if isinstance(value, (int, float)):
                params.atr_multiplier = float(value)
                return

<<<<<<< HEAD
=======
        # --- Momentum confirm toggle (fast/slow lengths handled above) ---
        if any(x in norm for x in ['momentumconfirm', 'usemomentum']):
            if isinstance(value, bool):
                params.use_momentum_confirm = value
                return

>>>>>>> origin/main
        # --- Order sizing ---
        if any(x in norm for x in ['ordersize', 'positionsize', 'qtyp']):
            if isinstance(value, (int, float)):
                params.order_size_pct = float(value)
                return

        # --- Direction ---
        if 'longonly' in norm:
            if isinstance(value, bool):
                params.long_only = value
                return
        if 'shortonly' in norm:
            if isinstance(value, bool):
                params.short_only = value
                return

        # Store unmatched parameters
        params.custom_params[original_name] = value

    def extract_entry_conditions(self) -> List[str]:
        """Extract entry condition patterns from PineScript."""
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
        """Extract exit condition patterns from PineScript."""
        conditions = []

        # Look for strategy.exit() calls
        exit_pattern = r'strategy\.exit\s*\([^)]*'
        for match in re.finditer(exit_pattern, self._clean_content):
            conditions.append(match.group(0).strip())

        # Look for strategy.close() calls
        close_pattern = r'strategy\.close\s*\([^)]*'
        for match in re.finditer(close_pattern, self._clean_content):
            conditions.append(match.group(0).strip())

        return conditions

    def get_indicator_calls(self) -> Dict[str, List[str]]:
        """Extract indicator function calls."""
        indicators = {}
        ta_pattern = r'ta\.(\w+)\s*\(([^)]+)\)'
        for match in re.finditer(ta_pattern, self._clean_content):
            func_name = match.group(1)
            args = match.group(2)
            if func_name not in indicators:
                indicators[func_name] = []
            indicators[func_name].append(args)
        return indicators

    def detect_exit_pattern(self, params: StrategyParams) -> None:
        """
        Detect strategy.exit() and strategy.close() patterns.
        Only enables trailing stop if actual trailing exit code exists.
        """
        content = self._clean_content

        # Check for strategy.close() usage (market close at next bar open)
        has_close = bool(re.search(r'strategy\.close\s*\(', content))
        has_exit = bool(re.search(r'strategy\.exit\s*\(', content))

        if has_close and not has_exit:
            params.exit_type = "strategy_close"
        elif has_close and has_exit:
            # Both used - strategy.close for signal exits, strategy.exit for stops
            params.exit_type = "mixed"

        # Check for strategy.exit with trail_points/trail_offset (TV trailing stop)
        has_trail_exit = bool(re.search(
            r'strategy\.exit\s*\([^)]*trail_(?:points|offset)\s*=',
            content, re.DOTALL
        ))

        # Check for manual trailing stop logic in exit section.
        # Matches comparisons with trail-stop variables on either side
        # (low <= trailStopPrice, trail_stop >= close) and trailing-active
        # state flags (trailingActive), which imply a manual trailing exit.
        has_manual_trail = bool(re.search(
            r'(?:trail(?:ing)?_?(?:stop|sl)\w*\s*[<>=])'
            r'|(?:[<>=]=?\s*trail(?:ing)?_?(?:stop|sl)\w*)'
            r'|(?:\btrail(?:ing)?_?active\b)',
            content, re.IGNORECASE
        ))

        # Only enable trailing if actual trailing exit code exists
        if has_trail_exit or has_manual_trail:
            params.use_trailing_stop = True
        else:
            # No trailing exit code found - don't use trailing even if
            # trailing_step_pct variable was defined (it may be unused)
            params.use_trailing_stop = False

        # Check for strategy.exit with stop and limit
        exit_with_stop_limit = re.search(
            r'strategy\.exit\s*\([^)]*stop\s*=\s*(\w+)[^)]*limit\s*=\s*(\w+)',
            content, re.DOTALL
        )
        if not exit_with_stop_limit:
            exit_with_stop_limit = re.search(
                r'strategy\.exit\s*\([^)]*limit\s*=\s*(\w+)[^)]*stop\s*=\s*(\w+)',
                content, re.DOTALL
            )

        # Check if strategy.exit is inside position_size check (dynamic recalculation)
        dynamic_pattern = re.search(
            r'if\s+strategy\.position_size\s*[><=!]+\s*0\s*\n.*?strategy\.exit',
            content, re.DOTALL
        )
        if dynamic_pattern:
            params.dynamic_exits = True

        # If we found ATR multipliers through input parsing, mark them
        if params.sl_atr_mult > 0 or params.tp_atr_mult > 0:
            params.dynamic_exits = True
            params.use_trailing_stop = False

        # Detect close-driven trailing activation patterns such as:
        # if currentPnL >= profit_target_pct
        #     trailingActive := true
        close_trail_activation = re.search(
            r'if\s+current\w*pnl\w*\s*[><=!]+\s*[\w\.\-\+]+\s*\n\s*\w*trail\w*\s*:?\=\s*true',
            content,
            re.IGNORECASE,
        )
        if close_trail_activation:
            params.trailing_activation_on_close = True

        # Check for percentage-based stop in strategy.close pattern
        # e.g., close <= entryPrice * (1 - stop_loss_pct / 100)
        pct_stop = re.search(
            r'close\s*[<>=]+\s*\w+\s*\*\s*\(\s*1\s*-\s*(\w+)\s*/\s*100',
            content
        )
        if pct_stop and not has_exit:
            # Percentage-based stop with strategy.close = check every bar at close
            params.exit_type = "strategy_close"

    def to_dict(self) -> Dict[str, Any]:
        """Export all parsed data as a dictionary."""
        return {
            'settings': self.parse_strategy_settings().__dict__,
            'params': self.parse_params().__dict__,
            'entry_conditions': self.extract_entry_conditions(),
            'exit_conditions': self.extract_exit_conditions(),
            'indicators': self.get_indicator_calls(),
        }


def parse_pine_file(path: str) -> tuple:
    """Convenience function to parse a PineScript file."""
    parser = PineScriptParser(pine_path=path)
    return parser.parse_params(), parser.parse_strategy_settings()
