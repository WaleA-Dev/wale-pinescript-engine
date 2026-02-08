"""
Main Window for PineScript Backtest Engine GUI
Professional desktop application with clean interface
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from data_providers.databento_provider import DatabentoProvider

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QDateEdit, QCheckBox, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QFrame,
    QGroupBox, QFormLayout, QProgressBar, QStatusBar, QMenuBar,
    QMenu, QToolBar, QStyle, QApplication, QScrollArea
)
from PySide6.QtCore import Qt, QDate, QThread, Signal, QSettings
from PySide6.QtGui import QFont, QColor, QPalette, QAction, QIcon

# Import application modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class BacktestWorker(QThread):
    """Worker thread for running backtests without freezing UI."""
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)
    settings_parsed = Signal(object)  # Emit parsed strategy settings

    def __init__(self, df, pine_code, config):
        super().__init__()
        self.df = df
        self.pine_code = pine_code
        self.config = config

    def run(self):
        try:
            from src.parser import PineScriptParser, StrategyParams
            from src.backtest import BacktestEngine, BacktestConfig

            self.progress.emit(20, "Parsing PineScript...")

            # Parse Pine Script
            parser = PineScriptParser(pine_content=self.pine_code)
            params = parser.parse_params()
            settings = parser.parse_strategy_settings()

            # Emit parsed settings so GUI can update
            self.settings_parsed.emit(settings)

            self.progress.emit(40, "Configuring backtest...")

            # Use PineScript strategy() settings, with GUI as fallback
            initial_capital = settings.initial_capital if settings.initial_capital != 100000.0 else self.config.get('initial_capital', 100000)
            commission_pct = settings.commission_value if settings.commission_value != 0.1 else self.config.get('commission_pct', 0.1)

            # Determine position sizing from strategy settings
            qty_type = "percent_of_equity"
            if settings.default_qty_type == "percent_of_equity":
                order_size_pct = settings.default_qty_value
                qty_type = "percent_of_equity"
            elif settings.default_qty_type == "fixed":
                order_size_pct = 100.0
                qty_type = "fixed"
            else:
                order_size_pct = self.config.get('order_size_pct', 100)

            config = BacktestConfig(
                initial_capital=initial_capital,
                commission_pct=commission_pct,
                order_size_pct=order_size_pct,
                qty_type=qty_type,
                pyramiding=settings.pyramiding,
            )

            self.progress.emit(60, "Running backtest...")

            # Run backtest
            engine = BacktestEngine(config=config, params=params)
            result = engine.run(self.df)

            self.progress.emit(100, "Complete!")
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class ValidateKeyWorker(QThread):
    """Worker thread for validating Databento API key (keeps UI responsive)."""
    result = Signal(bool, str)

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def run(self):
        try:
            # Uses pre-imported DatabentoProvider
            provider = DatabentoProvider(api_key=self.api_key)
            is_valid, message = provider.validate_api_key()
            self.result.emit(is_valid, message)
            
        except Exception as e:
            self.result.emit(False, f"Validation error: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PineScript Backtest Engine")
        self.setMinimumSize(1400, 900)

        # Settings
        self.settings = QSettings("WalePineScript", "BacktestEngine")

        # State
        self.current_data = None
        self.current_result = None
        self.worker = None
        self._validate_key_worker = None

        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._apply_dark_theme()
        self._load_settings()

    def _setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel - Configuration (40%)
        left_panel = self._create_config_panel()
        splitter.addWidget(left_panel)

        # Right Panel - Results (60%)
        right_panel = self._create_results_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([500, 900])
        main_layout.addWidget(splitter)

    def _create_config_panel(self) -> QWidget:
        """Create the configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # === Data Source Section ===
        data_group = QGroupBox("Data Source")
        data_layout = QVBoxLayout(data_group)

        # Databento API Key
        api_layout = QFormLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter Databento API Key...")
        api_layout.addRow("API Key:", self.api_key_input)

        # Validate button
        self.validate_btn = QPushButton("Validate Key")
        self.validate_btn.clicked.connect(self._validate_api_key)
        api_layout.addRow("", self.validate_btn)

        data_layout.addLayout(api_layout)

        # Symbol and exchange
        symbol_layout = QHBoxLayout()
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("e.g. NDAQ, SCHD, AAPL")
        self.symbol_input.setText("NDAQ")
        symbol_layout.addWidget(QLabel("Symbol:"))
        symbol_layout.addWidget(self.symbol_input)

        self.exchange_combo = QComboBox()
        self.exchange_combo.addItems(['AUTO', 'NASDAQ', 'NYSE', 'AMEX'])
        self.exchange_combo.setCurrentText('AUTO')
        self.exchange_combo.setToolTip("Auto-detect exchange from ticker, or select manually")
        symbol_layout.addWidget(QLabel("Exchange:"))
        symbol_layout.addWidget(self.exchange_combo)

        data_layout.addLayout(symbol_layout)

        # Timeframe
        tf_layout = QHBoxLayout()
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1H', '4H', '1D'])
        self.timeframe_combo.setCurrentText('1H')
        tf_layout.addWidget(QLabel("Timeframe:"))
        tf_layout.addWidget(self.timeframe_combo)
        tf_layout.addStretch()

        data_layout.addLayout(tf_layout)

        # Date range
        date_layout = QHBoxLayout()

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addYears(-2))
        date_layout.addWidget(QLabel("Start:"))
        date_layout.addWidget(self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        date_layout.addWidget(QLabel("End:"))
        date_layout.addWidget(self.end_date)

        data_layout.addLayout(date_layout)

        # Options
        options_layout = QHBoxLayout()
        self.rth_only = QCheckBox("RTH Only")
        self.rth_only.setChecked(True)
        options_layout.addWidget(self.rth_only)

        self.fetch_btn = QPushButton("Fetch Data")
        self.fetch_btn.clicked.connect(self._fetch_data)
        options_layout.addWidget(self.fetch_btn)

        self.load_csv_btn = QPushButton("Load CSV")
        self.load_csv_btn.clicked.connect(self._load_csv)
        options_layout.addWidget(self.load_csv_btn)

        data_layout.addLayout(options_layout)

        # Data status
        self.data_status = QLabel("No data loaded")
        self.data_status.setStyleSheet("color: #888;")
        data_layout.addWidget(self.data_status)

        layout.addWidget(data_group)

        # === Strategy Section ===
        strategy_group = QGroupBox("PineScript Strategy")
        strategy_layout = QVBoxLayout(strategy_group)

        # Load strategy file button
        load_layout = QHBoxLayout()
        self.load_pine_btn = QPushButton("Load .pine File")
        self.load_pine_btn.clicked.connect(self._load_pine_file)
        load_layout.addWidget(self.load_pine_btn)

        self.save_pine_btn = QPushButton("Save Strategy")
        self.save_pine_btn.clicked.connect(self._save_pine_file)
        load_layout.addWidget(self.save_pine_btn)

        strategy_layout.addLayout(load_layout)

        # Code editor
        self.code_editor = QTextEdit()
        self.code_editor.setFont(QFont("Consolas", 10))
        self.code_editor.setPlaceholderText(
            "// Paste your PineScript strategy here...\n"
            "//@version=5\n"
            "strategy(\"My Strategy\", overlay=true)\n\n"
            "// Your strategy code..."
        )
        self.code_editor.setMinimumHeight(200)
        strategy_layout.addWidget(self.code_editor)

        layout.addWidget(strategy_group)

        # === Backtest Settings ===
        settings_group = QGroupBox("Backtest Settings")
        settings_layout = QFormLayout(settings_group)

        self.capital_input = QDoubleSpinBox()
        self.capital_input.setRange(1000, 10000000)
        self.capital_input.setValue(100000)
        self.capital_input.setPrefix("$")
        self.capital_input.setSingleStep(10000)
        settings_layout.addRow("Initial Capital:", self.capital_input)

        self.commission_input = QDoubleSpinBox()
        self.commission_input.setRange(0, 5)
        self.commission_input.setValue(0.1)
        self.commission_input.setSuffix("%")
        self.commission_input.setSingleStep(0.01)
        self.commission_input.setDecimals(3)
        settings_layout.addRow("Commission:", self.commission_input)

        self.order_size_input = QDoubleSpinBox()
        self.order_size_input.setRange(1, 100)
        self.order_size_input.setValue(100)
        self.order_size_input.setSuffix("%")
        settings_layout.addRow("Position Size:", self.order_size_input)

        layout.addWidget(settings_group)

        # === Run Button ===
        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.setMinimumHeight(50)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.run_btn.clicked.connect(self._run_backtest)
        layout.addWidget(self.run_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Stretch to fill
        layout.addStretch()

        return panel

    def _create_results_panel(self) -> QWidget:
        """Create the results panel with tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Summary Tab
        summary_widget = self._create_summary_tab()
        self.results_tabs.addTab(summary_widget, "Summary")

        # Trades Tab
        trades_widget = self._create_trades_tab()
        self.results_tabs.addTab(trades_widget, "Trade List")

        # Charts Tab
        charts_widget = self._create_charts_tab()
        self.results_tabs.addTab(charts_widget, "Charts")

        layout.addWidget(self.results_tabs)

        return panel

    def _create_summary_tab(self) -> QWidget:
        """Create summary statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Metrics grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)

        # Performance metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QFormLayout(perf_group)

        self.metric_labels = {}
        metrics = [
            ('total_trades', 'Total Trades'),
            ('winning_trades', 'Winning Trades'),
            ('losing_trades', 'Losing Trades'),
            ('win_rate', 'Win Rate'),
            ('total_pnl', 'Total P&L'),
            ('avg_pnl', 'Average P&L'),
            ('avg_winner', 'Avg Winner'),
            ('avg_loser', 'Avg Loser'),
            ('profit_factor', 'Profit Factor'),
            ('max_drawdown', 'Max Drawdown'),
            ('max_drawdown_pct', 'Max Drawdown %'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('sortino_ratio', 'Sortino Ratio'),
            ('calmar_ratio', 'Calmar Ratio'),
        ]

        for key, label in metrics:
            value_label = QLabel("--")
            value_label.setFont(QFont("Arial", 12, QFont.Bold))
            value_label.setAlignment(Qt.AlignRight)
            self.metric_labels[key] = value_label
            perf_layout.addRow(f"{label}:", value_label)

        metrics_layout.addWidget(perf_group)
        metrics_layout.addStretch()

        scroll.setWidget(metrics_widget)
        layout.addWidget(scroll)

        # Export button
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self._export_results)
        layout.addWidget(export_btn)

        return widget

    def _create_trades_tab(self) -> QWidget:
        """Create trades list tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Trade table
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(10)
        self.trades_table.setHorizontalHeaderLabels([
            '#', 'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price',
            'Signal', 'Qty', 'P&L', 'P&L %', 'Bars'
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trades_table.setAlternatingRowColors(True)
        self.trades_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.trades_table)

        # Export trades button
        export_layout = QHBoxLayout()
        export_csv_btn = QPushButton("Export to CSV")
        export_csv_btn.clicked.connect(self._export_trades_csv)
        export_layout.addWidget(export_csv_btn)
        export_layout.addStretch()
        layout.addLayout(export_layout)

        return widget

    def _create_charts_tab(self) -> QWidget:
        """Create charts tab with image-based rendering."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.chart_label = QLabel("Run a backtest to see equity curve and charts")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setStyleSheet("color: #888; font-size: 14px;")
        scroll.setWidget(self.chart_label)
        layout.addWidget(scroll)

        return widget

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        load_data = QAction("Load CSV Data...", self)
        load_data.triggered.connect(self._load_csv)
        file_menu.addAction(load_data)

        load_pine = QAction("Load Pine Script...", self)
        load_pine.triggered.connect(self._load_pine_file)
        file_menu.addAction(load_pine)

        file_menu.addSeparator()

        export_results = QAction("Export Results...", self)
        export_results.triggered.connect(self._export_results)
        file_menu.addAction(export_results)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self):
        """Setup status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def _apply_dark_theme(self):
        """Apply dark theme to application."""
        palette = QPalette()

        # Dark colors
        dark = QColor(30, 30, 30)
        darker = QColor(20, 20, 20)
        mid = QColor(50, 50, 50)
        light = QColor(70, 70, 70)
        text = QColor(220, 220, 220)
        highlight = QColor(42, 130, 218)

        palette.setColor(QPalette.Window, dark)
        palette.setColor(QPalette.WindowText, text)
        palette.setColor(QPalette.Base, darker)
        palette.setColor(QPalette.AlternateBase, mid)
        palette.setColor(QPalette.ToolTipBase, text)
        palette.setColor(QPalette.ToolTipText, text)
        palette.setColor(QPalette.Text, text)
        palette.setColor(QPalette.Button, mid)
        palette.setColor(QPalette.ButtonText, text)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, highlight)
        palette.setColor(QPalette.Highlight, highlight)
        palette.setColor(QPalette.HighlightedText, Qt.black)

        self.setPalette(palette)

        # Additional styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTableWidget {
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #333;
                padding: 5px;
                border: 1px solid #444;
            }
            QTabWidget::pane {
                border: 1px solid #444;
            }
            QTabBar::tab {
                background-color: #333;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #444;
            }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton {
                background-color: #444;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
            }
        """)

    def _load_settings(self):
        """Load saved settings."""
        # Do not load API key automatically for security
        # api_key = self.settings.value("databento_api_key", "")
        # if api_key:
        #     self.api_key_input.setText(api_key)

        symbol = self.settings.value("last_symbol", "NDAQ")
        self.symbol_input.setText(symbol)

        timeframe = self.settings.value("last_timeframe", "1H")
        self.timeframe_combo.setCurrentText(timeframe)

    def _save_settings(self):
        """Save current settings."""
        # Do not save API key for security
        # self.settings.setValue("databento_api_key", self.api_key_input.text())
        self.settings.setValue("last_symbol", self.symbol_input.text())
        self.settings.setValue("last_timeframe", self.timeframe_combo.currentText())

    def closeEvent(self, event):
        """Handle window close."""
        self._save_settings()
        event.accept()

    # === Action Methods ===

    def _validate_api_key(self):
        """Validate Databento API key in a worker thread so UI stays responsive."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Error", "Please enter an API key.")
            return

        self.validate_btn.setEnabled(False)
        self.statusbar.showMessage("Pinging Databento...")
        self._validate_key_worker = ValidateKeyWorker(api_key)
        self._validate_key_worker.result.connect(self._on_validate_key_done)
        self._validate_key_worker.finished.connect(self._on_validate_key_worker_finished)
        self._validate_key_worker.start()

    def _on_validate_key_done(self, valid: bool, message: str):
        """Show validation result in a message box (called from main thread)."""
        if valid:
            box = QMessageBox(self)
            box.setWindowTitle("API Key Valid")
            box.setIcon(QMessageBox.Information)
            box.setText(message)
            box.setStandardButtons(QMessageBox.Ok)
        else:
            box = QMessageBox(self)
            box.setWindowTitle("Invalid API Key")
            box.setIcon(QMessageBox.Warning)
            box.setText(message)
            box.setStandardButtons(QMessageBox.Ok)
        box.raise_()
        box.activateWindow()
        box.exec()
        self.statusbar.showMessage("Databento responded. API key valid." if valid else "Databento request failed.")

    def _on_validate_key_worker_finished(self):
        """Re-enable button when worker thread finishes."""
        self.validate_btn.setEnabled(True)
        self._validate_key_worker = None

    def _fetch_data(self):
        """Fetch data from Databento."""
        from data_providers.databento_provider import DatabentoProvider

        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Error", "Please enter a Databento API key")
            return

        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Error", "Please enter a symbol")
            return

        exchange = self.exchange_combo.currentText()
        self.statusbar.showMessage(f"Fetching {symbol} data from {exchange}...")
        self.fetch_btn.setEnabled(False)

        try:
            provider = DatabentoProvider(api_key)

            start = datetime(
                self.start_date.date().year(),
                self.start_date.date().month(),
                self.start_date.date().day()
            )
            end = datetime(
                self.end_date.date().year(),
                self.end_date.date().month(),
                self.end_date.date().day()
            )

            # Auto-detect exchange for display
            if exchange == 'AUTO':
                detected = DatabentoProvider.detect_exchange(symbol)
                self.statusbar.showMessage(f"Auto-detected {symbol} on {detected}. Fetching...")

            df = provider.fetch_ohlcv(
                symbol=symbol,
                start_date=start,
                end_date=end,
                timeframe=self.timeframe_combo.currentText(),
                dataset=exchange,
                rth_only=self.rth_only.isChecked()
            )

            if df.empty:
                QMessageBox.warning(self, "No Data",
                    f"No data returned for {symbol}.\n\n"
                    f"Tips:\n"
                    f"- Databento data starts from May 2018\n"
                    f"- Try a different exchange (NASDAQ vs NYSE/AMEX)\n"
                    f"- Check the symbol spelling")
                self.data_status.setText("No data returned")
                self.data_status.setStyleSheet("color: #e74c3c;")
            else:
                self.current_data = df
                self.data_status.setText(
                    f"Loaded {len(df)} bars | {df['time'].min()} to {df['time'].max()}"
                )
                self.data_status.setStyleSheet("color: #2ecc71;")
                self.statusbar.showMessage(f"Loaded {len(df)} bars for {symbol}")

        except Exception as e:
            error_msg = str(e)
            # Provide helpful hints for common errors
            if 'data_start_before_available_start' in error_msg:
                error_msg = (
                    f"Start date is before data availability.\n\n"
                    f"Databento XNAS/XNYS data starts from May 2018.\n"
                    f"Please set a start date of 2018-05-01 or later."
                )
            elif 'not_found' in error_msg.lower():
                error_msg = (
                    f"Symbol '{symbol}' not found on selected exchange.\n\n"
                    f"Try switching exchange:\n"
                    f"- NASDAQ: tech stocks (AAPL, MSFT, NDAQ)\n"
                    f"- NYSE/AMEX: ETFs and blue chips (SCHD, SPY, JPM)"
                )
            QMessageBox.critical(self, "Error", f"Failed to fetch data: {error_msg}")
            self.statusbar.showMessage("Data fetch failed")

        finally:
            self.fetch_btn.setEnabled(True)

    def _load_csv(self):
        """Load data from CSV file."""
        from data_providers.databento_provider import CSVDataProvider

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load CSV Data", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not filepath:
            return

        try:
            provider = CSVDataProvider()
            df = provider.load_csv(filepath, rth_only=self.rth_only.isChecked())

            self.current_data = df
            self.data_status.setText(
                f"Loaded {len(df)} bars from CSV | {df['time'].min()} to {df['time'].max()}"
            )
            self.data_status.setStyleSheet("color: #2ecc71;")
            self.statusbar.showMessage(f"Loaded {len(df)} bars from {Path(filepath).name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")

    def _load_pine_file(self):
        """Load PineScript file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Pine Script", "", "Pine Script (*.pine *.txt);;All Files (*)"
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            self.code_editor.setText(content)
            self.statusbar.showMessage(f"Loaded strategy from {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def _save_pine_file(self):
        """Save PineScript to file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Pine Script", "", "Pine Script (*.pine);;All Files (*)"
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.code_editor.toPlainText())
            self.statusbar.showMessage(f"Saved strategy to {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

    def _run_backtest(self):
        """Run the backtest."""
        if self.current_data is None or self.current_data.empty:
            QMessageBox.warning(self, "Error", "Please load data first")
            return

        pine_code = self.code_editor.toPlainText().strip()
        if not pine_code:
            QMessageBox.warning(self, "Error", "Please enter or load a PineScript strategy")
            return

        # Disable run button
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Prepare config
        config = {
            'initial_capital': self.capital_input.value(),
            'commission_pct': self.commission_input.value(),
            'order_size_pct': self.order_size_input.value(),
        }

        # Run in background thread
        self.worker = BacktestWorker(self.current_data.copy(), pine_code, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_backtest_complete)
        self.worker.error.connect(self._on_backtest_error)
        self.worker.settings_parsed.connect(self._on_settings_parsed)
        self.worker.start()

    def _on_progress(self, value, message):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
        self.statusbar.showMessage(message)

    def _on_backtest_complete(self, result):
        """Handle backtest completion."""
        self.current_result = result
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Update metrics
        self._update_metrics(result)

        # Update trades table
        self._update_trades_table(result)

        # Render charts
        self._render_charts(result)

        # Show summary tab
        self.results_tabs.setCurrentIndex(0)

        self.statusbar.showMessage(
            f"Backtest complete: {result.total_trades} trades, "
            f"{result.win_rate:.1f}% win rate, "
            f"${result.total_pnl:,.2f} P&L"
        )

    def _on_backtest_error(self, error):
        """Handle backtest error."""
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Backtest Error", error)
        self.statusbar.showMessage("Backtest failed")

    def _render_charts(self, result):
        """Render equity curve and drawdown charts using Agg backend."""
        if result.equity_curve is None or len(result.equity_curve) == 0:
            self.chart_label.setText("No data to chart")
            return

        try:
            import matplotlib
            if matplotlib.get_backend().lower() != 'agg':
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor='#1e1e1e')

            for ax in [ax1, ax2]:
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='#cccccc')
                ax.xaxis.label.set_color('#cccccc')
                ax.yaxis.label.set_color('#cccccc')
                ax.title.set_color('#cccccc')
                for spine in ax.spines.values():
                    spine.set_color('#444444')

            x = range(len(result.equity_curve))

            # Equity curve
            eq = result.equity_curve
            ax1.plot(x, eq, color='#2ecc71', linewidth=1)
            ax1.set_ylabel('Equity ($)', color='#cccccc')
            ax1.set_title('Equity Curve', color='#cccccc')
            ax1.grid(True, alpha=0.2, color='#444444')
            initial = eq[0]
            ax1.fill_between(x, eq, initial,
                             where=[e >= initial for e in eq],
                             alpha=0.1, color='#2ecc71')
            ax1.fill_between(x, eq, initial,
                             where=[e < initial for e in eq],
                             alpha=0.1, color='#e74c3c')

            # Drawdown
            dd = result.drawdown_curve
            ax2.fill_between(x, dd, 0, color='#e74c3c', alpha=0.3)
            ax2.plot(x, dd, color='#e74c3c', linewidth=1)
            ax2.set_ylabel('Drawdown ($)', color='#cccccc')
            ax2.set_xlabel('Bar', color='#cccccc')
            ax2.set_title('Drawdown', color='#cccccc')
            ax2.grid(True, alpha=0.2, color='#444444')

            fig.tight_layout()

            # Render to QPixmap
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, facecolor='#1e1e1e')
            plt.close(fig)
            buf.seek(0)

            from PySide6.QtGui import QPixmap
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            self.chart_label.setPixmap(pixmap)
            self.chart_label.setAlignment(Qt.AlignCenter)

        except Exception as e:
            self.chart_label.setText(f"Chart rendering failed: {str(e)}")

    def _on_settings_parsed(self, settings):
        """Update GUI fields when strategy settings are parsed from PineScript."""
        self.capital_input.setValue(settings.initial_capital)
        self.commission_input.setValue(settings.commission_value)
        if settings.default_qty_type == "percent_of_equity":
            self.order_size_input.setValue(settings.default_qty_value)

    def _update_metrics(self, result):
        """Update metrics display."""
        formatters = {
            'total_trades': lambda x: str(int(x)),
            'winning_trades': lambda x: str(int(x)),
            'losing_trades': lambda x: str(int(x)),
            'win_rate': lambda x: f"{x:.2f}%",
            'total_pnl': lambda x: f"${x:,.2f}",
            'avg_pnl': lambda x: f"${x:,.2f}",
            'avg_winner': lambda x: f"${x:,.2f}",
            'avg_loser': lambda x: f"${x:,.2f}",
            'profit_factor': lambda x: f"{x:.2f}",
            'max_drawdown': lambda x: f"${x:,.2f}",
            'max_drawdown_pct': lambda x: f"{x:.2f}%",
            'sharpe_ratio': lambda x: f"{x:.2f}",
            'sortino_ratio': lambda x: f"{x:.2f}",
            'calmar_ratio': lambda x: f"{x:.2f}",
        }

        for key, label in self.metric_labels.items():
            value = getattr(result, key, 0)
            formatter = formatters.get(key, str)
            label.setText(formatter(value))

            # Color coding for P&L
            if 'pnl' in key.lower() and value != 0:
                color = "#2ecc71" if value > 0 else "#e74c3c"
                label.setStyleSheet(f"color: {color};")

    def _update_trades_table(self, result):
        """Update trades table."""
        self.trades_table.setRowCount(0)

        for trade in result.trades:
            row = self.trades_table.rowCount()
            self.trades_table.insertRow(row)

            items = [
                str(trade.trade_id),
                str(trade.entry_time) if trade.entry_time else "",
                f"${trade.entry_price:.2f}" if trade.entry_price else "",
                str(trade.exit_time) if trade.exit_time else "OPEN",
                f"${trade.exit_price:.2f}" if trade.exit_price else "",
                trade.exit_signal.value if trade.exit_signal else "",
                str(trade.qty),
                f"${trade.pnl:.2f}" if trade.pnl else "",
                f"{trade.pnl_pct:.2f}%" if trade.pnl_pct else "",
                str(trade.bars_in_trade),
            ]

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)

                # Color code P&L
                if col == 7 and trade.pnl:
                    color = QColor("#2ecc71") if trade.pnl > 0 else QColor("#e74c3c")
                    item.setForeground(color)

                self.trades_table.setItem(row, col, item)

    def _export_results(self):
        """Export results to file."""
        if self.current_result is None:
            QMessageBox.warning(self, "Error", "No results to export")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON Files (*.json);;All Files (*)"
        )

        if not filepath:
            return

        try:
            data = self.current_result.to_dict()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.statusbar.showMessage(f"Results exported to {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    def _export_trades_csv(self):
        """Export trades to CSV."""
        if self.current_result is None:
            QMessageBox.warning(self, "Error", "No trades to export")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Trades", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not filepath:
            return

        try:
            import pandas as pd
            trades_data = [t.to_dict() for t in self.current_result.trades]
            df = pd.DataFrame(trades_data)
            df.to_csv(filepath, index=False)
            self.statusbar.showMessage(f"Trades exported to {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About PineScript Backtest Engine",
            "PineScript Backtest Engine v1.0\n\n"
            "A professional desktop application for backtesting\n"
            "TradingView PineScript strategies locally.\n\n"
            "Features:\n"
            "- Databento data integration\n"
            "- PineScript strategy parsing\n"
            "- Accurate trade execution\n"
            "- Comprehensive metrics\n\n"
            "Built by WaleA-Dev"
        )


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
