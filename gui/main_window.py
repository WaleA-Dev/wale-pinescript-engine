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

            self.progress.emit(40, "Configuring backtest...")

            # Configure engine
            config = BacktestConfig(
                initial_capital=self.config.get('initial_capital', 100000),
                commission_pct=self.config.get('commission_pct', 0.1),
                order_size_pct=self.config.get('order_size_pct', 100),
            )

            self.progress.emit(60, "Running backtest...")

            # Run backtest
            engine = BacktestEngine(config=config, params=params)
            result = engine.run(self.df)

            self.progress.emit(100, "Complete!")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


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

        # Symbol and timeframe
        symbol_layout = QHBoxLayout()
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("NDAQ")
        self.symbol_input.setText("NDAQ")
        symbol_layout.addWidget(QLabel("Symbol:"))
        symbol_layout.addWidget(self.symbol_input)

        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1H', '4H', '1D'])
        self.timeframe_combo.setCurrentText('1H')
        symbol_layout.addWidget(QLabel("Timeframe:"))
        symbol_layout.addWidget(self.timeframe_combo)

        data_layout.addLayout(symbol_layout)

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
        """Create charts tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Placeholder for charts
        self.charts_label = QLabel("Run a backtest to see equity curve and charts")
        self.charts_label.setAlignment(Qt.AlignCenter)
        self.charts_label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self.charts_label)

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
        api_key = self.settings.value("databento_api_key", "")
        if api_key:
            self.api_key_input.setText(api_key)

        symbol = self.settings.value("last_symbol", "NDAQ")
        self.symbol_input.setText(symbol)

        timeframe = self.settings.value("last_timeframe", "1H")
        self.timeframe_combo.setCurrentText(timeframe)

    def _save_settings(self):
        """Save current settings."""
        self.settings.setValue("databento_api_key", self.api_key_input.text())
        self.settings.setValue("last_symbol", self.symbol_input.text())
        self.settings.setValue("last_timeframe", self.timeframe_combo.currentText())

    def closeEvent(self, event):
        """Handle window close."""
        self._save_settings()
        event.accept()

    # === Action Methods ===

    def _validate_api_key(self):
        """Validate Databento API key."""
        from data_providers.databento_provider import DatabentoProvider

        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Error", "Please enter an API key.")
            return

        self.validate_btn.setEnabled(False)
        self.statusbar.showMessage("Validating API key...")
        QApplication.processEvents()

        try:
            provider = DatabentoProvider(api_key)
            valid, message = provider.validate_api_key()
            if valid:
                QMessageBox.information(self, "API Key Valid", message)
                self.statusbar.showMessage("API key validated successfully.")
            else:
                QMessageBox.warning(self, "Invalid API Key", message)
                self.statusbar.showMessage("API key validation failed.")
        except Exception as e:
            err = str(e).strip() or repr(e)
            QMessageBox.critical(
                self,
                "Validation Error",
                f"Could not validate API key:\n\n{err}",
            )
            self.statusbar.showMessage("API key validation error.")
        finally:
            self.validate_btn.setEnabled(True)

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

        self.statusbar.showMessage(f"Fetching {symbol} data...")
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

            df = provider.fetch_ohlcv(
                symbol=symbol,
                start_date=start,
                end_date=end,
                timeframe=self.timeframe_combo.currentText(),
                rth_only=self.rth_only.isChecked()
            )

            self.current_data = df
            self.data_status.setText(
                f"Loaded {len(df)} bars | {df['time'].min()} to {df['time'].max()}"
            )
            self.data_status.setStyleSheet("color: #2ecc71;")
            self.statusbar.showMessage(f"Loaded {len(df)} bars for {symbol}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch data: {str(e)}")
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
