from PyQt6.QtWidgets import QLabel, QApplication, QButtonGroup, QPushButton, QMainWindow, QRadioButton,\
QButtonGroup, QTextEdit, QCheckBox, QComboBox
import sys
import os
import coins
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtGui import QFont
import finance_control

class myWindow(QMainWindow):
    def __init__(self):
        screen = QGuiApplication.primaryScreen()
        screen_g = screen.geometry()
        self.screen_width = screen_g.width()/5.5
        self.screen_height = screen_g.height()/8
        # 1.1
        super().__init__()
        self.setWindowTitle("Crypto trading")
        self.resize(800, 530)
        self.move(int(self.screen_width), int(self.screen_height))
        self.setStyleSheet("background-color: #303541")
        self.title = QLabel("Select indicator", self)
        self.title.move(int(self.screen_width/7.5), int(self.screen_height/2.5))
        self.title.resize(200, 35)
        self.title.setStyleSheet("color: #FFFFFF; font-family: 'Inter'; \
        font-size: 25px")
        # ----------------------------------------------------------
        self.sma_ = QRadioButton("SMA Trend", self)
        self.sma_.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.sma_.move(int(self.screen_width/8), int(self.screen_height/0.9))
        self.sma_.toggled.connect(self.checking_sma)

        self.sma_coin_lab = QLabel("Coin:", self)
        self.sma_coin_lab.setVisible(False)
        self.sma_coin_lab.move(int(self.screen_width/0.8), int(self.screen_height/0.9))
        self.sma_coin_lab.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")

        self.sma_coin = QComboBox(self)
        self.sma_coin.setVisible(False)
        self.sma_coin.setEnabled(True)
        self.sma_coin.move(int(self.screen_width/0.6), int(self.screen_height/0.9))
        self.sma_coin.resize(200, 30)
        coin = coins.coins_()
        self.sma_coin.addItems(coin)
        self.sma_coin.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; font-size: 15px;\
        border: none")
        # -
        self.sma_days_lab = QLabel("Days:", self)
        self.sma_days_lab.setVisible(False)
        self.sma_days_lab.move(int(self.screen_width/0.8), int(self.screen_height/0.6))
        self.sma_days_lab.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")

        self.sma_days_s = QTextEdit(self)
        self.sma_days_s.setVisible(False)
        self.sma_days_s.move(int(self.screen_width/0.6), int(self.screen_height/0.59))
        self.sma_days_s.setPlaceholderText("20")
        self.sma_days_s.setToolTip("Min period")
        self.sma_days_s.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; font-size: 15px;\
        border: none")

        self.sma_days_b = QTextEdit(self)
        self.sma_days_b.setVisible(False)
        self.sma_days_b.move(int(self.screen_width/0.52), int(self.screen_height/0.59))
        self.sma_days_b.setPlaceholderText("200")
        self.sma_days_s.setToolTip("Medium period")
        self.sma_days_b.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; font-size: 15px;\
        border: none")

        self.sma_days_all_checkbox = QCheckBox("Apply schedule", self)
        self.sma_days_all_checkbox.setVisible(False)
        self.sma_days_all_checkbox.move(int(self.screen_width/0.45), int(self.screen_height/0.9))
        self.sma_days_all_checkbox.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.sma_days_all_checkbox.resize(130, 30)
        self.sma_days_all_checkbox.toggled.connect(self.check_trend)

        self.sma_days_all = QTextEdit(self)
        self.sma_days_all.setVisible(False)
        # self.sma_days_all.setEnabled(False)
        self.sma_days_all.move(int(self.screen_width/0.45), int(self.screen_height/0.59))
        self.sma_days_all.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma';\
        border: none; font-size: 15px")
        self.sma_days_all.setPlaceholderText("2000")
        self.sma_days_all.setToolTip("Max period")

        self.sma_button = QPushButton("Submit", self)
        self.sma_button.setVisible(False)
        self.sma_button.move(int(self.screen_width/0.8), int(self.screen_height/0.45))
        self.sma_button.resize(200, 30)
        self.sma_button.setStyleSheet("QPushButton{border: none; color: black; border-radius: 10px 10px 10px 10px;\
        background-color: #D9D9D9; font-size: 13px; font-family: 'Inter'}\
        QPushButton:hover{background-color: #B9B9B9}\
        QPushButton:pressed{background-color: #9B9B9B}")
        self.sma_button.clicked.connect(self.sma_trend)


        # ----------------------------------------------------------


        self.rolling_ema_ = QRadioButton("Gibrid EMA, Rolling EMA Trend", self)
        self.rolling_ema_.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.rolling_ema_.resize(220, 30)
        self.rolling_ema_.move(int(self.screen_width/8), int(self.screen_height/2)+100)
        self.rolling_ema_.toggled.connect(self.checking_rolling_ema)

        self.rolling_ema_button = QPushButton("Submit", self)
        self.rolling_ema_button.setVisible(False)
        self.rolling_ema_button.move(int(self.screen_width/0.8), int(self.screen_height/0.45))
        self.rolling_ema_button.resize(200, 30)
        self.rolling_ema_button.setStyleSheet("QPushButton{border: none; color: black; border-radius: 10px 10px 10px 10px;\
        background-color: #D9D9D9; font-size: 13px; font-family: 'Inter'}\
        QPushButton:hover{background-color: #B9B9B9}\
        QPushButton:pressed{background-color: #9B9B9B}")
        self.rolling_ema_button.clicked.connect(self.rolling_ema_trend)

        
        # ----------------------------------------------------------

        self.ema_ = QRadioButton("Classic EMA Trend", self)
        self.ema_.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.ema_.toggled.connect(self.checking_classic_ema)

        self.ema_trend = QCheckBox("Check trend domination", self)
        self.ema_trend.setVisible(False)
        self.ema_trend.move(int(self.screen_width/0.45), int(self.screen_height/0.9))
        self.ema_trend.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.ema_trend.resize(190, 30)
        self.ema_trend.toggled.connect(self.check_box_ema)

        self.ema_.move(int(self.screen_width/8), int(self.screen_height/2)+145)
        self.ema_button = QPushButton("Submit", self)
        self.ema_button.setVisible(False)
        self.ema_button.move(int(self.screen_width/0.8), int(self.screen_height/0.45))
        self.ema_button.resize(200, 30)
        self.ema_button.setStyleSheet("QPushButton{border: none; color: black; border-radius: 10px 10px 10px 10px;\
        background-color: #D9D9D9; font-size: 13px; font-family: 'Inter'}\
        QPushButton:hover{background-color: #B9B9B9}\
        QPushButton:pressed{background-color: #9B9B9B}")
        self.ema_button.clicked.connect(self.classic_ema_trend)
        
        # ----------------------------------------------------------


        self.rsi_ = QRadioButton("RSI indicator", self)
        self.rsi_.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.rsi_.move(int(self.screen_width/8), int(self.screen_height/2)+190)
        self.rsi_.resize(200, 35)
        self.rsi_.toggled.connect(self.checking_rsi)

        self.rsi_button = QPushButton("Submit", self)
        self.rsi_button.setVisible(False)
        self.rsi_button.move(int(self.screen_width/0.8), int(self.screen_height/0.45))
        self.rsi_button.resize(200, 30)
        self.rsi_button.setStyleSheet("QPushButton{border: none; color: black; border-radius: 10px 10px 10px 10px;\
        background-color: #D9D9D9; font-size: 13px; font-family: 'Inter'}\
        QPushButton:hover{background-color: #B9B9B9}\
        QPushButton:pressed{background-color: #9B9B9B}")
        self.check_pair = QCheckBox("SMA+RSI", self)
        self.check_pair.setVisible(False)
        self.check_pair.move(int(self.screen_width/0.45), int(self.screen_height/0.9))
        self.check_pair.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.check_pair.resize(130, 30)
        self.check_pair.toggled.connect(self.check_box_pair)
        self.rsi_button.clicked.connect(self.rsi_trend)
        # self.sma_days_all_checkbox.toggled.connect(self.check_trend)

        # ----------------------------------------------------------


        self.ichimoku_cloud_ = QRadioButton("Ichimoku Cloud", self)
        self.ichimoku_cloud_.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.ichimoku_cloud_.move(int(self.screen_width/8), int(self.screen_height/2)+235)
        self.ichimoku_cloud_.resize(200, 35)
        self.ichimoku_cloud_.toggled.connect(self.checking_ichimoku)

        self.ichimoku_cloud_button = QPushButton("Submit", self)
        self.ichimoku_cloud_button.setVisible(False)
        self.ichimoku_cloud_button.move(int(self.screen_width/0.8), int(self.screen_height/0.45))
        self.ichimoku_cloud_button.resize(200, 30)
        self.ichimoku_cloud_button.setStyleSheet("QPushButton{border: none; color: black; border-radius: 10px 10px 10px 10px;\
        background-color: #D9D9D9; font-size: 13px; font-family: 'Inter'}\
        QPushButton:hover{background-color: #B9B9B9}\
        QPushButton:pressed{background-color: #9B9B9B}")
        self.ichimoku_cloud_button.clicked.connect(self.ichimoku_trend)


        # ----------------------------------------------------------


        self.macd_ = QRadioButton("MACD indicator", self)
        self.macd_.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.macd_.move(int(self.screen_width/8), int(self.screen_height/2)+280)
        self.macd_.resize(200, 35)

        self.macd_.toggled.connect(self.checking_macd)

        self.macd_button = QPushButton("Submit", self)
        self.macd_button.setVisible(False)
        self.macd_button.move(int(self.screen_width/0.8), int(self.screen_height/0.45))
        self.macd_button.resize(200, 30)
        self.macd_button.setStyleSheet("QPushButton{border: none; color: black; border-radius: 10px 10px 10px 10px;\
        background-color: #D9D9D9; font-size: 13px; font-family: 'Inter'}\
        QPushButton:hover{background-color: #B9B9B9}\
        QPushButton:pressed{background-color: #9B9B9B}")
        self.macd_button.clicked.connect(self.macd_trend)
        

        # ----------------------------------------------------------


        self.stoch_osc_ = QRadioButton("Stochastic oscillator", self)
        self.stoch_osc_.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.stoch_osc_.move(int(self.screen_width/8), int(self.screen_height/2)+325)
        self.stoch_osc_.resize(200, 35)
        self.stoch_osc_.toggled.connect(self.checking_ost)
        
        self.stoch_button = QPushButton("Submit", self)
        self.stoch_button.setVisible(False)
        self.stoch_button.move(int(self.screen_width/0.8), int(self.screen_height/0.45))
        self.stoch_button.resize(200, 30)
        self.stoch_button.setStyleSheet("QPushButton{border: none; color: black; border-radius: 10px 10px 10px 10px;\
        background-color: #D9D9D9; font-size: 13px; font-family: 'Inter'}\
        QPushButton:hover{background-color: #B9B9B9}\
        QPushButton:pressed{background-color: #9B9B9B}")
        
        self.stoch_button.clicked.connect(self.stoch_trend)


        # ----------------------------------------------------------


        self.cointegration_ = QRadioButton("Cointegration indicator", self)
        self.cointegration_.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.cointegration_.move(int(self.screen_width/8), int(self.screen_height/2)+370)
        self.cointegration_.resize(200, 35)
        self.cointegration_.toggled.connect(self.checking_coo)
        
        self.cointegration_coin = QComboBox(self)
        self.cointegration_coin.setVisible(False)
        # self.cointegration_coin.setPlaceholderText("TRX-USDT")
        self.cointegration_coin.addItems(coin)
        self.cointegration_coin.setToolTip("Select second coin for checking cointegration")
        self.cointegration_coin.move(int(self.screen_width/0.45), int(self.screen_height/0.9))
        self.cointegration_coin.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; font-size: 15px; \
        border: none")

        self.coo_button = QPushButton("Submit", self)
        self.coo_button.setVisible(False)
        self.coo_button.move(int(self.screen_width/0.8), int(self.screen_height/0.45))
        self.coo_button.resize(200, 30)
        self.coo_button.setStyleSheet("QPushButton{border: none; color: black; border-radius: 10px 10px 10px 10px;\
        background-color: #D9D9D9; font-size: 13px; font-family: 'Inter'}\
        QPushButton:hover{background-color: #B9B9B9}\
        QPushButton:pressed{background-color: #9B9B9B}")
        self.coo_button.clicked.connect(self.co_trend)


        # ----------------------------------------------------------


        self.elliot_waves_ = QRadioButton("Elliot waves theory", self)
        self.elliot_waves_.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.elliot_waves_.move(int(self.screen_width/8), int(self.screen_height/2)+415)
        self.elliot_waves_.resize(200, 35)

        self.ell_depth = QLabel("Depth: ", self)
        self.ell_depth.setVisible(False)
        self.ell_depth.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.ell_depth.move(int(self.screen_width/0.8), int(self.screen_height/0.45))

        self.ell_depth_value = QTextEdit(self)
        self.ell_depth_value.setVisible(False)
        self.ell_depth_value.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; font-size: 15px;\
        border: none")
        self.ell_depth_value.setPlaceholderText("3")
        self.ell_depth_value.setToolTip("Depth - count of candles, where script finding extremums")
        self.ell_depth_value.move(int(self.screen_width/0.6), int(self.screen_height/0.45))

        self.ell_diviation_lab = QLabel("Diviation: ", self)
        # self.ell_diviation_lab.setPlaceholderText("5")
        self.ell_diviation_lab.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        font-size: 15px")
        self.ell_diviation_lab.move(int(self.screen_width/0.5), int(self.screen_height/0.45))
        self.ell_diviation_lab.setVisible(False)

        self.ell_diviation = QTextEdit(self)
        self.ell_diviation.setPlaceholderText("5")
        self.ell_diviation.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; font-size: 15px;\
        border: none")
        self.ell_diviation.move(int(self.screen_width/0.41), int(self.screen_height/0.45))
        self.ell_diviation.setToolTip("Diviation in percents(for example, 5 percents)")
        self.ell_diviation.setVisible(False)
        # self.ell_div
        # self.sma_days_lab.setStyleSheet("color: #FFFFFF; font-family: 'Tahoma'; \
        # font-size: 15px")


        self.elliot_waves_.toggled.connect(self.elliot_check)

        self.elliot_button = QPushButton("Submit", self)
        self.elliot_button.setVisible(False)
        self.elliot_button.move(int(self.screen_width/0.8), int(self.screen_height/0.38))
        self.elliot_button.resize(200, 30)
        self.elliot_button.setStyleSheet("QPushButton{border: none; color: black; border-radius: 10px 10px 10px 10px;\
        background-color: #D9D9D9; font-size: 13px; font-family: 'Inter'}\
        QPushButton:hover{background-color: #B9B9B9}\
        QPushButton:pressed{background-color: #9B9B9B}")
        self.elliot_button.clicked.connect(self.elliot_trend)


        # ------------------------------------------------------------
        self.radio_group = QButtonGroup(self)
        self.radio_group.addButton(self.sma_)
        self.radio_group.addButton(self.rolling_ema_)
        self.radio_group.addButton(self.ema_)
        self.radio_group.addButton(self.rsi_)
        self.radio_group.addButton(self.ichimoku_cloud_)
        self.radio_group.addButton(self.macd_)
        self.radio_group.addButton(self.stoch_osc_)
        self.radio_group.addButton(self.cointegration_)
        self.radio_group.addButton(self.elliot_waves_)
        # ----------------------------------------------
        self.terminal = QLabel(self)
        self.terminal.resize(int(self.screen_width/0.4), int(self.screen_height/0.45))
        self.terminal.setStyleSheet("border: none; background-color: #D9D9D9; border: solid 1px #D9D9D9;\
        border-radius: 20px;")
        self.terminal.move(int(self.screen_width/0.7), int(self.screen_height/0.33))

        self.text_terminal = QLabel("Terminal", self)
        self.text_terminal.setWordWrap(True)
        self.text_terminal.move(int(self.screen_width/0.65), int(self.screen_height/0.3))
        self.text_terminal.resize(400, 170)
        self.text_terminal.setStyleSheet("color: #000000; font-size: 13px; background-color: rgba(0, 0, 0, 0);")

    def elliot_trend(self):
        coin = self.sma_coin.currentText()
        days = self.sma_days_s.toPlainText()
        depth = abs(int(self.ell_depth_value.toPlainText()))
        diviation = abs(int(self.ell_diviation.toPlainText()))
        response = finance_control.elliot_waves(coin, abs(int(days)), depth, diviation)
        self.text_terminal.setText(response)

    def elliot_check(self):
        if self.elliot_waves_.isChecked():
            self.sma_coin_lab.setVisible(True)
            self.sma_coin.setVisible(True)
            self.sma_coin.setEnabled(True)
            self.sma_days_lab.setVisible(True)
            self.sma_days_s.setPlaceholderText("2000")
            self.sma_days_s.setToolTip("Max period")
            self.sma_days_s.setVisible(True)
            self.ell_depth.setVisible(True)
            self.ell_depth_value.setVisible(True)
            self.ell_diviation_lab.setVisible(True)
            self.ell_diviation.setVisible(True)
            self.elliot_button.setVisible(True)
        else:
            self.sma_coin_lab.setVisible(False)
            self.sma_coin.setVisible(False)
            self.sma_days_lab.setVisible(False)
            self.sma_days_s.setVisible(False)
            self.sma_days_s.setPlaceholderText("20")
            self.sma_days_s.setToolTip("Min period")
            self.ell_depth.setVisible(False)
            self.ell_depth_value.setVisible(False)
            self.ell_diviation_lab.setVisible(False)
            self.ell_diviation.setVisible(False)
            self.elliot_button.setVisible(False)

    def co_trend(self):
        coin = self.sma_coin.currentText()
        coin_2 = self.cointegration_coin.currentText()
        days = self.sma_days_s.toPlainText()
        # p = self.p_value.toPlainText()
        response = finance_control.cointegration(coin, coin_2, abs(int(days)))
        self.text_terminal.setText(response)

    def checking_coo(self):
        if self.cointegration_.isChecked():
            self.sma_coin_lab.setVisible(True)
            self.sma_coin.setVisible(True)
            self.sma_days_lab.setVisible(True)
            self.sma_days_s.setToolTip("Max period")
            self.sma_days_s.setPlaceholderText("2000")
            self.sma_days_s.setVisible(True)
            self.cointegration_coin.setVisible(True)
            self.coo_button.setVisible(True)
        else:
            self.sma_coin_lab.setVisible(False)
            self.sma_coin.setVisible(False)
            self.sma_days_lab.setVisible(False)
            self.sma_days_s.setToolTip("Min period")
            self.sma_days_s.setPlaceholderText("20")
            self.sma_days_s.setVisible(False)
            self.cointegration_coin.setVisible(False)
            self.coo_button.setVisible(False)
            self.text_terminal.setText("Terminal")

    def checking_sma(self):
        if self.sma_.isChecked():
            self.sma_coin_lab.setVisible(True)
            self.sma_days_lab.setVisible(True)
            self.sma_coin.setVisible(True)
            self.sma_days_s.setVisible(True)
            self.sma_days_s.setToolTip("Min period")
            self.sma_days_b.setToolTip("Medium period")
            self.sma_days_b.setVisible(True)
            self.sma_button.setVisible(True)
            self.sma_days_all_checkbox.setVisible(True)
        else:
            self.sma_coin_lab.setVisible(False)
            self.sma_days_lab.setVisible(False)
            self.sma_coin.setVisible(False)
            self.sma_days_s.setVisible(False)
            self.sma_days_b.setVisible(False)
            self.sma_button.setVisible(False)
            self.sma_days_all_checkbox.setVisible(False)
            self.sma_days_all_checkbox.setChecked(False)
            self.sma_days_all.setVisible(False)
            self.text_terminal.setText("Terminal")
    def check_trend(self):
        if self.sma_days_all_checkbox.isChecked():
            self.sma_days_all.setVisible(True)
            self.sma_days_all.move(int(self.screen_width/0.45), int(self.screen_height/0.59))
        else:
            self.sma_days_all.setVisible(False)
    def sma_trend(self):
        coin = self.sma_coin.currentText()
        period_s = self.sma_days_s.toPlainText()
        period_b = self.sma_days_b.toPlainText()
        if self.sma_days_all_checkbox.isChecked():
            days = self.sma_days_all.toPlainText()
            check = finance_control.trends_walk_sma(coin, abs(int(days)), abs(int(period_s)), abs(int(period_b)))
            self.text_terminal.setText(check)
        else:
            string, _, _, diff = finance_control.checking_trend(coin, abs(int(period_s)), abs(int(period_b)))
            text = f"Result: {string}\nDifference: {diff}."
            self.text_terminal.setText(text)
            print(text)
    def checking_rolling_ema(self):
        if self.rolling_ema_.isChecked():
            self.sma_coin_lab.setVisible(True)
            self.sma_days_lab.setVisible(True)
            self.sma_coin.setVisible(True)
            self.sma_days_s.setVisible(True)
            self.sma_days_b.setVisible(True)
            self.rolling_ema_button.setVisible(True)
            self.sma_days_all.setEnabled(True)
            self.sma_days_all.setVisible(True)
        else:
            self.sma_coin_lab.setVisible(False)
            self.sma_days_lab.setVisible(False)
            self.sma_coin.setVisible(False)
            self.sma_days_s.setVisible(False)
            self.sma_days_b.setVisible(False)
            self.rolling_ema_button.setVisible(False)
            self.sma_days_all.setVisible(False)
            self.text_terminal.setText("Terminal")
    def rolling_ema_trend(self):
        coin = self.sma_coin.currentText()
        period_1 = self.sma_days_s.toPlainText()
        period_2 = self.sma_days_b.toPlainText()
        days = self.sma_days_all.toPlainText()
        trend_walks, check, diff, check_1, diff_1 = finance_control.trend_walk_rolling_ema(coin, abs(int(days)), abs(int(period_1)), abs(int(period_2)))
        text = f"Trend: {trend_walks}\nResult: {check}\nDifference: {diff}"
        text += f"\nTrend: {trend_walks}\nResult: {check_1}\nDifference: {diff_1}"
        self.text_terminal.setText(text)
        
    def checking_classic_ema(self):
        if self.ema_.isChecked():
            self.sma_coin_lab.setVisible(True)
            self.sma_days_lab.setVisible(True)
            self.sma_coin.setVisible(True)
            self.sma_days_s.setVisible(True)
            self.ema_button.setVisible(True)
            self.sma_days_all.setEnabled(True)
            self.sma_days_all.move(int(self.screen_width/0.6), int(self.screen_height/0.59))
            self.ema_trend.setVisible(True)
            self.sma_days_all.setVisible(True)
        else:
            self.sma_coin_lab.setVisible(False)
            self.sma_days_lab.setVisible(False)
            self.sma_coin.setVisible(False)
            self.sma_days_s.setVisible(False)
            self.sma_days_b.setVisible(False)
            self.sma_days_all.move(int(self.screen_width/0.45), int(self.screen_height/0.59))
            self.ema_button.setVisible(False)
            self.ema_trend.setVisible(False)
            self.ema_trend.setChecked(False)
            self.sma_days_all.setVisible(False)
            self.text_terminal.setText("Terminal")
    def check_box_ema(self):
        if self.ema_trend.isChecked():
            self.sma_days_b.setVisible(True)
            self.sma_days_all.move(int(self.screen_width/0.5), int(self.screen_height/0.59))
            self.sma_days_b.move(int(self.screen_width/0.6), int(self.screen_height/0.59))
        else:
            self.sma_days_b.setVisible(False)
            self.sma_days_all.setVisible(True)
            self.sma_days_all.move(int(self.screen_width/0.6), int(self.screen_height/0.59))
    def classic_ema_trend(self):
        coin = self.sma_coin.currentText()
        period_1 = self.sma_days_s.toPlainText()
        period_2 = self.sma_days_b.toPlainText()
        days = self.sma_days_all.toPlainText()
        if self.ema_trend.isChecked():
            finance_control.trend_walk_classic_ema(coin, abs(int(period_1)), abs(int(period_2)), abs(int(days)))
        else:
            response = finance_control.check_ema(coin, abs(int(period_1)), abs(int(days)))
            self.text_terminal.setText(response)
    def check_box_pair(self):
        if self.check_pair.isChecked():
            self.sma_days_s.setVisible(True)
            self.sma_days_b.setVisible(True)
            self.sma_days_all.move(int(self.screen_width/0.45), int(self.screen_height/0.59))
        else: 
            self.sma_days_s.setVisible(False)
            self.sma_days_b.setVisible(False)
            self.sma_days_all.move(int(self.screen_width/0.6), int(self.screen_height/0.59))
    def checking_rsi(self):
        if self.rsi_.isChecked():
            self.sma_coin_lab.setVisible(True)
            self.sma_days_lab.setVisible(True)
            self.sma_coin.setVisible(True)
            self.sma_days_all.move(int(self.screen_width/0.6), int(self.screen_height/0.59))
            self.sma_days_all.setEnabled(True)
            self.sma_days_all.setVisible(True)
            self.check_pair.setVisible(True)
            self.rsi_button.setVisible(True)
        else:
            self.sma_coin_lab.setVisible(False)
            self.sma_days_lab.setVisible(False)
            self.sma_coin.setEnabled(False)
            self.sma_coin.setVisible(False)
            self.sma_days_s.setVisible(False)
            self.sma_days_b.setVisible(False)
            self.sma_days_all.move(int(self.screen_width/0.45), int(self.screen_height/0.59))
            self.sma_days_all.setVisible(False)
            self.check_pair.setVisible(False)
            self.check_pair.setChecked(False)
            self.rsi_button.setVisible(False)
            self.text_terminal.setText("Terminal")

    def rsi_trend(self):
        coin = self.sma_coin.currentText()
        days = self.sma_days_all.toPlainText()
        period_1 = self.sma_days_s.toPlainText()
        period_2 = self.sma_days_b.toPlainText()
        if self.check_pair.isChecked():
            response = finance_control.rsi_sma(coin, abs(int(days)), abs(int(period_1)), abs(int(period_2)))
            self.text_terminal.setText(response)
        else:
            response, _ = finance_control.rsi(coin, abs(int(days)))
            self.text_terminal.setText(response)
    def checking_ichimoku(self):
        if self.ichimoku_cloud_.isChecked():
            self.sma_coin_lab.setVisible(True)
            self.sma_coin.setVisible(True)
            self.sma_coin.setEnabled(True)
            self.sma_days_lab.setVisible(True)
            self.sma_days_s.setToolTip("Max period")
            self.sma_days_s.setPlaceholderText("2000")
            self.sma_days_s.setVisible(True)
            self.ichimoku_cloud_button.setVisible(True)
        else:
            self.sma_coin_lab.setVisible(False)
            self.sma_coin.setVisible(False)
            self.sma_days_lab.setVisible(False)
            self.sma_days_s.setVisible(False)
            self.sma_days_s.setToolTip("Min period")
            self.sma_days_s.setPlaceholderText("20")
            self.ichimoku_cloud_button.setVisible(False)
            self.text_terminal.setText("Terminal")
    def ichimoku_trend(self):
        coin = self.sma_coin.currentText()
        days = self.sma_days_s.toPlainText()
        response = finance_control.ichimoku_cloud(coin, abs(int(days)))
        self.text_terminal.setText(response)
    
    def macd_trend(self):
        coin = self.sma_coin.currentText()
        days = self.sma_days_s.toPlainText()
        response  = finance_control.macd(coin, abs(int(days)))
        self.text_terminal.setText(response)

    def checking_macd(self):
        if self.macd_.isChecked():
            self.sma_coin_lab.setVisible(True)
            self.sma_coin.setVisible(True)
            self.sma_days_lab.setVisible(True)
            self.sma_days_s.setVisible(True)
            self.sma_days_s.setPlaceholderText("2000")
            self.sma_days_s.setToolTip("Max period")
            self.macd_button.setVisible(True)
        else:
            self.sma_coin_lab.setVisible(False)
            self.sma_coin.setVisible(False)
            self.sma_days_lab.setVisible(False)
            self.sma_days_s.setVisible(False)
            self.sma_days_s.setPlaceholderText("20")
            self.sma_days_s.setToolTip("Min period")
            self.text_terminal.setText("Terminal")
            self.macd_button.setVisible(False)

    def stoch_trend(self):
        coin = self.sma_coin.currentText()
        period = self.sma_days_s.toPlainText()
        days = self.sma_days_b.toPlainText()
        response = finance_control.stochastic_oscillator(coin, abs(int(period)), abs(int(days)))
        self.text_terminal.setText(response)

    def checking_ost(self):
        if self.stoch_osc_.isChecked():
            self.sma_coin_lab.setVisible(True)
            self.sma_coin.setVisible(True)
            self.sma_days_lab.setVisible(True)
            self.sma_days_s.setVisible(True)
            self.sma_days_s.setText("14")
            self.sma_days_s.setToolTip("Min period")
            self.sma_days_b.setVisible(True)
            self.sma_days_b.setPlaceholderText("2000")
            self.sma_days_b.setToolTip("Max period")
            self.stoch_button.setVisible(True)
        else:
            self.sma_coin_lab.setVisible(False)
            self.sma_coin.setVisible(False)
            self.sma_days_lab.setVisible(False)
            self.sma_days_s.setVisible(False)
            self.sma_days_b.setVisible(False)
            self.sma_days_b.setPlaceholderText("200")
            self.sma_days_b.setToolTip("Medium period")
            self.sma_days_s.setText("")
            self.text_terminal.setText("Terminal")
            self.stoch_button.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = myWindow()
    window.show()
    sys.exit(app.exec())