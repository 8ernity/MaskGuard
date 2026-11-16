"""
Modern styling and theme definitions for MaskGuard.
Features a polished dark charcoal palette with emerald green accents and a clean light mode.
"""

DARK_THEME_QSS = """
/* Global Base */
QWidget {
    background-color: #0D1117;
    color: #F0F6FC;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    selection-background-color: #10B981;
    selection-color: #FFFFFF;
}

/* Sidebar Navigation */
QFrame#Sidebar {
    background-color: #161B22;
    border-right: 1px solid #30363D;
    min-width: 220px;
    max-width: 240px;
}

QLabel#AppTitle {
    color: #10B981;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 18px 16px 6px 16px;
}

QLabel#AppSubtitle {
    color: #8B949E;
    font-size: 11px;
    font-weight: 500;
    padding: 0px 16px 16px 16px;
}

QPushButton.NavButton {
    background-color: transparent;
    color: #C9D1D9;
    font-size: 13px;
    font-weight: 600;
    text-align: left;
    padding: 10px 16px;
    border: none;
    border-radius: 6px;
    margin: 3px 10px;
}

QPushButton.NavButton:hover {
    background-color: #21262D;
    color: #F0F6FC;
}

QPushButton.NavButton:checked, QPushButton.NavButton[active="true"] {
    background-color: #10B981;
    color: #FFFFFF;
    font-weight: 700;
}

/* Card Containers */
QFrame.Card {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 16px;
}

QFrame.StatCard {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 16px;
    min-width: 140px;
}

QLabel.CardTitle {
    color: #8B949E;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

QLabel.StatValue {
    color: #F0F6FC;
    font-size: 26px;
    font-weight: 700;
    margin: 4px 0px;
}

QLabel.SectionHeader {
    color: #F0F6FC;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Buttons */
QPushButton.PrimaryButton {
    background-color: #10B981;
    color: #FFFFFF;
    font-weight: 600;
    font-size: 13px;
    padding: 9px 18px;
    border-radius: 6px;
    border: none;
}

QPushButton.PrimaryButton:hover {
    background-color: #059669;
}

QPushButton.PrimaryButton:pressed {
    background-color: #047857;
}

QPushButton.PrimaryButton:disabled {
    background-color: #21262D;
    color: #8B949E;
}

QPushButton.SecondaryButton {
    background-color: #21262D;
    color: #F0F6FC;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid #30363D;
}

QPushButton.SecondaryButton:hover {
    background-color: #30363D;
    border-color: #8B949E;
}

QPushButton.DangerButton {
    background-color: #EF4444;
    color: #FFFFFF;
    font-weight: 600;
    font-size: 13px;
    padding: 9px 18px;
    border-radius: 6px;
    border: none;
}

QPushButton.DangerButton:hover {
    background-color: #DC2626;
}

/* Status Badges */
QLabel.BadgeOnline {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border: 1px solid #10B981;
    border-radius: 12px;
}

QLabel.BadgeOffline {
    background-color: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border: 1px solid #EF4444;
    border-radius: 12px;
}

QLabel.BadgeWarning {
    background-color: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border: 1px solid #F59E0B;
    border-radius: 12px;
}

/* Video Preview Viewport */
QLabel#VideoViewport {
    background-color: #010409;
    border: 1px solid #30363D;
    border-radius: 8px;
}

/* Tables */
QTableWidget {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 6px;
    gridline-color: #30363D;
    color: #F0F6FC;
}

QHeaderView::section {
    background-color: #21262D;
    color: #8B949E;
    font-weight: 600;
    padding: 6px;
    border: 1px solid #30363D;
}

/* Inputs & Sliders */
QLineEdit, QComboBox, QSpinBox {
    background-color: #21262D;
    color: #F0F6FC;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 6px 10px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border-color: #10B981;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #30363D;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #10B981;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #10B981;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

/* Progress Bar */
QProgressBar {
    background-color: #21262D;
    border: 1px solid #30363D;
    border-radius: 6px;
    text-align: center;
    color: #FFFFFF;
    font-weight: 600;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #10B981;
    border-radius: 5px;
}

/* CheckBox */
QCheckBox {
    color: #F0F6FC;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #30363D;
    background-color: #21262D;
}

QCheckBox::indicator:checked {
    background-color: #10B981;
    border-color: #10B981;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #0D1117;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #30363D;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #484F58;
}
"""

LIGHT_THEME_QSS = """
/* Global Base Light Mode */
QWidget {
    background-color: #F8FAFC;
    color: #0F172A;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    selection-background-color: #10B981;
    selection-color: #FFFFFF;
}

/* Sidebar Navigation */
QFrame#Sidebar {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
    min-width: 220px;
    max-width: 240px;
}

QLabel#AppTitle {
    color: #059669;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 18px 16px 6px 16px;
}

QLabel#AppSubtitle {
    color: #64748B;
    font-size: 11px;
    font-weight: 500;
    padding: 0px 16px 16px 16px;
}

QPushButton.NavButton {
    background-color: transparent;
    color: #475569;
    font-size: 13px;
    font-weight: 600;
    text-align: left;
    padding: 10px 16px;
    border: none;
    border-radius: 6px;
    margin: 3px 10px;
}

QPushButton.NavButton:hover {
    background-color: #F1F5F9;
    color: #0F172A;
}

QPushButton.NavButton:checked, QPushButton.NavButton[active="true"] {
    background-color: #10B981;
    color: #FFFFFF;
    font-weight: 700;
}

/* Card Containers */
QFrame.Card, QFrame.StatCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px;
}

QLabel.CardTitle {
    color: #64748B;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

QLabel.StatValue {
    color: #0F172A;
    font-size: 26px;
    font-weight: 700;
    margin: 4px 0px;
}

QLabel.SectionHeader {
    color: #0F172A;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Buttons */
QPushButton.PrimaryButton {
    background-color: #10B981;
    color: #FFFFFF;
    font-weight: 600;
    font-size: 13px;
    padding: 9px 18px;
    border-radius: 6px;
    border: none;
}

QPushButton.PrimaryButton:hover {
    background-color: #059669;
}

QPushButton.SecondaryButton {
    background-color: #F1F5F9;
    color: #0F172A;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid #CBD5E1;
}

QPushButton.SecondaryButton:hover {
    background-color: #E2E8F0;
}

QPushButton.DangerButton {
    background-color: #EF4444;
    color: #FFFFFF;
    font-weight: 600;
    font-size: 13px;
    padding: 9px 18px;
    border-radius: 6px;
    border: none;
}

/* Status Badges */
QLabel.BadgeOnline {
    background-color: rgba(16, 185, 129, 0.15);
    color: #059669;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border: 1px solid #10B981;
    border-radius: 12px;
}

QLabel.BadgeOffline {
    background-color: rgba(239, 68, 68, 0.15);
    color: #DC2626;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border: 1px solid #EF4444;
    border-radius: 12px;
}

/* Video Preview Viewport */
QLabel#VideoViewport {
    background-color: #0F172A;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
}

/* Tables */
QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    gridline-color: #E2E8F0;
    color: #0F172A;
}

QHeaderView::section {
    background-color: #F1F5F9;
    color: #64748B;
    font-weight: 600;
    padding: 6px;
    border: 1px solid #E2E8F0;
}

QLineEdit, QComboBox, QSpinBox {
    background-color: #FFFFFF;
    color: #0F172A;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 10px;
}

QProgressBar {
    background-color: #E2E8F0;
    border-radius: 6px;
    text-align: center;
    color: #0F172A;
    font-weight: 600;
}

QProgressBar::chunk {
    background-color: #10B981;
    border-radius: 5px;
}
"""


def get_stylesheet(theme: str = "dark") -> str:
    """Return QSS stylesheet string for the requested theme."""
    return LIGHT_THEME_QSS if theme.lower() == "light" else DARK_THEME_QSS
