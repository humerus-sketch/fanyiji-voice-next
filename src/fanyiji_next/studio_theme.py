"""Warm, quiet studio surfaces and resolution-independent application icons."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


BG = "#F6F3EE"
PAPER = "#FFFEFB"
INK = "#27352F"
MUTED = "#7B847B"
ACCENT = "#5F826B"
SOFT = "#E7EDE3"
BORDER = "#E7E2D8"
PEACH = "#F1E4D5"

# Secondary foregrounds are slightly darker than decorative muted labels so that
# small controls remain legible on warm paper at normal desktop scaling.
SECONDARY = "#626F64"
ACCENT_HOVER = "#50735C"
ACCENT_PRESSED = "#42624D"
FOCUS = "#829B79"


STUDIO_STYLE = """
QProgressBar { border: none; background: #E6E9DF; border-radius: 3px; }
QProgressBar::chunk { background: #78967B; border: none; border-radius: 3px; }
QWidget {
    color: #27352F;
    selection-background-color: #DDE7D8;
    selection-color: #27352F;
}
QMainWindow#studioRoot, QWidget#studioRoot {
    background-color: #F6F3EE;
}
QWidget#topbar, QFrame#topbar {
    background-color: #F6F3EE;
    border: none;
    border-bottom: 1px solid #E7E2D8;
}
QLabel {
    background: transparent;
    border: none;
}
QLabel#brandMark {
    background: #E7EDE3;
    color: #5F826B;
    border: 1px solid #D8E1D2;
    border-radius: 13px;
    padding: 5px;
}
QLabel#brandName {
    color: #27352F;
    font-size: 15pt;
    font-weight: 600;
}
QLabel#smallCaps {
    color: #626F64;
    font-size: 8pt;
    font-weight: 600;
}
QLabel#pageTitle {
    color: #27352F;
    font-size: 25pt;
    font-weight: 600;
}
QLabel#muted {
    color: #626F64;
}
QLabel#footerHint {
    color: #626F64;
    font-size: 9pt;
}
QWidget#paper, QFrame#paper {
    background: #FFFEFB;
    border: 1px solid #E7E2D8;
    border-radius: 20px;
}
QWidget#currentCard, QFrame#currentCard {
    background: #FFFEFB;
    border: 1px solid #E7E2D8;
    border-radius: 23px;
}
QLabel#channelTag {
    color: #50735C;
    background: #E7EDE3;
    border: 1px solid #DCE5D7;
    border-radius: 11px;
    padding: 4px 10px;
    font-size: 9pt;
    font-weight: 600;
}
QLabel#originalText {
    color: #626F64;
    background: transparent;
    border: none;
}
QLabel#translationText {
    color: #27352F;
    background: transparent;
    border: none;
}
QWidget#historyRow, QFrame#historyRow {
    background: #FFFEFB;
    border: 1px solid transparent;
    border-radius: 13px;
}
QWidget#historyRow:hover, QFrame#historyRow:hover {
    background: #F4F3EC;
    border-color: #E7E2D8;
}
QWidget#historySelected, QFrame#historySelected {
    background: #E7EDE3;
    border: 1px solid #CBD8C5;
    border-radius: 13px;
}
QWidget#drawer, QFrame#drawer {
    background: #FFFEFB;
    border: 1px solid #E7E2D8;
    border-radius: 19px;
}
QWidget#dock, QFrame#dock {
    background: #FFFEFB;
    border: 1px solid #E3DED3;
    border-radius: 19px;
}
QPushButton {
    color: #626F64;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 8px 12px;
    min-height: 22px;
}
QPushButton:hover {
    color: #27352F;
    background: #EEEFE7;
}
QPushButton:pressed {
    background: #E1E7DA;
}
QPushButton:checked {
    color: #42624D;
    background: #E7EDE3;
    border-color: #CDD9C8;
}
QPushButton:checked:hover {
    background: #DEE7D8;
}
QPushButton:focus {
    border: 1px solid #829B79;
}
QPushButton:disabled {
    color: #9CA297;
    background: transparent;
    border-color: transparent;
}
QPushButton#primaryButton {
    color: #FFFEFB;
    background: #5F826B;
    border: 1px solid #5F826B;
    border-radius: 12px;
    padding: 10px 24px;
    min-height: 24px;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background: #50735C;
    border-color: #50735C;
}
QPushButton#primaryButton:pressed, QPushButton#primaryButton:checked {
    background: #42624D;
    border-color: #42624D;
}
QPushButton#primaryButton:focus {
    border: 2px solid #AEC0A6;
    padding: 9px 23px;
}
QPushButton#primaryButton:disabled {
    background: #DCE3D7;
    color: #8D9988;
    border-color: #DCE3D7;
}
QPushButton#quietButton {
    background: transparent;
    border: 1px solid #E7E2D8;
    border-radius: 11px;
    padding: 8px 13px;
}
QPushButton#quietButton:hover {
    background: #F1F1E9;
    border-color: #D9DFD0;
}
QPushButton#quietButton:pressed, QPushButton#quietButton:checked {
    color: #42624D;
    background: #E7EDE3;
    border-color: #CDD9C8;
}
QPushButton#quietButton:focus {
    border-color: #829B79;
}
QPushButton#quietButton:disabled {
    background: transparent;
    color: #9CA297;
    border-color: #ECE9E1;
}
QPushButton#iconButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 7px;
    min-width: 22px;
    min-height: 22px;
}
QPushButton#iconButton:hover {
    background: #E7EDE3;
}
QPushButton#iconButton:pressed, QPushButton#iconButton:checked {
    background: #DEE7D8;
    border-color: #CDD9C8;
}
QPushButton#iconButton:focus {
    border-color: #829B79;
}
QPushButton#iconButton:disabled {
    background: transparent;
    border-color: transparent;
}
QPushButton#chipButton {
    background: #FFFEFB;
    border: 1px solid #E7E2D8;
    border-radius: 14px;
    padding: 5px 12px;
    min-height: 18px;
    font-size: 9pt;
}
QPushButton#chipButton:hover {
    background: #F1F1E9;
    border-color: #D3DDCD;
}
QPushButton#chipButton:pressed, QPushButton#chipButton:checked {
    color: #42624D;
    background: #E7EDE3;
    border-color: #C6D5BE;
}
QPushButton#chipButton:checked:hover {
    background: #DEE7D8;
}
QPushButton#chipButton:focus {
    border-color: #829B79;
}
QPushButton#chipButton:disabled {
    color: #9CA297;
    background: #F5F3ED;
    border-color: #ECE9E1;
}
QLabel#statusChip {
    color: #50735C;
    background: #E7EDE3;
    border: 1px solid #DCE5D7;
    border-radius: 12px;
    padding: 5px 11px;
    font-size: 9pt;
}
QLabel#statusChip[active="true"] {
    color: #42624D;
    background: #DDE8D8;
    border-color: #C6D7BE;
}
QLabel#notice {
    color: #775C40;
    background: #F1E4D5;
    border: 1px solid #E7D5BE;
    border-radius: 11px;
    padding: 11px 14px;
}
QLineEdit {
    background: #FFFEFB;
    color: #27352F;
    border: 1px solid #E0DDD2;
    border-radius: 9px;
    padding: 8px 10px;
    min-height: 20px;
}
QLineEdit#searchField {
    background: #F6F3EE;
    border-color: #E7E2D8;
    border-radius: 11px;
    padding: 8px 12px;
}
QLineEdit:focus, QLineEdit#searchField:focus {
    border-color: #829B79;
    background: #FFFEFB;
}
QLineEdit:disabled {
    color: #9CA297;
    background: #F5F3ED;
    border-color: #ECE9E1;
}
QComboBox, QSpinBox {
    background: #FFFEFB;
    color: #27352F;
    border: 1px solid #E0DDD2;
    border-radius: 9px;
    padding: 7px 10px;
    min-height: 20px;
}
QComboBox {
    padding-right: 29px;
}
QComboBox:hover, QSpinBox:hover {
    border-color: #BDCDB5;
}
QComboBox:focus, QSpinBox:focus {
    border-color: #829B79;
}
QComboBox:disabled, QSpinBox:disabled {
    color: #9CA297;
    background: #F5F3ED;
    border-color: #ECE9E1;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border: none;
    border-top-right-radius: 9px;
    border-bottom-right-radius: 9px;
    background: transparent;
}
QComboBox QAbstractItemView {
    background: #FFFEFB;
    color: #27352F;
    border: 1px solid #D9DFD0;
    outline: 0;
    padding: 5px;
    selection-background-color: #E7EDE3;
    selection-color: #42624D;
}
QSpinBox::up-button, QSpinBox::down-button {
    background: #F4F3EC;
    border: none;
    width: 22px;
}
QSpinBox::up-button {
    border-top-right-radius: 8px;
}
QSpinBox::down-button {
    border-bottom-right-radius: 8px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background: #E7EDE3;
}
QSlider::groove:horizontal {
    background: #E6E6DC;
    border: none;
    border-radius: 3px;
    height: 5px;
}
QSlider::sub-page:horizontal {
    background: #8BA17D;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFEFB;
    border: 2px solid #5F826B;
    border-radius: 7px;
    width: 11px;
    margin: -5px 0;
}
QSlider::handle:horizontal:hover, QSlider::handle:horizontal:focus {
    background: #E7EDE3;
    border-color: #42624D;
}
QSlider::handle:horizontal:disabled {
    border-color: #BAC5B3;
    background: #F6F3EE;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    border: none;
    width: 7px;
    margin: 3px 0;
}
QScrollBar::handle:vertical {
    background: #CFD5C8;
    border: none;
    border-radius: 3px;
    min-height: 32px;
}
QScrollBar::handle:vertical:hover {
    background: #A8B69D;
}
QScrollBar:horizontal {
    background: transparent;
    border: none;
    height: 7px;
    margin: 0 3px;
}
QScrollBar::handle:horizontal {
    background: #CFD5C8;
    border: none;
    border-radius: 3px;
    min-width: 32px;
}
QScrollBar::handle:horizontal:hover {
    background: #A8B69D;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: transparent;
    border: none;
    width: 0;
    height: 0;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}
QCheckBox {
    color: #626F64;
    spacing: 8px;
    background: transparent;
}
QCheckBox:hover {
    color: #27352F;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #C9D0C2;
    border-radius: 5px;
    background: #FFFEFB;
}
QCheckBox::indicator:hover {
    border-color: #829B79;
    background: #F0F3EC;
}
QCheckBox::indicator:checked {
    background: #5F826B;
    border-color: #5F826B;
}
QCheckBox::indicator:checked:hover {
    background: #50735C;
    border-color: #50735C;
}
QCheckBox:focus {
    color: #42624D;
}
QCheckBox:disabled {
    color: #9CA297;
}
QCheckBox::indicator:disabled {
    background: #EFEEE7;
    border-color: #DDDCD2;
}
QCheckBox::indicator:checked:disabled {
    background: #BCCBB5;
    border-color: #BCCBB5;
}
QToolTip {
    color: #27352F;
    background: #FFFEFB;
    border: 1px solid #D9DFD0;
    border-radius: 5px;
    padding: 6px 9px;
}
QDialog {
    background: #F6F3EE;
}
QSplitter::handle {
    background: transparent;
}
"""


def icon(name: str, color: str = "#65786B", size: int = 20) -> QIcon:
    """Draw a compact line icon; call after creating a QApplication.

    Paths use a 24-unit view box and a double-density transparent pixmap, so no
    system symbol font or downloaded asset is involved.
    """
    supported = {
        "play", "pause", "replay", "settings", "history", "copy", "export",
        "close", "chevron_down", "check", "layout", "focus", "search",
        "arrow_right",
    }
    if name not in supported:
        raise ValueError(f"Unknown studio icon: {name}")
    if type(size) is not int or size < 1:
        raise ValueError("Icon size must be a positive integer")
    ink = QColor(color)
    if not ink.isValid():
        raise ValueError(f"Invalid icon color: {color}")

    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(size / 24.0, size / 24.0)
        painter.setPen(QPen(ink, 2.0, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        def line(x1: float, y1: float, x2: float, y2: float) -> None:
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        def path(points: tuple[tuple[float, float], ...], closed: bool = False) -> None:
            shape = QPainterPath(QPointF(*points[0]))
            for point in points[1:]:
                shape.lineTo(QPointF(*point))
            if closed:
                shape.closeSubpath()
            painter.drawPath(shape)

        if name == "play":
            painter.setBrush(ink)
            path(((8, 5), (19, 12), (8, 19)), closed=True)
        elif name == "pause":
            line(8, 5, 8, 19)
            line(16, 5, 16, 19)
        elif name == "replay":
            arc = QPainterPath(QPointF(5, 8))
            arc.cubicTo(8, 3, 16, 3, 19, 9)
            arc.cubicTo(22, 16, 15, 22, 9, 19)
            arc.cubicTo(6, 18, 4, 16, 4, 13)
            painter.drawPath(arc)
            path(((5, 3), (5, 8), (10, 8)))
        elif name == "settings":
            line(4, 6, 20, 6)
            line(4, 12, 20, 12)
            line(4, 18, 20, 18)
            painter.setBrush(ink)
            for x, y in ((9, 6), (15, 12), (8, 18)):
                painter.drawEllipse(QPointF(x, y), 2.3, 2.3)
        elif name == "history":
            painter.drawEllipse(QRectF(4, 4, 16, 16))
            line(12, 7, 12, 12)
            line(12, 12, 16, 14)
        elif name == "copy":
            painter.drawRoundedRect(QRectF(8, 8, 12, 12), 2, 2)
            shape = QPainterPath(QPointF(16, 5))
            shape.lineTo(16, 4)
            shape.lineTo(4, 4)
            shape.lineTo(4, 16)
            shape.lineTo(5, 16)
            painter.drawPath(shape)
        elif name == "export":
            path(((4, 14), (4, 20), (20, 20), (20, 14)))
            line(12, 4, 12, 14)
            path(((8, 8), (12, 4), (16, 8)))
        elif name == "close":
            line(6, 6, 18, 18)
            line(18, 6, 6, 18)
        elif name == "chevron_down":
            path(((6, 9), (12, 15), (18, 9)))
        elif name == "check":
            path(((5, 12), (10, 17), (19, 7)))
        elif name == "layout":
            painter.drawRoundedRect(QRectF(3.5, 5, 17, 14), 2, 2)
            line(12, 5, 12, 19)
        elif name == "focus":
            for corner in (
                ((4, 9), (4, 4), (9, 4)),
                ((15, 4), (20, 4), (20, 9)),
                ((20, 15), (20, 20), (15, 20)),
                ((9, 20), (4, 20), (4, 15)),
            ):
                path(corner)
        elif name == "search":
            painter.drawEllipse(QRectF(4, 4, 12, 12))
            line(15, 15, 20, 20)
        elif name == "arrow_right":
            line(4, 12, 19, 12)
            path(((13, 6), (19, 12), (13, 18)))
    finally:
        painter.end()
    return QIcon(pixmap)
