"""The new application's independent visual theme."""

STYLE = """
QWidget { color: #e6edf5; font-family: 'Microsoft YaHei UI'; font-size: 13px; }
QMainWindow, QDialog { background: #111820; }
QFrame#sidebar { background: #19232f; border-right: 1px solid #293747; }
QFrame#sourceCard { background: #202e3e; border: 1px solid #304155; border-radius: 10px; }
QFrame#captionPanel { background: #182431; border: 1px solid #304155; border-radius: 12px; }
QFrame#captionRow { background: #203044; border: 1px solid #2c4057; border-radius: 9px; }
QFrame#emptyState { background: transparent; border: none; }
QLabel { background: transparent; border: none; }
QLabel#brand { font-size: 25px; font-weight: 700; color: #f1f6fb; }
QLabel#brandTag { color: #71d9bc; font-size: 11px; }
QLabel#pageTitle { font-size: 23px; font-weight: 700; color: #f4f8fc; }
QLabel#sectionTitle { font-size: 15px; font-weight: 600; }
QLabel#muted { color: #91a5ba; }
QLabel#original { color: #a6b8ca; }
QLabel#statusPill { background: #26384a; color: #a9c0d6; border-radius: 12px; padding: 5px 12px; }
QLabel#statusPill[active="true"] { background: #193e37; color: #76e5bd; }
QLabel#notice { background: #383024; color: #f4d49e; padding: 8px 12px; border-radius: 7px; }
QPushButton { background: #26394d; border: 1px solid #3c536b; border-radius: 7px; padding: 8px 13px; }
QPushButton:hover { background: #304a64; border-color: #7298b8; }
QPushButton:pressed { background: #1d3042; }
QPushButton:disabled { color: #75879a; border-color: #2e4052; background: #1d2a38; }
QPushButton#primary { background: #78dfb8; border: 1px solid #8be8c7; color: #132c27; font-weight: 700; padding: 10px 24px; }
QPushButton#primary:hover { background: #96efcd; }
QPushButton#primary:disabled { background: #304b43; border-color: #47675b; color: #8aa89a; }
QPushButton#stop { background: #463035; color: #ffbcc0; border-color: #80505b; }
QPushButton#quiet { background: transparent; border-color: #3b5267; padding: 4px 9px; color: #a8c8df; }
QPushButton:checked { border-color: #78dfb8; color: #8ae8c7; }
QComboBox, QSpinBox, QLineEdit { background: #111f2e; border: 1px solid #3b5269; border-radius: 6px; padding: 6px 8px; selection-background-color: #315a76; }
QComboBox:focus, QSpinBox:focus, QLineEdit:focus { border-color: #78dfb8; }
QComboBox QAbstractItemView { background: #1b2b3c; selection-background-color: #315675; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #688098; border-radius: 4px; background: #142332; }
QCheckBox::indicator:checked { background: #78dfb8; border-color: #78dfb8; }
QProgressBar { background: #314254; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #78dfb8; border-radius: 3px; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 7px; margin: 0; }
QScrollBar::handle:vertical { background: #50697f; min-height: 24px; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background: transparent; }
QToolTip { background: #25384b; color: #f1f6fa; border: 1px solid #63819d; padding: 6px; }
"""
