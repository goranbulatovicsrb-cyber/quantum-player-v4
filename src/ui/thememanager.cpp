#include "thememanager.h"

ThemeManager* ThemeManager::s_instance = nullptr;

ThemeManager* ThemeManager::instance() {
    if (!s_instance) s_instance = new ThemeManager();
    return s_instance;
}

QString ThemeManager::themeName(Theme t) const {
    switch (t) {
    case Midnight:  return "Midnight";
    case Cyberpunk: return "Cyberpunk";
    case Ocean:     return "Ocean";
    case Aurora:    return "Aurora";
    case Carbon:    return "Carbon";
    default:        return "Unknown";
    }
}

QStringList ThemeManager::themeNames() const {
    QStringList names;
    for (int i = 0; i < ThemeCount; ++i)
        names << themeName(static_cast<Theme>(i));
    return names;
}

ThemeColors ThemeManager::colorsFor(Theme theme) {
    ThemeColors c;
    switch (theme) {
    case Midnight:
        c.background = QColor(0x08,0x08,0x18);
        c.surface    = QColor(0x0f,0x0f,0x2a);
        c.surfaceAlt = QColor(0x16,0x16,0x38);
        c.accent     = QColor(0x7b,0x68,0xff);
        c.accentDim  = QColor(0x3b,0x32,0x88);
        c.text       = QColor(0xe8,0xe8,0xff);
        c.textDim    = QColor(0x88,0x88,0xbb);
        c.border     = QColor(0x2a,0x2a,0x55);
        c.vizBar1    = QColor(0x22,0x22,0xff);
        c.vizBar2    = QColor(0x88,0x44,0xff);
        c.vizBar3    = QColor(0xff,0x44,0xff);
        c.vizPeak    = QColor(0xff,0xff,0xff);
        c.vizBg      = QColor(0x05,0x05,0x12);
        break;
    case Cyberpunk:
        c.background = QColor(0x04,0x05,0x0a);
        c.surface    = QColor(0x08,0x0c,0x14);
        c.surfaceAlt = QColor(0x0c,0x12,0x1e);
        c.accent     = QColor(0x00,0xff,0x88);
        c.accentDim  = QColor(0x00,0x88,0x44);
        c.text       = QColor(0xcc,0xff,0xee);
        c.textDim    = QColor(0x55,0x99,0x77);
        c.border     = QColor(0x00,0x44,0x33);
        c.vizBar1    = QColor(0x00,0x88,0xff);
        c.vizBar2    = QColor(0x00,0xff,0xcc);
        c.vizBar3    = QColor(0x00,0xff,0x44);
        c.vizPeak    = QColor(0x00,0xff,0xff);
        c.vizBg      = QColor(0x02,0x04,0x06);
        break;
    case Ocean:
        c.background = QColor(0x03,0x0c,0x18);
        c.surface    = QColor(0x05,0x14,0x28);
        c.surfaceAlt = QColor(0x08,0x1e,0x38);
        c.accent     = QColor(0x00,0xb4,0xff);
        c.accentDim  = QColor(0x00,0x5a,0x88);
        c.text       = QColor(0xcc,0xee,0xff);
        c.textDim    = QColor(0x55,0x88,0xaa);
        c.border     = QColor(0x08,0x30,0x55);
        c.vizBar1    = QColor(0x00,0x44,0x88);
        c.vizBar2    = QColor(0x00,0x99,0xcc);
        c.vizBar3    = QColor(0x00,0xff,0xff);
        c.vizPeak    = QColor(0xcc,0xff,0xff);
        c.vizBg      = QColor(0x02,0x08,0x10);
        break;
    case Aurora:
        c.background = QColor(0x06,0x04,0x14);
        c.surface    = QColor(0x0c,0x08,0x22);
        c.surfaceAlt = QColor(0x14,0x0e,0x32);
        c.accent     = QColor(0xff,0x44,0xcc);
        c.accentDim  = QColor(0x88,0x22,0x66);
        c.text       = QColor(0xff,0xee,0xff);
        c.textDim    = QColor(0x99,0x77,0xaa);
        c.border     = QColor(0x33,0x14,0x44);
        c.vizBar1    = QColor(0x44,0x00,0xcc);
        c.vizBar2    = QColor(0xcc,0x22,0x88);
        c.vizBar3    = QColor(0xff,0x88,0x22);
        c.vizPeak    = QColor(0xff,0xff,0xaa);
        c.vizBg      = QColor(0x04,0x02,0x0e);
        break;
    case Carbon:
    default:
        c.background = QColor(0x0c,0x0c,0x0c);
        c.surface    = QColor(0x14,0x14,0x14);
        c.surfaceAlt = QColor(0x1e,0x1e,0x1e);
        c.accent     = QColor(0xff,0x44,0x22);
        c.accentDim  = QColor(0x88,0x22,0x11);
        c.text       = QColor(0xee,0xee,0xee);
        c.textDim    = QColor(0x88,0x88,0x88);
        c.border     = QColor(0x2a,0x2a,0x2a);
        c.vizBar1    = QColor(0xff,0x22,0x00);
        c.vizBar2    = QColor(0xff,0x88,0x00);
        c.vizBar3    = QColor(0xff,0xff,0x00);
        c.vizPeak    = QColor(0xff,0xff,0xff);
        c.vizBg      = QColor(0x06,0x06,0x06);
        break;
    }
    return c;
}

void ThemeManager::apply(QApplication *app, Theme theme) {
    m_current = theme;
    m_colors  = colorsFor(theme);
    app->setStyleSheet(buildQSS(theme));
}

QString ThemeManager::buildQSS(Theme theme) {
    const ThemeColors &c = colorsFor(theme);
    auto bg    = c.background.name();
    auto surf  = c.surface.name();
    auto surf2 = c.surfaceAlt.name();
    auto acc   = c.accent.name();
    auto accD  = c.accentDim.name();
    auto txt   = c.text.name();
    auto txtD  = c.textDim.name();
    auto brd   = c.border.name();

    QString qss = QString(R"(
* { outline: 0; }

QMainWindow, QDialog {
    background-color: %1;
    color: %6;
}

QWidget {
    background-color: transparent;
    color: %6;
    font-family: "Segoe UI", "SF Pro Display", Ubuntu, sans-serif;
    font-size: 13px;
}

/* ---- Panels / Frames ---- */
QFrame#sidePanel, QFrame#equalizerPanel {
    background-color: %2;
    border-right: 1px solid %8;
}

QFrame#centerPanel {
    background-color: %1;
}

QFrame#controlBar {
    background-color: %2;
    border-top: 1px solid %8;
    min-height: 80px;
}

QFrame#topBar {
    background-color: %2;
    border-bottom: 1px solid %8;
}

/* ---- Buttons ---- */
QPushButton {
    background-color: %3;
    color: %6;
    border: 1px solid %8;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: %4;
    color: white;
    border-color: %4;
}
QPushButton:pressed {
    background-color: %5;
}
QPushButton:checked {
    background-color: %5;
    color: white;
    border-color: %4;
}
QPushButton#playBtn {
    background-color: %4;
    border-radius: 22px;
    min-width: 44px; max-width: 44px;
    min-height: 44px; max-height: 44px;
    font-size: 18px;
    border: none;
}
QPushButton#playBtn:hover { background-color: %4; }
QPushButton#ctrlBtn {
    background: transparent;
    border: none;
    color: %7;
    font-size: 16px;
    min-width: 36px; max-width: 36px;
    min-height: 36px; max-height: 36px;
    border-radius: 18px;
}
QPushButton#ctrlBtn:hover { background-color: %3; color: %6; }
QPushButton#ctrlBtn:checked { color: %4; }

/* ---- Sliders ---- */
QSlider::groove:horizontal {
    height: 4px;
    background: %8;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 %5, stop:1 %4);
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: %4;
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: white;
    width: 16px; height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::groove:vertical {
    width: 4px;
    background: %8;
    border-radius: 2px;
}
QSlider::add-page:vertical {
    background: qlineargradient(x1:0,y1:1,x2:0,y2:0, stop:0 %5, stop:1 %4);
    border-radius: 2px;
}
QSlider::handle:vertical {
    background: %4;
    width: 14px; height: 14px;
    margin: 0 -5px;
    border-radius: 7px;
}

/* ---- ScrollBar ---- */
QScrollBar:vertical {
    background: %2; width: 6px; margin: 0;
}
QScrollBar::handle:vertical {
    background: %8; border-radius: 3px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: %5; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: %2; height: 6px; margin: 0;
}
QScrollBar::handle:horizontal {
    background: %8; border-radius: 3px; min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: %5; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ---- TableWidget (Playlist) ---- */
QTableWidget {
    background-color: %1;
    gridline-color: %8;
    border: none;
    selection-background-color: %5;
    selection-color: white;
    alternate-background-color: %2;
}
QTableWidget::item { padding: 4px 8px; border: none; }
QTableWidget::item:hover { background-color: %3; }
QHeaderView::section {
    background-color: %2;
    color: %7;
    border: none;
    border-bottom: 1px solid %8;
    padding: 6px 8px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ---- TreeWidget (Library) ---- */
QTreeWidget {
    background-color: %1;
    border: none;
    outline: none;
}
QTreeWidget::item { padding: 4px 4px; border: none; }
QTreeWidget::item:hover { background-color: %3; border-radius: 4px; }
QTreeWidget::item:selected { background-color: %5; border-radius: 4px; color: white; }
QTreeWidget::branch { background: transparent; }

/* ---- ComboBox ---- */
QComboBox {
    background-color: %3;
    border: 1px solid %8;
    border-radius: 6px;
    padding: 5px 10px;
    color: %6;
    min-width: 120px;
}
QComboBox:hover { border-color: %4; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { color: %7; }
QComboBox QAbstractItemView {
    background: %2;
    border: 1px solid %8;
    selection-background-color: %5;
    color: %6;
    padding: 4px;
}

/* ---- Labels ---- */
QLabel#trackTitle { font-size: 16px; font-weight: 700; color: %6; }
QLabel#trackArtist { font-size: 12px; color: %7; }
QLabel#timeLabel { font-size: 12px; color: %7; font-family: monospace; }
QLabel#sectionTitle {
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: %7;
}

/* ---- Tab Widget ---- */
QTabWidget::pane { border: none; background: transparent; }
QTabBar::tab {
    background: transparent;
    color: %7;
    padding: 10px 18px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}
QTabBar::tab:selected { color: %4; border-bottom: 2px solid %4; }
QTabBar::tab:hover { color: %6; }

/* ---- Tooltip ---- */
QToolTip {
    background: %2;
    color: %6;
    border: 1px solid %8;
    border-radius: 4px;
    padding: 4px 8px;
}

/* ---- Splitter ---- */
QSplitter::handle { background: %8; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical   { height: 1px; }

/* ---- MenuBar ---- */
QMenuBar { background: %2; color: %6; border-bottom: 1px solid %8; }
QMenuBar::item:selected { background: %3; border-radius: 4px; }
QMenu { background: %2; border: 1px solid %8; color: %6; }
QMenu::item:selected { background: %5; color: white; }
QMenu::separator { height: 1px; background: %8; margin: 4px 8px; }
)").arg(bg, surf, surf2, acc, accD, txt, txtD, brd);

    return qss;
}
