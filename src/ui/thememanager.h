#pragma once
#include <QString>
#include <QMap>
#include <QColor>
#include <QApplication>

struct ThemeColors {
    QColor background;
    QColor surface;
    QColor surfaceAlt;
    QColor accent;
    QColor accentDim;
    QColor text;
    QColor textDim;
    QColor border;
    QColor vizBar1;    // bottom gradient
    QColor vizBar2;    // mid gradient
    QColor vizBar3;    // top gradient
    QColor vizPeak;
    QColor vizBg;
};

class ThemeManager {
public:
    enum Theme {
        Midnight = 0,
        Cyberpunk,
        Ocean,
        Aurora,
        Carbon,
        ThemeCount
    };

    static ThemeManager* instance();
    void apply(QApplication *app, Theme theme);
    Theme currentTheme() const { return m_current; }
    const ThemeColors& colors() const { return m_colors; }
    QString themeName(Theme t) const;
    QStringList themeNames() const;

private:
    ThemeManager() = default;
    static ThemeManager *s_instance;
    Theme m_current = Midnight;
    ThemeColors m_colors;

    QString buildQSS(Theme theme);
    ThemeColors colorsFor(Theme theme);
};
