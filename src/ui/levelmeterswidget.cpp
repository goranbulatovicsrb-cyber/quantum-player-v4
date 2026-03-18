#include "levelmeterswidget.h"
#include <QPainter>
#include <QLinearGradient>
#include <cmath>

LevelMetersWidget::LevelMetersWidget(QWidget *parent)
    : QWidget(parent)
    , m_decayTimer(new QTimer(this))
{
    setFixedHeight(14);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_decayTimer->setInterval(40);
    connect(m_decayTimer, &QTimer::timeout, this, &LevelMetersWidget::onDecay);
    m_decayTimer->start();
}

void LevelMetersWidget::setLevel(float l, float r) {
    m_leftDB  = qBound(MIN_DB, l, MAX_DB);
    m_rightDB = qBound(MIN_DB, r, MAX_DB);
    if (m_leftDB  > m_leftPeak)  m_leftPeak  = m_leftDB;
    if (m_rightDB > m_rightPeak) m_rightPeak = m_rightDB;
    update();
}

void LevelMetersWidget::reset() {
    m_leftDB = m_rightDB = m_leftPeak = m_rightPeak = MIN_DB;
    update();
}

void LevelMetersWidget::setAccentColor(const QColor &c) { m_accent = c; }

void LevelMetersWidget::onDecay() {
    m_leftPeak  -= 0.6f;
    m_rightPeak -= 0.6f;
    m_leftDB    -= 1.5f;
    m_rightDB   -= 1.5f;
    if (m_leftPeak  < MIN_DB) m_leftPeak  = MIN_DB;
    if (m_rightPeak < MIN_DB) m_rightPeak = MIN_DB;
    if (m_leftDB    < MIN_DB) m_leftDB    = MIN_DB;
    if (m_rightDB   < MIN_DB) m_rightDB   = MIN_DB;
    update();
}

float LevelMetersWidget::normalize(float db) const {
    return (db - MIN_DB) / (MAX_DB - MIN_DB);
}

QColor LevelMetersWidget::colorForLevel(float norm) const {
    if (norm > 0.85f) return QColor(0xff, 0x22, 0x22);
    if (norm > 0.65f) return QColor(0xff, 0xaa, 0x00);
    return m_accent;
}

void LevelMetersWidget::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int W = width(), H = height();
    int gap = 2, ch = (H - gap) / 2;

    // Background
    p.fillRect(rect(), QColor(0,0,0,0));

    auto drawMeter = [&](float db, float peak, int y) {
        float norm  = normalize(db);
        float pnorm = normalize(peak);
        int barW = static_cast<int>(norm  * W);
        int pkX  = static_cast<int>(pnorm * W);

        // Bg track
        p.fillRect(0, y, W, ch, QColor(30,30,50,120));

        // Gradient bar
        QLinearGradient grad(0,y,W,y);
        grad.setColorAt(0.0, m_accent);
        grad.setColorAt(0.65, QColor(0xff,0xaa,0x00));
        grad.setColorAt(0.85, QColor(0xff,0x44,0x00));
        grad.setColorAt(1.0, QColor(0xff,0x00,0x00));
        p.fillRect(0, y, barW, ch, grad);

        // Peak hold
        if (pkX > 1) {
            p.fillRect(qMax(0, pkX-2), y, 2, ch, QColor(255,255,255,200));
        }
    };

    drawMeter(m_leftDB,  m_leftPeak,  0);
    drawMeter(m_rightDB, m_rightPeak, ch + gap);
}
