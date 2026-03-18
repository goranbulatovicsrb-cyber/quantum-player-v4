#include "visualizerwidget.h"
#include <QPainter>
#include <QPainterPath>
#include <QLinearGradient>
#include <cmath>

VisualizerWidget::VisualizerWidget(QWidget *parent)
    : QWidget(parent)
    , m_timer(new QTimer(this))
    , m_heights(BARS, 0.0f)
    , m_targets(BARS, 0.0f)
    , m_peaks(BARS, 0.0f)
    , m_peakVel(BARS, 0.0f)
{
    setMinimumHeight(100);
    // Default: Midnight colors
    m_col1   = QColor(0x22, 0x22, 0xff);
    m_col2   = QColor(0x88, 0x44, 0xff);
    m_col3   = QColor(0xff, 0x44, 0xff);
    m_peakCol = Qt::white;
    m_bgCol  = QColor(0x05, 0x05, 0x12);

    connect(m_timer, &QTimer::timeout, this, &VisualizerWidget::onTimer);
    m_timer->start(1000 / FPS);
}

void VisualizerWidget::setPlaying(bool p) { m_playing = p; }
void VisualizerWidget::setStyle(Style s)  { m_style   = s; }

void VisualizerWidget::setColors(const QColor &c1, const QColor &c2, const QColor &c3,
                                  const QColor &peak, const QColor &bg) {
    m_col1 = c1; m_col2 = c2; m_col3 = c3;
    m_peakCol = peak; m_bgCol = bg;
}

float VisualizerWidget::generateTarget(int bar, int frame) const {
    // Multi-layered sine waves per bar, simulates realistic spectrum
    float t = frame * 0.016f;
    float b = static_cast<float>(bar) / BARS;

    // Low freqs (0-10): bigger, slower movement
    float low  = 0.9f * std::abs(std::sin(t * 1.2f + b * 1.5f))
               * std::exp(-b * 1.5f);
    // Mid freqs (8-24): moderate
    float mid  = 0.7f * std::abs(std::sin(t * 2.1f + b * 2.8f + 1.2f))
               * (1.0f - std::exp(-b * 2.0f)) * std::exp(-b * 0.8f);
    // High freqs (20+): smaller, faster
    float high = 0.4f * std::abs(std::sin(t * 4.5f + b * 5.0f + 2.4f))
               * (1.0f - std::exp(-b * 4.0f));
    // Kick transient
    float kick = 0.3f * std::max(0.0f, std::sin(t * 1.8f)) * std::exp(-b * 3.0f);

    float v = (low + mid + high + kick);
    v = std::min(v, 1.0f);
    // Add subtle noise
    int seed = bar * 1337 + frame * 7;
    float noise = ((seed * 13 + 7) % 100) * 0.001f;
    return std::min(v + noise * 0.1f, 1.0f);
}

QColor VisualizerWidget::lerpColor(const QColor &a, const QColor &b, float t) const {
    return QColor(
        static_cast<int>(a.red()   + (b.red()   - a.red())   * t),
        static_cast<int>(a.green() + (b.green() - a.green()) * t),
        static_cast<int>(a.blue()  + (b.blue()  - a.blue())  * t)
    );
}

void VisualizerWidget::onTimer() {
    ++m_frame;
    for (int i = 0; i < BARS; ++i) {
        if (m_playing) {
            m_targets[i] = generateTarget(i, m_frame);
        } else {
            // Decay to zero when not playing
            m_targets[i] *= 0.92f;
        }
        // Lerp current to target
        float diff = m_targets[i] - m_heights[i];
        m_heights[i] += diff * (diff > 0 ? 0.45f : SMOOTH);
        m_heights[i] = std::max(0.0f, std::min(1.0f, m_heights[i]));

        // Peak handling
        if (m_heights[i] >= m_peaks[i]) {
            m_peaks[i]   = m_heights[i];
            m_peakVel[i] = 0.0f;
        } else {
            m_peakVel[i] += PEAK_FALL;
            m_peaks[i]   -= m_peakVel[i];
            if (m_peaks[i] < 0) m_peaks[i] = 0;
        }
    }
    update();
}

void VisualizerWidget::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int W = width(), H = height();

    // Background
    p.fillRect(0, 0, W, H, m_bgCol);

    // Subtle grid lines
    p.setPen(QPen(QColor(255,255,255,8), 1));
    for (int y = H/4; y < H; y += H/4)
        p.drawLine(0, y, W, y);

    if (m_style == Mirror) {
        int mid = H / 2;
        float barW   = static_cast<float>(W) / BARS;
        float gap    = std::max(1.0f, barW * 0.25f);
        float bw     = barW - gap;

        for (int i = 0; i < BARS; ++i) {
            float x = i * barW + gap / 2.0f;
            float h = m_heights[i] * mid * 0.92f;
            float t = m_heights[i]; // normalized 0-1 for color

            QColor barColor = t < 0.5f
                ? lerpColor(m_col1, m_col2, t * 2.0f)
                : lerpColor(m_col2, m_col3, (t - 0.5f) * 2.0f);

            // Glow
            for (int g = 4; g >= 1; --g) {
                QColor gc = barColor;
                gc.setAlpha(20 * g);
                p.fillRect(QRectF(x - g, mid - h - g, bw + g*2, h*2 + g*2), gc);
            }

            // Top half
            QLinearGradient grad(x, mid, x, mid - h);
            grad.setColorAt(0.0, barColor);
            grad.setColorAt(1.0, m_col3);
            p.fillRect(QRectF(x, mid - h, bw, h), grad);

            // Bottom mirror
            QLinearGradient gradB(x, mid, x, mid + h);
            QColor dimColor = barColor; dimColor.setAlpha(120);
            gradB.setColorAt(0.0, dimColor);
            gradB.setColorAt(1.0, QColor(m_col3.red(), m_col3.green(), m_col3.blue(), 60));
            p.fillRect(QRectF(x, mid, bw, h), gradB);
        }
    } else {
        // Standard bars from bottom
        float barW = static_cast<float>(W) / BARS;
        float gap  = std::max(1.5f, barW * 0.22f);
        float bw   = barW - gap;

        for (int i = 0; i < BARS; ++i) {
            float x = i * barW + gap / 2.0f;
            float h = m_heights[i] * (H - 6) * 0.95f;
            float t = m_heights[i];

            if (h < 1.0f) continue;

            QColor barColor = t < 0.5f
                ? lerpColor(m_col1, m_col2, t * 2.0f)
                : lerpColor(m_col2, m_col3, (t - 0.5f) * 2.0f);

            // Glow layers
            for (int g = 3; g >= 1; --g) {
                QColor gc = barColor;
                gc.setAlpha(18 * g);
                p.fillRect(QRectF(x - g, H - h - g, bw + g*2, h + g*2), gc);
            }

            // Main bar gradient
            QLinearGradient grad(x, H - h, x, H);
            grad.setColorAt(0.0, m_col3);
            grad.setColorAt(0.4, barColor);
            grad.setColorAt(1.0, m_col1);
            p.fillRect(QRectF(x, H - h, bw, h), grad);

            // Top shine
            QLinearGradient shine(x, H - h, x, H - h + std::min(h * 0.3f, 12.0f));
            shine.setColorAt(0, QColor(255,255,255,60));
            shine.setColorAt(1, QColor(255,255,255,0));
            p.fillRect(QRectF(x, H - h, bw, std::min(h * 0.3f, 12.0f)), shine);

            // Peak dot
            float ph = m_peaks[i] * (H - 6) * 0.95f;
            if (ph > 2.0f) {
                QColor pc = m_peakCol;
                pc.setAlpha(200);
                p.fillRect(QRectF(x, H - ph - 2, bw, 2), pc);
            }
        }
    }

    // Bottom glow line
    QLinearGradient glow(0, H-2, W, H-2);
    glow.setColorAt(0.0, QColor(0,0,0,0));
    glow.setColorAt(0.5, m_col2);
    glow.setColorAt(1.0, QColor(0,0,0,0));
    glow.setSpread(QGradient::ReflectSpread);
    p.fillRect(0, H-2, W, 2, glow);
}
