#pragma once
#include <QWidget>
#include <QTimer>
#include <QVector>
#include <QColor>

class VisualizerWidget : public QWidget {
    Q_OBJECT
public:
    enum Style { Bars, Mirror, Waveform };

    explicit VisualizerWidget(QWidget *parent = nullptr);

    void setPlaying(bool playing);
    void setStyle(Style s);
    void setColors(const QColor &c1, const QColor &c2, const QColor &c3,
                   const QColor &peak, const QColor &bg);

    QSize minimumSizeHint() const override { return {200, 80}; }

protected:
    void paintEvent(QPaintEvent *event) override;

private slots:
    void onTimer();

private:
    static constexpr int BARS     = 36;
    static constexpr int FPS      = 60;
    static constexpr float SMOOTH = 0.25f;
    static constexpr float PEAK_FALL = 0.008f;

    QTimer   *m_timer;
    bool      m_playing = false;
    int       m_frame   = 0;
    Style     m_style   = Bars;

    QVector<float> m_heights;
    QVector<float> m_targets;
    QVector<float> m_peaks;
    QVector<float> m_peakVel;

    QColor m_col1, m_col2, m_col3, m_peakCol, m_bgCol;

    float generateTarget(int bar, int frame) const;
    QColor lerpColor(const QColor &a, const QColor &b, float t) const;
};
