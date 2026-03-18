#pragma once
#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QVector>
#include <QComboBox>
#include <QPushButton>
#include "../dsp/biquadfilter.h"

class EqualizerCurve;

class EqualizerWidget : public QWidget {
    Q_OBJECT
public:
    explicit EqualizerWidget(QWidget *parent = nullptr);

    QVector<float> gains() const;    // dB per band
    void setGains(const QVector<float> &gains);
    void loadPreset(const QString &name);

signals:
    void gainsChanged(const QVector<float> &gains);

private slots:
    void onSliderChanged(int band, int value);
    void onPresetSelected(int index);
    void resetAll();

private:
    static const int BANDS = 10;
    static const double FREQS[BANDS];

    QSlider         *m_sliders[BANDS];
    QLabel          *m_valueLabels[BANDS];
    QComboBox       *m_presetBox;
    EqualizerCurve  *m_curve;
    QPushButton     *m_resetBtn;

    void buildUI();
    QVector<float> presetGains(const QString &name) const;
};

// Mini widget that draws the EQ frequency response curve
class EqualizerCurve : public QWidget {
    Q_OBJECT
public:
    explicit EqualizerCurve(QWidget *parent = nullptr);
    void setGains(const QVector<float> &gains);
    QSize minimumSizeHint() const override { return {200, 60}; }
protected:
    void paintEvent(QPaintEvent *) override;
private:
    QVector<float> m_gains;
    QColor m_accent{0x7b, 0x68, 0xff};
};
