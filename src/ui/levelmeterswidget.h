#pragma once
#include <QWidget>
#include <QTimer>

class LevelMetersWidget : public QWidget {
    Q_OBJECT
public:
    explicit LevelMetersWidget(QWidget *parent = nullptr);
    void setLevel(float leftDB, float rightDB);
    void reset();
    void setAccentColor(const QColor &c);
    QSize minimumSizeHint() const override { return {60, 14}; }
protected:
    void paintEvent(QPaintEvent *) override;
private slots:
    void onDecay();
private:
    float  m_leftDB  = -96.f;
    float  m_rightDB = -96.f;
    float  m_leftPeak  = -96.f;
    float  m_rightPeak = -96.f;
    QColor m_accent{0x7b,0x68,0xff};
    QTimer *m_decayTimer;
    static constexpr float MIN_DB = -48.f;
    static constexpr float MAX_DB =   0.f;
    float normalize(float db) const;
    QColor colorForLevel(float norm) const;
};
