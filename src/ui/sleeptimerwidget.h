#pragma once
#include <QWidget>
#include <QTimer>
#include <QLabel>
#include <QSpinBox>
#include <QPushButton>

class SleepTimerWidget : public QWidget {
    Q_OBJECT
public:
    explicit SleepTimerWidget(QWidget *parent = nullptr);
    bool isActive() const { return m_active; }
signals:
    void sleepTriggered();
private slots:
    void onStart();
    void onCancel();
    void onTick();
private:
    QSpinBox    *m_spinBox;
    QPushButton *m_startBtn;
    QPushButton *m_cancelBtn;
    QLabel      *m_statusLabel;
    QTimer      *m_timer;
    int          m_remainingSecs = 0;
    bool         m_active = false;
    void updateLabel();
};
