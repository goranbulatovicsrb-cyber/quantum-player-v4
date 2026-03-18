#include "sleeptimerwidget.h"
#include <QHBoxLayout>
#include <QVBoxLayout>

SleepTimerWidget::SleepTimerWidget(QWidget *parent) : QWidget(parent) {
    auto *lay = new QHBoxLayout(this);
    lay->setContentsMargins(4, 2, 4, 2);
    lay->setSpacing(4);

    auto *lbl = new QLabel("💤", this);
    lbl->setToolTip("Sleep Timer");
    lay->addWidget(lbl);

    m_spinBox = new QSpinBox(this);
    m_spinBox->setRange(1, 180);
    m_spinBox->setValue(30);
    m_spinBox->setSuffix(" min");
    m_spinBox->setFixedWidth(75);
    m_spinBox->setToolTip("Stop playback after N minutes");
    lay->addWidget(m_spinBox);

    m_startBtn = new QPushButton("Set", this);
    m_startBtn->setFixedWidth(36);
    m_startBtn->setFixedHeight(22);
    m_startBtn->setStyleSheet("font-size:11px;");
    connect(m_startBtn, &QPushButton::clicked, this, &SleepTimerWidget::onStart);
    lay->addWidget(m_startBtn);

    m_cancelBtn = new QPushButton("✕", this);
    m_cancelBtn->setFixedWidth(24);
    m_cancelBtn->setFixedHeight(22);
    m_cancelBtn->setStyleSheet("font-size:11px;");
    m_cancelBtn->setVisible(false);
    connect(m_cancelBtn, &QPushButton::clicked, this, &SleepTimerWidget::onCancel);
    lay->addWidget(m_cancelBtn);

    m_statusLabel = new QLabel("", this);
    m_statusLabel->setStyleSheet("font-size:11px; color:#888;");
    m_statusLabel->setFixedWidth(52);
    lay->addWidget(m_statusLabel);

    m_timer = new QTimer(this);
    m_timer->setInterval(1000);
    connect(m_timer, &QTimer::timeout, this, &SleepTimerWidget::onTick);
}

void SleepTimerWidget::onStart() {
    m_remainingSecs = m_spinBox->value() * 60;
    m_active = true;
    m_timer->start();
    m_startBtn->setVisible(false);
    m_spinBox->setEnabled(false);
    m_cancelBtn->setVisible(true);
    updateLabel();
}

void SleepTimerWidget::onCancel() {
    m_timer->stop();
    m_active = false;
    m_startBtn->setVisible(true);
    m_spinBox->setEnabled(true);
    m_cancelBtn->setVisible(false);
    m_statusLabel->setText("");
}

void SleepTimerWidget::onTick() {
    --m_remainingSecs;
    if (m_remainingSecs <= 0) {
        onCancel();
        emit sleepTriggered();
        return;
    }
    updateLabel();
}

void SleepTimerWidget::updateLabel() {
    int m = m_remainingSecs / 60, s = m_remainingSecs % 60;
    m_statusLabel->setText(QString("%1:%2").arg(m).arg(s, 2, 10, QChar('0')));
}
