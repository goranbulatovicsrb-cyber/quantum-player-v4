#include "equalizerwidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPainter>
#include <QPainterPath>
#include <QLabel>
#include <cmath>

const double EqualizerWidget::FREQS[BANDS] =
    { 32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000 };

EqualizerWidget::EqualizerWidget(QWidget *parent) : QWidget(parent) {
    buildUI();
}

void EqualizerWidget::buildUI() {
    auto *vlay = new QVBoxLayout(this);
    vlay->setContentsMargins(8, 8, 8, 8);
    vlay->setSpacing(6);

    // Title
    auto *title = new QLabel("EQUALIZER", this);
    title->setObjectName("sectionTitle");
    title->setAlignment(Qt::AlignCenter);
    vlay->addWidget(title);

    // Preset combo
    m_presetBox = new QComboBox(this);
    m_presetBox->addItems({"Flat", "Bass Boost", "Bass Cut", "Treble Boost",
                           "Treble Cut", "Rock", "Pop", "Jazz",
                           "Classical", "Electronic", "Vocal"});
    vlay->addWidget(m_presetBox);

    // EQ Curve
    m_curve = new EqualizerCurve(this);
    vlay->addWidget(m_curve);

    // Sliders grid
    auto *grid = new QGridLayout();
    grid->setSpacing(4);
    grid->setHorizontalSpacing(2);

    for (int i = 0; i < BANDS; ++i) {
        // Value label (top)
        m_valueLabels[i] = new QLabel("0", this);
        m_valueLabels[i]->setAlignment(Qt::AlignHCenter | Qt::AlignBottom);
        m_valueLabels[i]->setFixedWidth(30);
        m_valueLabels[i]->setStyleSheet("font-size: 10px; color: #888;");
        grid->addWidget(m_valueLabels[i], 0, i);

        // Slider (vertical, -12 to +12 dB)
        m_sliders[i] = new QSlider(Qt::Vertical, this);
        m_sliders[i]->setRange(-120, 120);
        m_sliders[i]->setValue(0);
        m_sliders[i]->setFixedWidth(30);
        m_sliders[i]->setMinimumHeight(100);
        m_sliders[i]->setToolTip(QString("Band %1 Hz").arg(FREQS[i] < 1000
            ? QString::number(static_cast<int>(FREQS[i]))
            : QString::number(static_cast<int>(FREQS[i]/1000)) + "k"));
        connect(m_sliders[i], &QSlider::valueChanged, this, [this, i](int v){
            onSliderChanged(i, v);
        });
        grid->addWidget(m_sliders[i], 1, i);

        // Freq label (bottom)
        QString fLabel = FREQS[i] < 1000
            ? QString::number(static_cast<int>(FREQS[i]))
            : QString::number(static_cast<int>(FREQS[i]/1000)) + "k";
        auto *fl = new QLabel(fLabel, this);
        fl->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
        fl->setFixedWidth(30);
        fl->setStyleSheet("font-size: 9px; color: #666;");
        grid->addWidget(fl, 2, i);
    }

    vlay->addLayout(grid);

    // dB scale labels
    auto *scaleLayout = new QHBoxLayout();
    for (const QString &label : {"+12", "0", "-12"}) {
        auto *l = new QLabel(label, this);
        l->setStyleSheet("font-size: 9px; color: #555;");
        scaleLayout->addWidget(l);
        if (label != "-12") scaleLayout->addStretch();
    }
    vlay->addLayout(scaleLayout);

    // Reset button
    m_resetBtn = new QPushButton("Reset", this);
    m_resetBtn->setFixedHeight(28);
    connect(m_resetBtn, &QPushButton::clicked, this, &EqualizerWidget::resetAll);
    vlay->addWidget(m_resetBtn);

    connect(m_presetBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &EqualizerWidget::onPresetSelected);
}

QVector<float> EqualizerWidget::gains() const {
    QVector<float> g;
    for (int i = 0; i < BANDS; ++i)
        g << m_sliders[i]->value() / 10.0f;
    return g;
}

void EqualizerWidget::setGains(const QVector<float> &gains) {
    for (int i = 0; i < BANDS && i < gains.size(); ++i) {
        m_sliders[i]->blockSignals(true);
        m_sliders[i]->setValue(static_cast<int>(gains[i] * 10));
        m_sliders[i]->blockSignals(false);
        m_valueLabels[i]->setText(QString::number(gains[i], 'f', 1));
    }
    m_curve->setGains(gains);
}

void EqualizerWidget::onSliderChanged(int band, int value) {
    float db = value / 10.0f;
    m_valueLabels[band]->setText(QString::number(db, 'f', 1));
    m_curve->setGains(gains());
    emit gainsChanged(gains());
}

void EqualizerWidget::onPresetSelected(int index) {
    QString name = m_presetBox->itemText(index);
    setGains(presetGains(name));
    emit gainsChanged(gains());
}

void EqualizerWidget::resetAll() {
    for (int i = 0; i < BANDS; ++i) m_sliders[i]->setValue(0);
    m_presetBox->setCurrentIndex(0);
}

QVector<float> EqualizerWidget::presetGains(const QString &name) const {
    // {32, 64, 125, 250, 500, 1k, 2k, 4k, 8k, 16k}
    if (name == "Bass Boost")
        return {8,7,5,3,1,0,-1,-1,-1,-1};
    if (name == "Bass Cut")
        return {-8,-7,-5,-3,-1,0,0,0,0,0};
    if (name == "Treble Boost")
        return {0,0,0,0,0,1,3,5,7,8};
    if (name == "Treble Cut")
        return {0,0,0,0,0,-1,-3,-5,-7,-8};
    if (name == "Rock")
        return {6,5,4,1,-1,-1,1,3,4,5};
    if (name == "Pop")
        return {-2,-1,2,4,5,4,2,-1,-2,-2};
    if (name == "Jazz")
        return {4,3,1,2,0,-2,-1,0,2,3};
    if (name == "Classical")
        return {4,3,2,1,0,0,-1,-1,2,3};
    if (name == "Electronic")
        return {6,5,2,0,-2,-2,0,2,4,6};
    if (name == "Vocal")
        return {-3,-2,0,2,4,5,4,3,1,-1};
    // Flat
    return {0,0,0,0,0,0,0,0,0,0};
}

// ===== EqualizerCurve =====
EqualizerCurve::EqualizerCurve(QWidget *parent)
    : QWidget(parent)
    , m_gains(10, 0.0f)
{
    setFixedHeight(70);
    setStyleSheet("background: transparent;");
}

void EqualizerCurve::setGains(const QVector<float> &gains) {
    m_gains = gains;
    update();
}

void EqualizerCurve::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int W = width(), H = height();
    float midY = H / 2.0f;

    // Background grid
    p.setPen(QPen(QColor(255,255,255,12), 1, Qt::DashLine));
    p.drawLine(0, static_cast<int>(midY), W, static_cast<int>(midY));
    for (int db : {-6, 6}) {
        float y = midY - (db / 12.0f) * midY;
        p.drawLine(0, static_cast<int>(y), W, static_cast<int>(y));
    }

    if (m_gains.size() < 10) return;

    // Draw curve through points
    float bandW = static_cast<float>(W) / (m_gains.size() - 1);
    QPainterPath path;
    for (int i = 0; i < m_gains.size(); ++i) {
        float x = i * bandW;
        float y = midY - (m_gains[i] / 12.0f) * midY;
        if (i == 0) path.moveTo(x, y);
        else {
            float cx = (i - 0.5f) * bandW;
            QPointF prev(( i-1) * bandW, midY - (m_gains[i-1] / 12.0f) * midY);
            QPointF curr(x, y);
            path.cubicTo(cx, prev.y(), cx, curr.y(), curr.x(), curr.y());
        }
    }

    // Fill area under curve
    QPainterPath fill = path;
    fill.lineTo(W, midY);
    fill.lineTo(0, midY);
    fill.closeSubpath();
    QLinearGradient fillGrad(0, 0, 0, H);
    fillGrad.setColorAt(0, QColor(m_accent.red(), m_accent.green(), m_accent.blue(), 80));
    fillGrad.setColorAt(1, QColor(m_accent.red(), m_accent.green(), m_accent.blue(), 10));
    p.fillPath(fill, fillGrad);

    // Draw curve line
    QPen curvePen(m_accent, 2);
    p.setPen(curvePen);
    p.drawPath(path);

    // Draw band points
    p.setBrush(m_accent);
    p.setPen(Qt::NoPen);
    for (int i = 0; i < m_gains.size(); ++i) {
        float x = i * bandW;
        float y = midY - (m_gains[i] / 12.0f) * midY;
        p.drawEllipse(QPointF(x, y), 3.5f, 3.5f);
    }
}
