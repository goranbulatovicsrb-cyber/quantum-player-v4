#include "nowplayingwidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPainter>
#include <QPainterPath>
#include <QGraphicsBlurEffect>

NowPlayingWidget::NowPlayingWidget(QWidget *parent) : QWidget(parent) {
    auto *vlay = new QVBoxLayout(this);
    vlay->setContentsMargins(20, 20, 20, 12);
    vlay->setSpacing(10);
    vlay->addStretch(1);

    // Album Art
    m_artLabel = new QLabel(this);
    m_artLabel->setAlignment(Qt::AlignCenter);
    m_artLabel->setFixedSize(180, 180);
    m_artLabel->setStyleSheet(R"(
        background-color: #1a1a2a;
        border-radius: 12px;
        border: 1px solid #2a2a3a;
    )");
    auto *artContainer = new QHBoxLayout();
    artContainer->addStretch();
    artContainer->addWidget(m_artLabel);
    artContainer->addStretch();
    vlay->addLayout(artContainer);

    vlay->addSpacing(8);

    // Track info
    m_titleLabel = new QLabel("No Track Selected", this);
    m_titleLabel->setObjectName("trackTitle");
    m_titleLabel->setAlignment(Qt::AlignCenter);
    m_titleLabel->setWordWrap(true);
    vlay->addWidget(m_titleLabel);

    m_artistLabel = new QLabel("—", this);
    m_artistLabel->setObjectName("trackArtist");
    m_artistLabel->setAlignment(Qt::AlignCenter);
    vlay->addWidget(m_artistLabel);

    m_albumLabel = new QLabel("", this);
    m_albumLabel->setAlignment(Qt::AlignCenter);
    m_albumLabel->setStyleSheet("font-size: 11px; color: #555;");
    vlay->addWidget(m_albumLabel);

    vlay->addStretch(2);
}

void NowPlayingWidget::setTrack(const Track &track) {
    m_titleLabel->setText(track.title.isEmpty() ? "Unknown Track" : track.title);
    m_artistLabel->setText(track.artist.isEmpty() ? "Unknown Artist" : track.artist);
    m_albumLabel->setText(track.album.isEmpty() ? "" : QString("♫  %1").arg(track.album));

    if (!track.albumArt.isNull()) {
        m_albumArt = track.albumArt;
    } else {
        m_albumArt = QPixmap();
    }

    if (!m_albumArt.isNull()) {
        QPixmap scaled = m_albumArt.scaled(180, 180, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        // Round the pixmap
        QPixmap rounded(scaled.size());
        rounded.fill(Qt::transparent);
        QPainter p(&rounded);
        p.setRenderHint(QPainter::Antialiasing);
        QPainterPath path;
        path.addRoundedRect(QRectF(rounded.rect()), 12, 12);
        p.setClipPath(path);
        p.drawPixmap(0, 0, scaled);
        m_artLabel->setPixmap(rounded);
        m_artLabel->setStyleSheet("border-radius: 12px; background: transparent;");
    } else {
        m_artLabel->setPixmap(QPixmap());
        m_artLabel->setText("♫");
        m_artLabel->setStyleSheet(R"(
            background-color: #1a1a2a; border-radius: 12px;
            font-size: 48px; color: #2a2a4a;
            border: 1px solid #2a2a3a;
        )");
    }
    updateBlur();
    update();
}

void NowPlayingWidget::clearTrack() {
    m_titleLabel->setText("No Track Selected");
    m_artistLabel->setText("—");
    m_albumLabel->setText("");
    m_artLabel->setPixmap(QPixmap());
    m_artLabel->setText("♫");
    m_albumArt = QPixmap();
    m_bgBlur = QPixmap();
    update();
}

void NowPlayingWidget::setAccentColor(const QColor &color) {
    m_accentColor = color;
}

void NowPlayingWidget::updateBlur() {
    if (m_albumArt.isNull()) { m_bgBlur = QPixmap(); return; }
    // Create a blurred/scaled version for background
    m_bgBlur = m_albumArt.scaled(width(), height(),
                                  Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
}

void NowPlayingWidget::resizeEvent(QResizeEvent *e) {
    QWidget::resizeEvent(e);
    updateBlur();
}

void NowPlayingWidget::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::SmoothPixmapTransform);

    // Background
    p.fillRect(rect(), QColor(0x08, 0x08, 0x18));

    // Blurred album art background
    if (!m_bgBlur.isNull()) {
        p.setOpacity(0.12);
        p.drawPixmap(0, 0, m_bgBlur);
        p.setOpacity(1.0);
    }

    // Gradient overlay
    QLinearGradient overlay(0, 0, 0, height());
    overlay.setColorAt(0.0, QColor(0x08, 0x08, 0x18, 200));
    overlay.setColorAt(0.5, QColor(0x08, 0x08, 0x18, 140));
    overlay.setColorAt(1.0, QColor(0x08, 0x08, 0x18, 220));
    p.fillRect(rect(), overlay);
}
