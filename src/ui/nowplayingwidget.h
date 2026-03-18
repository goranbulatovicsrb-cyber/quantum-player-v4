#pragma once
#include <QWidget>
#include <QLabel>
#include <QPixmap>
#include "../library/track.h"

class NowPlayingWidget : public QWidget {
    Q_OBJECT
public:
    explicit NowPlayingWidget(QWidget *parent = nullptr);
    void setTrack(const Track &track);
    void clearTrack();
    void setAccentColor(const QColor &color);

protected:
    void paintEvent(QPaintEvent *) override;
    void resizeEvent(QResizeEvent *) override;

private:
    QLabel *m_artLabel;
    QLabel *m_titleLabel;
    QLabel *m_artistLabel;
    QLabel *m_albumLabel;

    QPixmap m_albumArt;
    QPixmap m_bgBlur;
    QColor  m_accentColor{0x7b, 0x68, 0xff};

    void updateBlur();
};
