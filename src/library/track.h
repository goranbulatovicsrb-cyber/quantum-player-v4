#pragma once
#include <QString>
#include <QUrl>
#include <QPixmap>

struct Track {
    int     id       = 0;
    QString title;
    QString artist;
    QString album;
    QString genre;
    int     year     = 0;
    qint64  duration = 0;   // milliseconds
    QUrl    fileUrl;
    QPixmap albumArt;

    QString durationString() const {
        qint64 secs = duration / 1000;
        int m = static_cast<int>(secs / 60);
        int s = static_cast<int>(secs % 60);
        return QString("%1:%2").arg(m).arg(s, 2, 10, QChar('0'));
    }
    bool isValid() const { return !fileUrl.isEmpty(); }
};
