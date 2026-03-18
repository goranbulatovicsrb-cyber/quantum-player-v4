#pragma once
#include <QString>
#include <QPixmap>
#include <QList>
#include "album.h"

struct Artist {
    int     id = 0;
    QString name;
    QPixmap photo;
    QString bio;
    QList<Album> albums;

    QList<Track> allTracks() const {
        QList<Track> result;
        for (const auto &a : albums)
            result += a.tracks;
        return result;
    }
};
