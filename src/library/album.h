#pragma once
#include <QString>
#include <QPixmap>
#include <QList>
#include "track.h"

struct Album {
    int     id = 0;
    QString title;
    QString artist;
    int     year  = 0;
    QPixmap cover;
    QList<Track> tracks;
};
