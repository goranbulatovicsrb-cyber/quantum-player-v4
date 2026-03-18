#include "musiclibrary.h"
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QBuffer>

MusicLibrary* MusicLibrary::s_instance = nullptr;

MusicLibrary* MusicLibrary::instance() {
    if (!s_instance) s_instance = new MusicLibrary();
    return s_instance;
}

MusicLibrary::MusicLibrary(QObject *parent) : QObject(parent) {}

void MusicLibrary::addArtist(const Artist &artist) {
    m_artists.append(artist);
    emit libraryChanged();
}

void MusicLibrary::removeArtist(int artistId) {
    m_artists.removeIf([&](const Artist &a){ return a.id == artistId; });
    emit libraryChanged();
}

void MusicLibrary::addAlbumToArtist(int artistId, const Album &album) {
    for (auto &a : m_artists) {
        if (a.id == artistId) {
            a.albums.append(album);
            emit libraryChanged();
            return;
        }
    }
}

void MusicLibrary::addTrackToAlbum(int artistId, int albumId, const Track &track) {
    for (auto &artist : m_artists) {
        if (artist.id == artistId) {
            for (auto &album : artist.albums) {
                if (album.id == albumId) {
                    album.tracks.append(track);
                    emit libraryChanged();
                    return;
                }
            }
        }
    }
}

QList<Track> MusicLibrary::allTracks() const {
    QList<Track> result;
    for (const auto &a : m_artists)
        result += a.allTracks();
    return result;
}

Artist* MusicLibrary::findArtist(int id) {
    for (auto &a : m_artists)
        if (a.id == id) return &a;
    return nullptr;
}

Album* MusicLibrary::findAlbum(int artistId, int albumId) {
    if (auto *artist = findArtist(artistId)) {
        for (auto &alb : artist->albums)
            if (alb.id == albumId) return &alb;
    }
    return nullptr;
}

void MusicLibrary::saveToFile(const QString &path) {
    QJsonArray artistsArr;
    for (const auto &artist : m_artists) {
        QJsonObject ao;
        ao["id"]   = artist.id;
        ao["name"] = artist.name;
        ao["bio"]  = artist.bio;
        QJsonArray albs;
        for (const auto &album : artist.albums) {
            QJsonObject albo;
            albo["id"]     = album.id;
            albo["title"]  = album.title;
            albo["artist"] = album.artist;
            albo["year"]   = album.year;
            // Save cover as base64
            if (!album.cover.isNull()) {
                QByteArray ba; QBuffer buf(&ba);
                buf.open(QIODevice::WriteOnly);
                album.cover.save(&buf, "PNG");
                albo["cover"] = QString::fromLatin1(ba.toBase64());
            }
            QJsonArray tracks;
            for (const auto &t : album.tracks) {
                QJsonObject to;
                to["id"]       = t.id;
                to["title"]    = t.title;
                to["artist"]   = t.artist;
                to["album"]    = t.album;
                to["genre"]    = t.genre;
                to["year"]     = t.year;
                to["duration"] = t.duration;
                to["url"]      = t.fileUrl.toString();
                tracks.append(to);
            }
            albo["tracks"] = tracks;
            albs.append(albo);
        }
        ao["albums"] = albs;
        artistsArr.append(ao);
    }
    QJsonObject root;
    root["artists"]       = artistsArr;
    root["nextTrackId"]   = m_nextTrackId;
    root["nextAlbumId"]   = m_nextAlbumId;
    root["nextArtistId"]  = m_nextArtistId;
    QFile f(path);
    if (f.open(QIODevice::WriteOnly))
        f.write(QJsonDocument(root).toJson());
}

void MusicLibrary::loadFromFile(const QString &path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return;
    auto doc = QJsonDocument::fromJson(f.readAll());
    auto root = doc.object();
    m_nextTrackId  = root["nextTrackId"].toInt();
    m_nextAlbumId  = root["nextAlbumId"].toInt();
    m_nextArtistId = root["nextArtistId"].toInt();
    m_artists.clear();
    for (auto av : root["artists"].toArray()) {
        auto ao = av.toObject();
        Artist artist;
        artist.id   = ao["id"].toInt();
        artist.name = ao["name"].toString();
        artist.bio  = ao["bio"].toString();
        for (auto albv : ao["albums"].toArray()) {
            auto albo = albv.toObject();
            Album album;
            album.id     = albo["id"].toInt();
            album.title  = albo["title"].toString();
            album.artist = albo["artist"].toString();
            album.year   = albo["year"].toInt();
            if (albo.contains("cover")) {
                QByteArray ba = QByteArray::fromBase64(albo["cover"].toString().toLatin1());
                album.cover.loadFromData(ba);
            }
            for (auto tv : albo["tracks"].toArray()) {
                auto to = tv.toObject();
                Track t;
                t.id       = to["id"].toInt();
                t.title    = to["title"].toString();
                t.artist   = to["artist"].toString();
                t.album    = to["album"].toString();
                t.genre    = to["genre"].toString();
                t.year     = to["year"].toInt();
                t.duration = to["duration"].toInt();
                t.fileUrl  = QUrl(to["url"].toString());
                album.tracks.append(t);
            }
            artist.albums.append(album);
        }
        m_artists.append(artist);
    }
    emit libraryChanged();
}
