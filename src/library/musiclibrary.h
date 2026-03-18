#pragma once
#include <QObject>
#include <QList>
#include "artist.h"
#include "track.h"

class MusicLibrary : public QObject {
    Q_OBJECT
public:
    static MusicLibrary* instance();

    void addArtist(const Artist &artist);
    void removeArtist(int artistId);
    void addAlbumToArtist(int artistId, const Album &album);
    void addTrackToAlbum(int artistId, int albumId, const Track &track);

    const QList<Artist>& artists() const { return m_artists; }
    QList<Track> allTracks() const;
    Artist* findArtist(int id);
    Album*  findAlbum(int artistId, int albumId);

    void saveToFile(const QString &path);
    void loadFromFile(const QString &path);

    int nextTrackId()  { return ++m_nextTrackId; }
    int nextAlbumId()  { return ++m_nextAlbumId; }
    int nextArtistId() { return ++m_nextArtistId; }

signals:
    void libraryChanged();

private:
    explicit MusicLibrary(QObject *parent = nullptr);
    static MusicLibrary *s_instance;

    QList<Artist> m_artists;
    int m_nextTrackId  = 0;
    int m_nextAlbumId  = 0;
    int m_nextArtistId = 0;
};
