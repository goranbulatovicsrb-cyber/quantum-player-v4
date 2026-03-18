#pragma once
#include <QObject>
#include <QMediaPlayer>
#include <QAudioOutput>
#include <QList>
#include "../library/track.h"

class AudioPlayer : public QObject {
    Q_OBJECT
public:
    enum RepeatMode { RepeatNone = 0, RepeatOne, RepeatAll };

    explicit AudioPlayer(QObject *parent = nullptr);

    void setPlaylist(const QList<Track> &tracks);
    void appendToPlaylist(const Track &track);
    void clearPlaylist();
    void removeFromPlaylist(int index);
    void moveInPlaylist(int from, int to);

    void playIndex(int index);
    void play();
    void pause();
    void stop();
    void next();
    void previous();
    void seek(qint64 positionMs);

    void setVolume(float v);       // 0.0 – 1.0
    void setMuted(bool m);
    void setShuffleMode(bool s);
    void setRepeatMode(RepeatMode r);

    bool isPlaying()   const;
    bool isPaused()    const;
    bool isStopped()   const;
    bool shuffleMode() const { return m_shuffle; }
    RepeatMode repeatMode() const { return m_repeat; }
    float volume()     const;
    bool  isMuted()    const;
    qint64 position()  const;
    qint64 duration()  const;
    int currentIndex() const { return m_currentIndex; }
    const QList<Track>& playlist() const { return m_playlist; }

signals:
    void positionChanged(qint64 pos);
    void durationChanged(qint64 dur);
    void stateChanged(QMediaPlayer::PlaybackState state);
    void trackChanged(int index);
    void volumeChanged(float v);
    void playlistChanged();
    void errorOccurred(const QString &msg);

private slots:
    void onMediaStatusChanged(QMediaPlayer::MediaStatus status);
    void onPositionChanged(qint64 pos);
    void onDurationChanged(qint64 dur);
    void onPlaybackStateChanged(QMediaPlayer::PlaybackState s);
    void onErrorOccurred(QMediaPlayer::Error error, const QString &errorString);

private:
    QMediaPlayer  *m_player;
    QAudioOutput  *m_audio;
    QList<Track>   m_playlist;
    int            m_currentIndex = -1;
    bool           m_shuffle      = false;
    RepeatMode     m_repeat       = RepeatNone;
    QList<int>     m_shuffleOrder;

    void rebuildShuffleOrder();
    int  nextIndex() const;
    int  prevIndex() const;
};
