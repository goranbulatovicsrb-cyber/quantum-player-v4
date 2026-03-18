#pragma once
#include <QObject>
#include <QAudioDecoder>
#include <QAudioSink>
#include <QAudioFormat>
#include <QIODevice>
#include <QTimer>
#include <QMutex>
#include <QByteArray>
#include <QVector>
#include <QList>
#include <QUrl>
#include "../dsp/biquadfilter.h"
#include "../library/track.h"

// ──────────────────────────────────────────────────────────────
//  EQDevice  –  QIODevice that holds decoded PCM and applies EQ
// ──────────────────────────────────────────────────────────────
class EQDevice : public QIODevice {
    Q_OBJECT
public:
    explicit EQDevice(QObject *parent = nullptr);

    void setFormat(const QAudioFormat &fmt);
    void setGains(const QVector<float> &gainsDB);   // -12 .. +12 dB per band
    void enqueue(const QByteArray &pcm);
    void clear();
    qint64 bytesBuffered() const;
    bool   atEnd()  const override { return false; }

protected:
    qint64 readData (char *data, qint64 maxlen) override;
    qint64 writeData(const char *data, qint64) override { Q_UNUSED(data); return 0; }

private:
    static const int    BANDS = 10;
    static const double FREQS[BANDS];

    QByteArray     m_buf;
    mutable QMutex m_mutex;
    QAudioFormat   m_fmt;
    QVector<float> m_gains;
    BiquadFilter   m_filters[BANDS][2];   // [band][channel: 0=L 1=R]

    void rebuildFilters();
    void applyEQ(float *samples, int frameCount, int channels);
};

// ──────────────────────────────────────────────────────────────
//  AudioEngine  –  decoder + EQ device + sink pipeline
// ──────────────────────────────────────────────────────────────
class AudioEngine : public QObject {
    Q_OBJECT
public:
    enum RepeatMode { RepeatNone = 0, RepeatOne, RepeatAll };
    enum State      { Stopped = 0, Playing, Paused };

    explicit AudioEngine(QObject *parent = nullptr);
    ~AudioEngine();

    // Playlist
    void setPlaylist(const QList<Track> &tracks);
    void appendTrack (const Track &t);
    void clearPlaylist();
    void removeTrack (int index);
    void moveTrack   (int from, int to);
    const QList<Track> &playlist() const { return m_playlist; }

    // Playback control
    void playIndex(int index);
    void play();
    void pause();
    void stop();
    void next();
    void previous();
    void seek(qint64 ms);

    // EQ
    void setEQGains(const QVector<float> &gainsDB);

    // Properties
    void  setVolume(float v);
    float volume()        const;
    void  setMuted(bool m);
    bool  isMuted()       const;
    void  setShuffleMode(bool s);
    bool  shuffleMode()   const { return m_shuffle; }
    void  setRepeatMode(RepeatMode r);
    RepeatMode repeatMode() const { return m_repeat; }
    void  setPlaybackRate(float r);
    float playbackRate()  const  { return m_rate; }

    State  state()         const { return m_state; }
    bool   isPlaying()     const { return m_state == Playing; }
    bool   isPaused()      const { return m_state == Paused;  }
    bool   isStopped()     const { return m_state == Stopped; }
    int    currentIndex()  const { return m_currentIndex; }
    qint64 position()      const;
    qint64 duration()      const { return m_duration; }

signals:
    void positionChanged(qint64 ms);
    void durationChanged(qint64 ms);
    void stateChanged(AudioEngine::State s);
    void trackChanged(int index);
    void playlistChanged();
    void errorOccurred(const QString &msg);
    void levelChanged(float leftDB, float rightDB);   // peak meters

private slots:
    void onDecoderBuffer();
    void onDecoderFinished();
    void onDecoderError(QAudioDecoder::Error err);
    void onSinkStateChanged(QAudio::State s);
    void onPositionTimer();

private:
    // Playlist
    QList<Track>  m_playlist;
    int           m_currentIndex = -1;
    bool          m_shuffle  = false;
    RepeatMode    m_repeat   = RepeatNone;
    QList<int>    m_order;            // playback order (shuffle-aware)

    // Pipeline
    QAudioDecoder *m_decoder  = nullptr;
    QAudioSink    *m_sink     = nullptr;
    EQDevice      *m_eqDev   = nullptr;
    QAudioFormat   m_fmt;

    // State
    State   m_state    = Stopped;
    float   m_volume   = 0.7f;
    bool    m_muted    = false;
    float   m_rate     = 1.0f;

    // Timing
    qint64  m_duration = 0;
    qint64  m_seekTarget = -1;      // -1 = no pending seek
    qint64  m_decodedMs  = 0;       // how many ms we've decoded so far
    qint64  m_bytesPlayed = 0;
    QTimer *m_posTimer   = nullptr;

    // Level metering
    float   m_peakL = -96.f, m_peakR = -96.f;

    void initPipeline();
    void startDecode(const QUrl &url, qint64 seekMs = 0);
    int  nextIndex() const;
    int  prevIndex() const;
    void rebuildOrder();
    qint64 bytesToMs(qint64 bytes) const;
    qint64 msToBytes(qint64 ms)    const;
    void   measureLevel(const float *samples, int frames, int ch);
};
