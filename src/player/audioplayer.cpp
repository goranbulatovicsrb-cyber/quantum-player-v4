#include "audioplayer.h"
#include <QRandomGenerator>
#include <algorithm>

AudioPlayer::AudioPlayer(QObject *parent)
    : QObject(parent)
    , m_player(new QMediaPlayer(this))
    , m_audio(new QAudioOutput(this))
{
    m_player->setAudioOutput(m_audio);
    m_audio->setVolume(0.7f);

    connect(m_player, &QMediaPlayer::mediaStatusChanged,
            this,     &AudioPlayer::onMediaStatusChanged);
    connect(m_player, &QMediaPlayer::positionChanged,
            this,     &AudioPlayer::onPositionChanged);
    connect(m_player, &QMediaPlayer::durationChanged,
            this,     &AudioPlayer::onDurationChanged);
    connect(m_player, &QMediaPlayer::playbackStateChanged,
            this,     &AudioPlayer::onPlaybackStateChanged);
    connect(m_player, &QMediaPlayer::errorOccurred,
            this,     &AudioPlayer::onErrorOccurred);
}

void AudioPlayer::setPlaylist(const QList<Track> &tracks) {
    m_playlist = tracks;
    m_currentIndex = -1;
    rebuildShuffleOrder();
    emit playlistChanged();
}

void AudioPlayer::appendToPlaylist(const Track &track) {
    m_playlist.append(track);
    rebuildShuffleOrder();
    emit playlistChanged();
}

void AudioPlayer::clearPlaylist() {
    m_player->stop();
    m_playlist.clear();
    m_currentIndex = -1;
    rebuildShuffleOrder();
    emit playlistChanged();
}

void AudioPlayer::removeFromPlaylist(int index) {
    if (index < 0 || index >= m_playlist.size()) return;
    m_playlist.removeAt(index);
    if (m_currentIndex > index) --m_currentIndex;
    else if (m_currentIndex == index) m_currentIndex = -1;
    rebuildShuffleOrder();
    emit playlistChanged();
}

void AudioPlayer::moveInPlaylist(int from, int to) {
    if (from == to) return;
    m_playlist.move(from, to);
    rebuildShuffleOrder();
    emit playlistChanged();
}

void AudioPlayer::playIndex(int index) {
    if (index < 0 || index >= m_playlist.size()) return;
    m_currentIndex = index;
    m_player->setSource(m_playlist[index].fileUrl);
    m_player->play();
    emit trackChanged(index);
}

void AudioPlayer::play() {
    if (m_currentIndex < 0 && !m_playlist.isEmpty())
        playIndex(0);
    else
        m_player->play();
}

void AudioPlayer::pause()    { m_player->pause(); }
void AudioPlayer::stop()     { m_player->stop(); }

void AudioPlayer::next() {
    int idx = nextIndex();
    if (idx >= 0) playIndex(idx);
}

void AudioPlayer::previous() {
    if (m_player->position() > 3000) { seek(0); return; }
    int idx = prevIndex();
    if (idx >= 0) playIndex(idx);
}

void AudioPlayer::seek(qint64 ms)  { m_player->setPosition(ms); }
void AudioPlayer::setVolume(float v) { m_audio->setVolume(v); emit volumeChanged(v); }
void AudioPlayer::setMuted(bool m)  { m_audio->setMuted(m); }
void AudioPlayer::setShuffleMode(bool s) { m_shuffle = s; rebuildShuffleOrder(); }
void AudioPlayer::setRepeatMode(RepeatMode r) { m_repeat = r; }

bool   AudioPlayer::isPlaying() const { return m_player->playbackState() == QMediaPlayer::PlayingState; }
bool   AudioPlayer::isPaused()  const { return m_player->playbackState() == QMediaPlayer::PausedState; }
bool   AudioPlayer::isStopped() const { return m_player->playbackState() == QMediaPlayer::StoppedState; }
float  AudioPlayer::volume()    const { return m_audio->volume(); }
bool   AudioPlayer::isMuted()   const { return m_audio->isMuted(); }
qint64 AudioPlayer::position()  const { return m_player->position(); }
qint64 AudioPlayer::duration()  const { return m_player->duration(); }

void AudioPlayer::onMediaStatusChanged(QMediaPlayer::MediaStatus status) {
    if (status == QMediaPlayer::EndOfMedia) {
        if (m_repeat == RepeatOne) {
            seek(0); play();
        } else {
            int idx = nextIndex();
            if (idx >= 0) playIndex(idx);
            else if (m_repeat == RepeatAll && !m_playlist.isEmpty()) playIndex(0);
        }
    }
}

void AudioPlayer::onPositionChanged(qint64 pos)  { emit positionChanged(pos); }
void AudioPlayer::onDurationChanged(qint64 dur)  { emit durationChanged(dur); }
void AudioPlayer::onPlaybackStateChanged(QMediaPlayer::PlaybackState s) { emit stateChanged(s); }
void AudioPlayer::onErrorOccurred(QMediaPlayer::Error, const QString &msg) { emit errorOccurred(msg); }

void AudioPlayer::rebuildShuffleOrder() {
    m_shuffleOrder.clear();
    for (int i = 0; i < m_playlist.size(); ++i) m_shuffleOrder.append(i);
    if (m_shuffle) {
        std::shuffle(m_shuffleOrder.begin(), m_shuffleOrder.end(),
                     *QRandomGenerator::global());
    }
}

int AudioPlayer::nextIndex() const {
    if (m_playlist.isEmpty()) return -1;
    if (m_shuffle) {
        int pos = m_shuffleOrder.indexOf(m_currentIndex);
        if (pos + 1 < m_shuffleOrder.size()) return m_shuffleOrder[pos + 1];
        return -1;
    }
    int n = m_currentIndex + 1;
    return (n < m_playlist.size()) ? n : -1;
}

int AudioPlayer::prevIndex() const {
    if (m_playlist.isEmpty()) return -1;
    if (m_shuffle) {
        int pos = m_shuffleOrder.indexOf(m_currentIndex);
        if (pos > 0) return m_shuffleOrder[pos - 1];
        return -1;
    }
    int n = m_currentIndex - 1;
    return (n >= 0) ? n : -1;
}
