#include "audioengine.h"
#include <QRandomGenerator>
#include <QAudioDevice>
#include <QMediaDevices>
#include <cmath>
#include <algorithm>

// ──────────────────────────────────────────────────────────────
//  EQDevice
// ──────────────────────────────────────────────────────────────
const double EQDevice::FREQS[BANDS] =
    { 32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000 };

EQDevice::EQDevice(QObject *parent)
    : QIODevice(parent)
    , m_gains(BANDS, 0.f)
{
    open(QIODevice::ReadOnly);
}

void EQDevice::setFormat(const QAudioFormat &fmt) {
    m_fmt = fmt;
    rebuildFilters();
}

void EQDevice::setGains(const QVector<float> &gainsDB) {
    QMutexLocker lk(&m_mutex);
    m_gains = gainsDB;
    rebuildFilters();
}

void EQDevice::rebuildFilters() {
    double sr = m_fmt.sampleRate() > 0 ? m_fmt.sampleRate() : 44100.0;
    for (int b = 0; b < BANDS; ++b) {
        float g = (b < m_gains.size()) ? m_gains[b] : 0.f;
        BiquadFilter::Type t = BiquadFilter::PeakEQ;
        if (b == 0)        t = BiquadFilter::LowShelf;
        if (b == BANDS-1)  t = BiquadFilter::HighShelf;
        for (int ch = 0; ch < 2; ++ch) {
            m_filters[b][ch].reset();
            m_filters[b][ch].setParams(t, FREQS[b], sr, static_cast<double>(g), 1.41421356);
        }
    }
}

void EQDevice::enqueue(const QByteArray &pcm) {
    QMutexLocker lk(&m_mutex);
    m_buf.append(pcm);
}

void EQDevice::clear() {
    QMutexLocker lk(&m_mutex);
    m_buf.clear();
    // Reset filter states to avoid pops
    for (int b = 0; b < BANDS; ++b)
        for (int ch = 0; ch < 2; ++ch)
            m_filters[b][ch].reset();
}

qint64 EQDevice::bytesBuffered() const {
    QMutexLocker lk(&m_mutex);
    return m_buf.size();
}

qint64 EQDevice::readData(char *data, qint64 maxlen) {
    QMutexLocker lk(&m_mutex);
    if (m_buf.isEmpty()) return 0;

    qint64 bytes = qMin(maxlen, static_cast<qint64>(m_buf.size()));
    // Align to float frame boundary
    int ch = qMax(1, m_fmt.channelCount());
    qint64 frameBytes = ch * sizeof(float);
    bytes = (bytes / frameBytes) * frameBytes;
    if (bytes <= 0) return 0;

    memcpy(data, m_buf.constData(), static_cast<size_t>(bytes));
    m_buf.remove(0, static_cast<int>(bytes));

    // Apply EQ in-place (float samples)
    int frames = static_cast<int>(bytes / frameBytes);
    applyEQ(reinterpret_cast<float*>(data), frames, ch);

    return bytes;
}

void EQDevice::applyEQ(float *samples, int frames, int channels) {
    // Check if all gains are effectively zero (skip processing)
    bool allFlat = true;
    for (int b = 0; b < BANDS && allFlat; ++b)
        if (b < m_gains.size() && std::abs(m_gains[b]) > 0.05f) allFlat = false;
    if (allFlat) return;

    channels = qMin(channels, 2);
    for (int f = 0; f < frames; ++f) {
        for (int ch = 0; ch < channels; ++ch) {
            double s = samples[f * channels + ch];
            for (int b = 0; b < BANDS; ++b)
                s = m_filters[b][ch].process(s);
            // Soft clip to prevent harsh clipping
            s = s / (1.0 + std::abs(s) * 0.1);
            samples[f * channels + ch] = static_cast<float>(s);
        }
    }
}

// ──────────────────────────────────────────────────────────────
//  AudioEngine
// ──────────────────────────────────────────────────────────────
AudioEngine::AudioEngine(QObject *parent) : QObject(parent) {
    // Audio format: 32-bit float, 44100 Hz, stereo
    m_fmt.setSampleRate(44100);
    m_fmt.setChannelCount(2);
    m_fmt.setSampleFormat(QAudioFormat::Float);

    m_eqDev  = new EQDevice(this);
    m_eqDev->setFormat(m_fmt);

    m_decoder = new QAudioDecoder(this);
    m_decoder->setAudioFormat(m_fmt);

    connect(m_decoder, &QAudioDecoder::bufferReady,
            this,      &AudioEngine::onDecoderBuffer);
    connect(m_decoder, &QAudioDecoder::finished,
            this,      &AudioEngine::onDecoderFinished);
    connect(m_decoder, QOverload<QAudioDecoder::Error>::of(&QAudioDecoder::error),
            this,      &AudioEngine::onDecoderError);

    m_posTimer = new QTimer(this);
    m_posTimer->setInterval(100);
    connect(m_posTimer, &QTimer::timeout, this, &AudioEngine::onPositionTimer);

    initPipeline();
}

AudioEngine::~AudioEngine() {
    if (m_sink) m_sink->stop();
    if (m_decoder) m_decoder->stop();
}

void AudioEngine::initPipeline() {
    if (m_sink) {
        m_sink->stop();
        delete m_sink;
    }

    QAudioDevice dev = QMediaDevices::defaultAudioOutput();
    m_sink = new QAudioSink(dev, m_fmt, this);
    m_sink->setVolume(m_muted ? 0.f : m_volume);
    connect(m_sink, &QAudioSink::stateChanged,
            this,   &AudioEngine::onSinkStateChanged);
}

void AudioEngine::startDecode(const QUrl &url, qint64 seekMs) {
    m_decoder->stop();
    m_eqDev->clear();
    m_bytesPlayed = 0;
    m_decodedMs   = 0;
    m_seekTarget  = seekMs;

    m_decoder->setSource(url);
    m_decoder->start();

    if (m_state != Playing) {
        m_sink->stop();
        m_sink->start(m_eqDev);
        m_posTimer->start();
        m_state = Playing;
        emit stateChanged(m_state);
    }
}

// ── Playlist management ──────────────────────────────────────
void AudioEngine::setPlaylist(const QList<Track> &tracks) {
    m_playlist = tracks;
    m_currentIndex = -1;
    rebuildOrder();
    emit playlistChanged();
}

void AudioEngine::appendTrack(const Track &t) {
    m_playlist.append(t);
    rebuildOrder();
    emit playlistChanged();
}

void AudioEngine::clearPlaylist() {
    stop();
    m_playlist.clear();
    m_currentIndex = -1;
    rebuildOrder();
    emit playlistChanged();
}

void AudioEngine::removeTrack(int index) {
    if (index < 0 || index >= m_playlist.size()) return;
    m_playlist.removeAt(index);
    if (m_currentIndex > index)       --m_currentIndex;
    else if (m_currentIndex == index)  m_currentIndex = -1;
    rebuildOrder();
    emit playlistChanged();
}

void AudioEngine::moveTrack(int from, int to) {
    if (from == to) return;
    m_playlist.move(from, to);
    rebuildOrder();
    emit playlistChanged();
}

// ── Playback control ─────────────────────────────────────────
void AudioEngine::playIndex(int index) {
    if (index < 0 || index >= m_playlist.size()) return;
    m_currentIndex = index;
    const Track &t = m_playlist[index];
    m_duration = t.duration > 0 ? t.duration : 0;
    emit durationChanged(m_duration);
    emit trackChanged(index);
    startDecode(t.fileUrl, 0);
}

void AudioEngine::play() {
    if (m_state == Paused) {
        m_sink->resume();
        m_decoder->start();
        m_posTimer->start();
        m_state = Playing;
        emit stateChanged(m_state);
    } else if (m_state == Stopped) {
        if (m_currentIndex < 0 && !m_playlist.isEmpty())
            playIndex(0);
    }
}

void AudioEngine::pause() {
    if (m_state != Playing) return;
    m_sink->suspend();
    m_decoder->stop();
    m_posTimer->stop();
    m_state = Paused;
    emit stateChanged(m_state);
}

void AudioEngine::stop() {
    m_decoder->stop();
    m_sink->stop();
    m_eqDev->clear();
    m_posTimer->stop();
    m_bytesPlayed = 0;
    m_state = Stopped;
    emit stateChanged(m_state);
    emit positionChanged(0);
}

void AudioEngine::next() {
    int idx = nextIndex();
    if (idx >= 0) playIndex(idx);
    else if (m_repeat == RepeatAll && !m_playlist.isEmpty()) playIndex(0);
    else stop();
}

void AudioEngine::previous() {
    if (position() > 3000) { seek(0); return; }
    int idx = prevIndex();
    if (idx >= 0) playIndex(idx);
}

void AudioEngine::seek(qint64 ms) {
    if (m_currentIndex < 0) return;
    // Restart decoding from beginning, skip to target
    m_decoder->stop();
    m_eqDev->clear();
    m_bytesPlayed = 0;
    m_seekTarget  = ms;
    m_decodedMs   = 0;
    m_decoder->setSource(m_playlist[m_currentIndex].fileUrl);
    m_decoder->start();
    emit positionChanged(ms);
}

// ── EQ ───────────────────────────────────────────────────────
void AudioEngine::setEQGains(const QVector<float> &gainsDB) {
    m_eqDev->setGains(gainsDB);
}

// ── Volume / Mute ────────────────────────────────────────────
void AudioEngine::setVolume(float v) {
    m_volume = v;
    if (m_sink && !m_muted) m_sink->setVolume(v);
}
float AudioEngine::volume() const { return m_sink ? m_sink->volume() : m_volume; }

void AudioEngine::setMuted(bool m) {
    m_muted = m;
    if (m_sink) m_sink->setVolume(m ? 0.f : m_volume);
}
bool AudioEngine::isMuted() const { return m_muted; }

void AudioEngine::setShuffleMode(bool s) { m_shuffle = s; rebuildOrder(); }
void AudioEngine::setRepeatMode(RepeatMode r) { m_repeat = r; }

void AudioEngine::setPlaybackRate(float r) {
    m_rate = qBound(0.25f, r, 3.0f);
    // Note: pitch-preserving rate change requires DSP beyond Qt6 scope.
    // This affects position reporting accuracy.
}

qint64 AudioEngine::position() const {
    if (!m_sink || m_state == Stopped) return 0;
    qint64 us = m_sink->processedUSecs();
    return us / 1000;
}

// ── Decoder slots ────────────────────────────────────────────
void AudioEngine::onDecoderBuffer() {
    if (!m_decoder->bufferAvailable()) return;
    QAudioBuffer buf = m_decoder->read();
    if (!buf.isValid()) return;

    // Update duration from decoder if not set
    if (m_duration == 0 && buf.duration() > 0) {
        // Will accumulate
    }

    // Track decoded position for seeking
    qint64 bufMs = buf.startTime() / 1000;

    // Skip frames if we have a pending seek
    if (m_seekTarget >= 0) {
        if (bufMs < m_seekTarget) {
            m_decodedMs = bufMs;
            return; // discard frames before seek target
        }
        m_seekTarget = -1;
    }

    m_decodedMs = bufMs;

    // Convert buffer to the expected format
    const float *samples = buf.constData<float>();
    int frameCount = buf.frameCount();
    int channels   = buf.format().channelCount();

    // Peak metering
    measureLevel(samples, frameCount, channels);

    // Enqueue to EQ device
    QByteArray pcm(reinterpret_cast<const char*>(samples),
                   frameCount * channels * sizeof(float));
    m_eqDev->enqueue(pcm);
}

void AudioEngine::onDecoderFinished() {
    // Wait until sink drains the buffer, then go to next track
    // Use a short delay (sink buffer drain time ~200ms)
    QTimer::singleShot(300, this, [this]{
        if (m_repeat == RepeatOne) {
            playIndex(m_currentIndex);
        } else {
            next();
        }
    });
}

void AudioEngine::onDecoderError(QAudioDecoder::Error err) {
    Q_UNUSED(err);
    emit errorOccurred(m_decoder->errorString());
}

void AudioEngine::onSinkStateChanged(QAudio::State s) {
    Q_UNUSED(s);
}

void AudioEngine::onPositionTimer() {
    qint64 pos = position();
    emit positionChanged(pos);

    // Update duration from decoded position + remaining
    if (pos > m_duration) {
        m_duration = pos;
        emit durationChanged(m_duration);
    }
}

void AudioEngine::measureLevel(const float *samples, int frames, int ch) {
    float peakL = 0, peakR = 0;
    ch = qMin(ch, 2);
    for (int f = 0; f < frames; ++f) {
        float l = std::abs(samples[f * ch]);
        float r = (ch > 1) ? std::abs(samples[f * ch + 1]) : l;
        if (l > peakL) peakL = l;
        if (r > peakR) peakR = r;
    }
    // Convert to dB
    auto todB = [](float v) -> float {
        return v > 0.00001f ? 20.f * std::log10(v) : -96.f;
    };
    // Smooth decay
    m_peakL = qMax(todB(peakL), m_peakL - 1.5f);
    m_peakR = qMax(todB(peakR), m_peakR - 1.5f);
    emit levelChanged(m_peakL, m_peakR);
}

// ── Helpers ──────────────────────────────────────────────────
void AudioEngine::rebuildOrder() {
    m_order.clear();
    for (int i = 0; i < m_playlist.size(); ++i) m_order << i;
    if (m_shuffle)
        std::shuffle(m_order.begin(), m_order.end(), *QRandomGenerator::global());
}

int AudioEngine::nextIndex() const {
    if (m_playlist.isEmpty()) return -1;
    if (m_shuffle) {
        int p = m_order.indexOf(m_currentIndex);
        return (p + 1 < m_order.size()) ? m_order[p + 1] : -1;
    }
    int n = m_currentIndex + 1;
    return (n < m_playlist.size()) ? n : -1;
}

int AudioEngine::prevIndex() const {
    if (m_playlist.isEmpty()) return -1;
    if (m_shuffle) {
        int p = m_order.indexOf(m_currentIndex);
        return (p > 0) ? m_order[p - 1] : -1;
    }
    int n = m_currentIndex - 1;
    return (n >= 0) ? n : -1;
}

qint64 AudioEngine::bytesToMs(qint64 bytes) const {
    int ch = m_fmt.channelCount();
    int sr = m_fmt.sampleRate();
    if (ch <= 0 || sr <= 0) return 0;
    return bytes / (ch * sizeof(float)) * 1000 / sr;
}

qint64 AudioEngine::msToBytes(qint64 ms) const {
    int ch = m_fmt.channelCount();
    int sr = m_fmt.sampleRate();
    return ms * sr / 1000 * ch * sizeof(float);
}
