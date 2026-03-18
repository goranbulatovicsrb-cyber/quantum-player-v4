#pragma once
#include <QMainWindow>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QAction>
#include <QComboBox>
#include <QSplitter>

#include "player/audioengine.h"
#include "ui/visualizerwidget.h"
#include "ui/equalizerwidget.h"
#include "ui/playlistwidget.h"
#include "ui/librarywidget.h"
#include "ui/nowplayingwidget.h"
#include "ui/thememanager.h"
#include "ui/levelmeterswidget.h"
#include "ui/sleeptimerwidget.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *) override;
    void dragEnterEvent(QDragEnterEvent *) override;
    void dropEvent(QDropEvent *) override;
    void keyPressEvent(QKeyEvent *) override;

private slots:
    void onPlayPause();
    void onNext();
    void onPrev();
    void onStop();
    void onVolumeChanged(int v);
    void onMute();
    void onShuffle();
    void onRepeat();

    void onPositionChanged(qint64 ms);
    void onDurationChanged(qint64 ms);
    void onEngineStateChanged(AudioEngine::State s);
    void onTrackChanged(int index);
    void onLevelChanged(float l, float r);
    void onPlayerError(const QString &msg);

    void onAddFiles();
    void onAddFolder();
    void onPlaylistDoubleClick(int index);
    void onAddToPlaylist(const QList<Track> &tracks);
    void onPlayNow(const QList<Track> &tracks);

    void onEQGainsChanged(const QVector<float> &gains);
    void onThemeChanged(int index);
    void onVisualizerStyleChanged(int index);
    void onSpeedChanged(int value);
    void onSleepTriggered();

    void openAbout();

private:
    AudioEngine      *m_engine;

    NowPlayingWidget *m_nowPlaying;
    VisualizerWidget *m_visualizer;
    EqualizerWidget  *m_equalizer;
    PlaylistWidget   *m_playlist;
    LibraryWidget    *m_library;
    LevelMetersWidget*m_levelMeters;
    SleepTimerWidget *m_sleepTimer;

    QPushButton *m_playBtn;
    QPushButton *m_prevBtn;
    QPushButton *m_nextBtn;
    QPushButton *m_stopBtn;
    QPushButton *m_shuffleBtn;
    QPushButton *m_repeatBtn;
    QPushButton *m_muteBtn;
    QSlider     *m_seekSlider;
    QSlider     *m_volSlider;
    QSlider     *m_speedSlider;
    QLabel      *m_posLabel;
    QLabel      *m_durLabel;
    QLabel      *m_speedLabel;
    QComboBox   *m_themeBox;
    QComboBox   *m_vizStyleBox;

    bool m_seeking = false;

    void buildUI();
    void buildMenuBar();
    void buildControlBar(QWidget *parent);
    void applyTheme(ThemeManager::Theme theme);
    void updateVisualizerColors();
    QString formatTime(qint64 ms) const;
    void saveSettings();
    void loadSettings();
};
