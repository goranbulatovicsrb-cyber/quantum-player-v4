#include "mainwindow.h"
#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QFrame>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMimeData>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QCloseEvent>
#include <QKeyEvent>
#include <QSettings>
#include <QLabel>
#include <QMessageBox>
#include <QTabWidget>
#include <QStatusBar>
#include <QFileInfo>
#include <QDir>
#include "library/musiclibrary.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_engine(new AudioEngine(this))
{
    setWindowTitle("Quantum Player");
    setMinimumSize(1100, 700);
    resize(1300, 800);
    setAcceptDrops(true);

    buildUI();
    buildMenuBar();

    connect(m_engine, &AudioEngine::positionChanged,  this, &MainWindow::onPositionChanged);
    connect(m_engine, &AudioEngine::durationChanged,  this, &MainWindow::onDurationChanged);
    connect(m_engine, &AudioEngine::stateChanged,     this, &MainWindow::onEngineStateChanged);
    connect(m_engine, &AudioEngine::trackChanged,     this, &MainWindow::onTrackChanged);
    connect(m_engine, &AudioEngine::levelChanged,     this, &MainWindow::onLevelChanged);
    connect(m_engine, &AudioEngine::errorOccurred,    this, &MainWindow::onPlayerError);

    loadSettings();
    applyTheme(ThemeManager::Midnight);
}

MainWindow::~MainWindow() {}

void MainWindow::buildUI() {
    auto *central = new QWidget(this);
    setCentralWidget(central);
    auto *mainLay = new QVBoxLayout(central);
    mainLay->setContentsMargins(0,0,0,0);
    mainLay->setSpacing(0);

    // ── TOP BAR ──────────────────────────────────────────────
    auto *topBar = new QFrame(central);
    topBar->setObjectName("topBar");
    topBar->setFixedHeight(46);
    auto *topLay = new QHBoxLayout(topBar);
    topLay->setContentsMargins(12,6,12,6);

    auto *logo = new QLabel("⬡ QUANTUM PLAYER", topBar);
    logo->setStyleSheet("font-size:15px;font-weight:800;letter-spacing:3px;color:#7b68ff;");
    topLay->addWidget(logo);
    topLay->addStretch();

    topLay->addWidget(new QLabel("Viz:", topBar));
    m_vizStyleBox = new QComboBox(topBar);
    m_vizStyleBox->addItems({"Bars","Mirror"});
    m_vizStyleBox->setFixedWidth(88);
    connect(m_vizStyleBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onVisualizerStyleChanged);
    topLay->addWidget(m_vizStyleBox);

    topLay->addSpacing(10);
    topLay->addWidget(new QLabel("Theme:", topBar));
    m_themeBox = new QComboBox(topBar);
    m_themeBox->addItems(ThemeManager::instance()->themeNames());
    m_themeBox->setFixedWidth(108);
    connect(m_themeBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onThemeChanged);
    topLay->addWidget(m_themeBox);

    // Sleep timer
    topLay->addSpacing(10);
    m_sleepTimer = new SleepTimerWidget(topBar);
    connect(m_sleepTimer, &SleepTimerWidget::sleepTriggered,
            this, &MainWindow::onSleepTriggered);
    topLay->addWidget(m_sleepTimer);

    mainLay->addWidget(topBar);

    // ── MAIN SPLITTER ────────────────────────────────────────
    auto *splitter = new QSplitter(Qt::Horizontal, central);
    splitter->setHandleWidth(1);

    // Left panel (Library / Playlist tabs)
    auto *sidePanel = new QFrame(splitter);
    sidePanel->setObjectName("sidePanel");
    sidePanel->setMinimumWidth(230);
    sidePanel->setMaximumWidth(340);
    auto *sideLay = new QVBoxLayout(sidePanel);
    sideLay->setContentsMargins(0,0,0,0);
    auto *tabs = new QTabWidget(sidePanel);
    m_library  = new LibraryWidget(tabs);
    m_playlist = new PlaylistWidget(tabs);
    tabs->addTab(m_library,  "Library");
    tabs->addTab(m_playlist, "Playlist");
    sideLay->addWidget(tabs);
    splitter->addWidget(sidePanel);

    // Center panel (Now Playing + Visualizer)
    auto *centerPanel = new QFrame(splitter);
    centerPanel->setObjectName("centerPanel");
    auto *centerLay = new QVBoxLayout(centerPanel);
    centerLay->setContentsMargins(0,0,0,0);
    centerLay->setSpacing(0);
    m_nowPlaying = new NowPlayingWidget(centerPanel);
    m_nowPlaying->setMinimumHeight(300);
    centerLay->addWidget(m_nowPlaying, 2);
    m_visualizer = new VisualizerWidget(centerPanel);
    m_visualizer->setMinimumHeight(110);
    centerLay->addWidget(m_visualizer, 1);
    splitter->addWidget(centerPanel);

    // Right panel (EQ)
    auto *eqPanel = new QFrame(splitter);
    eqPanel->setObjectName("equalizerPanel");
    eqPanel->setMinimumWidth(200);
    eqPanel->setMaximumWidth(290);
    auto *eqLay = new QVBoxLayout(eqPanel);
    eqLay->setContentsMargins(0,0,0,0);
    m_equalizer = new EqualizerWidget(eqPanel);
    eqLay->addWidget(m_equalizer);
    splitter->addWidget(eqPanel);

    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    splitter->setStretchFactor(2, 0);
    splitter->setSizes({260, 720, 240});
    mainLay->addWidget(splitter, 1);

    // ── CONTROL BAR ──────────────────────────────────────────
    auto *ctrlBar = new QFrame(central);
    ctrlBar->setObjectName("controlBar");
    buildControlBar(ctrlBar);
    mainLay->addWidget(ctrlBar);

    // ── Connections ───────────────────────────────────────────
    connect(m_playlist, &PlaylistWidget::trackDoubleClicked,
            this, &MainWindow::onPlaylistDoubleClick);
    connect(m_playlist, &PlaylistWidget::addFilesRequested,
            this, &MainWindow::onAddFiles);
    connect(m_playlist, &PlaylistWidget::clearRequested,
            this, [this]{ m_engine->clearPlaylist(); });
    connect(m_playlist, &PlaylistWidget::removeRequested,
            this, [this](int i){ m_engine->removeTrack(i); });

    connect(m_library, &LibraryWidget::addToPlaylist,
            this, &MainWindow::onAddToPlaylist);
    connect(m_library, &LibraryWidget::playNow,
            this, &MainWindow::onPlayNow);

    connect(m_equalizer, &EqualizerWidget::gainsChanged,
            this, &MainWindow::onEQGainsChanged);
}

void MainWindow::buildControlBar(QWidget *parent) {
    auto *lay = new QVBoxLayout(parent);
    lay->setContentsMargins(0,0,0,0);
    lay->setSpacing(0);

    // Level meters (full width, above controls)
    m_levelMeters = new LevelMetersWidget(parent);
    lay->addWidget(m_levelMeters);

    auto *row = new QWidget(parent);
    auto *rlay = new QHBoxLayout(row);
    rlay->setContentsMargins(14,6,14,8);
    rlay->setSpacing(6);

    // Playback buttons
    m_shuffleBtn = new QPushButton("⇄", row);
    m_shuffleBtn->setObjectName("ctrlBtn");
    m_shuffleBtn->setCheckable(true);
    m_shuffleBtn->setToolTip("Shuffle");
    connect(m_shuffleBtn, &QPushButton::toggled,
            this, [this](bool on){ m_engine->setShuffleMode(on); });
    rlay->addWidget(m_shuffleBtn);

    m_prevBtn = new QPushButton("⏮", row);
    m_prevBtn->setObjectName("ctrlBtn");
    connect(m_prevBtn, &QPushButton::clicked, this, &MainWindow::onPrev);
    rlay->addWidget(m_prevBtn);

    m_playBtn = new QPushButton("▶", row);
    m_playBtn->setObjectName("playBtn");
    connect(m_playBtn, &QPushButton::clicked, this, &MainWindow::onPlayPause);
    rlay->addWidget(m_playBtn);

    m_nextBtn = new QPushButton("⏭", row);
    m_nextBtn->setObjectName("ctrlBtn");
    connect(m_nextBtn, &QPushButton::clicked, this, &MainWindow::onNext);
    rlay->addWidget(m_nextBtn);

    m_stopBtn = new QPushButton("⏹", row);
    m_stopBtn->setObjectName("ctrlBtn");
    connect(m_stopBtn, &QPushButton::clicked, this, &MainWindow::onStop);
    rlay->addWidget(m_stopBtn);

    m_repeatBtn = new QPushButton("↩", row);
    m_repeatBtn->setObjectName("ctrlBtn");
    m_repeatBtn->setCheckable(true);
    connect(m_repeatBtn, &QPushButton::toggled, this, [this](bool on){
        m_engine->setRepeatMode(on ? AudioEngine::RepeatAll : AudioEngine::RepeatNone);
    });
    rlay->addWidget(m_repeatBtn);

    rlay->addSpacing(8);

    // Seek bar
    m_posLabel = new QLabel("0:00", row);
    m_posLabel->setObjectName("timeLabel");
    m_posLabel->setFixedWidth(40);
    m_posLabel->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    rlay->addWidget(m_posLabel);

    m_seekSlider = new QSlider(Qt::Horizontal, row);
    m_seekSlider->setRange(0, 1000);
    m_seekSlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(m_seekSlider, &QSlider::sliderPressed,  [this]{ m_seeking = true; });
    connect(m_seekSlider, &QSlider::sliderReleased, [this]{
        m_seeking = false;
        qint64 dur = m_engine->duration();
        m_engine->seek(dur * m_seekSlider->value() / 1000);
    });
    rlay->addWidget(m_seekSlider);

    m_durLabel = new QLabel("0:00", row);
    m_durLabel->setObjectName("timeLabel");
    m_durLabel->setFixedWidth(40);
    rlay->addWidget(m_durLabel);

    rlay->addSpacing(8);

    // Volume
    m_muteBtn = new QPushButton("🔊", row);
    m_muteBtn->setObjectName("ctrlBtn");
    connect(m_muteBtn, &QPushButton::clicked, this, &MainWindow::onMute);
    rlay->addWidget(m_muteBtn);

    m_volSlider = new QSlider(Qt::Horizontal, row);
    m_volSlider->setRange(0, 100);
    m_volSlider->setValue(70);
    m_volSlider->setFixedWidth(88);
    m_volSlider->setToolTip("Volume");
    connect(m_volSlider, &QSlider::valueChanged, this, &MainWindow::onVolumeChanged);
    rlay->addWidget(m_volSlider);

    rlay->addSpacing(8);

    // Speed control
    m_speedLabel = new QLabel("1.0×", row);
    m_speedLabel->setObjectName("timeLabel");
    m_speedLabel->setFixedWidth(32);
    m_speedLabel->setToolTip("Playback Speed");
    rlay->addWidget(m_speedLabel);

    m_speedSlider = new QSlider(Qt::Horizontal, row);
    m_speedSlider->setRange(25, 200);   // 0.25x to 2.0x (×100)
    m_speedSlider->setValue(100);
    m_speedSlider->setFixedWidth(80);
    m_speedSlider->setToolTip("Playback Speed (0.25× – 2.0×)");
    connect(m_speedSlider, &QSlider::valueChanged, this, &MainWindow::onSpeedChanged);
    rlay->addWidget(m_speedSlider);

    lay->addWidget(row);
}

void MainWindow::buildMenuBar() {
    auto *fileMenu = menuBar()->addMenu("File");
    fileMenu->addAction("Add Files to Playlist…",  this, &MainWindow::onAddFiles, QKeySequence::Open);
    fileMenu->addAction("Add Folder…",             this, &MainWindow::onAddFolder);
    fileMenu->addSeparator();
    fileMenu->addAction("Save Library", this, [this]{
        QString p = QFileDialog::getSaveFileName(this,"Save Library","","JSON (*.json)");
        if (!p.isEmpty()) MusicLibrary::instance()->saveToFile(p);
    });
    fileMenu->addAction("Load Library", this, [this]{
        QString p = QFileDialog::getOpenFileName(this,"Load Library","","JSON (*.json)");
        if (!p.isEmpty()) MusicLibrary::instance()->loadFromFile(p);
    });
    fileMenu->addSeparator();
    fileMenu->addAction("Exit", this, &QWidget::close, QKeySequence::Quit);

    auto *playMenu = menuBar()->addMenu("Playback");
    playMenu->addAction("Play / Pause", this, &MainWindow::onPlayPause, QKeySequence(Qt::Key_Space));
    playMenu->addAction("Next",     this, &MainWindow::onNext,  QKeySequence(Qt::Key_Right));
    playMenu->addAction("Previous", this, &MainWindow::onPrev,  QKeySequence(Qt::Key_Left));
    playMenu->addAction("Stop",     this, &MainWindow::onStop,  QKeySequence(Qt::Key_S));

    auto *viewMenu = menuBar()->addMenu("View");
    auto *themeMenu = viewMenu->addMenu("Theme");
    for (int i = 0; i < ThemeManager::ThemeCount; ++i) {
        auto *a = themeMenu->addAction(ThemeManager::instance()->themeName(
                      static_cast<ThemeManager::Theme>(i)));
        connect(a, &QAction::triggered, this, [this,i]{ m_themeBox->setCurrentIndex(i); });
    }

    menuBar()->addMenu("Help")->addAction("About", this, &MainWindow::openAbout);
}

// ── Slots ─────────────────────────────────────────────────────

void MainWindow::onPlayPause() {
    m_engine->isPlaying() ? m_engine->pause() : m_engine->play();
}
void MainWindow::onNext()  { m_engine->next(); }
void MainWindow::onPrev()  { m_engine->previous(); }
void MainWindow::onStop()  { m_engine->stop(); }
void MainWindow::onShuffle() {}
void MainWindow::onRepeat()  {}

void MainWindow::onVolumeChanged(int v) {
    m_engine->setVolume(v / 100.f);
    m_muteBtn->setText(v == 0 ? "🔇" : v < 40 ? "🔈" : v < 75 ? "🔉" : "🔊");
}

void MainWindow::onMute() {
    bool m = m_engine->isMuted();
    m_engine->setMuted(!m);
    m_muteBtn->setText(!m ? "🔇" : "🔊");
}

void MainWindow::onSpeedChanged(int v) {
    float rate = v / 100.f;
    m_speedLabel->setText(QString::number(rate, 'f', 2) + "×");
    m_engine->setPlaybackRate(rate);
}

void MainWindow::onPositionChanged(qint64 ms) {
    if (!m_seeking) {
        qint64 dur = m_engine->duration();
        if (dur > 0) m_seekSlider->setValue(static_cast<int>(ms * 1000 / dur));
        m_posLabel->setText(formatTime(ms));
    }
}

void MainWindow::onDurationChanged(qint64 ms) {
    m_durLabel->setText(formatTime(ms));
}

void MainWindow::onEngineStateChanged(AudioEngine::State s) {
    bool playing = (s == AudioEngine::Playing);
    m_playBtn->setText(playing ? "⏸" : "▶");
    m_visualizer->setPlaying(playing);
    if (s == AudioEngine::Stopped) {
        m_seekSlider->setValue(0);
        m_posLabel->setText("0:00");
        m_levelMeters->reset();
    }
}

void MainWindow::onTrackChanged(int index) {
    m_playlist->setCurrentIndex(index);
    const auto &pl = m_engine->playlist();
    if (index >= 0 && index < pl.size()) {
        m_nowPlaying->setTrack(pl[index]);
        setWindowTitle(QString("%1 – %2 | Quantum Player")
            .arg(pl[index].title, pl[index].artist));
    }
}

void MainWindow::onLevelChanged(float l, float r) {
    m_levelMeters->setLevel(l, r);
}

void MainWindow::onPlayerError(const QString &msg) {
    statusBar()->showMessage("⚠  " + msg, 6000);
}

void MainWindow::onEQGainsChanged(const QVector<float> &gains) {
    m_engine->setEQGains(gains);
}

void MainWindow::onAddFiles() {
    QStringList files = QFileDialog::getOpenFileNames(this, "Add Audio Files", "",
        "Audio Files (*.mp3 *.flac *.wav *.ogg *.m4a *.aac *.opus *.wma)");
    for (const QString &f : files) {
        QFileInfo fi(f);
        Track t;
        t.id      = MusicLibrary::instance()->nextTrackId();
        t.title   = fi.baseName();
        t.fileUrl = QUrl::fromLocalFile(f);
        m_engine->appendTrack(t);
        m_playlist->addTrack(t);
    }
    if (!files.isEmpty() && m_engine->currentIndex() < 0)
        m_engine->playIndex(0);
}

void MainWindow::onAddFolder() {
    QString dir = QFileDialog::getExistingDirectory(this, "Add Folder");
    if (dir.isEmpty()) return;
    QStringList filters = {"*.mp3","*.flac","*.wav","*.ogg","*.m4a","*.aac","*.opus"};
    QDir d(dir);
    auto entries = d.entryInfoList(filters, QDir::Files, QDir::Name);
    for (const auto &fi : entries) {
        Track t;
        t.id      = MusicLibrary::instance()->nextTrackId();
        t.title   = fi.baseName();
        t.fileUrl = QUrl::fromLocalFile(fi.absoluteFilePath());
        m_engine->appendTrack(t);
        m_playlist->addTrack(t);
    }
    if (!entries.isEmpty() && m_engine->currentIndex() < 0)
        m_engine->playIndex(0);
}

void MainWindow::onPlaylistDoubleClick(int index) { m_engine->playIndex(index); }

void MainWindow::onAddToPlaylist(const QList<Track> &tracks) {
    for (const auto &t : tracks) {
        m_engine->appendTrack(t);
        m_playlist->addTrack(t);
    }
}

void MainWindow::onPlayNow(const QList<Track> &tracks) {
    m_engine->clearPlaylist();
    m_playlist->clearPlaylist();
    onAddToPlaylist(tracks);
    m_engine->playIndex(0);
}

void MainWindow::onThemeChanged(int i) {
    applyTheme(static_cast<ThemeManager::Theme>(i));
}

void MainWindow::onVisualizerStyleChanged(int i) {
    m_visualizer->setStyle(i == 0 ? VisualizerWidget::Bars : VisualizerWidget::Mirror);
}

void MainWindow::onSleepTriggered() {
    m_engine->stop();
    statusBar()->showMessage("💤 Sleep timer: playback stopped.", 5000);
}

void MainWindow::applyTheme(ThemeManager::Theme theme) {
    ThemeManager::instance()->apply(qApp, theme);
    updateVisualizerColors();
    m_nowPlaying->setAccentColor(ThemeManager::instance()->colors().accent);
    m_levelMeters->setAccentColor(ThemeManager::instance()->colors().accent);
    m_nowPlaying->update();
}

void MainWindow::updateVisualizerColors() {
    const auto &c = ThemeManager::instance()->colors();
    m_visualizer->setColors(c.vizBar1, c.vizBar2, c.vizBar3, c.vizPeak, c.vizBg);
}

void MainWindow::openAbout() {
    QMessageBox::about(this, "About Quantum Player",
        "<b>⬡ Quantum Player v1.1</b><br><br>"
        "Ultra-modern C++/Qt6 music player.<br><br>"
        "<b>Features:</b><br>"
        "• Real DSP 10-band equalizer (biquad IIR filters)<br>"
        "• Live L/R peak level meters<br>"
        "• Animated spectrum visualizer (Bars &amp; Mirror)<br>"
        "• 5 themes: Midnight · Cyberpunk · Ocean · Aurora · Carbon<br>"
        "• Music library (Artist / Album / Track)<br>"
        "• Shuffle, Repeat, Sleep Timer<br>"
        "• Playback speed control<br>"
        "• Drag &amp; Drop support<br><br>"
        "Built with Qt " QT_VERSION_STR);
}

QString MainWindow::formatTime(qint64 ms) const {
    qint64 s = ms / 1000;
    return QString("%1:%2").arg(s/60).arg(s%60, 2, 10, QChar('0'));
}

void MainWindow::saveSettings() {
    QSettings s("QuantumSoft","QuantumPlayer");
    s.setValue("volume",    m_volSlider->value());
    s.setValue("theme",     m_themeBox->currentIndex());
    s.setValue("shuffle",   m_shuffleBtn->isChecked());
    s.setValue("repeat",    m_repeatBtn->isChecked());
    s.setValue("vizStyle",  m_vizStyleBox->currentIndex());
    s.setValue("speed",     m_speedSlider->value());
    s.setValue("geometry",  saveGeometry());
    s.setValue("state",     saveState());
}

void MainWindow::loadSettings() {
    QSettings s("QuantumSoft","QuantumPlayer");
    m_volSlider->setValue(s.value("volume", 70).toInt());
    m_themeBox->setCurrentIndex(s.value("theme", 0).toInt());
    m_shuffleBtn->setChecked(s.value("shuffle", false).toBool());
    m_repeatBtn->setChecked(s.value("repeat", false).toBool());
    m_vizStyleBox->setCurrentIndex(s.value("vizStyle", 0).toInt());
    m_speedSlider->setValue(s.value("speed", 100).toInt());
    if (s.contains("geometry")) restoreGeometry(s.value("geometry").toByteArray());
}

void MainWindow::closeEvent(QCloseEvent *e) { saveSettings(); e->accept(); }

void MainWindow::dragEnterEvent(QDragEnterEvent *e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent *e) {
    static const QStringList EXT = {".mp3",".flac",".wav",".ogg",".m4a",".aac",".opus"};
    for (const auto &url : e->mimeData()->urls()) {
        QString path = url.toLocalFile();
        bool ok = false;
        for (const auto &ext : EXT) if (path.endsWith(ext, Qt::CaseInsensitive)) { ok=true; break; }
        if (!ok) continue;
        Track t;
        t.id      = MusicLibrary::instance()->nextTrackId();
        t.title   = QFileInfo(path).baseName();
        t.fileUrl = url;
        m_engine->appendTrack(t);
        m_playlist->addTrack(t);
    }
    if (m_engine->currentIndex() < 0 && !m_engine->playlist().isEmpty())
        m_engine->playIndex(0);
}

void MainWindow::keyPressEvent(QKeyEvent *e) {
    switch (e->key()) {
    case Qt::Key_Space:      onPlayPause(); break;
    case Qt::Key_Right:      m_engine->seek(m_engine->position() + 5000);  break;
    case Qt::Key_Left:       m_engine->seek(m_engine->position() - 5000);  break;
    case Qt::Key_Up:         m_volSlider->setValue(qMin(100, m_volSlider->value()+5)); break;
    case Qt::Key_Down:       m_volSlider->setValue(qMax(0,   m_volSlider->value()-5)); break;
    case Qt::Key_N:          onNext();  break;
    case Qt::Key_P:          onPrev();  break;
    case Qt::Key_S:          onStop();  break;
    case Qt::Key_M:          onMute();  break;
    default: QMainWindow::keyPressEvent(e);
    }
}
