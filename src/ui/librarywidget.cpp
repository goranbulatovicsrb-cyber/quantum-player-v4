#include "librarywidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QAction>
#include <QInputDialog>
#include <QFileDialog>
#include <QMessageBox>
#include <QTreeWidgetItem>
#include <QHeaderView>
#include <QLineEdit>

static const int ROLE_TYPE     = Qt::UserRole + 1; // "artist","album","track"
static const int ROLE_ART_ID   = Qt::UserRole + 2;
static const int ROLE_ALB_ID   = Qt::UserRole + 3;
static const int ROLE_TRACK_ID = Qt::UserRole + 4;

LibraryWidget::LibraryWidget(QWidget *parent) : QWidget(parent) {
    buildUI();
    refresh();
    connect(MusicLibrary::instance(), &MusicLibrary::libraryChanged,
            this, &LibraryWidget::refresh);
}

void LibraryWidget::buildUI() {
    auto *vlay = new QVBoxLayout(this);
    vlay->setContentsMargins(0, 0, 0, 0);
    vlay->setSpacing(0);

    // Toolbar
    auto *toolbar = new QWidget(this);
    toolbar->setFixedHeight(38);
    toolbar->setStyleSheet("border-bottom: 1px solid #1a1a2a;");
    auto *tlay = new QHBoxLayout(toolbar);
    tlay->setContentsMargins(8, 4, 8, 4);

    auto *lbl = new QLabel("LIBRARY", toolbar);
    lbl->setObjectName("sectionTitle");
    tlay->addWidget(lbl);
    tlay->addStretch();

    auto *addArtBtn = new QPushButton("+ Artist", toolbar);
    addArtBtn->setFixedHeight(26);
    addArtBtn->setStyleSheet("font-size: 11px;");
    connect(addArtBtn, &QPushButton::clicked, this, &LibraryWidget::onAddArtist);
    tlay->addWidget(addArtBtn);

    auto *addAlbBtn = new QPushButton("+ Album", toolbar);
    addAlbBtn->setFixedHeight(26);
    addAlbBtn->setStyleSheet("font-size: 11px;");
    connect(addAlbBtn, &QPushButton::clicked, this, &LibraryWidget::onAddAlbum);
    tlay->addWidget(addAlbBtn);

    vlay->addWidget(toolbar);

    // Tree
    m_tree = new QTreeWidget(this);
    m_tree->setColumnCount(1);
    m_tree->header()->hide();
    m_tree->setIndentation(16);
    m_tree->setAnimated(true);
    m_tree->setContextMenuPolicy(Qt::CustomContextMenu);

    connect(m_tree, &QTreeWidget::customContextMenuRequested,
            this, &LibraryWidget::onContextMenu);
    connect(m_tree, &QTreeWidget::itemDoubleClicked,
            this, &LibraryWidget::onItemDoubleClicked);

    vlay->addWidget(m_tree);
}

void LibraryWidget::refresh() {
    m_tree->clear();
    populateTree();
}

void LibraryWidget::populateTree() {
    auto *lib = MusicLibrary::instance();
    for (const auto &artist : lib->artists()) {
        auto *artItem = new QTreeWidgetItem(m_tree);
        artItem->setText(0, QString("♪  %1").arg(artist.name));
        artItem->setData(0, ROLE_TYPE,   "artist");
        artItem->setData(0, ROLE_ART_ID, artist.id);
        artItem->setExpanded(true);

        // Optional artist photo
        if (!artist.photo.isNull())
            artItem->setIcon(0, QIcon(artist.photo.scaled(20,20,Qt::KeepAspectRatio,Qt::SmoothTransformation)));

        for (const auto &album : artist.albums) {
            auto *albItem = new QTreeWidgetItem(artItem);
            QString albLabel = QString("  %1  (%2)")
                .arg(album.title)
                .arg(album.year > 0 ? QString::number(album.year) : "");
            albItem->setText(0, albLabel);
            albItem->setData(0, ROLE_TYPE,   "album");
            albItem->setData(0, ROLE_ART_ID, artist.id);
            albItem->setData(0, ROLE_ALB_ID, album.id);

            if (!album.cover.isNull())
                albItem->setIcon(0, QIcon(album.cover.scaled(20,20,Qt::KeepAspectRatio,Qt::SmoothTransformation)));

            for (const auto &track : album.tracks) {
                auto *trkItem = new QTreeWidgetItem(albItem);
                trkItem->setText(0, QString("    %1").arg(track.title.isEmpty() ? "Unknown" : track.title));
                trkItem->setData(0, ROLE_TYPE,     "track");
                trkItem->setData(0, ROLE_ART_ID,   artist.id);
                trkItem->setData(0, ROLE_ALB_ID,   album.id);
                trkItem->setData(0, ROLE_TRACK_ID, track.id);
            }
        }
    }
}

QList<Track> LibraryWidget::tracksForItem(QTreeWidgetItem *item) const {
    if (!item) return {};
    auto *lib = MusicLibrary::instance();
    QString type  = item->data(0, ROLE_TYPE).toString();
    int artId = item->data(0, ROLE_ART_ID).toInt();
    int albId = item->data(0, ROLE_ALB_ID).toInt();
    int trkId = item->data(0, ROLE_TRACK_ID).toInt();

    if (type == "artist") {
        if (auto *a = lib->findArtist(artId)) return a->allTracks();
    } else if (type == "album") {
        if (auto *alb = lib->findAlbum(artId, albId)) return alb->tracks;
    } else if (type == "track") {
        if (auto *alb = lib->findAlbum(artId, albId)) {
            for (const auto &t : alb->tracks)
                if (t.id == trkId) return {t};
        }
    }
    return {};
}

void LibraryWidget::onItemDoubleClicked(QTreeWidgetItem *item, int) {
    auto tracks = tracksForItem(item);
    if (!tracks.isEmpty()) emit addToPlaylist(tracks);
}

void LibraryWidget::onContextMenu(const QPoint &pos) {
    auto *item = m_tree->itemAt(pos);
    QMenu menu;
    if (item) {
        menu.addAction("▶  Play Now",        [this, item]{ emit playNow(tracksForItem(item)); });
        menu.addAction("+ Add to Playlist",  [this, item]{ emit addToPlaylist(tracksForItem(item)); });
        menu.addSeparator();

        QString type = item->data(0, ROLE_TYPE).toString();
        if (type == "artist") {
            int artId = item->data(0, ROLE_ART_ID).toInt();
            menu.addAction("Remove Artist", [artId]{
                MusicLibrary::instance()->removeArtist(artId);
            });
        }
    }
    menu.addSeparator();
    menu.addAction("Add Artist...", this, &LibraryWidget::onAddArtist);
    menu.addAction("Add Album...", this, &LibraryWidget::onAddAlbum);
    menu.exec(m_tree->viewport()->mapToGlobal(pos));
}

void LibraryWidget::onAddArtist() {
    bool ok;
    QString name = QInputDialog::getText(this, "New Artist", "Artist name:", QLineEdit::Normal, "", &ok);
    if (!ok || name.isEmpty()) return;

    Artist a;
    a.id   = MusicLibrary::instance()->nextArtistId();
    a.name = name;

    // Optional photo
    QString photoPath = QFileDialog::getOpenFileName(this, "Artist Photo (optional)", "",
        "Images (*.png *.jpg *.jpeg *.webp)");
    if (!photoPath.isEmpty()) a.photo = QPixmap(photoPath);

    MusicLibrary::instance()->addArtist(a);
}

void LibraryWidget::onAddAlbum() {
    auto *lib = MusicLibrary::instance();
    if (lib->artists().isEmpty()) {
        QMessageBox::information(this, "No Artists", "Please add an artist first.");
        return;
    }
    // Artist selection
    QStringList artistNames;
    for (const auto &a : lib->artists()) artistNames << a.name;
    bool ok;
    QString artistName = QInputDialog::getItem(this, "Select Artist", "Artist:",
                                               artistNames, 0, false, &ok);
    if (!ok) return;
    int artId = -1;
    for (const auto &a : lib->artists())
        if (a.name == artistName) { artId = a.id; break; }
    if (artId < 0) return;

    QString albumTitle = QInputDialog::getText(this, "Album Title", "Title:", QLineEdit::Normal, "", &ok);
    if (!ok || albumTitle.isEmpty()) return;

    int year = QInputDialog::getInt(this, "Album Year", "Year:", 2024, 1900, 2100, 1, &ok);
    if (!ok) year = 0;

    Album album;
    album.id     = lib->nextAlbumId();
    album.title  = albumTitle;
    album.artist = artistName;
    album.year   = year;

    // Album cover
    QString coverPath = QFileDialog::getOpenFileName(this, "Album Cover (optional)", "",
        "Images (*.png *.jpg *.jpeg *.webp)");
    if (!coverPath.isEmpty()) album.cover = QPixmap(coverPath);

    // Add tracks
    QStringList trackFiles = QFileDialog::getOpenFileNames(this, "Add Tracks to Album", "",
        "Audio Files (*.mp3 *.flac *.wav *.ogg *.m4a *.aac *.opus)");

    for (const QString &tf : trackFiles) {
        QFileInfo fi(tf);
        Track t;
        t.id       = lib->nextTrackId();
        t.title    = fi.baseName();
        t.artist   = artistName;
        t.album    = albumTitle;
        t.fileUrl  = QUrl::fromLocalFile(tf);
        t.albumArt = album.cover;
        album.tracks.append(t);
    }

    lib->addAlbumToArtist(artId, album);
}
