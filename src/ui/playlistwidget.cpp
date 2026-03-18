#include "playlistwidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMimeData>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QLabel>

PlaylistWidget::PlaylistWidget(QWidget *parent) : QWidget(parent) {
    buildUI();
    setAcceptDrops(true);
}

void PlaylistWidget::buildUI() {
    auto *vlay = new QVBoxLayout(this);
    vlay->setContentsMargins(0, 0, 0, 0);
    vlay->setSpacing(0);

    // Toolbar
    auto *toolbar = new QWidget(this);
    toolbar->setObjectName("playlistToolbar");
    toolbar->setStyleSheet("QWidget#playlistToolbar { border-bottom: 1px solid #222; }");
    toolbar->setFixedHeight(38);
    auto *tlay = new QHBoxLayout(toolbar);
    tlay->setContentsMargins(8, 4, 8, 4);

    auto *titleLbl = new QLabel("PLAYLIST", toolbar);
    titleLbl->setObjectName("sectionTitle");
    tlay->addWidget(titleLbl);
    tlay->addStretch();

    auto *addBtn = new QPushButton("+ Add Files", toolbar);
    addBtn->setFixedHeight(26);
    addBtn->setStyleSheet("font-size: 11px;");
    connect(addBtn, &QPushButton::clicked, this, &PlaylistWidget::onAddFiles);
    tlay->addWidget(addBtn);

    auto *clearBtn = new QPushButton("Clear", toolbar);
    clearBtn->setFixedHeight(26);
    clearBtn->setStyleSheet("font-size: 11px;");
    connect(clearBtn, &QPushButton::clicked, this, &PlaylistWidget::onClear);
    tlay->addWidget(clearBtn);

    vlay->addWidget(toolbar);

    // Table
    m_table = new QTableWidget(this);
    m_table->setColumnCount(5);
    m_table->setHorizontalHeaderLabels({"#", "Title", "Artist", "Album", "Time"});
    m_table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Fixed);
    m_table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    m_table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    m_table->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    m_table->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Fixed);
    m_table->setColumnWidth(0, 38);
    m_table->setColumnWidth(4, 52);
    m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_table->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_table->verticalHeader()->hide();
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->setAlternatingRowColors(true);
    m_table->setShowGrid(false);
    m_table->setDragDropMode(QAbstractItemView::InternalMove);
    m_table->setDragEnabled(true);
    m_table->setDropIndicatorShown(true);
    m_table->verticalHeader()->setDefaultSectionSize(34);

    connect(m_table, &QTableWidget::cellDoubleClicked,
            this, &PlaylistWidget::onDoubleClick);
    m_table->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_table, &QTableWidget::customContextMenuRequested,
            this, &PlaylistWidget::onContextMenu);

    vlay->addWidget(m_table);
}

void PlaylistWidget::setTracks(const QList<Track> &tracks) {
    m_tracks = tracks;
    refreshTable();
}

void PlaylistWidget::addTrack(const Track &track) {
    m_tracks.append(track);
    int row = m_table->rowCount();
    m_table->insertRow(row);
    m_table->setItem(row, 0, new QTableWidgetItem(QString::number(row + 1)));
    m_table->setItem(row, 1, new QTableWidgetItem(track.title.isEmpty() ? "Unknown" : track.title));
    m_table->setItem(row, 2, new QTableWidgetItem(track.artist));
    m_table->setItem(row, 3, new QTableWidgetItem(track.album));
    m_table->setItem(row, 4, new QTableWidgetItem(track.durationString()));
    for (int c = 0; c < 5; ++c)
        if (auto *it = m_table->item(row, c))
            it->setTextAlignment(c == 0 || c == 4
                ? Qt::AlignCenter : (Qt::AlignLeft | Qt::AlignVCenter));
}

void PlaylistWidget::removeTrack(int index) {
    if (index < 0 || index >= m_tracks.size()) return;
    m_tracks.removeAt(index);
    m_table->removeRow(index);
    // Re-number
    for (int r = 0; r < m_table->rowCount(); ++r)
        m_table->setItem(r, 0, new QTableWidgetItem(QString::number(r+1)));
}

void PlaylistWidget::clearPlaylist() {
    m_tracks.clear();
    m_table->setRowCount(0);
    m_currentIndex = -1;
}

void PlaylistWidget::setCurrentIndex(int index) {
    m_currentIndex = index;
    highlightCurrent();
}

int PlaylistWidget::currentIndex() const {
    return m_table->currentRow();
}

void PlaylistWidget::refreshTable() {
    m_table->setRowCount(0);
    for (const auto &t : m_tracks) addTrack(t);
    highlightCurrent();
}

void PlaylistWidget::highlightCurrent() {
    for (int r = 0; r < m_table->rowCount(); ++r) {
        bool cur = (r == m_currentIndex);
        for (int c = 0; c < m_table->columnCount(); ++c) {
            if (auto *it = m_table->item(r, c)) {
                QFont f = it->font();
                f.setBold(cur);
                it->setFont(f);
                if (cur) {
                    it->setForeground(QColor(0x7b, 0x68, 0xff));
                } else {
                    it->setForeground(QBrush());
                }
            }
        }
    }
}

void PlaylistWidget::onDoubleClick(int row, int) {
    emit trackDoubleClicked(row);
}

void PlaylistWidget::onContextMenu(const QPoint &pos) {
    int row = m_table->rowAt(pos.y());
    QMenu menu;
    if (row >= 0) {
        menu.addAction("▶  Play Now", [this, row]{ emit trackDoubleClicked(row); });
        menu.addSeparator();
        menu.addAction("✕  Remove", [this, row]{ onRemoveSelected(); });
    }
    menu.addSeparator();
    menu.addAction("+ Add Files", this, &PlaylistWidget::onAddFiles);
    menu.addAction("Clear Playlist", this, &PlaylistWidget::onClear);
    menu.exec(m_table->viewport()->mapToGlobal(pos));
}

void PlaylistWidget::onAddFiles() {
    emit addFilesRequested();
}

void PlaylistWidget::onClear() {
    clearPlaylist();
    emit clearRequested();
}

void PlaylistWidget::onRemoveSelected() {
    QList<int> rows;
    for (auto *item : m_table->selectedItems()) {
        int r = item->row();
        if (!rows.contains(r)) rows.prepend(r);
    }
    std::sort(rows.rbegin(), rows.rend());
    for (int r : rows) {
        emit removeRequested(r);
        removeTrack(r);
    }
}
