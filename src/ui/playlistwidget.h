#pragma once
#include <QWidget>
#include <QTableWidget>
#include <QPushButton>
#include <QLabel>
#include "../library/track.h"

class PlaylistWidget : public QWidget {
    Q_OBJECT
public:
    explicit PlaylistWidget(QWidget *parent = nullptr);

    void setTracks(const QList<Track> &tracks);
    void addTrack(const Track &track);
    void removeTrack(int index);
    void clearPlaylist();
    void setCurrentIndex(int index);
    int  currentIndex() const;
    QList<Track> tracks() const { return m_tracks; }

signals:
    void trackDoubleClicked(int index);
    void tracksReordered(const QList<Track> &newOrder);
    void removeRequested(int index);
    void addFilesRequested();
    void clearRequested();

private slots:
    void onDoubleClick(int row, int col);
    void onContextMenu(const QPoint &pos);
    void onAddFiles();
    void onClear();
    void onRemoveSelected();

private:
    QTableWidget *m_table;
    QList<Track>  m_tracks;
    int           m_currentIndex = -1;

    void buildUI();
    void refreshTable();
    void highlightCurrent();
};
