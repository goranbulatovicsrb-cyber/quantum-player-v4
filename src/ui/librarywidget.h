#pragma once
#include <QWidget>
#include <QTreeWidget>
#include <QPushButton>
#include "../library/musiclibrary.h"

class LibraryWidget : public QWidget {
    Q_OBJECT
public:
    explicit LibraryWidget(QWidget *parent = nullptr);
    void refresh();

signals:
    void addToPlaylist(const QList<Track> &tracks);
    void playNow(const QList<Track> &tracks);

private slots:
    void onAddArtist();
    void onAddAlbum();
    void onContextMenu(const QPoint &pos);
    void onItemDoubleClicked(QTreeWidgetItem *item, int col);

private:
    QTreeWidget *m_tree;
    void buildUI();
    void populateTree();
    QList<Track> tracksForItem(QTreeWidgetItem *item) const;
};
