#include <QApplication>
#include <QFileInfo>
#include <QUrl>
#include "mainwindow.h"
#include "library/musiclibrary.h"
#include "library/track.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("Quantum Player");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("QuantumSoft");
    app.setOrganizationDomain("quantumsoft.app");
    app.setStyle("Fusion");

    MainWindow window;
    window.show();
    return app.exec();
}
