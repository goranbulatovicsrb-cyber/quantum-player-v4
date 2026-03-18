cmake_minimum_required(VERSION 3.16)
project(QuantumPlayer VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 6.2 REQUIRED COMPONENTS Core Widgets Multimedia MultimediaWidgets)

qt_add_executable(QuantumPlayer
    src/main.cpp
    src/mainwindow.cpp               src/mainwindow.h

    src/player/audioengine.cpp       src/player/audioengine.h
    src/player/audioplayer.cpp       src/player/audioplayer.h

    src/dsp/biquadfilter.h

    src/ui/visualizerwidget.cpp      src/ui/visualizerwidget.h
    src/ui/equalizerwidget.cpp       src/ui/equalizerwidget.h
    src/ui/playlistwidget.cpp        src/ui/playlistwidget.h
    src/ui/librarywidget.cpp         src/ui/librarywidget.h
    src/ui/nowplayingwidget.cpp      src/ui/nowplayingwidget.h
    src/ui/thememanager.cpp          src/ui/thememanager.h
    src/ui/levelmeterswidget.cpp     src/ui/levelmeterswidget.h
    src/ui/sleeptimerwidget.cpp      src/ui/sleeptimerwidget.h

    src/library/track.h
    src/library/album.h
    src/library/artist.h
    src/library/musiclibrary.cpp     src/library/musiclibrary.h

    src/resources/resources.qrc
)

target_include_directories(QuantumPlayer PRIVATE src)

target_link_libraries(QuantumPlayer PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::MultimediaWidgets
)

set_target_properties(QuantumPlayer PROPERTIES
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS QuantumPlayer
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
