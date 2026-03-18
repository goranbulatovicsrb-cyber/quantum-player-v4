# ⬡ Quantum Player

**Ultra-modern C++/Qt6 music player with stunning visuals and powerful features.**

![Quantum Player Screenshot](docs/screenshot.png)

## ✨ Features

- 🎵 **Full audio playback** — MP3, FLAC, WAV, OGG, M4A, AAC, OPUS
- 🌈 **5 Ultra-modern themes** — Midnight, Cyberpunk, Ocean, Aurora, Carbon
- 📊 **Animated spectrum visualizer** — Bars & Mirror modes with peak indicators
- 🎛️ **10-band Equalizer** — With 11 presets (Bass Boost, Rock, Jazz, Electronic...)
- 📚 **Music Library** — Organize by Artist → Album → Track with album art
- 📋 **Playlist management** — Add files, drag-to-reorder, context menus
- 🔀 **Shuffle & Repeat** — All repeat modes
- 🖱️ **Drag & Drop** — Drop audio files directly onto the player
- 💾 **Persistent settings** — Theme, volume, and window state saved
- 📁 **Library save/load** — Export your library to JSON

---

## 🔧 Requirements

| Dependency | Version |
|------------|---------|
| Qt         | 6.2+    |
| CMake      | 3.16+   |
| C++ compiler | C++17 (MSVC 2019+, GCC 10+, Clang 12+) |

---

## 🚀 Build Instructions

### Windows (recommended: Visual Studio + Qt)

```bash
# 1. Install Qt 6 from https://www.qt.io/download
#    Make sure to include: Qt Multimedia, Qt MultimediaWidgets

# 2. Clone / unzip the project
unzip QuantumPlayer.zip
cd QuantumPlayer

# 3. Configure and build
cmake -B build -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/msvc2022_64"
cmake --build build --config Release

# 4. Run
build\Release\QuantumPlayer.exe
```

### Linux (Ubuntu/Debian)

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y qt6-base-dev qt6-multimedia-dev cmake build-essential

# 2. Build
cd QuantumPlayer
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 3. Run
./build/QuantumPlayer
```

### macOS

```bash
# 1. Install Qt via Homebrew or from qt.io
brew install qt@6

# 2. Build
cd QuantumPlayer
cmake -B build -DCMAKE_PREFIX_PATH=$(brew --prefix qt@6)
cmake --build build -j$(sysctl -n hw.ncpu)

# 3. Run
./build/QuantumPlayer.app/Contents/MacOS/QuantumPlayer
```

---

## 🎮 Usage

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `→` | Next track |
| `←` | Previous track |
| `S` | Stop |
| `Ctrl+O` | Add files |
| `Ctrl+Q` | Quit |

### Getting Started
1. **Add music**: File → Add Files (or drag & drop audio files onto the window)
2. **Build library**: Library tab → `+ Artist` → `+ Album` (add album cover + tracks)
3. **Change theme**: Use the Theme dropdown in the top bar
4. **Equalizer**: Adjust the 10-band EQ on the right, or pick a preset
5. **Visualizer style**: Switch between Bars and Mirror mode in the top bar

---

## 🏗️ Architecture

```
QuantumPlayer/
├── src/
│   ├── main.cpp                  # Entry point
│   ├── mainwindow.h/cpp          # Main window + layout
│   ├── player/
│   │   └── audioplayer.h/cpp     # QMediaPlayer wrapper
│   ├── dsp/
│   │   └── biquadfilter.h        # IIR biquad DSP filter
│   ├── ui/
│   │   ├── visualizerwidget      # Animated spectrum bars
│   │   ├── equalizerwidget       # 10-band EQ + curve
│   │   ├── playlistwidget        # Playlist table
│   │   ├── librarywidget         # Artist/Album/Track tree
│   │   ├── nowplayingwidget      # Album art + track info
│   │   └── thememanager          # QSS themes
│   └── library/
│       ├── track.h               # Track data
│       ├── album.h               # Album data
│       ├── artist.h              # Artist data
│       └── musiclibrary.h/cpp    # Library manager + JSON persistence
└── CMakeLists.txt
```

---

## 📜 License

MIT License — free to use, modify, and distribute.
