#!/bin/sh

if [ ! -x ".venv/bin/python" ]; then
    echo "ERROR: Create the project virtual environment and install PyInstaller first." >&2
    exit 1
fi

".venv/bin/python" -m PyInstaller --clean --noconfirm battleship_console.spec || exit $?

echo "Build complete: dist/BattleshipPyGame/BattleshipConsole"