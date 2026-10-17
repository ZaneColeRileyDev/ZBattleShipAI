@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Create the project virtual environment and install PyInstaller first.
    exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller --clean --noconfirm battleship_pygame.spec
if errorlevel 1 exit /b %errorlevel%

echo Build complete: dist\Battleship\Battleship.exe
