# ZBattleShip

A Battleship game where you play against a PyTorch AI. The project includes a
Pygame interface, a console version, and scripts for generating game data,
training the AI, exporting it to ONNX, and packaging the apps with PyInstaller.

## Requirements

- Python 3.10 or newer
- The trained model file `models/BattleShipAgentV2.pth` to play the game
- A CUDA-capable PyTorch installation to train the AI (the training script
  currently selects the CUDA device directly)

Install the project dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy matplotlib torch pygame pillow polars tqdm scikit-learn onnx pyinstaller
```

On Linux or macOS, activate the environment with:

```sh
source .venv/bin/activate
```

For GPU training, install the PyTorch build that matches your CUDA setup using
the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
The provided `requirements.txt` is currently empty, so install the packages
using the command above rather than `pip install -r requirements.txt`.

## Play

Start the graphical game:

```powershell
python pygame_app.py
```

Start the console game:

```powershell
python console_app.py
```

In the Pygame version, click a square on the left grid to fire at the AI. The
right grid shows your own board. When the game ends, select **Play Again** to
start another round. In the console version, enter a tile number from `0` to
`99` when prompted.

Both games use the model at `models/BattleShipAgentV2.pth` and run inference on
the CPU. If the model file is missing, generate training data and train a model
using the steps below.

## Generate training data

The data generator simulates games and writes Parquet files under `data/`.
By default it generates 8,000 games in chunks of 500:

```powershell
python datagen.py
```

Ensure the `data` directory exists before running the generator. The dataset
loader reads files matching `data/battleship_*.parquet`.

## Train the AI

After generating data, run:

```powershell
python training.py
```

Training uses the Parquet dataset, trains for 15 epochs, and saves epoch
checkpoints to `checkpoints/`. The final model weights are saved to
`models/BattleShipAgentV2.pth`. The training script currently uses CUDA
directly, so a compatible CUDA-enabled PyTorch installation and GPU are needed.

## Export the model to ONNX

With trained weights available, run:

```powershell
python export.py
```

This exports `models/BattleShipAgent.onnx` and validates the exported model.

## Build Windows executables

With PyInstaller installed in the project's `.venv`, run either build script
from the repository root:

```powershell
.\build_pygame_app.bat
```

The Pygame build is created at
`dist/BattleshipPyGame/BattleshipPygame.exe`.

```powershell
.\build_console_app.bat
```

The console build is created at `dist/BattleshipConsole/Battleship.exe`.
Distribute the entire corresponding folder, not just the `.exe`; each build
bundles its Python dependencies and the trained model.

## Project files

| File | Purpose |
| --- | --- |
| `pygame_app.py` | Pygame game interface |
| `console_app.py` | Console game interface |
| `model.py` | PyTorch AI model definition |
| `datagen.py` | Simulates games and writes Parquet training data |
| `dataset.py` | Loads and prepares the training dataset |
| `training.py` | Trains the AI and saves checkpoints and final weights |
| `export.py` | Exports and validates the ONNX model |
| `battleship_pygame.spec` | PyInstaller configuration for the Pygame app |
| `battleship_console.spec` | PyInstaller configuration for the console app |
