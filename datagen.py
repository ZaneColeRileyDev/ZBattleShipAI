import numpy as np
import matplotlib.pyplot as plt
import random 
import polars as pl

from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

destroyer = np.array([1, 1], dtype=np.float32)
submarine = np.array([2, 2, 2], dtype=np.float32)
crusier = np.array([[3, 3, 3],
                    [0, 0, 3]], dtype=np.float32)

battleship = np.array([4, 4, 4, 4], dtype=np.float32)

aircraft_carrier = np.array([[5, 5, 5, 5],
                             [0, 0, 5, 5]])

ships = [destroyer, submarine, crusier, battleship, aircraft_carrier]

colors = {0: "#cfe8f3", 3: "#e07b39", 5: "#3b5b92", 2: "#8C9BB4", 4: "#4716A9", 1: "#91C67F", 8: "#FF0000", 7: "#2BFF00"}
labels = {0: "Water", 3: "Crusier", 5: "Aircraft Carrier", 1: "Destroyer", 2: "Submarine", 4: "Battleship", 8: "Hit", 7: "Miss"}

rng = np.random.default_rng()

def create_board() -> np.ndarray:
    return np.array(np.zeros((10, 10), dtype=np.float32), dtype=np.float32)



def check_tile_against_ship_shape(tile: dict, board: np.ndarray, ship_shape: tuple) -> bool:
    try:
        row = tile["row"]
        col = tile["col"]
        if len(ship_shape) > 1:
            board[row + ship_shape[0], col + ship_shape[1]]
            return True
        else:
            board[row + ship_shape[0], col]
            return True
    except IndexError:
        return False

def pick_shot(visible: np.ndarray) -> tuple[int, int]:
    # Target mode: shoot next to existing hits
    cands = []
    for r, c in np.argwhere(visible == 8):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 10 and 0 <= cc < 10 and visible[rr, cc] == 0:
                cands.append((rr, cc))
    if cands:
        return random.choice(cands)

    # Hunt mode: checkerboard parity (smallest ship is length 2)
    unknown = [tuple(x) for x in np.argwhere(visible == 0)]
    parity = [(r, c) for r, c in unknown if (r + c) % 2 == 0]
    return random.choice(parity or unknown)

def insert_ship(ship: np.ndarray, board: np.ndarray, rotate: bool = True,
                max_attempts: int = 1000) -> np.ndarray:
    ship = np.atleast_2d(ship)              # 1D (n,) -> 2D (1, n)
    if rotate:
        ship = np.rot90(ship, np.random.randint(4))

    h, w = ship.shape
    mask = ship != 0                        # only the cells the ship occupies

    for _ in range(max_attempts):
        row = np.random.randint(0, board.shape[0] - h + 1)
        col = np.random.randint(0, board.shape[1] - w + 1)

        region = board[row:row + h, col:col + w]   # a view into the board
        if np.all(region[mask] == 0):              # no overlap
            region[mask] = ship[mask]              # writes into board
            return board

    raise RuntimeError("Could not place ship; board too crowded.")


def add_ships(board: np.ndarray, ship_num: int = 3) -> np.ndarray:
    ships_copy = ships.copy()
    for i in range(ship_num):
        ship_idx = np.random.randint(0, len(ships_copy))
        ship = ships_copy[ship_idx]
        board = insert_ship(ship, board)
        ships_copy.pop(ship_idx)
    return board

def search_for_ships(board: np.ndarray, ship_num: int = 3) -> dict:
    ships = {
        1: "destroyer",
        2: "submarine",
        3: "crusier",
        4: "battleship",
        5: "aircraft carrier"
    }
    locations = {}
    for s in ships.keys():
        locations[ships[s]] = []
    for v in [1, 2, 3, 4, 5]:
        ship = np.argwhere(board == v)
        if ship.shape[0] == 0:
            continue
        else:
            locations[ships[v]].append(ship) 
    return locations


def extract_board_data(board, board_data, turn, row, col, player):
    visible = np.where(board >= 7, board, 0)   # keep only hits/misses
    board_data.append((turn, int(row) * 10 + int(col), visible.astype(np.uint8).ravel(), player))


def build_game_frame(game_rows, winner, game_id) -> pl.DataFrame:
    n = len(game_rows)
    cells = np.stack([r[2] for r in game_rows])          # (n, 100) uint8
    df = pl.DataFrame(cells, schema=[f"cell_{i}" for i in range(100)], orient="row")
    return df.with_columns(
        pl.Series("game",   np.full(n, game_id, dtype=np.uint32)),
        pl.Series("turn",   np.array([r[0] for r in game_rows], dtype=np.uint16)),
        pl.Series("action", np.array([r[1] for r in game_rows], dtype=np.uint8)),
        pl.Series("player", np.array([r[3] for r in game_rows], dtype=np.uint8)),
        pl.Series("winner", np.full(n, winner, dtype=np.uint8)),
    )


def take_shot(target: np.ndarray, locations: dict, threshold: float) -> tuple[int, int]:
    # Cells that are still fair game
    open_cells = np.flatnonzero((target.ravel() != 7) & (target.ravel() != 8))
    if open_cells.size == 0:
        raise RuntimeError(f"Board full. sunk state: {locations.keys()}")
    if np.random.random() > threshold:
        # "Smart" shot: pick a ship cell that hasn't been hit yet
        remaining = [(r, c) for ship_list in locations.values()
                            for ship in ship_list
                            for r, c in ship if target[r, c] != 8]
        if remaining:
            row, col = random.choice(remaining)
            target[row, col] = 8
            return int(row), int(col)

    # Random shot at an untouched cell
    row, col = np.unravel_index(rng.choice(open_cells), target.shape)
    hit = target[row, col] in (1, 2, 3, 4, 5)   # still holds a ship value
    target[row, col] = 8 if hit else 7
    return int(row), int(col)

def apply_shot(board: np.ndarray, r: int, c: int) -> None:
    hit = board[r, c] in (1, 2, 3, 4, 5)
    board[r, c] = 8 if hit else 7

def auto_play(p1, p2, board_data, turn_num, loc1=None, loc2=None, threshold=None):
    # Player 1 shoots at p2's board
    before = p2.copy()
    r, c = pick_shot(np.where(before >= 7, before, 0))
    extract_board_data(before, board_data, turn_num, r, c, player=1)
    apply_shot(p2, r, c)

    # Player 2 shoots at p1's board
    before = p1.copy()
    r, c = pick_shot(np.where(before >= 7, before, 0))
    extract_board_data(before, board_data, turn_num, r, c, player=2)
    apply_shot(p1, r, c)
    return board_data
                    

def check_sunk(board: np.ndarray, locations: dict, sunk: dict) -> dict:
    for name, ship_list in locations.items():
        for ship in ship_list:                       # ship is an (n, 2) array of coords
            if all(board[r, c] == 8 for r, c in ship):
                sunk[name] = True
    return sunk
                

def check_win(sunkp1: dict, sunkp2: dict, ship_num: int) -> list:
    if sum(sunkp1.values()) == ship_num:   # all of P1's ships sunk -> P2 wins
        return [True, 2]
    if sum(sunkp2.values()) == ship_num:
        return [True, 1]
    return [False, 0]



def visulize(board: np.ndarray) -> None:
    values = sorted(colors)
    cmap = ListedColormap([colors[v] for v in values])
    norm = BoundaryNorm([v - 0.5 for v in values] + [values[-1] + 0.5], cmap.N)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(board, cmap=cmap, norm=norm)

    n = board.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)

    # Battleship-style labels: columns A-J on top, rows 1-10 on the left
    ax.set_xticks(range(n), labels=list("ABCDEFGHIJ"))
    ax.set_yticks(range(n), labels=range(1, n + 1))
    ax.xaxis.tick_top()

    ax.legend(handles=[Patch(color=colors[v], label=labels[v]) for v in values],
          loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()


def create_game():
    player_1_board = create_board()
    player_2_board = create_board()

    add_ships(board=player_1_board, ship_num=ship_num)
    add_ships(board=player_2_board, ship_num=ship_num)

    p1_locations = search_for_ships(player_1_board)
    p2_locations = search_for_ships(player_2_board)

    for k in p1_locations.copy().keys():
        if len(p1_locations[k]) == 0:
            p1_locations.pop(k)
    for k in p2_locations.copy().keys():
        if len(p2_locations[k]) == 0:
            p2_locations.pop(k)

    return player_1_board, player_2_board, p1_locations, p2_locations

games = 8000
ship_num = 3

CHUNK_GAMES = 500

def main() -> None:
    player_1_board, player_2_board, p1_locations, p2_locations = create_game()
    chunk, chunk_idx = [], 0
    sunk_p1 = {}
    sunk_p2 = {}
    board_data = []
    player = 1
    games_played = 0
    turn_num = 0
    with tqdm(total=games, desc="Generating Games") as pbar:
        while games_played < games: 
            if player > 2:
                player = 1
            board_data = auto_play(player_1_board, player_2_board, loc1=p1_locations, loc2=p2_locations, turn_num=turn_num, board_data=board_data)
            sunk_p1 = check_sunk(player_1_board, p1_locations, sunk_p1)
            sunk_p2 = check_sunk(player_2_board, p2_locations, sunk_p2)
            player += 1
            turn_num += 1
            is_win = check_win(sunk_p1, sunk_p2, ship_num)
            winner = is_win[1] if is_win[0] else 0
            game_over = is_win[0]
            if game_over:
                chunk.append(build_game_frame(board_data, winner, games_played))
                games_played += 1
                player_1_board, player_2_board, p1_locations, p2_locations = create_game()
                sunk_p1, sunk_p2 = {}, {}
                player = 1
                board_data = []
                turn_num = 0
                pbar.update(1)

                if len(chunk) >= CHUNK_GAMES:
                    pl.concat(chunk).write_parquet(f"data/battleship_{chunk_idx:04d}.parquet")
                    chunk, chunk_idx = [], chunk_idx + 1
            else:
                continue
    if chunk:
        pl.concat(chunk).write_parquet(f"data/battleship_{chunk_idx:04d}.parquet")

if "__main__" == __name__:
    main()
