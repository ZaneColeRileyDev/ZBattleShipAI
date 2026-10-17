import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

from model import BattleShipModel
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


def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def load_model(path: str) -> BattleShipModel:
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model = BattleShipModel()
    model.load_state_dict(weights)
    model.eval()
    return model

def ai_move(board: np.ndarray, model: BattleShipModel) -> np.ndarray:

    visible = torch.tensor(np.where(board >= 7, board, 0), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(visible).flatten()
        flat_board = board.flatten("C").copy()
        mask = torch.tensor(np.isin(flat_board, [7, 8]), dtype=torch.bool)   # True = already shot
        logits = logits.masked_fill(mask, float("-inf"))
        idx = torch.argmax(logits).item()
    
    hit = flat_board[idx] in (1, 2, 3, 4, 5)
    if hit:
        print(f"Hit Ship: Tile {idx}")
    else:
        print(f"Miss: Tile {idx}")
    flat_board[idx] = 8 if hit else 7
    return flat_board.reshape((10, 10))



def take_shot(target: np.ndarray, tile: int):
    # Cells that are still fair game
    open_cells = np.flatnonzero((target.ravel() != 7) & (target.ravel() != 8))
    if open_cells.size == 0:
        return target

    # Random shot at an untouched cell
    try:
        shot_posible = True if open_cells[tile] else False
    except IndexError:
        return target
    target = target.flatten("C")
    if shot_posible:
        hit = target[tile] in (1, 2, 3, 4, 5)   # still holds a ship value
        if hit:
            print(f"Hit Ship: Tile {tile}")
        else:
            print(f"Miss: Tile {tile}")
        target[tile] = 8 if hit else 7
        return target
    return target.reshape((10, 10))

def do_move(op: np.ndarray, tile: int) -> np.ndarray:
    board = take_shot(op, tile)
    
    return board
                  

def check_sunk(board: np.ndarray, locations: dict, sunk: dict) -> dict:
    for name, ship_list in locations.items():
        for ship in ship_list:                       # ship is an (n, 2) array of coords
            if all(board[r, c] == 8 for r, c in ship):
                sunk[name] = True
                print(f"Sunk Ship: {name}")
    return sunk
                

def check_win(sunkp1: dict, sunkp2: dict, ship_num: int) -> list:
    if sum(sunkp1.values()) == ship_num:   # all of P1's ships sunk -> P2 wins
        return [True, 2]
    if sum(sunkp2.values()) == ship_num:
        return [True, 1]
    return [False, 0]

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

def create_game(ship_num: int = 3):
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

def is_board_same(b1, b2) -> bool:
    if np.array_equal(b1, b2):
        return True
    else:
        return False

def main():
    player_1_board, player_2_board, p1_locations, p2_locations = create_game()
    sunk_p1 = {}
    sunk_p2 = {}
    player = 1
    turn_num = 0
    ship_num = 3
    model_path = resource_path(os.path.join("models", "BattleShipAgentV2.pth"))
    model = load_model(model_path)
    turn_over = False
    while True:
        vaild_tiles = np.flatnonzero((player_2_board.ravel() != 7) & (player_2_board.ravel() != 8))
        if player > 2:
            player = 1

        if player == 1:
            print(f"Turn Num: {turn_num}")
            print(f"Vaild Tile Numbers: {vaild_tiles}")
            tile = int(input("Enter a Tile Number: "))
            past_p2_board = player_2_board.copy()
            player_2_board = do_move(player_2_board, tile)
            same_board_p2 = is_board_same(b1=player_2_board, b2=past_p2_board)
            if not same_board_p2:
                player += 1 
                sunk_p2 = check_sunk(player_2_board.reshape((10, 10)), p2_locations, sunk_p2)
                
            else:
                continue


        if player == 2:
            past_p1_board = player_1_board.copy()
            player_1_board = ai_move(player_1_board, model)
            

            same_board_p1 = is_board_same(b1=player_1_board, b2=past_p1_board)

            if not same_board_p1:
                player += 1
                turn_num += 1
                sunk_p1 = check_sunk(player_1_board.reshape((10, 10)), p1_locations, sunk_p1)
                turn_over = True
            else:
                continue
        if turn_over:
            turn_num += 1
            is_win = check_win(sunk_p1, sunk_p2, ship_num)
            winner = is_win[1] if is_win[0] else 0
            game_over = is_win[0]
            if game_over:
                print(f"Player {winner} Wins!!!")
                if is_win[1] == 1:
                    visulize(player_2_board)
                if is_win[1] == 2:
                    visulize(player_1_board)
                player = 1
            turn_over = False
            visulize(player_1_board)


if "__main__" == __name__:
    main()
