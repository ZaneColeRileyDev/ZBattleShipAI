import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

import pygame

from PIL import Image
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



def take_shot(target, display_board, tile):
    flat = target.flatten("C")
    disp = display_board.flatten("C")
    if not (0 <= tile < 100) or flat[tile] in (7, 8):
        return target, display_board            # invalid shot, nothing changes

    hit = flat[tile] in (1, 2, 3, 4, 5)
    print(f"{'Hit Ship' if hit else 'Miss'}: Tile {tile}")
    flat[tile] = disp[tile] = 8 if hit else 7
    return flat.reshape((10, 10)), disp.reshape((10, 10))

def do_move(op: np.ndarray, display_board: np.ndarray, tile: int) -> list[np.ndarray, np.ndarray]:
    board, display_board = take_shot(op, display_board, tile)
    
    return board, display_board
                  

def check_sunk(board: np.ndarray, locations: dict, sunk: dict) -> dict:
    for name, ship_list in locations.items():
        for ship in ship_list:                       # ship is an (n, 2) array of coords
            if all(board[r, c] == 8 for r, c in ship):
                sunk[name] = True
                # print(f"Sunk Ship: {name}")
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


def visulize(board: np.ndarray, player: int) -> None:
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
    plt.savefig(f"Board_{player}")
    img = Image.open(f"Board_{player}.png")
    img.show()


def is_board_same(b1, b2) -> bool:
    if np.array_equal(b1, b2):
        return True
    else:
        return False

def new_game():
    display_board = create_board()
    p1_board, p2_board, p1_loc, p2_loc = create_game()
    return display_board, p1_board, p2_board, p1_loc, p2_loc, {}, {}, 0

def game():
    GRID_SIZE = 10
    CELL_SIZE = 50  # Size of each square in pixels
    GRID_WIDTH = GRID_SIZE * CELL_SIZE
    PADDING = 40    # Gap between the two boards

    # Left board starts at X=0. Right board starts after left grid + padding.
    RIGHT_GRID_OFFSET = GRID_WIDTH + PADDING

    WINDOW_WIDTH = (GRID_WIDTH * 2) + PADDING
    WINDOW_HEIGHT = GRID_WIDTH + 60  # Extra bottom spacing for text labels
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("ZBattle Ship, by ZGamingStudio")
    display_board = create_board()
    (display_board, player_1_board, player_2_board,
    p1_locations, p2_locations, sunk_p1, sunk_p2, turn_num) = new_game()

    game_over = False
    winner = 0
    font = pygame.font.SysFont(None, 48)
    button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 90, WINDOW_HEIGHT // 2 - 10, 180, 55)
    player = 1
    turn_num = 0
    ship_num = 3
    font_small = pygame.font.SysFont(None, 32)
    MSG_MS = 2500                      # how long a message stays up
    left_msg, left_until = "", 0       # player 2's ships (left board)
    right_msg, right_until = "", 0     # player 1's ships (right board)
    model_path = resource_path(os.path.join("models", "BattleShipAgentV2.pth"))
    model = load_model(model_path)
    turn_over = False
    running = True
    while running:
        if player > 2:
            player = 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.K_RETURN and game_over:
                game()


            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game_over:
                    if button_rect.collidepoint(event.pos):
                        (display_board, player_1_board, player_2_board,
                        p1_locations, p2_locations, sunk_p1, sunk_p2, turn_num) = new_game()
                        game_over = False
                        winner = 0
                else:
                    mouse_x, mouse_y = event.pos
                    if 0 <= mouse_x < GRID_WIDTH and 0 <= mouse_y < GRID_WIDTH:
                        tile = (mouse_y // CELL_SIZE) * GRID_SIZE + (mouse_x // CELL_SIZE)

                        past_p2_board = player_2_board.copy()
                        player_2_board, display_board = do_move(player_2_board, display_board, tile)

                        if not is_board_same(player_2_board, past_p2_board):
                            before = set(sunk_p2)
                            sunk_p2 = check_sunk(player_2_board, p2_locations, sunk_p2)
                            newly = [n for n in sunk_p2 if n not in before]
                            if newly:
                                left_msg = f"{', '.join(newly).title()} sunk!"
                                left_until = pygame.time.get_ticks() + MSG_MS

                            if sum(sunk_p2.values()) == ship_num:
                                winner, game_over = 1, True
                            else:
                                player_1_board = ai_move(player_1_board, model)
                                before = set(sunk_p1)
                                sunk_p1 = check_sunk(player_1_board, p1_locations, sunk_p1)
                                newly = [n for n in sunk_p1 if n not in before]
                                if newly:
                                    right_msg = f"{', '.join(newly).title()} sunk!"
                                    right_until = pygame.time.get_ticks() + MSG_MS

                                if sum(sunk_p1.values()) == ship_num:
                                    winner, game_over = 2, True
                            turn_num += 1

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                y = row * CELL_SIZE
                x_left = col * CELL_SIZE
                x_right = RIGHT_GRID_OFFSET + (col * CELL_SIZE)

                pygame.draw.rect(screen, colors[0], (x_left, y, CELL_SIZE, CELL_SIZE))


                cell_state_l = display_board[row, col]

                for _ in colors.keys():
                    if colors[cell_state_l]:
                        pygame.draw.rect(screen, colors[cell_state_l], (x_left, y, CELL_SIZE, CELL_SIZE))
                        break
                pygame.draw.rect(screen, (0, 0, 0), (x_left, y, CELL_SIZE, CELL_SIZE), 1)    


                state_right = player_1_board[row, col]
                for _ in colors.keys():
                    if colors[state_right]:
                        pygame.draw.rect(screen, colors[state_right], (x_right, y, CELL_SIZE, CELL_SIZE))
                        break
                pygame.draw.rect(screen, (0, 0, 0), (x_right, y, CELL_SIZE, CELL_SIZE), 1)  

        if turn_over:
            turn_num += 1
            is_win = check_win(sunk_p1, sunk_p2, ship_num)
            winner = is_win[1] if is_win[0] else 0
            game_over = is_win[0]
            if game_over:
                print(f"Player {winner} Wins!!!")
                player = 1
            turn_over = False
        strip_y = GRID_WIDTH
        pygame.draw.rect(screen, (0, 0, 0), (0, strip_y, WINDOW_WIDTH, WINDOW_HEIGHT - strip_y))

        now = pygame.time.get_ticks()
        text_y = strip_y + (WINDOW_HEIGHT - strip_y) // 2

        if left_msg and now < left_until:
            surf = font_small.render(left_msg, True, (255, 255, 255))
            screen.blit(surf, surf.get_rect(center=(GRID_WIDTH // 2, text_y)))

        if right_msg and now < right_until:
            surf = font_small.render(right_msg, True, (255, 255, 255))
            screen.blit(surf, surf.get_rect(center=(RIGHT_GRID_OFFSET + GRID_WIDTH // 2, text_y)))
        if game_over:
            text = font.render(f"Player {winner} Wins!", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50)))

            hovered = button_rect.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(screen, (70, 160, 70) if hovered else (50, 120, 50), button_rect, border_radius=8)
            label = font.render("Play Again", True, (255, 255, 255))
            screen.blit(label, label.get_rect(center=button_rect.center))
            left_msg = right_msg = ""
        pygame.display.flip()
    pygame.quit()

if "__main__" == __name__:
    game()
