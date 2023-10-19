from __future__ import annotations
import argparse
from ast import List
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar, Union
import random
import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3


##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None


##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True
    game_trace: List[str] = field(default_factory=list)  # New attribute to store game events

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

        # Record the initial game parameters and board configuration
        self.game_trace.append(
            f"timeout: {'currently unimplemented'}")
        self.game_trace.append(f"max number of turns: {self.options.max_turns}")
        self.game_trace.append(f"alpha-beta: {'off' if self.options.alpha_beta else 'off'}")
        self.game_trace.append(
            f"player 1: {'Human'} & player 2: {'Human'}")
        self.game_trace.append("Initial board configuration:")
        self.game_trace.append(self.to_string())

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        self.set(coord, None)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, mv: CoordPair) -> bool:
        dst = mv.dst
        src = mv.src
        # If coords are out of bound, move is not valid
        if not self.is_valid_coord(src) or not self.is_valid_coord(dst):
            return False

        unit = self.get(src)
        # If there is no friendly unit on the source square, move is not valid
        if unit is None or unit.player != self.next_player:
            return False

        isValid = (
                self.is_dst_valid_square(unit, src, dst) and  # Is the destination square a valid square to move to
                self.is_moving_unit_allowed_to_move(unit, src, dst) and # Is the unit allowed to move to the destination square
                self.can_dst_unit_be_targeted(unit, dst)  # Can the unit on the destination square be targeted
        )

        return isValid

    def is_dst_valid_square(self, unit: Unit, src: Coord, dst: Coord) -> bool:
        if self.is_unit_tech_or_virus(unit):  # Virus and tech can move in all directions
            return (
                    src.row == dst.row + 1 and src.col == dst.col or
                    dst.col == src.col + 1 or src.row == dst.row or
                    src.row + 1 == dst.row and src.col == dst.col or
                    dst.col == src.col + 1 and src.row == dst.row
            )

        player = unit.player
        if player == Player.Attacker:
            if src.row == dst.row and src.col == dst.col:  # If self destruct, valid move.
                return True
            # If source is an attacker, then it can only move up or left
            if (src.row == dst.row + 1 and src.col == dst.col) or \
                    (src.col == dst.col + 1 and src.row == dst.row):
                return True
            return False
        else:
            if src.row == dst.row and src.col == dst.col:  # If self destruct, valid move.
                return True
            # If source is a defender, then it can only move down or right
            if (src.row + 1 == dst.row and src.col == dst.col) or \
                    (dst.col == src.col + 1 and src.row == dst.row):
                return True
        return False

    def is_moving_unit_allowed_to_move(self, unit: Unit, src: Coord, dst: Coord) -> bool:
        if self.is_unit_tech_or_virus(unit):  # Tech or virus can't be engaged in combat
            return True

        adjacent_units: List[Union[Unit, None]] = self.get_adjacent_units(src, dst)

        for au in adjacent_units:
            if au is not None and au.player != unit.player:
                return False

        return True

    def can_dst_unit_be_targeted(self, unit: Unit, dst: Coord) -> bool:
        src_unit = self.get(dst)
        if src_unit and src_unit.player == unit.player and src_unit.type == unit.type:
            return True  # Allow self-destruct actions

        target_unit: Unit = self.board[dst.row][dst.col]
        if target_unit is None:
            return True  # No unit, no repair, move is valid

        if unit.player != target_unit.player:
            return True  # can always attack if target unit is opponent
        if target_unit.health == 9:  # max health doesn't allow repairing
            return False

        return unit.repair_amount(target_unit) > 0  # Move is valid if unit is allowed to repair target

    def is_unit_tech_or_virus(self, unit: Unit) -> bool:
        return unit.type == UnitType.Virus or unit.type == UnitType.Tech

    def get_adjacent_units(self, src: Coord, dst: Coord) -> List(Unit | None):
        topCoord = Coord(src.row - 1, src.col)
        bottomCoord = Coord(src.row + 1, src.col)
        rightCoord = Coord(src.row, src.col + 1)
        leftCoord = Coord(src.row, src.col - 1)

        top = self.board[src.row - 1][src.col] if self.is_valid_coord(topCoord) and topCoord != dst else None
        bottom = self.board[src.row + 1][src.col] if self.is_valid_coord(bottomCoord) and bottomCoord != dst else None
        right = self.board[src.row][src.col + 1] if self.is_valid_coord(rightCoord) and rightCoord != dst else None
        left = self.board[src.row][src.col - 1] if self.is_valid_coord(leftCoord) and leftCoord != dst else None

        return [left, right, bottom, top]

    def apply_self_destruct_damage(self, coords):
        # Get the self-destruct damage from the moving unit
        moving_unit = self.get(coords.src)
        self_destruct_damage = moving_unit.health

        # Get the coordinates for all the pieces surrounding the moving unit
        x, y = coords.src.row, coords.src.col
        adjacent_coords = [
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
            (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)
        ]

        # Reduce the health of surrounding units by 2
        for i, j in adjacent_coords:
            if self.is_valid_coord(Coord(i, j)):
                target_unit = self.board[i][j]
                if target_unit:
                    target_unit.mod_health(-2)
                    if not target_unit.is_alive():
                        self.set(Coord(i, j), None)

        # Remove the self-destructing unit from the board
        moving_unit.mod_health(-moving_unit.health)
        self.remove_dead(coords.src)

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str]:
        # Record the action at the start
        self.game_trace.append(f"turn #{self.turns_played + 1}")
        self.game_trace.append(f"{self.next_player.name}")
        self.game_trace.append(f"move from {coords.src.to_string()} to {coords.dst.to_string()}")

        if self.is_valid_move(coords):
            moving_unit = self.get(coords.src)
            target_unit = self.get(coords.dst)

            print(f"Moving unit: {moving_unit}")  # Debug print
            print(f"Target unit: {target_unit}")  # Debug print

            # If the source and destination are the same (self-destruct)
            if coords.src == coords.dst:
                self.apply_self_destruct_damage(coords)
            # If there's a unit at the destination
            elif target_unit:
                # If it's an opponent's unit, apply damage to both units
                if target_unit.player != moving_unit.player:
                    damage = moving_unit.damage_amount(target_unit)

                    print(f"Calculated damage: {damage}")  # Debug print

                    # Reduce health of the target unit
                    print(f"Target unit health before: {target_unit.health}")  # Debug print
                    self.mod_health(coords.dst, -damage)
                    print(f"Target unit health after: {target_unit.health}")  # Debug print

                    # Reduce health of the attacking unit
                    print(f"Moving unit health before: {moving_unit.health}")  # Debug print
                    self.mod_health(coords.src, -damage)
                    print(f"Moving unit health after: {moving_unit.health}")  # Debug print

                # If it's a friendly unit, repair if move is valid
                else:
                    repair = moving_unit.repair_amount(target_unit)
                    print(f"Repair amount: {repair}")  # Debug print
                    print(f"Target unit health before: {target_unit.health}")  # Debug print
                    self.mod_health(coords.dst, +repair)
                    print(f"Target unit health after: {target_unit.health}")  # Debug print

            else:
                # Move the unit to the destination if the target unit is
                # not alive or if there's no unit at the destination
                self.set(coords.dst, moving_unit)
                self.set(coords.src, None)
            if self.is_finished():
                self.game_trace.append("New board configuration:")
                self.game_trace.append(self.to_string())
                return True, ""
        return False, "invalid move"

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ", end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ", end='')
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def write_game_trace_to_file(self, filename: str):
        with open(filename, 'w') as file:
            for line in self.game_trace:
                file.write(line + "\n")

    def is_finished(self) -> bool:
        # Game ends if 100 moves have been played or if any AI is destroyed
        if self.turns_played >= 100:
            print("Max number of turns (100) has passed")
        return self.turns_played >= 2 or not self._attacker_has_ai or not self._defender_has_ai

    def has_winner(self) -> Player | None:
        """Determines if there's a winner and returns the winner."""
        # If the game hasn't reached its end conditions yet, return None
        if not self.is_finished():
            return None
        # Check if the attacker's AI is destroyed
        if not self._attacker_has_ai:
            self.game_trace.append(f"{Player.Defender.name} wins in {self.turns_played} turns")
            return Player.Defender
        # Check if the defender's AI is destroyed
        elif not self._defender_has_ai:
            self.game_trace.append(f"{Player.Attacker.name} wins in {self.turns_played} turns")
            return Player.Attacker
        # If neither AI is destroyed and 10 turns have been played, the defender wins
        else:
            self.game_trace.append(f"{Player.Defender.name} wins because max turns (100) have passed")
            return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src, _) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()
        (score, move, avg_depth) = self.random_move()
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        print(f"Average recursive depth: {avg_depth:0.1f}")
        print(f"Evals per depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None


##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--max_turns', type=float, help='maximum number of turn for a game')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--alpha_beta', type=bool, default=False, help='use alpha-beta pruning')
    args = parser.parse_args()

    # Construct the file name
    b_value = str(args.alpha_beta).lower()
    t_value = str(args.max_time) if args.max_time else 'off'
    m_value = str(args.max_turns) if args.max_turns else '100'

    file_name = f"gameTrace-{b_value}-{t_value}-{m_value}.txt"

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    if args.max_turns is not None:
        options.max_turns = args.max_turns

    # create a new game
    game = Game(options=options)

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!\nGame Over!!")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

    game.write_game_trace_to_file(file_name)



##############################################################################################################

if __name__ == '__main__':
    main()
