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
from collections import deque


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
    debug: bool = False
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True

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

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.deepcopy(self)
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

        if self.is_self_destruct_move(src, dst):
            return True

        isValid = (
                self.is_dst_valid_square(unit, src, dst) and  # Is the destination square a valid square to move to
                self.is_moving_unit_allowed_to_move(unit, src, dst) and # Is the unit allowed to move to the destination square
                self.can_dst_unit_be_targeted(unit, dst)  # Can the unit on the destination square be targeted
        )

        if not isValid:
            self.log(f"Move is not valid: {mv.to_string()}")

        return isValid
    
    def is_self_destruct_move(self, src: Coord, dst: Coord) -> bool:
        return src.row == dst.row and src.col == dst.col

    def is_dst_valid_square(self, unit: Unit, src: Coord, dst: Coord) -> bool:
        if self.is_unit_tech_or_virus(unit):  # Virus and tech can move in all directions
            sameCol = (src.col == dst.col);
            sameRow = (src.row == dst.row);
           
            oneSquareUp = src.row == dst.row + 1 and sameCol;
            oneSquareDown = (src.row + 1 == dst.row) and sameCol;
            oneSquareLeft = (dst.col == src.col + 1) and sameRow;
            oneSquareRight = (src.col == dst.col + 1) and sameRow;

            up = oneSquareUp and sameCol
            down = oneSquareDown and sameCol
            left = oneSquareLeft and sameRow
            right = oneSquareRight and sameRow

            return up or down or left or right 

        player = unit.player
        if player == Player.Attacker:
            
            # If source is an attacker, then it can only move up or left
            if (src.row == dst.row + 1 and src.col == dst.col) or \
                    (src.col == dst.col + 1 and src.row == dst.row):
                return True
            return False
        else:
            # If source is a defender, then it can only move down or right
            if (src.row + 1 == dst.row and src.col == dst.col) or \
                    (dst.col == src.col + 1 and src.row == dst.row):
                return True
        return False

    def is_moving_unit_allowed_to_move(self, unit: Unit, src: Coord, dst: Coord) -> bool:
        if self.is_unit_tech_or_virus(unit):  # Tech or virus can't be engaged in combat
            return True

        dst_unit = self.get(dst)
        if dst_unit is not None:
            return True  # If there's a unit on the destination square, can target
        adjacent_units: List[Union[Unit, None]] = self.get_adjacent_units(src, dst)

        for au in adjacent_units:
            if au is not None and au.player != unit.player:
                return False

        return True

    def can_dst_unit_be_targeted(self, unit: Unit, dst: Coord) -> bool:

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
        ...
        if self.is_valid_move(coords):
            moving_unit = self.get(coords.src)
            target_unit = self.get(coords.dst)

            self.log(f"Moving unit: {moving_unit}")  # Debug print
            self.log(f"Target unit: {target_unit}")  # Debug print

            # If the source and destination are the same (self-destruct)
            if coords.src == coords.dst:
                self.apply_self_destruct_damage(coords)
            # If there's a unit at the destination
            elif target_unit:
                # If it's an opponent's unit, apply damage to both units
                if target_unit.player != moving_unit.player:
                    damage = moving_unit.damage_amount(target_unit)

                    self.log(f"Calculated damage: {damage}")  # Debug print

                    # Reduce health of the target unit
                    self.log(f"Target unit health before: {target_unit.health}")  # Debug print
                    self.mod_health(coords.dst, -damage)
                    self.log(f"Target unit health after: {target_unit.health}")  # Debug print

                    # Reduce health of the attacking unit
                    self.log(f"Moving unit health before: {moving_unit.health}")  # Debug print
                    self.mod_health(coords.src, -damage)
                    self.log(f"Moving unit health after: {moving_unit.health}")  # Debug print

                # If it's a friendly unit, repair if move is valid
                else:
                    repair = moving_unit.repair_amount(target_unit)
                    self.log(f"Repair amount: " + str(repair))  # Debug print
                    self.log(f"Target unit health before: {target_unit.health}")  # Debug print
                    self.mod_health(coords.dst, +repair)
                    self.log(f"Target unit health after: {target_unit.health}")  # Debug print

            else:
                # Move the unit to the destination if the target unit is
                # not alive or if there's no unit at the destination
                self.set(coords.dst, moving_unit)
                self.set(coords.src, None)
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
            if s == "m":
                return self.suggest_move()
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                self.log('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            self.log("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    self.log(f"Broker {self.next_player.name}: ")
                    self.log(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv)
                if success:
                    self.log(f"Player {self.next_player.name}: ")
                    self.log(result)
                    self.next_turn()
                    break
                else:
                    self.log("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                self.log(f"Computer {self.next_player.name}: ")
                self.log(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        # Game ends if 100 moves have been played or if any AI is destroyed
        if self.turns_played >= 100:
            self.log("Max number of turns (100) has passed")
        return self.turns_played >= 100 or not self._attacker_has_ai or not self._defender_has_ai

    def has_winner(self) -> Player | None:
        """Determines if there's a winner and returns the winner."""
        # If the game hasn't reached its end conditions yet, return None
        if not self.is_finished():
            return None

        # Check if the attacker's AI is destroyed
        if not self._attacker_has_ai:
            return Player.Defender
        # Check if the defender's AI is destroyed
        elif not self._defender_has_ai:
            return Player.Attacker
        # If neither AI is destroyed and 10 turns have been played, the defender wins
        else:
            return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()

        player_units = self.player_units(self.next_player);
        for (src, _) in player_units:
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self, no_self_destruct: bool = False) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        if no_self_destruct:
            move_candidates = [move for move in move_candidates if move.src != move.dst]
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta."""
        start_time = datetime.now()
        move = self.get_best_move()
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        # print(f"Heuristic score: {score}")
        # print(f"Average recursive depth: {avg_depth:0.1f}")
        # print(f"Evals per depth: ", end='')
        # for k in sorted(self.stats.evaluations_per_depth.keys()):
        #     print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end='')
        # print()
        # total_evals = sum(self.stats.evaluations_per_depth.values())
        # if self.stats.total_seconds > 0:
        #     print(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
        # print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move
    
    def get_best_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha-beta pruning."""
        start_time = datetime.now()
        isMaximizingPlayer = self.next_player == Player.Attacker
        isMinimizingPlayer = not isMaximizingPlayer
        best_move = None
        worst_move = None
        best_score = MIN_HEURISTIC_SCORE if isMaximizingPlayer else MAX_HEURISTIC_SCORE
        worst_score = MAX_HEURISTIC_SCORE if isMaximizingPlayer else MIN_HEURISTIC_SCORE
        max_depth = 3  # Adjust this depth based on your game's complexity
       
        for move in self.move_candidates():  # Implement this function to get available moves
            # Call the minimax function for each possible move
            new_state = self.clone()
            new_state.perform_move(move)
            score = self.minimax(new_state, max_depth, isMinimizingPlayer, MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE)
            if isMaximizingPlayer and score > best_score:
                best_score = score
                best_move = move
            elif not isMaximizingPlayer and score < best_score:
                best_score = score
                best_move = move

            if isMaximizingPlayer and score < worst_score:
                worst_score = score
                worst_move = move
            elif not isMaximizingPlayer and score > worst_score:
                worst_score = score
                worst_move = move

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        # Print any relevant statistics or debugging information
        print(f"Best move: {best_move}")
        print(f"Best score: {best_score}")

        print(f"Worst move: {worst_move}")
        print(f"Worst score: {worst_score}")
        return best_move
    

    def minimax(self, state : Game, depth : int, maximizing_player : bool, alpha : int, beta : int):
        if depth == 0 or state.is_finished():
            return state.heuristic_e0()
        
        playerTurn = Player.Attacker if maximizing_player else Player.Defender

        if maximizing_player:
            best_value = MIN_HEURISTIC_SCORE
            for move in state.move_candidates():
                new_state = state.clone()
                new_state.next_player = playerTurn
                new_state.perform_move(move)
                value = self.minimax(new_state, depth - 1, False, alpha, beta)
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return best_value
        else:
            best_value = MAX_HEURISTIC_SCORE
            for move in state.move_candidates():
                new_state = state.clone()
                new_state.next_player = playerTurn
                new_state.perform_move(move)
                value = self.minimax(new_state, depth - 1, True, alpha, beta)
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return best_value
        
    def heuristic_e0(self):
        """ Calculates the heuristic value for e0. """
        # e0 = (3VP1 + 3TP1 + 3FP1 + 3PP1 + 9999AIP1) − (3VP2 + 3TP2 + 3FP2 + 3PP2 + 9999AIP2)

        # Weights for each unit type
        unit_weights = {
            "Program": 3,
            "Firewall": 3,
            "Tech": 3,
            "Virus": 3,
            "AI": 9999
        }

        # Initialize player scores
        p1_score = 0
        p2_score = 0

        # Extract units from the board and put them in units list
        units = [cell for row in self.board for cell in row if cell is not None]

        # Calculate scores based on unit types and weights
        for unit in units:
            weight = unit_weights[unit.type.name]
            if unit.player == Player.Attacker:  # Attacker represents player 1 (P1)
                p1_score += weight
            else:  # Defender represents player 2 (P2)
                p2_score += weight

        # The heuristic value for e0
        e0 = p1_score - p2_score

        return e0
    
    def heuristic_e1(self):
        """ Calculates the heuristic value for e1. """
        # weights for our heuristic components
        w1, w2, w3, w4, w5 = 1000, 3, -5, 20, 99999999999999

        # Unit-specific weights for board control factor
        unit_weights = {
            "Program": 1,
            "Firewall": 1,
            "Tech": 3,
            "Virus": 3,
            "AI": 100
        }

        player = self.next_player

        # Extract units and their positions from the board
        units_with_positions = [(cell, (x, y)) for x, row in enumerate(self.board) for y, cell in enumerate(row) if
                                cell is not None]

        # Get the health and position of our AI and the opponent's AI
        our_ai = next((unit for unit in units_with_positions if unit[0].type == "AI" and unit[0].player == player),
                      None)
        opponent_ai = next((unit for unit in units_with_positions if unit[0].type == "AI" and unit[0].player != player),
                           None)

        our_ai_health = our_ai[0].health if our_ai else 0
        opponent_ai_health = opponent_ai[0].health if opponent_ai else 0
        our_ai_position = our_ai[1] if our_ai else None
        opponent_ai_position = opponent_ai[1] if opponent_ai else None

        # [w1] AI Health Factor
        # Shows the difference in health between current player's AI and the opponent's AI.
        h_ai = our_ai_health - opponent_ai_health

        # [w2] Board Control Factor with adjusted unit weights
        # Evaluates the overall strength and presence of current player's units versus the opponent's on the board.
        our_units_value = sum(
            unit_weights[unit.type.name] * unit.health for unit, _ in units_with_positions if unit.player == player)
        opponent_units_value = sum(
            unit_weights[unit.type.name] * unit.health for unit, _ in units_with_positions if unit.player != player)
        h_control = our_units_value - opponent_units_value

        # [w3] Virus(es) near AI Factor
        # Penalizes based on the number of enemy Virus units adjacent to our AI, emphasizing immediate threats.
        our_ai_positions = [pos for unit, pos in units_with_positions if unit.type == "AI" and unit.player == player]
        our_ai_position = our_ai_positions[0] if our_ai_positions else None
        enemy_ai_position = [pos for unit, pos in units_with_positions if unit.type == "AI" and unit.player != player]
        enemy_ai_position = enemy_ai_position[0] if enemy_ai_position else None

        # Check if two board positions are adjacent to each other (horizontally or vertically).
        def is_adjacent(pos1, pos2):
            return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1

        # Counts the number of enemy Virus units adjacent to current player's AI, emphasizing immediate threats.
        bad_virus_near_ai = sum(1 for unit, pos in units_with_positions if
                                unit.player != player and unit.type == "Virus" and our_ai_position and is_adjacent(
                                    our_ai_position, pos))

        # Counts the number of friendly Virus units adjacent to opponent's AI, emphasizing immediate adjacency.
        good_virus_near_ai = -sum(1 for unit, pos in units_with_positions if
                                  unit.player == player and unit.type == "Virus" and enemy_ai_position and is_adjacent(
                                      our_ai_position, pos))

        # If Player is attacker the effect is positive, if Defender it is negative
        if player == Player.Attacker:
            virus_near_ai = good_virus_near_ai
        else:
            virus_near_ai = bad_virus_near_ai

        # [w4] Tech Support Factor
        # Reward positions where a friendly Tech is near the current player's AI if its health is low
        # Additional bonus if the Tech is near when the threat of a Virus attack is high
        our_ai_low_health = our_ai_health <= 7
        tech_supporting_ai = any(
            unit.type == "Tech" and unit.player == player and is_adjacent(our_ai_position, pos) for unit, pos in
            units_with_positions)

        # Compute the tech support heuristic component using w4
        h_tech_support = w4 if our_ai_low_health and tech_supporting_ai else 0

        # Adjust the tech support heuristic component [potential threat factor] if h_threat is high
        if virus_near_ai >= 1:
            h_tech_support += w4

        # [w5] Victory Factor
        # Gives near infinity bonus if current player's AI is alive and opponent's AI is defeated
        our_ai_alive = any(unit.type == "AI" and unit.player == player for unit, _ in units_with_positions)
        opponent_ai_defeated = not any(unit.type == "AI" and unit.player != player for unit, _ in units_with_positions)
        h_victory = w5 if our_ai_alive and opponent_ai_defeated else 0

        # The heuristic value for e1
        e1 = w1 * h_ai + w2 * h_control + w3 * virus_near_ai + h_tech_support + h_victory

        return e1 if player == Player.Attacker else -e1
        
    def heuristic_e2(self) -> int:
        attacker_ai = None
        defender_ai = None

        # Find the AI units for both players
        for coord, unit in self.player_units(Player.Attacker):
            if unit.type == UnitType.AI:
                attacker_ai = unit

        for coord, unit in self.player_units(Player.Defender):
            if unit.type == UnitType.AI:
                defender_ai = unit

        if attacker_ai is None or defender_ai is None:
            # The AI units are not on the board; the game is over
            return MAX_HEURISTIC_SCORE if attacker_ai else -MAX_HEURISTIC_SCORE
        
        weights = {
            "ai_healths_score": 5,
            "moves_for_virus_to_reach_ai": 5,
            "total_units_health": 1,
            "end_of_game_score": 1
        }

        moves_for_virus_to_reach_ai_score = self.moves_for_virus_to_reach_ai()
        ai_healths_score = self.calculate_ai_healths_score(attacker_ai, defender_ai)
        total_units_health_score = self.calculate_total_units_health_score(attacker_ai, defender_ai)
        end_of_game_score = self.calculate_end_of_game_score()

        # Combine scores with weights
        total_score = (
            weights["total_units_health"] * total_units_health_score + 
            weights["moves_for_virus_to_reach_ai"] * moves_for_virus_to_reach_ai_score + 
            weights["ai_healths_score"] * ai_healths_score + 
            weights["end_of_game_score"] * end_of_game_score
        )


        return total_score
    
    def calculate_end_of_game_score(self) -> int:
        # Calculate the heuristic score based on the end of the game
        winner = self.has_winner()

        if winner == Player.Attacker:
            return MAX_HEURISTIC_SCORE
        elif winner == Player.Defender:
            return -MAX_HEURISTIC_SCORE
        else:
            return 0
    
    def calculate_total_units_health_score(self, attacker_ai: Unit, defender_ai: Unit) -> int:
        # Calculate the heuristic score based on the total health of units
        attacker_health = 0
        defender_health = 0


        for coord, unit in self.player_units(Player.Attacker):
            attacker_health += unit.health

        for coord, unit in self.player_units(Player.Defender):
            defender_health += unit.health

        score = attacker_health - defender_health

        return score
    
    def calculate_ai_healths_score(self, attacker_ai: Unit, defender_ai: Unit) -> int:
        # Calculate the heuristic score based on the health of AI units
        attacker_ai_health = attacker_ai.health
        defender_ai_health = defender_ai.health

        # You can adjust these weights based on the importance of AI health
        # in your game's strategy
        attacker_weight = 1
        defender_weight = 1

        score = attacker_ai_health - defender_ai_health

        return score
    
    def moves_for_virus_to_reach_ai(self) -> int:
        # Find the positions of the Virus and the defender's AI
        virus_position = None
        defender_ai_position = None

        for coord, unit in self.player_units(Player.Defender):
            if unit.type == UnitType.Virus:
                virus_position = coord
            elif unit.type == UnitType.AI:
                defender_ai_position = coord

        if virus_position is None:
            return -5

        # Calculate the distance between the Virus and the AI using your calculate_distance function
        distance = self.shortest_distance(virus_position, defender_ai_position)

        if distance == -1:
            return -5
        
        # Calculate the score based on the number of moves required
        score = 5 - distance

        return score
    
    def shortest_distance(self, start, end):
        # Define the possible movements: up, down, left, right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        rows, cols = len(self.board), len(self.board[0])

        # Create a visited set to keep track of visited cells
        visited = set()

        # Create a queue for BFS
        queue = deque([(start, 0)])

        while queue:
            (x, y), distance = queue.popleft()

            if (x, y) == end:
                return distance

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy

                # Check if the new position is valid and not visited
                if 0 <= new_x < rows and 0 <= new_y < cols and grid[new_x][new_y] == 0 and (new_x, new_y) not in visited:
                    queue.append(((new_x, new_y), distance + 1))
                    visited.add((new_x, new_y))

        # If the end point is not reachable, return a large distance (or -1 to indicate unreachable)
        return -1
    

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
    
    def log(self, *args):
        """Log a message to the console."""
        if self.debug:
            print(*args)



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
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.AttackerVsDefender

    # set up game options
    options = Options(game_type=GameType.CompVsComp)

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


##############################################################################################################

if __name__ == '__main__':
    main()
