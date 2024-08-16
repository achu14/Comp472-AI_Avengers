# Comp472-Project
AI Wargame Project Description

a 2-player game played by an attacker (a) and a defender (d) on a 5 × 5 board.

Each player has 6 units on the board. These units can be of different types:
AI (A) each player has only 1 AI unit. The goal of the game is to destroy the opponent’s AI.
Viruses (V) are very offensive units; they can destroy the AI in 1 attack.
Techs (T) are very defensive units, Tech and Virus are equal in combat against each other.
Programs (P) are generic soldiers.
Firewalls (F) are strong at absorbing attacks, but weak at damaging other units.
Each unit has an associated health level, represented by an integer between [0. . . 9]. Initially, all units have
full health (9). When the health level of a Virus, a Tech, a Program or a Firewall reaches 0 or below 0, the unit
is destroyed and removed from the board. If the health level of an AI reaches 0, then the player loses the game.
1.1 Initial Configuration
At the beginning of a game, the attacker has 1×AI, 2×Viruses, 2× Programs and 1×Firewall; while the defender
has 1×AI, 2×Techs, 2×Firewalls and 1×Program. The game starts with the following initial configuration:

0 1 2 3 4
A: dA9 dT9 dF9 . .
B: dT9 dP9 . . .
C: dF9 . . . aP9
D: . . . aF9 aV9
E: . . aP9 aV9 aA9

dA9, at A0, represents the defender (d)’s AI (A) and has a health of 9.
aP9, at C4, represents the attacker (a)’s Program (P) and has a health of 9.
. . .

The attacker starts the game.

1.2 Actions
Each player take turn to play any of the following actions.
Movement A single unit can move to an adjacent position on the board. Rules to move a unit are the following:
1. The destination must be free (no other unit is on it).
2. Units are said to be engaged in combat if an adversarial unit is adjacent (in any of the 4 directions).
If an AI, a Firewall or a Program is engaged in combat, they cannot move.
The Virus and the Tech can move even if engaged in combat.
3. The attacker’s AI, Firewall and Program can only move up or left.
The Tech and Virus can move left, top, right, bottom.
4. The defender’s AI, Firewall and Program can only move down or right.
The Tech and Virus can move left, top, right, bottom.

Attack A unit S can attack another unit T. Rules to attack a unit are the following:
1. T must be adjacent to S in any of the 4 directions (up, down, left or right).
2. T and S must be adversarial units (i.e. belong to different players).
A combat is bi-directional. This means that if S attacks T, S damages T but T also damages S.
Table 1 shows the damages inflicted to an adversary’s health during a combat.
Health values cannot go above 9 or below 0. If a unit’s health reaches 0, it is destroyed and eliminated
from the board.
Repair A unit S can repair another unit T. Rules to repair a unit are the following:
1. T must be adjacent to S in any of the 4 directions (up, down, left or right).
2. T and S must be friendly units (i.e. belong to the same player).
3. The repair must lead to a change of health on T. Table 2 shows the change in health resulting from
repairs. So for example,
• a Tech cannot repair a Virus (see value 0 in Table 2). This is would be an invalid action.
• S cannot repair T if T’s health is already at 9. This is would be an invalid action.
Self-Destruct Any unit can kill itself (and be removed from the board) and damage surrounding units.
Self-destruction removes the unit and inflicts 2 points of damage to all 8 surrounding units (if present).
This includes diagonals and friendly units. Remember that health values cannot go above 9 or below 0. If
the health of a unit reaches 0 or below 0, it is removed from the board.
1.3 End of the Game
The game ends when a player loses their AI, or a pre-determined number of moves has been reached (e.g. 100). A
player wins if their AI is alive while the other AI is destroyed; otherwise the defender wins (because the attacker
started playing first).
Given that any unit can self-destruct, no player should be left in a position where no action is available to
them.
