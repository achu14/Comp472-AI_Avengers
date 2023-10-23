How to run the program :
The program runs in a command line using Python. To run it, run the following command in a command line prompt that's open in the folder of the program: 
- `python ai_wargame.py`

Additionaly, you may add arguments to the program : 

- --max_depth <int>: To indicate the max depth of the minimax algorithm
- --max_time <float>: To indicate the maximum amount of seconds the minimax algorithm - can seach before returning a move
- --max_turns <float>: The maximum number of turn the game can last, after which the defender wins.
- --game_type <string>: The type of game.
- --disable_alpha_beta <bool>: To indicate whether to use Alpha Beta pruning during the minimax algorithm
- --heuristic <int>: Which heuristic, between 0, 1 and 2, to choose for as the evaluator function of a terminal node of the minimax algorith
- --broker <string>: To play via a game broker.

Here's an example of how to run the program with the above arguments : 
`python ai_wargame.py --max_depth 6 --max_time 6 --max_turn 50 --game_type computer --disable_alpha_beta False --heuristic 0`

When a game finishes, a log file is produced in the same folder as the program file.