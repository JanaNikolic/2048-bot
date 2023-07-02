from game import Game2048
from board import Board

import random

NUM_TRIALS = 100

def random_run():
    game = Game2048()
    game_board = Board()
    game_board.update_grid_cells(game.matrix, 0, 0, 0)

    while not game.game_end:
        move = random.randint(0, 3)
        game.make_move(move)
        game_board.update_grid_cells(game.matrix, game.get_merge_score(), game.get_sum(), game.max_num())
        
    game_board.close_window()
    return game.max_num(), game.get_sum(), game.get_merge_score()

def main():
    max_val_results = [0] * NUM_TRIALS
    total_sum_results = [0] * NUM_TRIALS
    total_merge_score = [0] * NUM_TRIALS
    
    for i in range(NUM_TRIALS):
        max_val_results[i], total_sum_results[i], total_merge_score[i] = random_run()
        
    total_sum_avg = sum(total_sum_results) / NUM_TRIALS
    max_val_avg = sum(max_val_results) / NUM_TRIALS
    total_merge_avg = sum(total_merge_score) / NUM_TRIALS

    f = open("random.txt", "w")
    f.write("avg max val: " + str(max_val_avg) + "\n") 
    f.write("avg total sum: " + str(total_sum_avg) + "\n")
    f.write("avg merge score: " + str(total_merge_avg) + "\n")
    f.write("max vals: " + str(max_val_results) + "\n") 
    f.write("total sums: " + str(total_sum_results) + "\n")
    f.write("total merge score: " + str(total_merge_score) + "\n")
    f.close()

if __name__ == '__main__':
    main()