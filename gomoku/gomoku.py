###############################################################################
#
# gomoku reinforced deep learning agents
#
# ouh - 2020-05-18
#
###############################################################################

import numpy as np
import time
from gomoku import gglobals as gg
from gomoku import player

class Gomoku:
    """
    Game class
    """
    def __init__(self, p1, p2,
                 board_rows=gg.BOARD_ROWS,
                 board_cols=gg.BOARD_COLS,
                 win_num=gg.WIN_NUM):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.win_num = win_num
        self.board = None
        self.available_positions = None

        self.p1 = p1
        self.p2 = p2
        self.curr_player = None

        self.p1_symbol = gg.P1_SYMBOL
        self.p2_symbol = gg.P2_SYMBOL

        self.p1.game_init(self.board_rows, self.board_cols,
                          self.win_num, self.p1_symbol)
        self.p2.game_init(self.board_rows, self.board_cols,
                          self.win_num, self.p2_symbol)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_rows, self.board_cols), dtype=np.int8)
        self.available_positions = [(i,j) for i in range(self.board_rows)
                                          for j in range(self.board_cols)]
        self.curr_player = self.p1

    def winner(self):
        for rr in range(self.board_rows):
            for cc in range(self.board_cols-self.win_num+1):
                ss = sum(self.board[rr,cc:cc+self.win_num])
                if ss == self.win_num:
                    return 1
                if ss == -self.win_num:
                    return -1

        for cc in range(self.board_cols):
            for rr in range(self.board_rows-self.win_num+1):
                ss = sum(self.board[rr:rr+self.win_num,cc])
                if ss == self.win_num:
                    return 1
                if ss == -self.win_num:
                    return -1

        for rr in range(self.board_rows-self.win_num+1):
            for cc in range(self.board_cols-self.win_num+1):
                ss = sum([self.board[rr+i,cc+i] for i in range(self.win_num)])
                if ss == self.win_num:
                    return 1
                if ss == -self.win_num:
                    return -1

        for rr in range(self.board_rows-self.win_num+1):
            for cc in range(self.win_num-1,self.board_cols):
                ss = sum([self.board[rr+i,cc-i] for i in range(self.win_num)])
                if ss == self.win_num:
                    return 1
                if ss == -self.win_num:
                    return -1

        if len(self.available_positions) == 0:
            return 0

        return None

    def play_train(self, rounds, onlyplay=False, show=False):
        t1 = time.time()
        p1_wins = p2_wins = draws = total_count = 0
        stat_hist_win = list()
        stat_hist_count = list()
        for i in range(rounds):
            print(i)
            count = 0
            win = None
            while win is None:
                count += 1
                action = self.curr_player.make_move()
                if show:
                    if (count+1)%2==0:
                        self.print_board()
                        input()
                    print(str(count)+":", action)
                self.board[action] = self.curr_player.params['symbol']
                self.available_positions.remove(action)
                self.curr_player = \
                    self.p1 if self.curr_player is self.p2 else self.p2
                self.curr_player.accept_move(action)
                win = self.winner()

            if win == 1:
                p1_wins += 1
                if not onlyplay:
                    self.p1.feed_reward('win', count)
                    self.p2.feed_reward('loss', count)
            elif win == -1:
                p2_wins +=1
                if not onlyplay:
                    self.p1.feed_reward('loss', count)
                    self.p2.feed_reward('win', count)
            else:
                draws += 1
                if not onlyplay:
                    self.p1.feed_reward('draw', count)
                    self.p2.feed_reward('draw', count)

            total_count += count
            stat_hist_win.append(win)
            stat_hist_count.append(count)
            self.reset()
            self.p1.reset()
            self.p2.reset()
        self.p1.params['epochs'] += 1
        self.p2.params['epochs'] += 1
        stat_data = {
                     'rounds': rounds,
                     'time': int(time.time()-t1),
                     'p1_wins': p1_wins,
                     'p2_wins': p2_wins,
                     'draws': draws,
                     'total_count': total_count,
                     'hist_win': stat_hist_win,
                     'hist_count': stat_hist_count,
                     'onlyplay': onlyplay,
                     'p1_params': self.p1.params,
                     'p1_cnn_params': self.p1.cnn_params,
                     'p2_params': self.p2.params,
                     'p2_cnn_params': self.p2.cnn_params
                    }
        return stat_data

    def print_board(self):
        row_string = "   | "
        for cc in range(self.board_cols):
            row_string += str(cc).rjust(2," ") + " | "
        print(row_string)
        print("-"*len(row_string))

        for rr in range(self.board_rows):
            row_string = str(rr).rjust(2," ") + " | "
            for cc in range(self.board_cols):
                row_string += str(self.board[(rr,cc)]).rjust(2," ") + " | "
            print(row_string)
            print("-"*len(row_string))

        print("\n")
