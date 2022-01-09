###############################################################################
#
# gomoku reinforced deep learning agents
#
# ouh - 2020-05-017
#
###############################################################################

import numpy as np
import tensorflow as tf
import pexpect
import re
import pickle
import os
import shutil
import time

from gomoku import gglobals as gg

class DRLPlayer:
    """
    Deep reinforcement learning player
    """
    def __init__(self, name,
                 exp_rate=gg.EXP_RATE,
                 decay_gamma=gg.D_GAMMA,
                 win_reward=gg.WIN_REWARD,
                 loss_reward=gg.LOSS_REWARD,
                 draw_reward=gg.DRAW_REWARD):

        self.params = {
                       'name': name,
                       'exp_rate': exp_rate,
                       'decay_gamma': decay_gamma,
                       'reward_value': {'win': win_reward,
                                        'loss': loss_reward,
                                        'draw': draw_reward},
                       'is_deep_trained': False,
                       'board_rows': None,
                       'board_cols': None,
                       'win_num': None,
                       'symbol': None,
                       'epochs': 0
                       }

        self.current_board = None
        self.available_positions = None
        self.states_hist = list()

        self.cnn_params = None
        self.cnn = None

        self.states_value = dict()
        self.states_count = dict()

    def reset(self):
        board_rows = self.params['board_rows']
        board_cols = self.params['board_cols']
        self.current_board = np.zeros((board_rows,board_cols), dtype=np.int8)
        self.available_positions = [(i,j) for i in range(board_rows)
                                          for j in range(board_cols)]
        self.states_hist = list()

    def game_init(self, board_rows, board_cols, win_num, symbol):
        if (self.params['board_rows'] is None
            and self.params['board_cols'] is None
            and self.params['win_num'] is None
            and self.params['symbol'] is None):
            self.params['board_rows'] = board_rows
            self.params['board_cols'] = board_cols
            self.params['win_num'] = win_num
            self.params['symbol'] = symbol
            self.reset()
        elif (self.params['board_rows'] != board_rows
              or self.params['board_cols'] != board_cols
              or self.params['win_num'] != win_num
              or self.params['symbol'] != symbol):
            print("board geometry mismatch")
            exit()

    def init_cnn(self):
        self.cnn_params = {
                            'layer_conv2d_1': {
                                'filters': 256,
                                'kernel_size': (self.params['win_num'],
                                                self.params['win_num']),
                                'padding': 'same',
                                'activation': 'relu',
                                'input_shape': (self.params['board_rows'],
                                                self.params['board_cols'], 1)
                            },
                            'layer_dense_1': {
                                'units': 512,
                                'activation': 'relu'
                            },
                            'layer_dense_2': {
                                'units': 1,
                                'activation': 'tanh'
                            },
                            'compile': {
                                'optimizer': 'rmsprop',
                                'loss': 'mse',
                                'metrics': ['mae']
                            },
                            'fit': {
                                'epochs': 100,
                                'batch_size': 1024
                            },
                            'summary': None
                          }

        self.cnn = tf.keras.models.Sequential()
        self.cnn.add(tf.keras.layers.Conv2D(**self.cnn_params['layer_conv2d_1']))
        self.cnn.add(tf.keras.layers.Flatten())
        self.cnn.add(tf.keras.layers.Dense(**self.cnn_params['layer_dense_1']))
        self.cnn.add(tf.keras.layers.Dense(**self.cnn_params['layer_dense_2']))

        self.cnn.compile(**self.cnn_params['compile'])

        summary_stringlist = list()
        self.cnn.summary(print_fn=lambda x: summary_stringlist.append(x))
        self.cnn_params['summary'] = '\n'.join(summary_stringlist)

    def make_move(self):
        exp_rate = self.params['exp_rate']
        board_rows = self.params['board_rows']
        board_cols = self.params['board_cols']
        symbol = self.params['symbol']
        is_deep_trained = self.params['is_deep_trained']
        if np.random.rand() < exp_rate:
            idx = np.random.choice(len(self.available_positions))
            action = self.available_positions[idx]
        else:
            act_val = []
            pred_act = []
            pred_bb = []
            value_max = -999
            for p in self.available_positions:
                next_board = self.current_board.copy()
                next_board[p] = symbol

                win = self.check_win(next_board, symbol)
                if win:
                    action = p
                    self.current_board[action] = symbol
                    self.states_hist.append(self.current_board.tobytes())
                    self.available_positions.remove(action)
                    return action

                op_symbol = -1*symbol
                next_op_board = self.current_board.copy()
                next_op_board[p] = op_symbol

                op_win = self.check_win(next_op_board, op_symbol)
                if op_win:
                    action = p
                    self.current_board[action] = symbol
                    self.states_hist.append(self.current_board.tobytes())
                    self.available_positions.remove(action)
                    return action

                next_board_hash = next_board.tobytes()
                value = self.states_value.get(next_board_hash)
                if value is None:
                    if is_deep_trained is True:
                        bb = np.frombuffer(next_board_hash, dtype=np.int8)
                        bb = bb.reshape(board_rows, board_cols, 1)
                        pred_act += [p]
                        pred_bb += [bb]
                    else:
                        act_val += [(p, 0)]
                else:
                    act_val += [(p, value)]

            if pred_act:
                bb = np.array(pred_bb, dtype=np.int8)
                pred = self.cnn.predict(bb)
                act_val += [(pred_act[i], pred[i,0]) for i in range(len(pred))]

            np.random.shuffle(act_val)
            for pp, vv in act_val:
                if vv >= value_max:
                    value_max = vv
                    action = pp

        self.current_board[action] = symbol
        self.states_hist.append(self.current_board.tobytes())
        self.available_positions.remove(action)

        return action

    def check_win(self, board, symbol):
        brows = self.params['board_rows']
        bcols = self.params['board_cols']
        wnum = self.params['win_num']
        wsum = symbol*wnum

        for rr in range(brows):
            for cc in range(bcols-wnum+1):
                ss = sum(board[rr,cc:cc+wnum])
                if ss == wsum:
                    return True

        for cc in range(bcols):
            for rr in range(brows-wnum+1):
                ss = sum(board[rr:rr+wnum,cc])
                if ss == wsum:
                    return True

        for rr in range(brows-wnum+1):
            for cc in range(bcols-wnum+1):
                ss = sum([board[rr+i,cc+i] for i in range(wnum)])
                if ss == wsum:
                    return True

        for rr in range(brows-wnum+1):
            for cc in range(wnum-1,bcols):
                ss = sum([board[rr+i,cc-i] for i in range(wnum)])
                if ss == wsum:
                    return True

        return False

    def accept_move(self, action):
        self.current_board[action] = -1*self.params['symbol']
        self.available_positions.remove(action)

    def feed_reward(self, outcome, count=gg.WIN_NUM):
        # brows = self.params['board_rows']
        # bcols = self.params['board_cols']
        decay_gamma = self.params['decay_gamma']
        reward = self.params['reward_value'][outcome]
        for st in reversed(self.states_hist):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
                self.states_count[st] = 0

            self.states_count[st] += 1
            self.states_value[st] += \
                        (reward-self.states_value[st])/self.states_count[st]
            reward *= decay_gamma

    def deep_train(self):
        t1 = time.time()
        board_rows = self.params['board_rows']
        board_cols = self.params['board_cols']
        if not self.params['is_deep_trained']:
            self.init_cnn()
        train_boards = np.zeros((len(self.states_value),
                                 board_rows, board_cols, 1), dtype=np.int8)
        train_labels = np.zeros(len(self.states_value))
        for nn, (kk, vv) in enumerate(self.states_value.items()):
            tmp_board = np.frombuffer(kk, dtype=np.int8)
            train_boards[nn,:,:,0] = tmp_board.reshape(board_rows, board_cols)
            train_labels[nn] = vv
        history = self.cnn.fit(train_boards, train_labels, **self.cnn_params['fit'])
        self.params['is_deep_trained'] = True
        stat_data = history.history
        stat_data['time'] = int(time.time()-t1)
        self.params['epochs'] += 1
        return stat_data

    def save(self, fdir, exist_ok=False):
        ext1 = gg.P_SAVE_EXT_PARAMS
        ext2 = gg.P_SAVE_EXT_CNN
        sdir = os.path.join(fdir, self.params['name'])
        if os.path.exists(sdir):
            if exist_ok is False:
                print("directory exists. not saved.")
                return
            else:
                shutil.rmtree(sdir)

        os.mkdir(sdir)
        param_path = os.path.join(sdir, self.params['name'] + ext1)
        with open(param_path, 'wb') as fout:
            data_dump = {
                         'params': self.params,
                         'cnn_params': self.cnn_params,
                         'states_values': self.states_value,
                         'states_count': self.states_count
                        }
            pickle.dump(data_dump, fout)

        cnn_path = os.path.join(sdir, self.params['name'] + ext2)
        if self.params["is_deep_trained"]:
            self.cnn.save(cnn_path)

    def load(self, fdir):
        ext1 = gg.P_SAVE_EXT_PARAMS
        ext2 = gg.P_SAVE_EXT_CNN
        fpath = os.path.join(fdir, self.params['name'] + ext1)
        if not os.path.exists(fpath):
            print("file does not exist")
            return

        with open(fpath, 'rb') as fin:
            data_dump = pickle.load(fin)
            self.params = data_dump['params']
            self.cnn_params = data_dump['cnn_params']
            self.states_value = data_dump['states_values']
            self.states_count = data_dump['states_count']

        if self.params['is_deep_trained'] is True:
            cnn_path = os.path.join(fdir, self.params['name'] + ext2)
            self.cnn = tf.keras.models.load_model(cnn_path)

        self.reset()

class ExtAIPlayer:
    """
    Wrapper for external Gomoku AI
    http://petr.lastovicka.sweb.cz/protocl2en.htm
    uses wine for .exe files
    board is transposed compared to internal classes (x,y) -> (y,x)
    """
    def __init__(self, ai_cmd):
        self.proc = None
        self.params = {
                       'symbol': None,
                       'board_size': None,
                       'started': False,
                       'ai_cmd': ai_cmd,
                       'epochs': 0
                      }

        self.cnn_params = None
        self.move_pat = re.compile(b'\n([0-9]+),([0-9]+)')

    def game_init(self, board_rows, board_cols, win_num, symbol):
        if win_num != 5:
            print("external ai only supports win_num=5")
            exit()

        if board_rows != board_cols:
            print("external ai only supports square boards")
            exit()

        self.params['symbol'] = symbol
        self.params['board_size'] = board_rows

        self.proc = pexpect.spawn(self.params['ai_cmd'])
        self.proc.sendline("START " + str(self.params['board_size']))
        res = self.proc.expect(["\n(OK)", "\n(ERROR)"])
        if res != 0:
            print("board size not supported by external ai")
            exit()

    def reset(self):
        self.proc.close()
        self.proc = pexpect.spawn(self.params['ai_cmd'])
        self.proc.sendline("START " + str(self.params['board_size']))
        self.params['started'] = False

    def make_move(self):
        if not self.params['started']:
            self.proc.sendline("BEGIN")
            self.params['started'] = True

        self.proc.expect(self.move_pat, timeout=gg.AI_TIMEOUT)
        yy = int(self.proc.match.groups()[0])
        xx = int(self.proc.match.groups()[1])

        return (xx,yy)

    def accept_move(self, action):
        if not self.params['started']:
            self.params['started'] = True

        self.proc.sendline("TURN " + str(action[1]) + "," + str(action[0]))

    def feed_reward(self, outcome, count):
        pass

    def save(self, fdir, exist_ok=False):
        pass



class HumanPlayer:
    """
    Class for manual playing
    """
    def __init__(self, name):
        self.name = name
        self.current_board = None
        self.available_positions = None
        self.board_rows = None
        self.board_cols = None
        self.symbol = None

    def game_init(self, board_rows, board_cols, win_num, symbol):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.symbol = symbol
        self.reset()

    def reset(self):
        self.current_board = np.zeros((self.board_rows,self.board_cols),
                                      dtype=np.int8)
        self.available_positions = [(i,j) for i in range(self.board_rows)
                                          for j in range(self.board_cols)]

    def make_move(self):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in self.available_positions:
                self.current_board[action] = self.symbol
                self.available_positions.remove(action)
                return action

    def accept_move(self, action):
        self.current_board[action] = -1*self.symbol
        self.available_positions.remove(action)