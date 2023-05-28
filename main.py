###############################################################################
#
# gomoku reinforced deep learning agents
#
# ouh - 2020-05-01
#
###############################################################################

from gomoku import gomoku
from gomoku import player
from gomoku import gglobals as gg
from gomoku import utils
import time
import numpy as np

t1 = time.time()

p1 = player.DRLPlayer("p1")
# p1.load("data/report_01/mod_player/p1_m2")
# p1.load("data/report_03-01-02-01-01-01-01-01-01/mod_player/p1_m1")
p2 = player.DRLPlayer("p2")
# p2 = player.ExtAIPlayer("wine ext_brains/pbrain-Tito2014.exe")
# p2 = player.ExtAIPlayer("wine ext_brains/pbrain-embryo19_s.exe")

command_list = [
                ("train", p1, p2, 8, {"exp_rate": 0.3},
                                        None),
                ("deep_train", p1),
                ("train", p1, p2, 8, {"exp_rate": 0.1},
                                        None),
                # ("deep_train", p1),
                ("train", p1, p2, 8, {"exp_rate": 0.05},
                                        None),
                # ("deep_train", p1),
                ("train", p1, p2, 8, {"exp_rate": 0.01},
                                        None),
                # ("deep_train", p1),
                # ("train", p1, p2, 8, {"exp_rate": 0.001},
                                        # None),
                # ("deep_train", p1),
                # ("train", p1, p2, 8, {"exp_rate": 0},
                                        # None)
                # ("train", p1, p2, 2000, {"exp_rate": 0,
                #                          "lr": 0.4},
                #                         None)
                # # ("deep_train", p1),
                # ("train", p1, p2, 2000, None, None)
                # ("deep_train", p1),
                # ("deep_train", p2),
                # ("train", p1, p2, 200, None, None),
                # ("deep_train", p1),
                # ("train", p1, p2, 1000, None, None)
               ]

utils.universal_train_play(command_list, "report_02")

# game1 = gomoku.Gomoku(p1,p2)
# res = game1.play_train(10, show=True)

print("time: ", time.time()-t1)
