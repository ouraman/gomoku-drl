###############################################################################
#
# gomoku reinforced deep learning agents
#
# ouh - 2020-05-01
#
###############################################################################

import os

BOARD_ROWS = 7
BOARD_COLS = 7

WIN_NUM = 5

WIN_REWARD = 1
LOSS_REWARD = -1
DRAW_REWARD = 0.5

D_GAMMA = 0.9
EXP_RATE = 0.3

P1_SYMBOL = 1
P2_SYMBOL = -1

P_SAVE_EXT_PARAMS = ".params"
P_SAVE_EXT_CNN = ".cnn"

BASE_DATA_DIR = os.path.expanduser("~/Documents/work/science/projects/gomoku/gomoku-drl_run_experiments/data/")

STATS_SAVE_EXT = ".gomokustat"
TEMPLATES_DIR = "templates"
TEMPLATES_CSS_DIR = "css"
TEMPLATES_JS_DIR = "js"
REPORT_TEMPLATE = "report_template.html"

RUNNING_MEAN_LEN = 100

FIG_SIZE_1 = (8, 8)
FIG_SIZE_2 = (16, 8)
FIG_SIZE_3 = (30, 8)

FIG_MARKER_SIZE_1 = 2

FIG_WIDTH_DATA_PER_INCH = 1000

AI_TIMEOUT = 120
