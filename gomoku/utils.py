###############################################################################
#
# gomoku reinforced deep learning agents
#
# ouh - 2020-05-19
#
###############################################################################

import matplotlib.pyplot as plt
import os
import pickle
import pathlib
import shutil
from jinja2 import Environment, FileSystemLoader
from gomoku import gglobals as gg
from gomoku import gomoku

def running_mean(data, ll=2):
    ndata = list()
    for ii in range(ll, len(data)):
        ndata.append(sum(data[ii-ll:ii])/ll)
    return ndata

def running_win(data, win_symbol, ll=2):
    ndata = list()
    for ii in range(ll, len(data)):
        tmp = [1 if elem == win_symbol else 0 for elem in data[ii-ll:ii]]
        ndata.append(sum(tmp)/ll)
    return ndata

def save_game_stats(res, fpath):
    bdir = os.path.split(fpath)[0]
    pathlib.Path(bdir).mkdir(parents=True, exist_ok=True)
    with open(fpath, "ab") as fout:
        pickle.dump(("res", res), fout)

def save_deep_train_stats(player, res, fpath):
    bdir = os.path.split(fpath)[0]
    pathlib.Path(bdir).mkdir(parents=True, exist_ok=True)
    with open(fpath, "ab") as fout:
        dump_dict = {
                     'p_params': player.params,
                     'p_cnn_params': player.cnn_params,
                     'history': res
                    }
        pickle.dump(("deep_train", dump_dict), fout)

def load_game_stats(fpath):
    reslist = list()
    with open(fpath, 'rb') as fin:
        while True:
            try:
                reslist.append(pickle.load(fin))
            except EOFError:
                break
    return reslist

def generate_game_stats_report(rep_name, stats_path):
    root = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(root, gg.TEMPLATES_DIR)
    env = Environment(loader = FileSystemLoader(templates_dir))
    template = env.get_template(gg.REPORT_TEMPLATE)
    rep_dir = os.path.join(gg.BASE_DATA_DIR, rep_name)
    pathlib.Path(rep_dir).mkdir(parents=True, exist_ok=True)

    shutil.copytree(os.path.join(templates_dir, gg.TEMPLATES_CSS_DIR),
                    os.path.join(rep_dir,gg.TEMPLATES_CSS_DIR),
                    dirs_exist_ok=True)
    shutil.copytree(os.path.join(templates_dir, gg.TEMPLATES_JS_DIR),
                    os.path.join(rep_dir,gg.TEMPLATES_JS_DIR),
                    dirs_exist_ok=True)

    stat_list = load_game_stats(stats_path)

    figsize1 = gg.FIG_SIZE_1
    figsize2 = gg.FIG_SIZE_2
    figsize3 = gg.FIG_SIZE_3
    marker_size = gg.FIG_MARKER_SIZE_1
    mean_len = gg.RUNNING_MEAN_LEN

    cumdata_count = list()
    cumdata_win = list()
    cum_rounds = 0
    cum_time = 0
    summary_plot_data = list()
    reslist = list()
    for ii, (action, data) in enumerate(stat_list):
        if action == "deep_train":
            reslist.append((action, data))
            summary_plot_data.append((ii+1, action, cum_rounds))
            cum_time += data['history']['time']
        elif action == "res":
            stat_data = data
            figlist = list()

            fig, ax = plt.subplots(figsize=figsize1)
            pltdata = running_mean(stat_data['hist_win'], mean_len)
            xvals = range(len(pltdata))
            ax.set_ylabel("wins (+1/0/-1) running mean")
            ax.scatter(xvals, pltdata, s=marker_size)
            figname = "fig_"+str(ii)+"_win_run_mean.png"
            figpath = os.path.join(rep_dir, figname)
            fig.savefig(figpath)
            plt.close()
            figlist.append(figname)

            fig, ax = plt.subplots(figsize=figsize1)
            pltdata = running_mean(stat_data['hist_count'], mean_len)
            xvals = range(len(pltdata))
            ax.set_ylabel("number of game moves running mean")
            ax.scatter(xvals, pltdata, s=marker_size)
            figname = "fig_"+str(ii)+"_count_run_mean.png"
            figpath = os.path.join(rep_dir, figname)
            fig.savefig(figpath)
            plt.close()
            figlist.append(figname)

            fig, ax = plt.subplots(figsize=figsize2)
            pltdata1 = running_win(stat_data['hist_win'], 1, mean_len)
            pltdata2 = running_win(stat_data['hist_win'], -1, mean_len)
            pltdata3 = running_win(stat_data['hist_win'], 0, mean_len)
            xvals = range(mean_len, mean_len+len(pltdata1))
            ax.set_ylabel("ratios running mean")
            ax.scatter(xvals, pltdata1, label="p1 wins", s=marker_size)
            ax.scatter(xvals, pltdata2, label="p2 wins", s=marker_size)
            ax.scatter(xvals, pltdata3, label="draws", s=marker_size)
            ax.legend()
            ax.grid(True)
            figname = "fig_"+str(ii)+"_ratios_run_mean.png"
            figpath = os.path.join(rep_dir, figname)
            fig.savefig(figpath)
            plt.close()
            figlist.append(figname)

            stat_data['figures'] = figlist

            reslist.append((action, stat_data))

            cumdata_count.extend(stat_data['hist_count'])
            cumdata_win.extend(stat_data['hist_win'])
            summary_plot_data.append((ii+1, action, cum_rounds))
            cum_rounds += stat_data['rounds']
            cum_time += stat_data['time']

    summary = dict()
    figlist = list()

    pltdata1 = running_mean(cumdata_count, mean_len)
    xvals = range(mean_len, mean_len+len(pltdata1))
    figsize = (max(figsize3[0], int(len(pltdata1)/gg.FIG_WIDTH_DATA_PER_INCH)),
               figsize3[1])
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylabel("number of game moves running mean")
    ax.scatter(xvals, pltdata1, label="p1 wins", s=marker_size)
    ax.grid(True)
    last_dt_xx = -1
    for epoch, action, xx in summary_plot_data:
        if action == 'deep_train':
            if xx == last_dt_xx:
                valign = 'top'
            else:
                valign = 'bottom'
                last_dt_xx = xx
            ax.axvline(xx-10, color='b', linewidth=7)
            ax.text(xx-10, 1.05, epoch, transform=ax.get_xaxis_transform(),
                    fontsize='large', fontweight='bold', color='b',
                    horizontalalignment='right', verticalalignment=valign)
        else:
            ax.axvline(xx, color='r', linewidth=4)
            ax.text(xx, 1.05, epoch, transform=ax.get_xaxis_transform(),
                    fontsize='large', fontweight='bold', color='r',
                    horizontalalignment='left', verticalalignment='bottom')
    figname = "fig_summary_count_run_mean.png"
    figpath = os.path.join(rep_dir, figname)
    fig.savefig(figpath)
    plt.close()
    figlist.append(figname)

    pltdata1 = running_win(cumdata_win, 1, mean_len)
    pltdata2 = running_win(cumdata_win, -1, mean_len)
    pltdata3 = running_win(cumdata_win, 0, mean_len)
    xvals = range(mean_len, mean_len+len(pltdata1))
    figsize = (max(figsize3[0], int(len(pltdata1)/1000)), figsize3[1])
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylabel("ratios running mean")
    ax.scatter(xvals, pltdata1, label="p1 wins", s=marker_size)
    ax.scatter(xvals, pltdata2, label="p2 wins", s=marker_size)
    ax.scatter(xvals, pltdata3, label="draws", s=marker_size)
    ax.legend()
    ax.grid(True)
    last_dt_xx = -1
    for epoch, action, xx in summary_plot_data:
        if action == 'deep_train':
            if xx == last_dt_xx:
                valign = 'top'
            else:
                valign = 'bottom'
                last_dt_xx = xx
            ax.axvline(xx-10, color='b', linewidth=7)
            ax.text(xx-10, 1.05, epoch, transform=ax.get_xaxis_transform(),
                    fontsize='large', fontweight='bold', color='b',
                    horizontalalignment='right', verticalalignment=valign)
        else:
            ax.axvline(xx, color='r', linewidth=4)
            ax.text(xx, 1.05, epoch, transform=ax.get_xaxis_transform(),
                    fontsize='large', fontweight='bold', color='r',
                    horizontalalignment='left', verticalalignment='bottom')
    figname = "fig_summary_ratios_run_mean.png"
    figpath = os.path.join(rep_dir, figname)
    fig.savefig(figpath)
    plt.close()
    figlist.append(figname)

    summary['figures'] = figlist
    summary['time'] = cum_time

    fpath = os.path.join(rep_dir, rep_name+".html")
    with open(fpath, 'w') as fout:
        fout.write(template.render(rep_name = rep_name,
                                   data = reslist,
                                   summary = summary))

def universal_train_play(commands, rep_name):
    """
    universal wrapper for play, train, deep_train and generate

    commands: list of tupels
              tupels: ("play", p1, p2, rounds,
                       p1_params_update, p2_params_update)
                      ("train", p1, p2, rounds,
                       p1_params_update, p2_params_update)
                      ("deep_train", p)
                      params_update can be None
    rep_name: directory for report, stats and player saves
    """
    rep_base_dir = os.path.join(gg.BASE_DATA_DIR, rep_name)
    pathlib.Path(rep_base_dir).mkdir(parents=True, exist_ok=True)
    stats_path = os.path.join(rep_base_dir, rep_name + gg.STATS_SAVE_EXT)

    for action, *command in commands:
        onlyplay = False
        if action == "deep_train":
            pp = command[0]
            res = pp.deep_train()
            save_deep_train_stats(pp, res, stats_path)
            pp.save(rep_base_dir, exist_ok=True)
            continue
        if action == "play":
            onlyplay = True

        p1 = command[0]
        p2 = command[1]
        rounds = command[2]
        p1_params_update = command[3]
        p2_params_update = command[4]

        if p1_params_update is not None:
            p1.params.update(p1_params_update)
        if p2_params_update is not None:
            p2.params.update(p2_params_update)

        game = gomoku.Gomoku(p1, p2)
        res = game.play_train(rounds, onlyplay=onlyplay)
        save_game_stats(res, stats_path)

        p1.save(rep_base_dir, exist_ok=True)
        p2.save(rep_base_dir, exist_ok=True)
        generate_game_stats_report(rep_name, stats_path)
