import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RADIUS = 10
convkernel = np.ones(2 * RADIUS + 1)
COLORS = ['red', 'blue', 'green', 'm', 'magenta', 'brown', 'yellow', 'purple', 'black']

LAST_EPI = 10

def collect_file_folder(file_path, config_prefix):
    file_dir_list = []
    file_num = 0
    for dir in os.listdir(file_path):
        if dir.startswith(config_prefix):
            file_dir_list.append(os.path.join(file_path, dir))
            file_num += 1

    return file_dir_list, file_num

def load_total_results(file_path, config_prefix, items):
    file_dir_list, num_seeds = collect_file_folder(file_path, config_prefix)
    total_seed_data = []
    curr_min_length = np.inf
    # load all results needs to be plotted for one configuration
    for file_dir in file_dir_list:
        filename = os.path.join(file_dir, 'progress.csv')
        assert os.path.exists(filename), 'no such result files'
        one_seed_data = pd.read_csv(filename, encoding='utf-8', usecols=items)
        total_seed_data.append(one_seed_data.to_numpy())
        curr_min_length = np.minimum(curr_min_length, one_seed_data.shape[0]).astype(int)

    return total_seed_data, curr_min_length

def smooth_plot(total_data, curr_len, item_idx, color, label):
    tot_smooth_data = []
    for single_data in total_data:
        smooth_data = np.convolve(single_data[:, item_idx], convkernel, mode='same') \
                      / np.convolve(np.ones_like(single_data[:, item_idx]), convkernel, mode='same')
        tot_smooth_data.append(smooth_data[:curr_len])

    x_ = np.arange(0, curr_len) / curr_len * (1 * curr_len / 1000)
    y_mean = np.mean(tot_smooth_data, axis=0)
    y_std = np.std(tot_smooth_data, axis=0)
    # y_std_err = y_std / np.sqrt(len(tot_smooth_data))
    y_std_err = y_std / 1.
    plt.plot(x_, y_mean, color=color, linestyle='-', label=label)
    plt.fill_between(x_, y_mean - y_std_err / 1., y_mean + y_std_err / 1., color=color, alpha=.2)

    return np.mean(y_mean[-LAST_EPI:]), np.mean(y_std[-LAST_EPI:])



env_types = [
    'medium',
    'medium-replay',
    'medium-expert',
    'expert',
]
env_names = [
    'halfcheetah',
    'hopper',
    'walker2d',
]


if __name__ == '__main__':
    file_path = '/home/zy/zz/all_logs/offorRL/{}-{}-v2/IQL'
    prefix_name = 'beta=3.0_quantile=0.7_clip_score=100'
    ITEMS = [
        ['eval/Normalized Returns'], # 0
    ]

    item_plot = ITEMS[0]

    plt.figure(figsize=(8, 6), dpi=80)
    color_idx = 0
    for env_type in env_types:
        for env_name in env_names:
            try:
                total_data, curr_len = load_total_results(file_path.format(env_name, env_type), prefix_name, item_plot)
                print('current length: {}; total seeds: {}'.format(curr_len, len(total_data)))
                score_mean, score_std = smooth_plot(total_data, curr_len, item_idx=0, color=COLORS[color_idx], label=env_type)
                color_idx+=1
                print('performance on {}-{}-v2: {}+-{}'.format(env_name, env_type, score_mean, score_std))

            except ValueError:
                print('todo....')
    plt.legend(loc=4, fontsize=10)
    plt.xlim()
    plt.show()

