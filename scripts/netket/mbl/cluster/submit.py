import pathlib
import os.path
import pandas as pd
import numpy as np


run_type = 'short'

path = '/data/biophys/denysov/yusipov/qs'

N = 8
Ws = np.linspace(0.0, 20.0, 101)
U = 1.0
J = 1.0
diss_type = 1
diss_gamma = 0.1

seed_start = 1
seed_shift = 1
seed_num = 1

alpha = 2
beta = 2
n_samples = 10000
n_iter = 1000

for W_id, W in enumerate(Ws):
    print(f"W={W:0.4f}")

    curr_path = path \
                + '/' + f"NDM({alpha:d}_{beta:d}_{n_samples:d}_{n_iter:d})" \
                + '/' + f"H({W:0.4f}_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f})"

    config_dict = {'experiment_id': [0]}
    config_df = pd.DataFrame(config_dict).set_index('experiment_id')
    config_df['N'] = N
    config_df['W'] = W
    config_df['U'] = U
    config_df['J'] = J
    config_df['diss_type'] = diss_type
    config_df['diss_gamma'] = diss_gamma
    config_df['seed_start'] = seed_start
    config_df['seed_shift'] = seed_shift
    config_df['seed_num'] = seed_num
    config_df['alpha'] = alpha
    config_df['beta'] = beta
    config_df['n_samples'] = n_samples
    config_df['n_iter'] = n_iter

    pathlib.Path(curr_path).mkdir(parents=True, exist_ok=True)

    fn_test = f"{curr_path}/metrics_seeds({seed_start}_{seed_shift}_{seed_num}).xlsx"

    if not os.path.isfile(fn_test):
        print(f"{fn_test} does not exist!")

        config_df.to_excel(f"{curr_path}/config.xlsx", index=True)

        if run_type == 'short':
            os.system(f"sbatch run_short.sh \"{curr_path}\"")
        elif run_type == 'medium':
            os.system(f"sbatch run_medium.sh \"{curr_path}\"")
