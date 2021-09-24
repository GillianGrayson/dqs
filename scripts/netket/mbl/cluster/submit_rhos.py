import pathlib
import os.path
import pandas as pd
import numpy as np
import socket

host_name = socket.gethostname()
print(host_name)

run_type = 'short'

if host_name == "newton":
    path = '/data/biophys/denysov/yusipov/qs'
elif host_name == "master":
    path = '/common/home/yusipov_i/data/qs'

N = 8
Ws = np.linspace(0.0, 20.0, 101)
U = 1.0
J = 1.0
diss_type = 1
diss_gamma = 0.1

seed_start = 1
seed_shift = 1
seed_num = 1
seed_chunks = 50
seed_start_chunks = np.linspace(seed_start, seed_start + (seed_chunks-1) * seed_num, seed_chunks, dtype=int)

alpha = 2.0
beta = 2.0
n_samples = 10000
n_iter = 500

for W_id, W in enumerate(Ws):
    print(f"W={W:0.4f}")

    for seed_start_chunk in seed_start_chunks:

        curr_path = path \
                    + '/' + f"NDM({alpha:0.4f}_{beta:0.4f}_{n_samples:d}_{n_iter:d})" \
                    + '/' + f"H({W:0.4f}_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f})" \
                    + '/' + f"seeds({seed_start_chunk}_{seed_shift}_{seed_num})"

        config_dict = {'experiment_id': [0]}
        config_df = pd.DataFrame(config_dict).set_index('experiment_id')
        config_df['N'] = N
        config_df['W'] = W
        config_df['U'] = U
        config_df['J'] = J
        config_df['diss_type'] = diss_type
        config_df['diss_gamma'] = diss_gamma
        config_df['seed_start'] = seed_start_chunk
        config_df['seed_shift'] = seed_shift
        config_df['seed_num'] = seed_num
        config_df['alpha'] = alpha
        config_df['beta'] = beta
        config_df['n_samples'] = n_samples
        config_df['n_iter'] = n_iter

        pathlib.Path(curr_path).mkdir(parents=True, exist_ok=True)

        fn_test = f"{curr_path}/metrics_{seed_start_chunk}.xlsx"

        if not os.path.isfile(fn_test):
            print(f"{fn_test} does not exist!")
            config_df.to_excel(f"{curr_path}/config.xlsx", index=True)
            if host_name == "newton":
                if run_type == 'short':
                    os.system(f"sbatch mpipks_run_short.sh \"{curr_path}\"")
                elif run_type == 'medium':
                    os.system(f"sbatch mpipks_run_medium.sh \"{curr_path}\"")
            elif host_name == "master":
                if run_type == 'short':
                    os.system(f"sbatch unn_run_short.sh \"{curr_path}\"")
                elif run_type == 'medium':
                    os.system(f"sbatch unn_run_medium.sh \"{curr_path}\"")
        else:
            test_df = pd.read_excel(fn_test, index_col='metrics')
            if test_df.isnull().values.any():
                print("Need recalc")
                config_df.to_excel(f"{curr_path}/config.xlsx", index=True)
                if host_name == "newton":
                    if run_type == 'short':
                        os.system(f"sbatch mpipks_run_short.sh \"{curr_path}\"")
                    elif run_type == 'medium':
                        os.system(f"sbatch mpipks_run_medium.sh \"{curr_path}\"")
                elif host_name == "master":
                    if run_type == 'short':
                        os.system(f"sbatch unn_run_short.sh \"{curr_path}\"")
                    elif run_type == 'medium':
                        os.system(f"sbatch unn_run_medium.sh \"{curr_path}\"")
