import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scripts.netket.plot.layout import add_layout
from scripts.netket.plot.save import save_figure
import pathlib
import os.path

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

target_seed = seed_start
target_iteration = n_iter

metric = 'norm_rho_diff'

for W_id, W in enumerate(Ws):
    print(f"W={W:0.4f}")

    curr_path = path \
                + '/' + f"NDM({alpha:d}_{beta:d}_{n_samples:d}_{n_iter:d})" \
                + '/' + f"H({W:0.4f}_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f})"

    fn = f"{curr_path}/metrics_seeds({seed_start}_{seed_shift}_{seed_num}).xlsx"
    curr_df = pd.read_excel(fn, index_col='iteration')
    curr_df.rename(columns = {col: col + f"_W({W:0.4f})" for col in curr_df.columns}, inplace=True)

    if W_id == 0:
        df_all = curr_df
    else:
        df_all = df_all.merge(curr_df, how='inner', left_index=True, right_index=True)

columns = [f"{x}_{target_seed:d}_W({W:0.4f})" for x in [metric] for W in Ws]
norm_rho_diffs = [df_all.loc[target_iteration, col] for col in columns]

path_save = path + '/plot/ndm_vs_exact'
pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=Ws,
        y=norm_rho_diffs,
        showlegend=True,
        name=f"$\\alpha={alpha} \\beta={beta} samples={n_samples}$",
        mode="lines",
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(
                width=1
            )
        )
    )
)
add_layout(fig, r"$W$", r"$\left\Vert \rho^{\text{exact}} - \rho^{\text{neural}} \right\Vert$",  "")
fig.update_layout({'colorway': ['red', 'blue', "red", "red"]})
save_figure(fig, f"{path_save}/{metric}_NDM({alpha:d}_{beta:d}_{n_samples:d})_H(var_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f})_seed({target_seed})_it({target_iteration})")
df_all.to_excel(f"{path_save}/{metric}_NDM({alpha:d}_{beta:d}_{n_samples:d})_H(var_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f}).xlsx", index=True)
