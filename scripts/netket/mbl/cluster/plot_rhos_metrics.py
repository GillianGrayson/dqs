import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scripts.netket.plot.layout import add_layout
from scripts.netket.plot.save import save_figure
import pathlib
import socket
import matplotlib.pyplot as plt


host_name = socket.gethostname()
print(host_name)

run_type = 'short'

if host_name == "newton":
    path = '/data/biophys/denysov/yusipov/qs'
elif host_name == "master":
    path = '/common/home/yusipov_i/data/qs'

is_plot_mtx = True

N = 8
Ws = np.linspace(0.0, 20.0, 101)
U = 1.0
J = 1.0
diss_type = 1
diss_gamma = 0.1

seed_start = 1
seed_shift = 1
seed_num = 1
seeds = np.linspace(seed_start, seed_start + seed_shift * (seed_num - 1), seed_num, dtype=int)

alpha = 2
beta = 2
n_samples = 10000
n_iter = 500

target_seed = seed_start
target_iteration = n_iter

metric_keys = {
    'iteration_best': r"$\mathrm{Iterations}$", 
    'ldagl_mean': r"$L^\dagger L$",
    'norm_rho_diff': r"$\left\Vert \rho^{\text{exact}} - \rho^{\text{neural}} \right\Vert$",
    'norm_rho_diff_conj': r"$\left\Vert \rho^{\text{exact}} - \rho^{\text{neural}} \right\Vert$"
}

metrics_df = pd.DataFrame(data=np.zeros(shape=(len(Ws), 1 + len(metric_keys))), columns=['W'] + list(metric_keys.keys()))
metrics_df.loc[:, 'W'] = Ws
metrics_df.set_index('W', inplace=True)

for W_id, W in enumerate(Ws):
    print(f"W={W:0.4f}")

    curr_path = path \
                + '/' + f"NDM({alpha:0.4f}_{beta:0.4f}_{n_samples:d}_{n_iter:d})" \
                + '/' + f"H({W:0.4f}_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f})" \
                + '/' + f"seeds({seed_start}_{seed_shift}_{seed_num})"

    for seed in seeds:
        
        if is_plot_mtx:
            exact = np.load(f"{curr_path}/rho_exact_{seed}.npy")
            neural = np.load(f"{curr_path}/rho_neural_{seed}.npy")

            cmax = np.amax([np.abs(exact), np.abs(neural)])
            cmap = plt.get_cmap('Blues')

            plt.imshow(np.abs(exact), origin='lower', cmap=cmap)
            plt.clim(0, cmax)
            clb = plt.colorbar()
            clb.ax.set_title(r"$\left| \rho^{\mathrm{exact}}_{n,n} \right|$")
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.savefig(f"{curr_path}/rho_exact_{seed}.pdf")
            plt.savefig(f"{curr_path}/rho_exact_{seed}.png")
            plt.close()

            plt.imshow(np.abs(neural), origin='lower', cmap=cmap)
            plt.clim(0, cmax)
            clb = plt.colorbar()
            clb.ax.set_title(r"$\left| \rho^{\mathrm{neural}}_{n,n} \right|$")
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.savefig(f"{curr_path}/rho_neural_{seed}.pdf")
            plt.savefig(f"{curr_path}/rho_neural_{seed}.png")
            plt.close()

            diff_rho = exact - neural
            plt.imshow(np.abs(diff_rho), origin='lower', cmap=cmap)
            clb = plt.colorbar()
            clb.ax.set_title(r"$\left| \rho^{\mathrm{exact}}_{n,n} - \rho^{\mathrm{neural}}_{n,n} \right|$")
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.savefig(f"{curr_path}/rho_diff_{seed}.pdf")
            plt.savefig(f"{curr_path}/rho_diff_{seed}.png")
            plt.close()

        fn = f"{curr_path}/metrics_{seed}.xlsx"
        curr_df = pd.read_excel(fn, index_col='metrics')
        for metric_key in metric_keys:
            metrics_df.loc[W, metric_key] += curr_df.loc[metric_key, 'values'] / seed_num

path_save = path + '/plot/rhos/metrics'
pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

for metric_key in metric_keys:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Ws,
            y=metrics_df.loc[:, metric_key].values,
            showlegend=True,
            name=f"$\\alpha={alpha} \\quad \\beta={beta} \\quad samples={n_samples}$",
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
    add_layout(fig, r"$W$", metric_keys[metric_key],  "")
    fig.update_layout({'colorway': ['red', 'blue', "red", "red"]})
    save_figure(fig, f"{path_save}/{metric_key}_NDM({alpha:d}_{beta:d}_{n_samples:d}_{n_iter:d})_H(var_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f})_seed({target_seed})_it({target_iteration})")
metrics_df.to_excel(f"{path_save}/NDM({alpha:d}_{beta:d}_{n_samples:d}_{n_iter:d})_H(var_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f}).xlsx", index=True)
