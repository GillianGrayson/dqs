import netket as nk
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import linalg as la
from scipy.linalg import sqrtm
from scripts.netket.mbl.model import get_mbl_model
import plotly.graph_objects as go
from scripts.plot.routines.layout import add_layout
from scripts.plot.routines.save import save_figure

config_df = pd.read_excel('config.xlsx', index_col='experiment_id')

print(config_df.to_string())

N = int(config_df.at[0, 'N'])
W = float(config_df.at[0, 'W'])
U = float(config_df.at[0, 'U'])
J = float(config_df.at[0, 'J'])
diss_type = int(config_df.at[0, 'diss_type'])
diss_gamma = float(config_df.at[0, 'diss_gamma'])

alpha = float(config_df.at[0, 'alpha'])
beta = float(config_df.at[0, 'beta'])
n_samples = int(config_df.at[0, 'n_samples'])

seed_start = int(config_df.at[0, 'seed_start'])
seed_shift = int(config_df.at[0, 'seed_shift'])
seed_num = int(config_df.at[0, 'seed_num'])

n_iter = int(config_df.at[0, 'n_iter'])

result_df = pd.DataFrame(data=np.zeros(shape=(n_iter, seed_num)))

seeds = list(range(seed_start, seed_start + seed_num, seed_shift))
metrics = {
    'ldagl_mean': r"$L^\dagger L$",
    'norm_rho_diff': r"$\left\Vert \rho^{\text{exact}} - \rho^{\text{neural}} \right\Vert$",
    'trace_norm': r"$\left\Vert \rho^{\text{exact}} - \rho^{\text{neural}} \right\Vert_{\mathrm{TR}}$",
    'fidelity_diff': r"$1 - \mathrm{Fidelity}$",
}

metrics_df = pd.DataFrame(
    data=np.zeros(shape=(n_iter, len(metrics)), dtype=float),
    index=np.linspace(1, n_iter, n_iter, dtype=int),
    columns=list(metrics.keys())
)
metrics_df.index.name = 'iteration'

for seed in seeds:
    print(f"seed = {seed}")

    lind, ss = get_mbl_model(config_df, seed)

    # Calculate exact rho
    rho_exact = nk.exact.steady_state(lind, method="iterative", sparse=True, tol=1e-10)
    rho_exact = np.asmatrix(rho_exact)

    for it in tqdm(range(n_iter), mininterval=300.0):
        out = ss.run(n_iter=1, show_progress=False)
        rho_neural = np.array(ss.state.to_matrix())
        rho_neural = np.asmatrix(rho_neural)
        rho_diff = rho_exact - rho_neural
        ldagl_mean = ss.ldagl.mean
        norm_rho_diff = la.norm(rho_diff)
        trace_norm = np.trace(sqrtm(np.matmul(rho_diff.getH(), rho_diff))).real
        fidelity = np.trace(sqrtm(np.matmul(np.matmul(sqrtm(rho_exact), rho_neural), sqrtm(rho_exact)))) ** 2
        fidelity_diff = 1.0 - fidelity.real

        metrics_df.loc[it + 1, f"ldagl_mean"] = ss.ldagl.mean
        metrics_df.loc[it + 1, f"norm_rho_diff"] = la.norm(rho_diff)
        metrics_df.loc[it + 1, f"trace_norm"] = la.norm(trace_norm)
        metrics_df.loc[it + 1, f"fidelity_diff"] = la.norm(fidelity_diff)

        print(f"iteration = {it}")
        print(f"ldagl = {str(ss.ldagl)}")
        print(f"norm_rho_diff = {norm_rho_diff}")
        print(f"trace_norm = {trace_norm}")
        print(f"fidelity_diff = {fidelity_diff}")

    metrics_df.to_excel(f"metrics_seed({seed}).xlsx", index=True)

    for m in metrics:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=metrics_df.index.values,
                y=metrics_df[m].values,
                showlegend=False,
                mode="lines",
            )
        )
        add_layout(fig, r"$\mathrm{Iterations}$", metrics[m],  "")
        fig.update_layout({'colorway': ['red']})
        save_figure(fig, f"{m}_NDM({alpha:0.4f}_{beta:0.4f}_{n_samples:d}_{n_iter:d})_H(var_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f})_seed({seed})")

