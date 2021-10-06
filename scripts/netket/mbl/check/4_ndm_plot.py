import pandas as pd
import plotly.graph_objects as go
from scripts.plot.routines.layout import add_layout
from scripts.plot.routines.save import save_figure


# Model params
model = 'mbl'
N = 8
seed = 3
W = 1.0
U = 1.0
J = 1.0
dt = 1
gamma = 0.1

path = f"/media/sf_Work/dl/netket/{model}/N({N})_rnd({seed})_H({W:0.4f}_{U:0.4f}_{J:0.4f})_D({dt}_{gamma:0.4f})"

# Ansatz params

params = [
    {"alpha": 2, "beta": 2, 'n_samples': 5000, 'n_iter': 1000},
]

fig_ldagl = go.Figure()
fig_diff_norm = go.Figure()
for p in params:
    df = pd.read_excel(f"{path}/NDM({p['alpha']}_{p['beta']}_{p['n_samples']}_{p['n_iter']}).xlsx")

    fig_ldagl.add_trace(
        go.Scatter(
            x=df['iteration'].values,
            y=df['ldagl_mean'].values,
            showlegend=True,
            name=f"alpha={p['alpha']} beta={p['beta']} nsamp={p['n_samples']}",
            mode="lines",
        )
    )

    fig_diff_norm.add_trace(
        go.Scatter(
            x=df['iteration'].values,
            y=df['norm_rho_diff'].values,
            showlegend=True,
            name=f"alpha={p['alpha']} beta={p['beta']} nsamp={p['n_samples']}",
            mode="lines",
        )
    )

add_layout(fig_ldagl, "Iterations", r"$L^\dagger L$", f"MBL  (N={N}, W={W}, U={U}, J={J})")
fig_ldagl.update_layout({'colorway': ['red', 'blue', 'green', 'orange', 'brown']})
fig_ldagl.update_yaxes(type="log")
save_figure(fig_ldagl, f"{path}/ldagl")


add_layout(fig_diff_norm, "Iterations", r"$\left\Vert \rho^{\text{exact}} - \rho^{\text{neural}} \right\Vert$", f"MBL  (N={N}, W={W}, U={U}, J={J})")
fig_diff_norm.update_layout({'colorway': ['red', 'blue', 'green', 'orange', 'brown']})
fig_diff_norm.update_yaxes(type="log")
save_figure(fig_diff_norm, f"{path}/diff_norm")
