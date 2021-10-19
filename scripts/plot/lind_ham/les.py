import pandas as pd
from scripts.plot.routines.save import save_figure
from scripts.plot.routines.scatter import add_scatter_trace
from scripts.plot.routines.layout import add_layout
from scripts.plot.routines.violin import add_violin_trace
import os
import plotly.graph_objects as go

path = f"E:/YandexDisk/Work/os_lnd/draft/mbl/2/figures/lndham/lambda/100_100"

N = 100
alpha = 0.5
n_seeds = 50
T = 1.0

lpn_type = -1 # energy
lpn_log_deltas = [-6.0, -5.0, -4.0]
Ts = [1, 2, 5]

vio = go.Figure()
for T in Ts:
    for log_delta in lpn_log_deltas:
        fn = f"{path}/lambda_N({N})_numSeeds({n_seeds})_alpha({alpha:0.4f})_T({T:0.4f})_lpn({lpn_type}_{log_delta:0.4f}).csv"
        les = pd.read_csv(fn, header=None)
        les['mean'] = les.mean(axis=1)
        les['period'] = fr"Period={T}"
        if T == 1:
            showlegend = True
        else:
            showlegend = False
        add_violin_trace(vio, les.loc[:, 'mean'], fr"$\log_{{{10}}}\Delta={{{log_delta}}}$", x=les.loc[:, 'period'], showlegend=showlegend)

add_layout(vio, "", fr"$\lambda$", f"")
vio.update_layout({'colorway': ['blue', 'red', 'green']})
save_figure(vio, f"{path}/lambda_N({N})_numSeeds({n_seeds})_alpha({alpha:0.4f})_lpn({lpn_type})_vio")


