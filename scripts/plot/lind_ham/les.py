import pandas as pd
from scripts.plot.routines.save import save_figure
from scripts.plot.routines.scatter import add_scatter_trace
from scripts.plot.routines.layout import add_layout
from scripts.plot.routines.violin import add_violin_trace
import os
import plotly.graph_objects as go


path = f"E:/YandexDisk/Work/os_lnd/draft/mbl/2/figures/lndham/lambda"

N = 100
alpha = 0.5
n_seeds = 100
T = 1.0

lpn_type = -1 # energy
lpn_log_delta = -6.0

fn = f"{path}/lambda_N({N})_numSeeds({n_seeds})_alpha({alpha:0.4f})_T({T:0.4f})_lpn({lpn_type}_{lpn_log_delta:0.4f}_{lpn_log_delta:0.4f}_{lpn_log_delta:0.4f}).csv"

les = pd.read_csv(fn, header=None)
les = les.T
les['mean'] = les.mean(axis=1)

vio = go.Figure()
add_violin_trace(vio, les.loc[:, 'mean'].values, 'Random model')
add_layout(vio, "", r"$\lambda$", f"")
vio.update_layout({'colorway': ['blue']})
save_figure(vio, f"{path}/lambda_N({N})_numSeeds({n_seeds})_alpha({alpha:0.4f})_T({T:0.4f})_lpn({lpn_type}_{lpn_log_delta:0.4f}_{lpn_log_delta:0.4f}_{lpn_log_delta:0.4f})_vio")


