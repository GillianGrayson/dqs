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

is_cmax = True

N = 8
W = 0.0
U = 1.0
J = 1.0
diss_type = 1
diss_gamma = 0.1

seed_start_chunk = 25
seed = seed_start_chunk
seed_shift = 1
seed_num = 1

alpha = 2
beta = 2
n_samples = 10000
n_iter = 500

curr_path = path \
            + '/' + f"NDM({alpha:0.4f}_{beta:0.4f}_{n_samples:d}_{n_iter:d})" \
            + '/' + f"H({W:0.4f}_{U:0.4f}_{J:0.4f})_D({diss_type:d}_{diss_gamma:0.4f})" \
            + '/' + f"seeds({seed_start_chunk}_{seed_shift}_{seed_num})"

exact = np.load(f"{curr_path}/rho_exact_{seed}.npy")
neural = np.load(f"{curr_path}/rho_neural_{seed}.npy")

cmax = np.amax([np.abs(exact), np.abs(neural)])
cmap = plt.get_cmap('Blues')

plt.imshow(np.abs(exact), origin='lower', cmap=cmap)
if is_cmax:
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
