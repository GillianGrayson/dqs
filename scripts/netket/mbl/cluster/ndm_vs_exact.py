import netket as nk
import numpy as np
import pandas as pd
from netket.hilbert import Fock
from tqdm import tqdm
from numpy import linalg as la


config_df = pd.read_excel('config.xlsx', index_col='experiment_id')

N = int(config_df.at[0, 'N'])
W = float(config_df.at[0, 'W'])
U = float(config_df.at[0, 'U'])
J = float(config_df.at[0, 'J'])
diss_type = int(config_df.at[0, 'diss_type'])
diss_gamma = float(config_df.at[0, 'diss_gamma'])

seed_start = int(config_df.at[0, 'seed_start'])
seed_shift = int(config_df.at[0, 'seed_shift'])
seed_num = int(config_df.at[0, 'seed_num'])

alpha = float(config_df.at[0, 'alpha'])
beta = float(config_df.at[0, 'beta'])
n_samples = int(config_df.at[0, 'n_samples'])
n_iter = int(config_df.at[0, 'n_iter'])

result_df = pd.DataFrame(data=np.zeros(shape=(n_iter, seed_num)))

seeds = list(range(seed_start, seed_start + seed_num, seed_shift))
metrics_names = ['ldagl_mean', 'norm_rho_diff', 'norm_rho_diff_conj']
columns = [f"{x}_{y:d}" for x in metrics_names for y in seeds]

metrics_df = pd.DataFrame(
    data=np.zeros(shape=(n_iter, len(metrics_names) * len(seeds)), dtype=float),
    index=np.linspace(1, n_iter, n_iter, dtype=int),
    columns=columns
)
metrics_df.index.name = 'iteration'

for seed in seeds:
    print(f"seed = {seed}")

    np.random.seed(seed)

    # Generate on-site energies
    energies = np.random.uniform(-1.0, 1.0, N)

    # Hilbert space
    hi = Fock(n_max=1, n_particles=N // 2, N=N)

    # The Hamiltonian
    ha = nk.operator.LocalOperator(hi)
    # List of dissipative jump operators
    j_ops = []
    for boson_id in range(N - 1):
        ha += W * energies[boson_id] * nk.operator.boson.number(hi, boson_id)
        ha += U * nk.operator.boson.number(hi, boson_id) * nk.operator.boson.number(hi, boson_id + 1)
        ha -= J * (nk.operator.boson.create(hi, boson_id) * nk.operator.boson.destroy(hi, boson_id + 1) + nk.operator.boson.create(hi, boson_id + 1) * nk.operator.boson.destroy(hi, boson_id))
        if diss_type == 1:
            A = (nk.operator.boson.create(hi, boson_id + 1) + nk.operator.boson.create(hi, boson_id)) * (
                        nk.operator.boson.destroy(hi, boson_id + 1) - nk.operator.boson.destroy(hi, boson_id))
        elif diss_type == 0:
            A = nk.operator.boson.create(hi, boson_id) * nk.operator.boson.destroy(hi, boson_id)
        j_ops.append(np.sqrt(diss_gamma) * A)
    ha += W * energies[N - 1] * nk.operator.boson.number(hi, N - 1)
    if diss_type == 0:
        A = nk.operator.boson.create(hi, N - 1) * nk.operator.boson.destroy(hi, N - 1)
        j_ops.append(np.sqrt(diss_gamma) * A)

    # Create the Liouvillian
    lind = nk.operator.LocalLiouvillian(ha, j_ops)

    # Neural quantum state model: Positive-Definite Neural Density Matrix using the ansatz from Torlai and Melko
    ndm = nk.models.NDM(
        alpha=alpha,
        beta=beta,
    )

    # Metropolis Local Sampling
    graph = nk.graph.Hypercube(N, n_dim=1, pbc=False)
    sa_graph = nk.graph.disjoint_union(graph, graph)
    sa = nk.sampler.MetropolisExchange(lind.hilbert, graph=sa_graph)

    # Optimizer
    op = nk.optimizer.Sgd(0.01)
    sr = nk.optimizer.SR(diag_shift=0.01)

    # Variational state
    vs = nk.vqs.MCMixedState(sa, ndm, n_samples=n_samples)
    vs.init_parameters(nk.nn.initializers.normal(stddev=0.01))

    # Driver
    ss = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr)

    # Calculate exact rho
    rho_exact = nk.exact.steady_state(lind, method="iterative", sparse=True, tol=1e-10)

    for it in tqdm(range(n_iter), mininterval=300.0):
        out = ss.run(n_iter=1, show_progress=False)
        metrics_df.loc[it + 1, f"ldagl_mean_{seed:d}"] = ss.ldagl.mean
        rho_neural = np.array(ss.state.to_matrix())
        rho_diff = rho_exact - rho_neural
        rho_diff_conj = rho_exact - rho_neural.conjugate()
        metrics_df.loc[it + 1, f"norm_rho_diff_{seed:d}"] = la.norm(rho_diff)
        metrics_df.loc[it + 1, f"norm_rho_diff_conj_{seed:d}"] = la.norm(rho_diff_conj)

    metrics_df.to_excel(f"metrics_seeds({seed_start}_{seed_shift}_{seed_num}).xlsx", index=True)
