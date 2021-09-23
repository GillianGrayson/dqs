import netket as nk
import numpy as np
import pandas as pd
from netket.hilbert import Fock
from numpy import linalg as la
from scipy.linalg import sqrtm


config_df = pd.read_excel('config.xlsx', index_col='experiment_id')

print(config_df.to_string())

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

seeds = list(range(seed_start, seed_start + seed_num, seed_shift))

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
    op = nk.optimizer.Adam(0.01)
    sr = nk.optimizer.SR(diag_shift=0.01)

    # Variational state
    vs = nk.vqs.MCMixedState(sa, ndm, n_samples=n_samples)
    vs.init_parameters(nk.nn.initializers.normal(stddev=0.01))

    # Driver
    #ss = nk.SteadyState(lind, optimizer=op, variational_state=vs, preconditioner=sr)
    ss = nk.SteadyState(lind, optimizer=op, variational_state=vs)

    # Calculate exact rho
    rho_exact = nk.exact.steady_state(lind, method="iterative", sparse=True, tol=1e-10)
    rho_exact = np.asmatrix(rho_exact)

    rho_neural_best = np.zeros(shape=rho_exact.shape)
    iteration_best = -1
    ldagl_mean_best = 1e10
    norm_rho_diff_best = 1e10
    norm_rho_diff_conj_best = 1e10
    trace_norm_best = 1e10
    trace_norm_conj_best = 1e10
    fidelity_diff_best = 1e10
    fidelity_diff_conj_best = 1e10
    for it in range(n_iter):
        out = ss.run(n_iter=1, show_progress=False)
        rho_neural = np.array(ss.state.to_matrix())
        rho_neural = np.asmatrix(rho_neural)
        rho_diff = rho_exact - rho_neural
        rho_diff_conj = rho_exact - rho_neural.getT()
        ldagl_mean = ss.ldagl.mean
        ldagl_error_of_mean = ss.ldagl.error_of_mean
        ldagl_variance = ss.ldagl.variance
        norm_rho_diff = la.norm(rho_diff)
        norm_rho_diff_conj = la.norm(rho_diff_conj)
        trace_norm = np.trace(sqrtm(np.matmul(rho_diff.getH(), rho_diff))).real
        trace_norm_conj = np.trace(sqrtm(np.matmul(rho_diff_conj.getH(), rho_diff_conj))).real
        fidelity = np.trace(sqrtm(np.matmul(np.matmul(sqrtm(rho_exact), rho_neural), sqrtm(rho_exact)))) ** 2
        fidelity_diff = 1.0 - fidelity.real
        fidelity_conj = np.trace(sqrtm(np.matmul(np.matmul(sqrtm(rho_exact), rho_neural.getT()), sqrtm(rho_exact)))) ** 2
        fidelity_diff_conj = 1.0 - fidelity_conj.real
        print(f"iteration = {it}")
        print(f"ldagl = {str(ss.ldagl)}")
        print(f"norm_rho_diff = {norm_rho_diff}")
        print(f"norm_rho_diff_conj = {norm_rho_diff_conj}")
        print(f"trace_norm = {trace_norm}")
        print(f"trace_norm_conj = {trace_norm_conj}")
        print(f"fidelity_diff = {fidelity_diff}")
        print(f"fidelity_diff_conj = {fidelity_diff_conj}")

        if ldagl_mean_best > ldagl_mean:
            iteration_best = it
            ldagl_mean_best = ldagl_mean
            norm_rho_diff_best = norm_rho_diff
            norm_rho_diff_conj_best = norm_rho_diff_conj
            rho_neural_best = rho_neural
            trace_norm_best = trace_norm
            trace_norm_conj_best = trace_norm_conj
            fidelity_diff_best = fidelity_diff
            fidelity_diff_conj_best = fidelity_diff_conj

    np.save(f"rho_exact_{seed}.npy", rho_exact)
    np.save(f"rho_neural_{seed}.npy", rho_neural_best)

    metrics_dict = {'metrics': ['iteration_best', 'ldagl_mean', 'norm_rho_diff', 'norm_rho_diff_conj', 'trace_norm', 'trace_norm_conj', 'fidelity_diff', 'fidelity_diff_conj']}
    metrics_dict['values'] = [iteration_best, ldagl_mean_best, norm_rho_diff_best, norm_rho_diff_conj_best, trace_norm_best, trace_norm_conj_best, fidelity_diff_best, fidelity_diff_conj_best]
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.set_index('metrics', inplace=True)
    metrics_df.to_excel(f"metrics_{seed}.xlsx", index=True)
