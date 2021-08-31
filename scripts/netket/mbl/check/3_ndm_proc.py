import netket as nk
import numpy as np
from numpy import linalg as la
import pandas as pd
import os
from netket.hilbert import Fock
from tqdm import tqdm

# Model params
model = 'mbl'
N = 8
seed = 3
W = 15.0
U = 1.0
J = 1.0
dt = 1
gamma = 0.1

# Ansatz params
beta = 2
alpha = 2
n_samples = 5000
n_iter = 1000

np.random.seed(seed)

cpp_path = f"/media/sf_Work/dl/netket/{model}/test/cpp"
save_path = f"/media/sf_Work/dl/netket/{model}/N({N})_rnd({seed})_H({W:0.4f}_{U:0.4f}_{J:0.4f})_D({dt}_{gamma:0.4f})"
if not os.path.exists(f"{save_path}"):
    os.makedirs(f"{save_path}")

energies = np.random.uniform(-1.0, 1.0, N)

# Hilbert space
hi = Fock(n_max=1, n_particles=N//2, N=N)

# The Hamiltonian
ha = nk.operator.LocalOperator(hi)

# List of dissipative jump operators
j_ops = []

for boson_id in range(N - 1):
    ha += W * energies[boson_id] * nk.operator.boson.number(hi, boson_id)
    ha += U * nk.operator.boson.number(hi, boson_id) * nk.operator.boson.number(hi, boson_id + 1)
    ha -= J * (nk.operator.boson.create(hi, boson_id) * nk.operator.boson.destroy(hi, boson_id + 1) + nk.operator.boson.create(hi, boson_id + 1) * nk.operator.boson.destroy(hi, boson_id))
    if dt == 1:
        A = (nk.operator.boson.create(hi, boson_id + 1) + nk.operator.boson.create(hi, boson_id)) * (nk.operator.boson.destroy(hi, boson_id + 1) - nk.operator.boson.destroy(hi, boson_id))
    elif dt == 0:
        A = nk.operator.boson.create(hi, boson_id) * nk.operator.boson.destroy(hi, boson_id)
    j_ops.append(np.sqrt(gamma) * A)
ha += W * energies[N - 1] * nk.operator.boson.number(hi, N - 1)
if dt == 0:
    A = nk.operator.boson.create(hi, N - 1) * nk.operator.boson.destroy(hi, N - 1)
    j_ops.append(np.sqrt(gamma) * A)

# Create the Liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)

# Neural quantum state model: Positive-Definite Neural Density Matrix using the ansatz from Torlai and Melko
ndm = nk.models.NDM(
    alpha=alpha,
    beta=beta,
)

# Metropolis Sampling
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

metrics_dict = {
    'iteration': np.linspace(1, n_iter, n_iter),
    'ldagl_mean': [],
    'ldagl_error_of_mean': [],
    'norm_rho_diff': [],
    'norm_rho_diff_conj': []
}

# Calculate exact rho
rho_exact = nk.exact.steady_state(lind, method="iterative", sparse=True, tol=1e-10)

for it in tqdm(range(n_iter)):
    out = ss.run(n_iter=1)
    metrics_dict['ldagl_mean'].append(ss.ldagl.mean)
    metrics_dict['ldagl_error_of_mean'].append(ss.ldagl.error_of_mean)
    rho_neural = np.array(ss.state.to_matrix())
    rho_diff = rho_exact - rho_neural
    rho_diff_conj = rho_exact - rho_neural.conjugate()
    metrics_dict['norm_rho_diff'].append(la.norm(rho_diff))
    metrics_dict['norm_rho_diff_conj'].append(la.norm(rho_diff_conj))

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_excel(f"{save_path}/NDM({alpha}_{beta}_{n_samples}_{n_iter}).xlsx", index=False)
