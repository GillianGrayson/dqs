import netket as nk
import numpy as np
import pandas as pd
from netket.hilbert import Fock


def get_mbl_model(config: pd.DataFrame, seed: int,):
    np.random.seed(seed)

    N = int(config.at[0, 'N'])
    W = float(config.at[0, 'W'])
    U = float(config.at[0, 'U'])
    J = float(config.at[0, 'J'])
    diss_type = int(config.at[0, 'diss_type'])
    diss_gamma = float(config.at[0, 'diss_gamma'])

    alpha = float(config.at[0, 'alpha'])
    beta = float(config.at[0, 'beta'])
    n_samples = int(config.at[0, 'n_samples'])

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
        ha -= J * (nk.operator.boson.create(hi, boson_id) * nk.operator.boson.destroy(hi,
                                                                                      boson_id + 1) + nk.operator.boson.create(
            hi, boson_id + 1) * nk.operator.boson.destroy(hi, boson_id))
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

    return lind, ss