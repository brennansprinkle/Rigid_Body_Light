import numpy as np
from numba import njit, prange
from scipy import spatial


def build_neighbor_list(r_vectors, L, r_cut, eps=0.0):
    r_vectors = np.reshape(r_vectors, (-1, 3))
    r_vectors = periodize_r_vecs(r_vectors, L, np.shape(r_vectors)[0])

    r_tree = spatial.cKDTree(
        r_vectors, boxsize=L, balanced_tree=False, compact_nodes=False
    )

    pairs = r_tree.query_ball_point(
        r_vectors, r_cut, return_sorted=False, workers=1, eps=eps
    )  # eps has a large effect on performance and can affect accuracy if set incorrectly

    offsets = np.cumsum([0] + [len(p) for p in pairs], dtype=int)
    list_of_neighbors = np.fromiter(
        (item for sublist in pairs for item in sublist), dtype=int
    )
    return offsets, list_of_neighbors


@njit(parallel=True, fastmath=True)
def periodize_r_vecs(r_vecs_np, L, Nb):
    r_vecs = np.copy(r_vecs_np)
    for k in prange(Nb):
        for i in range(3):
            if L[i] > 0:
                while r_vecs[k, i] < 0:
                    r_vecs[k, i] += L[i]
                while r_vecs[k, i] > L[i]:
                    r_vecs[k, i] -= L[i]
    return r_vecs


@njit(parallel=True, fastmath=True)
def blob_blob_sterics(
    r_vectors,
    L,
    a,
    repulsion_strength,
    debye_length,
    list_of_neighbors,
    offsets,
):
    """
    The force is derived from the potential

    U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
    U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a

    with
    eps = potential strength
    r_norm = distance between blobs
    b = Debye length
    a = blob_radius
    """

    N = r_vectors.size // 3
    r_vectors = r_vectors.reshape((N, 3))
    force = np.zeros((N, 3))

    for i in prange(N):
        for kk in range(offsets[i + 1] - offsets[i]):
            j = list_of_neighbors[offsets[i] + kk]

            if i == j:
                continue

            dr = np.zeros(3)
            for k in range(3):
                dr[k] = r_vectors[j, k] - r_vectors[i, k]
                if L[k] > 0:
                    dr[k] -= (
                        int(dr[k] / L[k] + 0.5 * (int(dr[k] > 0) - int(dr[k] < 0)))
                        * L[k]
                    )

            # Compute force
            r_norm = np.sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2])

            for k in range(3):
                offset = 2.0 * a
                if r_norm > (offset):
                    force[i, k] += (
                        -(
                            (repulsion_strength / debye_length)
                            * np.exp(-(r_norm - (offset)) / debye_length)
                            / np.maximum(r_norm, 1.0e-12)
                        )
                        * dr[k]
                    )
                else:
                    force[i, k] += (
                        -(
                            (repulsion_strength / debye_length)
                            / np.maximum(r_norm, 1.0e-12)
                        )
                        * dr[k]
                    )

    return force
