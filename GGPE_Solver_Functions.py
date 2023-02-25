import numpy as np
import scipy
import numba
import matplotlib.pyplot as plt

from scipy.linalg import solve
from scipy.linalg import lu
from tqdm import tqdm


def magnetization(A_vec):
    '''For given vector compute mangetization'''
    conjugate_A = np.conjugate(A_vec)
    M_list = [conjugate_A[i] * InitialA[i + 1] for i in range(0, 2 * kMax)]
    M_list = np.array(M_list)
    M = np.sum(M_list)
    return M


def pseudo_hamiltonian(chi, A_vec, kMax):
    '''Construct tridiagonal matrix'''
    from scipy.sparse import diags
    KineticTerm = chi ** 2 / 2 * np.arange(-kMax, kMax + 1, 1) ** 2
    Mag = magnetization(A_vec) * np.ones(len(A_vec) - 1)
    ConjMag = np.conjugate(Mag) * np.ones(len(A_vec) - 1)
    H_matrix = scipy.sparse.diags([KineticTerm, -1 / 2 * Mag, -1 / 2 * ConjMag], [0, -1, 1]).toarray()
    return H_matrix


def evolution_operator(chi, A_vec, kMax, dT):
    '''Construct evolution opertator'''
    H_matrix = pseudo_hamiltonian(chi, A_vec, kMax)
    U = np.identity(len(H_matrix)) - 1j * dT * H_matrix
    return U


def equation_solver(A_prev, A_prev_T):
    '''Construct operator in LHS that acts on unknown vector A and known RHS'''
    left_U = evolution_operator(chi, A_prev, kMax, -dT / 2)
    right_U = evolution_operator(chi, A_prev_T, kMax, dT / 2)
    lhs = left_U
    rhs = np.dot(right_U, A_prev_T)
    # LHS_L, LHS_U = lu(LHS, permute_l=True)
    # U_A = solve(LHS_L, RHS)
    A = solve(lhs, rhs)

    return A


def time_evolution(InitialA, Tmax):
    '''Perform evolutioon from initial vector InitialA until Tmax will be achieved'''
    U0 = evolution_operator(chi, InitialA, kMax, dT)
    A_prev = np.dot(U0, InitialA)
    A_prev_T = InitialA
    mode_vals = []
    for _ in tqdm(np.arange(dT, Tmax + dT, dT)):
        RightU_T = evolution_operator(chi, A_prev_T, kMax, dT / 2)
        RHS_T = np.dot(RightU_T, A_prev_T)
        while True:
            ANewApprox = equation_solver(A_prev, A_prev_T)
            LeftU_New = evolution_operator(chi, ANewApprox, kMax, -dT / 2)
            LHS_New = np.dot(LeftU_New, ANewApprox)
            Residual = LHS_New - RHS_T
            if np.linalg.norm(Residual) / np.linalg.norm(A_prev_T) < 10 ** (-7):
                A_prev = ANewApprox
                break
        A_prev_T = A_prev
        mode_vals.append(A_prev_T)
    return np.array(mode_vals)
