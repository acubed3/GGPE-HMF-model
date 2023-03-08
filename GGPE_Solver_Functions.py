import numpy as np
import scipy
import numba
import matplotlib.pyplot as plt

from scipy.linalg import solve
from scipy.linalg import lu
from tqdm import tqdm


def magnetization(a_vec):
    """For given vector compute magnetization"""
    conj_a_vec = np.conjugate(a_vec)
    roll_a_vec = np.roll(a_vec, -1, axis=0)
    roll_a_vec[-1] = 0
    m_vec = conj_a_vec * roll_a_vec
    m = np.sum(m_vec)
    return m


def pseudo_hamiltonian(chi, a_vec, k_max):
    """Construct tridiagonal matrix"""
    from scipy.sparse import diags
    kinetic_term = chi ** 2 / 2 * np.arange(-k_max, k_max + 1, 1) ** 2
    mag = magnetization(a_vec) * np.ones(len(a_vec) - 1)
    conj_mag = np.conjugate(mag) * np.ones(len(a_vec) - 1)
    h_matrix = scipy.sparse.diags([kinetic_term, -1 / 2 * mag, -1 / 2 * conj_mag], [0, -1, 1]).toarray()
    return h_matrix


def evolution_operator(chi, a_vec, k_max, t_step):
    """Construct evolution operator"""
    h_matrix = pseudo_hamiltonian(chi, a_vec, k_max)
    evolution_matrix = np.identity(len(h_matrix)) - 1j * t_step * h_matrix
    return evolution_matrix


def equation_solver(a_prev, a_prev_t):
    """Construct operator in LHS that acts on unknown vector A and known RHS"""
    left_u = evolution_operator(chi, a_prev, k_max, -t_step / 2)
    right_u = evolution_operator(chi, a_prev, k_max, t_step / 2)
    lhs = left_u
    rhs = np.dot(right_u, a_prev_t)
    lhs_l, lhs_u = lu(lhs, permute_l=True)
    u_a = solve(lhs_l, rhs)
    a = solve(lhs, rhs)
    return a


def converge_approx(a_prev, a_prev_t):
    while True:
        a_new_approx = a_prev
        a_prev = equation_solver(a_prev, a_prev_t)
        if np.allclose(a_new_approx, a_prev, rtol=1e-7):
            a_new_t = a_prev
            break
    return a_new_t


def time_evolution(initial_a, t_max):
    '''Perform evolutioon from initial vector InitialA until Tmax will be achieved'''
    u0 = evolution_operator(chi, initial_a, k_max, t_step)
    a_prev = np.dot(u0, initial_a)
    a_prev_t = initial_a
    mode_vals = []
    for i in tqdm(np.arange(t_step, t_max + t_step, t_step)):
        #         #RightU_T = EvolutionOperator(chi, A_prev_T, kMax, dT/2)
        #         #RHS_T = np.dot(RightU_T, A_prev_T)
        #         while True:
        #             A_new_approx = A_prev
        #             A_prev = EquationSolver(A_prev, A_prev_T)
        #             #LeftU_New = EvolutionOperator(chi, A_prev, kMax, -dT/2)
        #             #LHS_New = np.dot(LeftU_New, A_prev)
        #             #Residual = LHS_New - RHS_T
        #             if np.allclose(A_new_approx, A_prev, rtol=1e-7):
        #                 A_prev_T = A_prev
        #                 break
        a_prev_t = converge_approx(a_prev, a_prev_t)
        mode_vals.append(a_prev_t)
        u0 = evolution_operator(chi, a_prev_t, k_max, t_step)
        a_prev = np.dot(u0, a_prev_t)
    return np.array(mode_vals)
