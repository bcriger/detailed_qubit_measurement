import qutip as qt, numpy as np
from scipy.sparse.linalg import expm_multiply
import scipy as scp
import scipy.constants as cons
import sde_solve as ss
import cProfile as prof

#------------------------------convenience functions------------------#
def _updown_sigmas(sigma):
    #assert not isinstance(sigma, basestring)
    if hasattr(sigma, '__iter__'):
        sigma_tpl = sigma
    else:
        sigma_tpl = (sigma, sigma)
    return sigma_tpl

eye, x, y, z = qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()

# Different definition
p, m = qt.sigmam(), qt.sigmap()

cav_op = lambda op, nq: qt.tensor(op, *(qt.identity(2) for _ in range(nq)))

num = lambda nc, nq: cav_op(qt.num(nc), nq)
ident = lambda nc, nq: cav_op(qt.identity(nc), nq)
ann = lambda nc, nq: cav_op(qt.destroy(nc), nq)

def qub_op(op, nc, nq, dx):
    """
    Embeds a qubit operator into a register of qubits which come after
    a cavity.
    """
    tens_lst = [qt.identity(nc)]
    
    for pos in range(dx):
        tens_lst.append(qt.identity(2))
    
    tens_lst.append(op)
    
    for pos in range(nq - dx - 1):
        tens_lst.append(qt.identity(2))
    
    return qt.tensor(*tens_lst)

s_x = lambda nc, nq, dx: qub_op(x, nc, nq, dx)
s_y = lambda nc, nq, dx: qub_op(y, nc, nq, dx)
s_z = lambda nc, nq, dx: qub_op(z, nc, nq, dx)
s_p = lambda nc, nq, dx: qub_op(p, nc, nq, dx)
s_m = lambda nc, nq, dx: qub_op(m, nc, nq, dx)

def vec2mat(vec, sparse=False):
    """
    QuTiP complains about calling len on a sparse matrix.
    I use shape[0] here.
    I also convert to dense.
    """
    n = int(np.sqrt(vec.shape[0]))
    if sparse:
        return vec.todense().reshape((n, n)).T
    else:
        return vec.reshape((n, n)).T


#---------------------------------------------------------------------#

#-----------------------------constants-------------------------------#

units = 'MHz'
assumptions = ['measurement exactly on resonance with cavity',
                'hbar = 1']

#'''
# Dickel Params
# delta = 2. * np.pi * 1000.  #MHz
# g =  2. * np.pi * 50.0  # since chi = g^2/delta, and chi = 2pi * 5 MHz
# kappa = 2. * np.pi * 5. 

#equation 3.5 and our definition differ by a factor of sqrt(kappa)
# amp = 2. * np.pi * 1. #/ np.sqrt(kappa)
# amp = 0.
# t_1 = 7. #microseconds, includes purcell
# t_2 = 0.5 #microseconds
# gamma_1 = 1. / t_1
# gamma_phi = 1. / t_2 - gamma_1 / 2.

#Poletto Surface-7 Params

def tanh_updown(t, e_ss, sigma, t_on, t_off):
    sigma_up, sigma_down = _updown_sigmas(sigma)
    return e_ss / 2. * (np.tanh((t - t_on) / sigma_up) - 
                            np.tanh((t - t_off) / sigma_down))

def tanh_updown_dot(t, e_ss, sigma, t_on, t_off):
    sigma_up, sigma_down = _updown_sigmas(sigma)
    assert sigma_up == sigma_down
    sigma = sigma_up
    shape = np.cosh((t - t_on) / sigma_up) ** -2. -\
             np.cosh((t - t_off) / sigma_down) ** -2. 
    return e_ss / (2. * sigma) * shape

# pulse = lambda t, args: tanh_updown(t, amp, sigma, t_on, t_off) 
# times = np.linspace(0., tau / 2., steps)

#-----------------lil ass simulation method---------------------------#

def on_off_sim(ham_off, pulse_ham, c_lst, amp, init_states,
                time, steps, qutip=False, args=None):
    """
    For a piecewise-constant Lindbladian, you can use
    scipy.sparse.linalg.expm_multiply to compute the action of exp(Lt)
    on vec(rho) at various times. I'm going to try to use this to
    simulate measurements.
    """
    out_lst = []
    
    if qutip:
        options = qt.Options(store_states=True, num_cpus=3)
        ham_on = [ham_off[0] + amp * pulse_ham] + ham_off[1:]
        times = np.linspace(0., time/2., steps)
        for state in init_states:
            sol_on = qt.mesolve(ham_on, state, times, c_lst, options=options, args=args).states
            sol_off = qt.mesolve(ham_off, sol_on[-1], times, c_lst, options=options, args=args).states
            out_lst.append([qt.mat2vec(state.data.todense()) for state in sol_on + sol_off])
    else:
        ham_on = ham_off + amp * pulse_ham

        lind_on = qt.liouvillian(ham_on, c_lst).data
        lind_off = qt.liouvillian(ham_off, c_lst).data

        for state in init_states:
            sol_on = expm_multiply(lind_on, qt.mat2vec(state.data.todense()),
                                    0., time/2., steps, True)
            sol_off = expm_multiply(lind_off, sol_on[-1],
                                    0., time/2., steps, True)
            out_lst.append(np.vstack([sol_on, sol_off]).copy())
        
    return out_lst

#---------------------------------------------------------------------#

#------------------------other simulation methods---------------------#

def five_checks(rho_input, is_mat=False):
    """
    Calculates the sanity checks for a density matrix:
    + min eigenvalue
    + max eigenvalue
    + trace
    + purity
    + deviation from hermiticity
    """
    rho = rho_input if is_mat else vec2mat(rho_input)
    herm_dev = np.amax(np.abs(rho - rho.conj().T))
    eigs = scp.linalg.eigvals(rho)
    tr = sum(eigs)
    output = np.array([np.amin(eigs) / tr, np.amax(eigs) / tr, tr / tr,
                                sum(eigs**2) / tr**2, herm_dev / tr])
    return output

def expectation_cb(vec_e_ops, rho_vec):
    dim = int(np.sqrt(rho_vec.shape[0]))
    tr_rho = sum(rho_vec[_] for _ in range(0, dim ** 2, dim + 1))
    return [(e_vec * rho_vec) / tr_rho for e_vec in vec_e_ops]

def trace_row(dim):
    """
    Returns a vector that, when you take the inner product with a 
    column-stacked density matrix, gives the trace of that matrix.
    """
    data = np.ones(dim)
    row_ind = np.zeros(dim)
    col_ind = np.arange(0, dim ** 2, dim + 1)
    return scp.sparse.csr_matrix((data, (row_ind, col_ind)),
                                            shape=(1, dim ** 2))

def sme_trajectories(ham_off, pulse_ham, pulse_fun, pulse_dot_fun,
                        c_ops, cb_func, meas_op, rho_init, times,
                        n_traj):
    """
    To visualise individual trajectories, and see some quantum
    jumps/latching/etc., we use sde_solve.
    """

    lind_off = qt.liouvillian(ham_off, c_ops).data
    lind_pulse = qt.liouvillian(pulse_ham, []).data
    
    dim = int(np.sqrt(rho_init.shape[0]))
    sp_id = scp.sparse.identity(dim, format='csr')
    lin_meas_op = scp.sparse.kron(sp_id, meas_op.data) + \
                    scp.sparse.kron(meas_op.data.conj(), sp_id)
    lin_meas_op_sq = lin_meas_op ** 2
    l1_lin_meas_op = 0.5 * lin_meas_op_sq # see Jacobs eq 6.17

    def det_fun(t, rho):
        return (lind_off + pulse_fun(t) * lind_pulse) * rho

    def stoc_fun(t, rho):
        return lin_meas_op * rho # linear SME
    
    def l1_stoc_fun(t, rho):
        return l1_lin_meas_op * rho 

    save_data = [[] for _ in range(n_traj)]

    for _ in range(n_traj):
        rho = rho_init.copy()
        for t_dx, t in enumerate(times[:-1]):
            t_now, t_fut = t, times[t_dx + 1]
            dt = t_fut - t_now
            mat_now = lind_off + pulse_fun(t_now) * lind_pulse
            mat_fut = lind_off + pulse_fun(t_fut) * lind_pulse
            d_mat_now = pulse_dot_fun(t_now) * lind_pulse
            save_data[_].append(cb_func(rho))
            dW = np.sqrt(dt) * np.random.randn()
            rho = ss.im_platen_15_step(t, rho, mat_now, mat_fut, stoc_fun,
                                        dt, dW, alpha = 0.5)
            # rho = ss.platen_15_step(t, rho, det_fun, stoc_fun, dt, dW)
            # rho = ss.im_milstein_1_step(t, rho, mat_now, mat_fut,
                                        # stoc_fun, dt, dW, l1_stoc_fun,
                                        # alpha=1.)
            # rho = ss.lin_taylor_15_step(t, rho, mat_now, lin_meas_op,
            #                             d_mat_now, lin_meas_op_sq,
            #                             dt, dW)
        save_data[_].append(cb_func(rho))
    return save_data


#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
