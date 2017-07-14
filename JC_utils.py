import qutip as qt, numpy as np
from scipy.sparse.linalg import expm_multiply

#------------------------------convenience functions------------------#
def _updown_sigmas(sigma):
    #assert not isinstance(sigma, basestring)
    if hasattr(sigma, '__iter__'):
        sigma_tpl = sigma
    else:
        sigma_tpl = (sigma, sigma)
    return sigma_tpl

eye, x, y, z = qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()
m, p = qt.sigmam(), qt.sigmap()

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

tau = 0.6
steps = 2e5
t_on = 0.02 
t_off = 0.7
sigma = 0.01

gamma_1 = (1000. * tau) ** -1. #ridic low, just trying for stability
gamma_phi = (1000. * tau) ** -1. #ridic low, just trying for stability


def tanh_updown(t, e_ss, sigma, t_on, t_off):
    sigma_up, sigma_down = _updown_sigmas(sigma)
    return e_ss / 2. * (np.tanh((t - t_on) / sigma_up) - 
                            np.tanh((t - t_off) / sigma_down))

pulse = lambda t, args: tanh_updown(t, amp, sigma, t_on, t_off) 
times = np.linspace(0., tau / 2., steps)

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

#---------------------------------------------------------------------#
