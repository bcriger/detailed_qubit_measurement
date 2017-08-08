import homodyne_sim as hs
import JC_utils as jc
import numpy as np
import qutip as qt
import scipy as scp
import cProfile as prof

"""
The purpose of this file is to simulate one-qubit measurement in the
presence of confounding factors:
 + high power
 + counter-rotating terms
 + additional bus resonators coupled to the qubit being measured
 + additional qubits coupled to those bus resonators 
 + (maybe) additional qubits coupled to the readout resonator 
"""

tau = 0.6
steps = 2e4
t_on = 0.02 
t_off = 0.7
sigma = 0.01

gamma_1 = (1000. * tau) ** -1. #ridic low, just trying for stability
gamma_phi = (1000. * tau) ** -1. #ridic low, just trying for stability

delta = 2. * np.pi * 1600.
g = 2. * np.pi * 50.
kappa = 2. * np.pi * 5.
omega_cav = 2. * np.pi * 7400.

n_c = 60 #number of cavity states
n_q = 1

plus_state = qt.Qobj( np.ones( (2, 2) ) ) / 2.

zero_state = qt.fock_dm(2, 0)
one_state = qt.fock_dm(2, 1)
vacuum = qt.fock_dm(n_c, 0)

zero_ket = qt.fock(2, 0)
one_ket = qt.fock(2, 1)
vacuum_ket = qt.fock(n_c, 0)

sz = jc.s_z(n_c, n_q, 0)
rest_ham = delta * ( -sz / 2.)
a = jc.ann(n_c, n_q)
a_dag = a.dag()
sp = jc.s_p(n_c, n_q, 0)
sm = sp.dag()
jc_ham = g * (a * sp + a_dag * sm)
rabi_ham = [
            rest_ham + jc_ham,
            [g * a_dag * sp, 'exp(2 * 1j * w * t)'],
            [g * a * sm, 'exp(-2 * 1j * w * t)']
            ]
pulse_ham = a + a_dag

rho0 = qt.tensor(vacuum, zero_state)
rho1 = qt.tensor(vacuum, one_state)

# Stueckelberg Angles & Dressed States
theta = scp.arctan(2. * g / delta) / 2.
dressed_1_ket = scp.cos(theta) * qt.tensor(vacuum_ket, zero_ket) + \
                scp.sin(theta) * qt.tensor(qt.fock(n_c, 1), one_ket)
dressed_1 = dressed_1_ket * dressed_1_ket.dag()

c_lst = [np.sqrt(kappa) * a, jc.gamma_1 * sm, jc.gamma_phi * sz]

e_lst = [0.5 * pulse_ham, 0.5j * (a_dag - a),
            jc.num(n_c, n_q), jc.s_x(n_c, n_q, 0), jc.s_y(n_c, n_q, 0),
            jc.s_z(n_c, n_q, 0), jc.ident(n_c, n_q)]

e_lst_sm = [e_lst[0]]

ham_off = rest_ham + jc_ham 
opts_on = qt.Options(store_final_state=True)

pulse_fun = lambda t, amp: jc.tanh_updown(t, amp, jc.sigma,
                                            5. * jc.sigma,
                                            jc.tau / 2. + 5. * jc.sigma)
times = np.linspace(0., jc.tau, jc.steps)

if __name__ == '__main__':
    e_vecs = [qt.operator_to_vector(e_op).data.H for e_op in e_lst]
    e_cb = lambda rho: jc.expectation_cb(e_vecs, rho)
    
    sim_dict = {'ham_off' : ham_off, 'pulse_ham' : pulse_ham,
                'pulse_fun': lambda t: pulse_fun(t, 480.), 'c_ops': c_lst,
                'cb_func': jc.five_checks, 'meas_op': np.sqrt(kappa) * a, 'rho_init': qt.operator_to_vector(dressed_1).data.todense(),
                'times': times, 'n_traj' : 1}

    sme_list = jc.sme_trajectories(**sim_dict)
    
    pass