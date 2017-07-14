import JC_utils as jc
import numpy as np
import qutip as qt
import scipy as scp
import scipy.constants as cons
import sde_solve as ss

"""
The purpose of this file is to simulate one-qubit measurement in the
presence of confounding factors:
 + high power
 + counter-rotating terms
 + additional bus resonators coupled to the qubit being measured
 + additional qubits coupled to those bus resonators 
 + (maybe) additional qubits coupled to the readout resonator 
"""

delta = 2. * np.pi * 1600.
g = 2. * np.pi * 50.
kappa = 2. * np.pi * 5.
omega_cav = 2. * np.pi * 7400.

n_c = 10 #number of cavity states
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
# theta = scp.arctan(-2. * g / delta) / 2.
theta = scp.arctan(2. * g / delta) / 2. #guess to avoid negative eigenvalue
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

def hi_power_check(amps, test_rhos, check_ops):
    """
    This function runs some one-qubit, one-mode ME simulations to look
    at traces of the qubit 0 and 1
    """
    big_list = []
    
    for amp in amps:
        state_histories = jc.on_off_sim(ham_off, pulse_ham, c_lst, amp,
                                            test_rhos, jc.tau, jc.steps)
        trace_histories = []
        
        for test_rho, state_hist in zip(test_rhos, state_histories):
            vec_rho = qt.operator_to_vector(test_rho)
            trace_hist = [np.inner(vec_rho.data.todense().T, state.T)[0,0] for state in state_hist]
            trace_histories.append(trace_hist)

        for state_hist in state_histories:
            for op in check_ops:
                vec_op = qt.operator_to_vector(op)
                trace_hist = [np.inner(vec_op.data.todense().T, state.T)[0,0] for state in state_hist]
                trace_histories.append(trace_hist)

        big_list.append(trace_histories)

    return big_list

def rabi_check(amps, test_rhos, check_ops):
    """
    Have to use qutip.mesolve, since the Hamiltonian is now
    time-dependent in the cavity frame. 
    """
    big_list = []
    
    for amp in amps:
        state_histories = jc.on_off_sim(rabi_ham, pulse_ham, c_lst, amp,
                                        test_rhos, jc.tau, jc.steps,
                                        qutip=True,
                                        args={'w':omega_cav})
        trace_histories = []
        
        for test_rho, state_hist in zip(test_rhos, state_histories):
            vec_rho = qt.operator_to_vector(test_rho)
            trace_hist = [np.inner(vec_rho.data.todense().T, state.T)[0,0] for state in state_hist]
            trace_histories.append(trace_hist)

        for state_hist in state_histories:
            for op in check_ops:
                vec_op = qt.operator_to_vector(op)
                trace_hist = [np.inner(vec_op.data.todense().T, state.T)[0,0] for state in state_hist]
                trace_histories.append(trace_hist)
        
        big_list.append(trace_histories)
        # big_list.append(state_histories)

    return big_list

def five_checks(rho_vec):
    """
    Calculates the sanity checks for a density matrix:
    + min eigenvalue
    + max eigenvalue
    + trace
    + purity
    + deviation from hermiticity
    """
    rho = jc.vec2mat(rho_vec)
    herm_dev = np.amax(np.abs(rho - rho.H))
    eigs = scp.linalg.eigvals(rho)
    output = (np.amin(eigs), np.amax(eigs), sum(eigs),
                                sum(eigs**2), herm_dev)
    return output

def trace_row(dim):
    """
    Returns a vector that, when you take the inner product with a 
    column-stacked density matrix, gives the trace of that matrix.
    """
    data = np.ones(dim)
    row_ind = np.zeros(dim)
    col_ind = np.arange(0, dim**2, dim + 1)
    return scp.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(1, dim**2))

def sme_trajectories(amps, test_rhos, n_traj):
    """
    To visualise individual trajectories, and see some quantum
    jumps/latching/etc., we use sde_solve.
    """
    lind_off = qt.liouvillian(ham_off, c_lst).data.todense()
    lind_pulse = qt.liouvillian(pulse_ham, []).data.todense()
    
    dim = test_rhos[0].shape[0]
    tr = trace_row(dim).todense()
    sp_id = scp.sparse.identity(20, format='csr')
    lin_meas_op = scp.sparse.kron(sp_id, a.data.todense()) + \
                    scp.sparse.kron(a.data.conj().todense(), sp_id)
    
    times = np.linspace(0., jc.tau, jc.steps)

    def det_fun(t, rho):
        return lind_on * rho if (t < jc.tau / 2.) else lind_off * rho

    def stoc_fun(t, rho):
        lin_state = lin_meas_op * rho
        try:
            return lin_state - (tr * lin_state)[0,0] * rho
        except IndexError: #correction term is 0
            return lin_state

    save_data = [[[[] for traj in range(n_traj)]
                        for rho in test_rhos]
                            for amp in amps]
    
    for amp_dx, amp in enumerate(amps):
        pulse_fun = lambda t: jc.tanh_updown(t, amp, jc.sigma, 5.*jc.sigma, jc.tau/2. + 5. * jc.sigma)
        for rho_dx, test_rho in enumerate(test_rhos):
            rho_init = qt.operator_to_vector(test_rho).data.todense()
            for _ in range(n_traj):
                rho = rho_init
                for t_dx, t in enumerate(times[:-1]):
                    t_now, t_fut = t, times[t_dx + 1]
                    dt = t_fut - t_now
                    mat_now = lind_off + pulse_fun(t_now) * lind_pulse
                    mat_fut = lind_off + pulse_fun(t_fut) * lind_pulse
                    save_data[amp_dx][rho_dx][_].append(five_checks(rho))
                    rho = ss.im_platen_15_step(t, rho, mat_now, mat_fut, stoc_fun,
                                                dt, np.sqrt(dt) * np.random.randn())
                save_data[amp_dx][rho_dx][_].append(five_checks(rho))
    return times, save_data

if __name__ == '__main__':
    # hi_power_list = hi_power_check([10., 20., 30.], [rho0, rho1, dressed_1], [a_dag * a])
    # rabi_list = rabi_check([10., 20., 30.], [rho0, rho1, dressed_1], [a_dag * a])
    times, sme_list = sme_trajectories([30.], [dressed_1], 1)
    pass