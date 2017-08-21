import sde_solve as ss 
from JC_utils import tanh_updown, tanh_updown_dot

README = """
    I'd like to compare analytical solutions to numerical ones in the
    same way that's done in the numerical exercises in Kloeden/Platen
    '95. 
    My problem isn't analytically soluble, though, so I'll have to use
    two; one which is scalar/nonautonomous, the other 
    vector/autonomous. 
    I'll try to draw as many functions as possible from the actual
    problem I want to solve, so that problems with the solver that
    really exist will (with luck) show up in the test cases.

    I'm going to stick, for now, to testing the linear Taylor scheme
    that I came up with most recently.
"""

def analytical_sol_sn(times, dWs, a_func, a_dot_func, ):
    """
    analytical solution for scalar non-autonomous problem. 
    Problem is:
    dX = a(t) X dt + b X dW (4.4.10 in KP95, special case)
    Solution is:
    X_t = X_0 exp[ int_0^t a(s)ds - 1/2 b^2 t + bW ]

    """
    pass

if __name__ == '__main__':
    x_0 = 1.
    main()