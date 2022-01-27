import numpy as np

def nll_s(x, T, C):
    return T / 2 * np.log(x) - C / (2 * x ** 2)

def init_ab(f, T, C):
    b0 = 10
    a0 = 0.1
    while f(a0, T, C) > 0:
        a0 /= 10
    while f(b0, T, C) < 0 :
        b0 *= 2
    return a0, b0

def bisection(f, T, C, eps=1e-6):
    a0, b0 = init_ab(f, T, C)
    dif = np.inf
    while np.abs(dif) > eps:
        cur_half = (a0 + b0) / 2
        dif = f(cur_half, T, C)
        if dif > 0:
            b0 = cur_half
        else:
            a0 = cur_half
    return cur_half
