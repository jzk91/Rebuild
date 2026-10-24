import numpy as np
from math import floor
from core.roots import newton

def _time_grid(t0, tf, h):
   
    if h > tf - t0:
        return np.array([t0, tf], float)

    r = (tf - t0) / h
    m = round(r)
    scale = max(1, abs(r))

    if abs(m - r) <= 1e-9 * scale:
        N = m + 1
    else:
        N = floor(r) + 2

    t = np.empty(N)
    for i in range(N - 1):
        t[i] = t0 + i * h
    t[-1] = tf

    return t


def _setup(y0, t_span, h):
    
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"步长必须是有限的正数, 现在h = {h}，错误。")

    t0, tf = t_span
    if not np.isfinite(t0) or not np.isfinite(tf) or tf <= t0:
        raise ValueError(f"时序区间端点数值大小错误, 或者数值非有限。现在 t0 = {t0}, tf = {tf}")
    t = _time_grid(t0, tf, h)
    N = len(t)

    y = np.atleast_1d(np.asarray(y0, dtype=float)).copy()
    if y.ndim > 1:
        raise ValueError(f"y0 必须是标量或 1 维数组, 收到 shape {y.shape}")
    if y.size == 0:
        raise ValueError("y0 为空, 请检查y0的输入")
    if not np.all(np.isfinite(y)):
        raise ValueError("y0 存在非有限的元素, 请检查y0的输入是否合理")

    n = y.size
    Y = np.empty((N, n))
    Y[0] = y

    return t, Y


def _call_f(f, t, y, args, n):
    
    dy = f(t, y, *args)
    dy_array = np.asarray(dy, dtype=float)
    if dy_array.shape != (n,):
        raise ValueError(f"f 返回 shape {dy_array.shape}, 应为 ({n},)")
    return dy_array


def euler_explicit(f, y0, t_span, h, *, args=()):

    t, Y = _setup(y0, t_span, h)
    N = Y.shape[0]
    n = Y.shape[1]

    for k in range(N - 1):
        dy_array = _call_f(f, t[k], Y[k], args, n)
        Y[k + 1] = Y[k] + dy_array * (t[k + 1] - t[k])

    return t, Y


def rk4(f, y0, t_span, h, *, args=()):

    t, Y = _setup(y0, t_span, h)
    N = Y.shape[0]
    n = Y.shape[1]

    for k in range(N - 1):
        delta = t[k + 1] - t[k]
        k1 = delta * _call_f(f, t[k], Y[k], args, n)
        k2 = delta * _call_f(f, t[k] + delta / 2, Y[k] + k1 / 2, args, n)
        k3 = delta * _call_f(f, t[k] + delta / 2, Y[k] + k2 / 2, args, n)
        k4 = delta * _call_f(f, t[k] + delta, Y[k] + k3, args, n)
        Y[k + 1] = Y[k] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, Y



def euler_implicit(f, y0, t_span, h, *, args=(), tol=1e-10):

    t, Y = _setup(y0, t_span, h)
    N = Y.shape[0]
    n = Y.shape[1]

    for k in range(N - 1):
        delta = t[k + 1] - t[k]
        
        def G(y):
            return y - Y[k] - delta * _call_f(f, t[k + 1], y, args, n)
        try:
            Y[k + 1] = newton(G, Y[k], tol=tol)[0]
        except RuntimeError as e:
            raise RuntimeError(
                f"隐式 Euler 第 {k} 步 (t = {t[k]:.6g} -> {t[k + 1]:.6g}, Δ = {delta:.3g}) newton 失败, "
                f"起点状态 Y[k] = {Y[k]!r}。检查 f 在这附近是否返回 nan/inf, 或减小 h。"
            ) from e

    return t, Y