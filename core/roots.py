import numpy as np

def newton(g, x0, tol=1e-10, rtol_resid=1e-14, max_iter=50, dx=None, jac=None, trace=None):
    # 内部一律按 1 维数组算: 标量就是 n = 1, J 是 1x1, solve 等价于除法
    scalar = np.ndim(x0) == 0
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n = x.size

    def _out(v):
        """内部数组 -> 对外类型（跟随 x0)。副本, 防止调用方拿到后被后续迭代改掉。"""
        return float(v[0]) if scalar else v.copy()

    def _g(v):
        """按 x0 的类型调用用户的 g, 结果转成 1 维数组。"""
        return np.atleast_1d(np.asarray(g(_out(v)), dtype=float))

    def _jacobian(v, gv):
        """J[i, j] = ∂g_i/∂x_j。gv 是 v 处已算好的 g, 差分时复用, 不再多调一次。"""
        if jac is not None:
            return np.asarray(jac(_out(v)), dtype=float).reshape(n, n)

        J = np.empty((n, n))
        for j in range(n):
            delta = 1e-8 * max(1.0, abs(v[j])) if dx is None else dx
            v_pert = v.copy()                  # 扰动副本, 不动当前迭代点
            v_pert[j] += delta
            J[:, j] = (_g(v_pert) - gv) / delta   # 扰动第 j 个分量 -> 第 j 列
        return J

    gx = _g(x)                                 # x0 处只算这一次, 同时当 scale
    scale = np.max(np.abs(gx))
    if trace is not None:
        trace.append(_out(x))

    for i in range(1, max_iter + 1):
        if not (np.all(np.isfinite(gx)) and np.all(np.isfinite(x))):
            raise RuntimeError(f"Newton 第 {i} 轮出现 nan/inf: x={_out(x)!r}, g={gx!r}。迭代发散。")

        # 残差判据: 查的是当前点, 返回的也是当前点, 此时已做了 i-1 次更新
        if np.all(gx == 0.0) or np.max(np.abs(gx)) <= rtol_resid * scale:
            return _out(x), i - 1

        J = _jacobian(x, gx)
        try:
            step = np.linalg.solve(J, -gx)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"Newton 第 {i} 轮 Jacobian 奇异: x={_out(x)!r}, J={J!r}。"
                f" 导数为零或初值落在平台上, 换初值。"
            ) from None

        x = x + step
        if trace is not None:
            trace.append(_out(x))

        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"Newton 第 {i} 轮更新后出现 nan/inf: x={_out(x)!r}。迭代发散。")

        # 步长判据: |Δx| 近似等于更新前那一点的误差, 新点误差更小
        if np.max(np.abs(step)) <= tol:
            return _out(x), i

        gx = _g(x)                             # 新点上算一次, 下一轮开头直接用

    raise RuntimeError(
        f"Newton 迭代 {max_iter} 次仍未收敛: 当前 x={_out(x)!r}, max|g|={np.max(np.abs(gx)):.3e}。"
        f" 检查初值是否离根太远, 或方程是否无实根。"
    )