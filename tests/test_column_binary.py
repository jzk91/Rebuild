"""models/column_binary.py 的验收测试 (书 §5.4)。

结构:
  1. 参数 == Table 5.7 的 DATA
  2. equilibrium (Eq 5.31)
  3. pack / unpack: 状态布局与下标
  4. outputs: 水力学 Eq 5.32、两个 PI、D 与 B
  5. f: 契约
  6. f: 任意状态下的守恒恒等式与"只影响相邻级" —— 查下标错一格
  7. 稳态: 降维求根、与 DATA 初始分布、与长时间积分交叉核对
  8. 复现 Table 5.7 (显式 Euler, DELTA = 0.005 min)
  9. 闭环与开环、显式 Euler 的稳定上限

本文件里的 `_y()` 按模块 docstring 的布局直接拼状态向量, 不经过 `pack`:
f 与 outputs 的测试不依赖 pack 写得对不对, pack 本身单独测。

python -m pytest tests/test_column_binary.py
"""

import dataclasses
from dataclasses import replace

import numpy as np
import pytest

from core.integrators import euler_explicit, euler_implicit, rk4
from models.column_binary import (ColumnBinaryParams, ColumnOutputs, equilibrium, f,
                                  outputs, pack, steady_state, unpack)

P = ColumnBinaryParams()                  # 阶跃后: z = 0.55
P50 = replace(P, z=0.50)                  # 阶跃前: DATA 初始分布对应的进料
PO = replace(P, KcD=0.0, KcB=0.0)         # 开环: R ≡ R0, V ≡ V0
NT, NF = P.NT, P.NF
N_STATE = 2 * NT + 4

# Table 5.7 DATA XB, X(1..20), XD —— z = 0.50 时的稳态, 印到 5 位 (300/500 dpi 重抄过)
DATA_X = np.array([.02,
                   .035, .05719, .08885, .1318, .18622, .24951, .31618, .37948, .43391, .47688,
                   .51526, .56295, .61896, .68052, .74345, .80319, .85603, .89995, .93458, .96079,
                   .98])


def _y(x, M_trays, erintD=0.0, erintB=0.0, p=P):
    """按模块 docstring 的布局直接拼 y (不调 pack)。x 长 NT+2, M_trays 长 NT。"""
    x = np.asarray(x, dtype=float)
    M_trays = np.asarray(M_trays, dtype=float)
    M_all = np.concatenate([[p.MB], M_trays, [p.MD]])
    return np.concatenate([M_all * x, M_trays, [erintD, erintB]])


# 书上程序的初始状态: DATA 组成、每块板持液 MO、两个误差积分 0
BOOK_Y0 = _y(DATA_X, np.full(NT, P.M0))


def _random_state(seed, p=P):
    """一个与稳态无关的任意状态, 用来测"任意状态都成立"的恒等式。"""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.01, 0.99, p.NT + 2))
    M = rng.uniform(8.0, 12.0, p.NT)
    eD, eB = rng.uniform(-0.01, 0.01, 2)
    return _y(x, M, eD, eB, p)


# Table 5.7 的 20 行: TIME, XB, X10, XD, R, V
TABLE_5_7 = [
    (0.0, .02000, .47688, .98000, 128.01, 178.01),
    (0.5, .02014, .51325, .98000, 128.01, 178.16),
    (1.0, .02108, .52434, .98010, 127.90, 179.32),
    (1.5, .02218, .53030, .98035, 127.64, 181.08),
    (2.0, .02276, .53229, .98061, 127.33, 182.67),
    (2.5, .02268, .53141, .98076, 127.11, 183.69),
    (3.0, .02212, .52879, .98077, 127.02, 184.10),
    (3.5, .02132, .52560, .98065, 127.06, 183.99),
    (4.0, .02051, .52282, .98048, 127.18, 183.55),
    (4.5, .01987, .52108, .98030, 127.32, 182.98),
    (5.0, .01950, .52056, .98019, 127.41, 182.47),
    (5.5, .01939, .52104, .98014, 127.44, 182.13),
    (6.0, .01950, .52207, .98016, 127.41, 182.01),
    (6.5, .01972, .52318, .98022, 127.33, 182.07),
    (7.0, .01995, .52399, .98029, 127.24, 182.24),
    (7.5, .02012, .52433, .98034, 127.15, 182.43),
    (8.0, .02019, .52423, .98036, 127.09, 182.56),
    (8.5, .02016, .52382, .98035, 127.07, 182.60),
    (9.0, .02007, .52332, .98032, 127.06, 182.57),
    (9.5, .01997, .52290, .98028, 127.07, 182.47),
]
DELTA = 0.005


def fortran_print_steps(n_rows=20, delta=DELTA, every=0.5):
    """书上程序打印的是第几步。

    TIME、TPRINT 都是 FORTRAN REAL (单精度), TIME = TIME + DELTA 累加;
    每一步先判断 TIME ≥ TPRINT 就打印, 再做 Euler。0.005 在单精度里略小,
    前 400 步的 TIME 落在整数倍下方, 于是 0.5 / 1.0 / 1.5 / 2.0 四行其实是第 101 / 201 / 301 / 401 步
    (t = 0.505 ...), 之后累加误差变号, 2.5 以后才是整点。同 §5.2 Table 5.4 的 TIME 坑。
    """
    t32, tp32 = np.float32(0.0), np.float32(0.0)
    steps, k = [], 0
    while len(steps) < n_rows:
        if t32 >= tp32:
            steps.append(k)
            tp32 = np.float32(tp32 + np.float32(every))
        t32 = np.float32(t32 + np.float32(delta))
        k += 1
    return steps


def run_table_5_7():
    """书上程序那一次: 初值 = DATA, 参数 z = 0.55, 显式 Euler, DELTA = 0.005 min, 到 10 min。"""
    return euler_explicit(f, BOOK_Y0, (0.0, 10.0), DELTA, args=(P,))


def table_row(y, p=P):
    o = outputs(y, p)
    return o.x_L[0], o.x_L[p.NF], o.x_L[p.NT + 1], o.R, o.V


# ===========================================================================
# 1. 参数
# ===========================================================================

def test_default_params_match_table_5_7_data():
    assert (P.NT, P.NF, P.F, P.z) == (20, 10, 100.0, 0.55)
    assert (P.alpha, P.beta, P.M0, P.MD, P.MB) == (2.0, 0.1, 10.0, 100.0, 100.0)
    assert (P.R0, P.V0, P.xD_set, P.xB_set) == (128.01, 178.01, 0.98, 0.02)
    assert (P.KcD, P.KcB, P.tauD, P.tauB) == (1000.0, 1000.0, 5.0, 1.25)
    assert isinstance(P.NT, int) and isinstance(P.NF, int)


def test_params_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        P.z = 0.5


def test_book_bias_values_balance_the_column_at_z_050():
    # R0、V0 不是随便取的: D = V0 − R0 = 50, B = R0 + F − V0 = 50,
    # 而 z = 0.50 时 F·z = D·xD + B·xB 正好成立
    D, B = P.V0 - P.R0, P.R0 + P.F - P.V0
    assert (D, B) == (pytest.approx(50.0), pytest.approx(50.0))
    assert P.F * 0.50 == pytest.approx(D * P.xD_set + B * P.xB_set)


# ===========================================================================
# 2. equilibrium —— Eq 5.31
# ===========================================================================

def test_equilibrium_values():
    y = equilibrium(np.array([0.0, 0.5, 1.0]), 2.0)
    assert y == pytest.approx([0.0, 2.0 / 3.0, 1.0], abs=1e-15)


def test_equilibrium_is_vectorized_and_keeps_shape():
    x = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    y = equilibrium(x, 2.0)
    assert isinstance(y, np.ndarray) and y.shape == (3, 4) and y.dtype == np.float64
    assert equilibrium(x, 1.0) == pytest.approx(x, abs=1e-15)    # α = 1 时汽液同组成


def test_equilibrium_relative_volatility_is_constant():
    # 不抄 Eq 5.31: 相对挥发度的定义 [y/(1−y)] / [x/(1−x)] 必须处处等于 α
    x = np.linspace(0.01, 0.99, 50)
    for alpha in (1.5, 2.0, 3.7):
        y = equilibrium(x, alpha)
        assert (y / (1 - y)) / (x / (1 - x)) == pytest.approx(np.full(50, alpha), rel=1e-12)


# ===========================================================================
# 3. pack / unpack —— 布局与下标
# ===========================================================================

def test_unpack_shapes_and_types():
    x, M, eD, eB = unpack(BOOK_Y0, P)
    assert x.shape == (NT + 2,) and M.shape == (NT + 2,)
    assert x.dtype == np.float64 and M.dtype == np.float64
    assert isinstance(eD, float) and isinstance(eB, float)


def test_unpack_reads_the_documented_layout():
    y = _random_state(1)
    x, M, eD, eB = unpack(y, P)
    assert (M[0], M[NT + 1]) == (P.MB, P.MD)                    # 两端来自参数
    assert M[1:NT + 1] == pytest.approx(y[NT + 2:2 * NT + 2], abs=0)
    assert x[0] == pytest.approx(y[0] / P.MB, rel=1e-15)         # 塔釜
    assert x[5] == pytest.approx(y[5] / y[NT + 1 + 5], rel=1e-15)  # 塔板 5: M₅ 在 y[NT+1+5]
    assert x[NT + 1] == pytest.approx(y[NT + 1] / P.MD, rel=1e-15)  # 回流罐
    assert (eD, eB) == (y[2 * NT + 2], y[2 * NT + 3])


def test_unpack_of_book_data_gives_book_compositions():
    x, _, _, _ = unpack(BOOK_Y0, P)
    assert x == pytest.approx(DATA_X, abs=1e-15)
    assert x[NF] == pytest.approx(0.47688, abs=1e-15)            # Table 5.7 TIME 0 的 X10


def test_pack_writes_the_documented_layout():
    y_ref = _random_state(2)
    x, M, eD, eB = unpack(y_ref, P)
    y = pack(x, M, eD, eB, P)
    assert isinstance(y, np.ndarray) and y.shape == (N_STATE,) and y.dtype == np.float64
    assert y == pytest.approx(y_ref, rel=1e-14)


def test_pack_uses_only_tray_holdups():
    # M 两端被忽略, 塔釜/回流罐持液取参数
    M = np.concatenate([[np.nan], np.full(NT, 11.0), [np.nan]])
    y = pack(DATA_X, M, 0.1, -0.2, P)
    assert np.all(np.isfinite(y))
    assert y == pytest.approx(_y(DATA_X, np.full(NT, 11.0), 0.1, -0.2), rel=1e-15)


def test_pack_unpack_roundtrip():
    y = _random_state(3)
    assert pack(*unpack(y, P), P) == pytest.approx(y, rel=1e-14)


def test_unpack_does_not_alias_y():
    y = BOOK_Y0.copy()
    x, M, _, _ = unpack(y, P)
    x[:] = -1.0
    M[:] = -1.0
    assert np.array_equal(y, BOOK_Y0)


# ===========================================================================
# 4. outputs
# ===========================================================================

def test_outputs_types_and_shapes():
    o = outputs(BOOK_Y0, P)
    assert isinstance(o, ColumnOutputs)
    assert o.x_L.shape == (NT + 2,) and o.M.shape == (NT + 2,) and o.L.shape == (NT + 2,)
    assert o.y_V.shape == (NT + 1,)                    # 回流罐没有汽相
    assert all(isinstance(v, float) for v in (o.R, o.V, o.D, o.B))


def test_outputs_at_book_initial_state():
    # DATA 初始状态: 持液 = MO, 误差和积分都是 0 -> 两个控制器都在偏置上
    o = outputs(BOOK_Y0, P)
    assert (o.R, o.V) == (pytest.approx(128.01, abs=1e-12), pytest.approx(178.01, abs=1e-12))
    assert (o.D, o.B) == (pytest.approx(50.0, abs=1e-10), pytest.approx(50.0, abs=1e-10))
    assert o.L[1:NF + 1] == pytest.approx(np.full(NF, 228.01), abs=1e-12)        # 进料板及以下
    assert o.L[NF + 1:NT + 1] == pytest.approx(np.full(NT - NF, 128.01), abs=1e-12)  # 进料板以上
    assert o.L[NT + 1] == o.R
    assert o.L[0] == o.B


def test_outputs_vapor_is_equilibrium_on_every_stage_including_reboiler():
    y = _random_state(4)
    o = outputs(y, P)
    assert o.y_V == pytest.approx(equilibrium(o.x_L[:NT + 1], P.alpha), rel=1e-15)


def test_outputs_hydraulics_eq_5_32():
    # 只把第 5 块板的持液加 1 mol: 只有 L[5] 变, 变化量 = 1/β
    M = np.full(NT, P.M0)
    M5 = M.copy()
    M5[4] += 1.0
    L0 = outputs(_y(DATA_X, M), P).L
    L1 = outputs(_y(DATA_X, M5), P).L
    dL = L1 - L0
    assert dL[5] == pytest.approx(1.0 / P.beta, rel=1e-10)
    assert np.all(np.delete(dL, [0, 5]) == 0.0)
    assert dL[0] == pytest.approx(0.0, abs=1e-12)   # L[0] = B = L[1] − V, 塔板 1 没动


def test_outputs_controller_directions_and_gains():
    base = outputs(BOOK_Y0, P)
    x = DATA_X.copy()
    x[NT + 1] -= 0.001                                # 塔顶变稀 -> 多回流
    o = outputs(_y(x, np.full(NT, P.M0)), P)
    assert o.R - base.R == pytest.approx(P.KcD * 0.001, rel=1e-8)
    assert o.V == base.V
    x = DATA_X.copy()
    x[0] += 0.001                                     # 塔釜变浓 -> 多蒸
    o = outputs(_y(x, np.full(NT, P.M0)), P)
    assert o.V - base.V == pytest.approx(P.KcB * 0.001, rel=1e-8)
    assert o.R == base.R


def test_outputs_integral_action():
    o = outputs(_y(DATA_X, np.full(NT, P.M0), erintD=0.01, erintB=0.01), P)
    assert o.R - P.R0 == pytest.approx(P.KcD * 0.01 / P.tauD, rel=1e-10)    # +2
    assert o.V - P.V0 == pytest.approx(-P.KcB * 0.01 / P.tauB, rel=1e-10)   # −8


def test_outputs_perfect_level_control_d_and_b():
    for seed in range(5):
        o = outputs(_random_state(seed), P)
        assert o.D == pytest.approx(o.V - o.R, rel=1e-14)       # 回流罐持液恒定
        assert o.B == pytest.approx(o.L[1] - o.V, rel=1e-14)    # 塔釜持液恒定


def test_open_loop_holds_r_and_v_at_bias():
    for seed in range(3):
        o = outputs(_random_state(seed), PO)
        assert (o.R, o.V) == (P.R0, P.V0)


# ===========================================================================
# 5. f —— 契约
# ===========================================================================

def test_f_returns_float_shape_44():
    out = f(0.0, BOOK_Y0, P)
    assert isinstance(out, np.ndarray)
    assert out.shape == (N_STATE,) and out.dtype == np.float64


def test_f_does_not_modify_y():
    y = _random_state(5)
    y_copy = y.copy()
    f(0.0, y, P)
    assert np.array_equal(y, y_copy)


def test_f_is_autonomous_and_repeatable():
    y = _random_state(6)
    ref = f(0.0, y, P)
    assert np.array_equal(f(0.0, y, P), ref)
    assert np.array_equal(f(3.7, y, P), ref)


def test_f_last_two_are_the_controller_errors():
    y = _random_state(7)
    x = y[:NT + 2] / np.concatenate([[P.MB], y[NT + 2:2 * NT + 2], [P.MD]])   # 不经过 unpack
    d = f(0.0, y, P)
    assert d[2 * NT + 2] == pytest.approx(P.xD_set - x[NT + 1], abs=1e-15)
    assert d[2 * NT + 3] == pytest.approx(P.xB_set - x[0], abs=1e-15)


def test_book_initial_profile_is_nearly_steady_at_z_050():
    # DATA 分布是 z = 0.50 的稳态, 只印到 5 位: 持液导数严格为 0, 组成导数只剩舍入量级
    d = f(0.0, BOOK_Y0, P50)
    assert np.max(np.abs(d[NT + 2:2 * NT + 2])) < 1e-10
    M_all = np.concatenate([[P.MB], np.full(NT, P.M0), [P.MD]])
    dxdt = d[:NT + 2] / M_all                    # 持液不变时 dx/dt = d(Mx)/dt / M
    assert np.max(np.abs(dxdt)) < 2e-4
    # 对照: 同一个状态, 进料已阶跃到 0.55, 进料板一上来就以 F·Δz/M = 0.5 /min 变浓
    d55 = f(0.0, BOOK_Y0, P)
    assert d55[NF] / P.M0 == pytest.approx(0.5 + dxdt[NF], abs=1e-12)


# ===========================================================================
# 6. f —— 守恒恒等式与相邻性 (查下标错一格)
# ===========================================================================

@pytest.mark.parametrize("seed", range(5))
def test_total_holdup_balance_holds_at_any_state(seed):
    y = _random_state(seed)
    d = f(0.0, y, P)
    o = outputs(y, P)
    assert np.sum(d[NT + 2:2 * NT + 2]) == pytest.approx(P.F - o.D - o.B, abs=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_light_component_balance_holds_at_any_state(seed):
    # 各级 d(Mx)/dt 求和, 内部流全部抵消, 只剩进料和两股产品。
    # 哪一级漏了一项、哪一项错了一格, 这里就不闭合
    y = _random_state(seed)
    d = f(0.0, y, P)
    o = outputs(y, P)
    assert np.sum(d[:NT + 2]) == pytest.approx(
        P.F * P.z - o.D * o.x_L[NT + 1] - o.B * o.x_L[0], abs=1e-10)


def _changed(d1, d0):
    return set(np.nonzero(np.abs(d1 - d0) > 1e-12)[0].tolist())


def test_feed_composition_enters_only_the_feed_tray():
    y = _random_state(8)
    changed = _changed(f(0.0, y, replace(P, z=0.60)), f(0.0, y, replace(P, z=0.50)))
    assert changed == {NF}


def test_tray_composition_affects_only_neighbours():
    # 开环 (R、V 不随 x 变) 下只改第 5 块板的 x: 只有第 4/5/6 块板的组分方程变
    y = _random_state(9, PO)
    y2 = y.copy()
    y2[5] *= 1.01
    assert _changed(f(0.0, y2, PO), f(0.0, y, PO)) == {4, 5, 6}


def test_tray_holdup_affects_only_its_own_and_the_tray_below():
    # 只改第 5 块板的 M (x 不变): L[5] 变, 影响第 5 块板 (流出) 和第 4 块板 (流入)
    y = _random_state(10, PO)
    x5 = y[5] / y[NT + 1 + 5]
    y2 = y.copy()
    y2[NT + 1 + 5] += 0.5
    y2[5] = y2[NT + 1 + 5] * x5
    changed = _changed(f(0.0, y2, PO), f(0.0, y, PO))
    assert changed == {4, 5, NT + 1 + 4, NT + 1 + 5}


def test_reboiler_composition_affects_reboiler_and_tray_1():
    # 塔釜的汽相 (Eq 5.31 也用在塔釜) 进第 1 块板; 另外只动 xB 的误差
    y = _random_state(11, PO)
    y2 = y.copy()
    y2[0] *= 1.01
    assert _changed(f(0.0, y2, PO), f(0.0, y, PO)) == {0, 1, 2 * NT + 3}


def test_drum_composition_affects_drum_and_top_tray():
    # 回流以 xD 进顶板; 另外只动 xD 的误差
    y = _random_state(12, PO)
    y2 = y.copy()
    y2[NT + 1] *= 0.999
    assert _changed(f(0.0, y2, PO), f(0.0, y, PO)) == {NT, NT + 1, 2 * NT + 2}


def test_drum_is_steady_when_distillate_equals_top_vapor():
    # 全凝: 回流罐组成不变, 当且仅当 xD 等于顶板汽相组成
    y = _random_state(13, PO)
    yv_top = outputs(y, PO).y_V[NT]
    y[NT + 1] = P.MD * yv_top
    assert f(0.0, y, PO)[NT + 1] == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# 7. 稳态
# ===========================================================================

def test_steady_state_shape_and_dtype():
    ss = steady_state(P)
    assert isinstance(ss, np.ndarray)
    assert ss.shape == (N_STATE,) and ss.dtype == np.float64


@pytest.mark.parametrize("p", [P, P50, replace(P, z=0.45), replace(P, KcD=200.0, tauD=2.0)],
                         ids=["z055", "z050", "z045", "retuned"])
def test_steady_state_makes_all_derivatives_vanish(p):
    assert np.max(np.abs(f(0.0, steady_state(p), p))) < 1e-9


def test_steady_state_sits_on_the_setpoints():
    for p in (P, P50):
        o = outputs(steady_state(p), p)
        assert o.x_L[0] == pytest.approx(p.xB_set, abs=1e-12)
        assert o.x_L[NT + 1] == pytest.approx(p.xD_set, abs=1e-10)


def test_steady_state_closes_the_material_balance():
    # README §5.4 的验收: F = D + B 与 F·z = D·xD + B·xB 残差到 1e-10
    o = outputs(steady_state(P), P)
    assert P.F - o.D - o.B == pytest.approx(0.0, abs=1e-10)
    assert P.F * P.z - o.D * o.x_L[NT + 1] - o.B * o.x_L[0] == pytest.approx(0.0, abs=1e-10)
    assert o.D == pytest.approx(P.F * (0.55 - 0.02) / (0.98 - 0.02), rel=1e-12)   # 55.2083


def test_steady_state_holdups_follow_constant_molal_overflow():
    # 稳态时进料板以上 L = R, 进料板及以下 L = R + F, 持液由 Eq 5.32 反推
    ss = steady_state(P)
    o = outputs(ss, P)
    assert o.L[1:NF + 1] == pytest.approx(np.full(NF, o.R + P.F), rel=1e-12)
    assert o.L[NF + 1:NT + 1] == pytest.approx(np.full(NT - NF, o.R), rel=1e-12)
    assert o.M[NF + 1] == pytest.approx(P.M0 + P.beta * (o.R - P.R0), rel=1e-12)


def test_steady_state_at_z_050_reproduces_the_data_profile():
    o = outputs(steady_state(P50), P50)
    assert o.x_L == pytest.approx(DATA_X, abs=2e-5)        # 5 位小数的舍入
    assert o.R == pytest.approx(128.01, abs=5e-3)
    assert o.V == pytest.approx(178.01, abs=5e-3)


def test_steady_state_at_z_055_values():
    o = outputs(steady_state(P), P)
    assert o.R == pytest.approx(126.7745, abs=1e-3)
    assert o.V == pytest.approx(181.9828, abs=1e-3)
    assert o.x_L[NF] == pytest.approx(0.52288, abs=1e-5)


def test_long_closed_loop_run_lands_on_the_new_steady_state():
    # 从 DATA 出发 (z 已阶跃到 0.55), RK4 积 200 min: 两端 PI 把 xD、xB 拉回 0.98 / 0.02,
    # 整个状态落到 steady_state(P) 上 (最慢的闭环模态时间常数约 10.6 min)
    _, Y = rk4(f, BOOK_Y0, (0.0, 200.0), 0.01, args=(P,))
    assert np.max(np.abs(Y[-1] - steady_state(P))) < 1e-8


def test_exact_steady_state_does_not_move():
    ss = steady_state(P)
    for integ in (euler_explicit, rk4, euler_implicit):
        _, Y = integ(f, ss, (0.0, 5.0), DELTA, args=(P,))
        assert np.max(np.abs(Y - ss)) < 1e-10, integ.__name__


# ===========================================================================
# 8. 复现 Table 5.7
# ===========================================================================

def test_fortran_print_steps():
    # 测试自身的辅助函数: 单精度 TIME 让前四行晚打一步
    steps = fortran_print_steps()
    assert steps[:6] == [0, 101, 201, 301, 401, 500]
    assert steps[-1] == 1900


def test_table_5_7_all_rows():
    t, Y = run_table_5_7()
    assert len(t) == 2001 and t[-1] == pytest.approx(10.0)
    for (t_print, XB, X10, XD, R, V), k in zip(TABLE_5_7, fortran_print_steps()):
        xb, x10, xd, r, v = table_row(Y[k])
        assert xb == pytest.approx(XB, abs=1e-5), t_print
        assert x10 == pytest.approx(X10, abs=3e-5), t_print
        assert xd == pytest.approx(XD, abs=1e-5), t_print
        assert r == pytest.approx(R, abs=0.015), t_print
        assert v == pytest.approx(V, abs=0.015), t_print


def test_table_5_7_row_at_5_min():
    # README 卡片上的那一行
    _, Y = run_table_5_7()
    xb, x10, xd, r, v = table_row(Y[1000])
    assert (xb, x10, xd) == (pytest.approx(0.01950, abs=1e-5), pytest.approx(0.52056, abs=2e-5),
                             pytest.approx(0.98019, abs=1e-5))
    assert (r, v) == (pytest.approx(127.41, abs=0.01), pytest.approx(182.47, abs=0.01))


def test_table_5_7_needs_the_single_precision_print_times():
    # 按 0.5 的整数倍取行 (第 100、200 … 步), 0.5–2.0 这几行 X10 差到 1e-4 以上:
    # 进料板在阶跃初期以约 0.05 /min 变化, 差一步 (0.005 min) 就差 2.5e-4
    _, Y = run_table_5_7()
    naive = max(abs(table_row(Y[100 * i])[1] - row[2]) for i, row in enumerate(TABLE_5_7))
    fortran = max(abs(table_row(Y[k])[1] - row[2])
                  for k, row in zip(fortran_print_steps(), TABLE_5_7))
    assert naive > 1e-4
    assert fortran < 3e-5


def test_table_5_7_holdups_barely_move():
    # 组成扰动几乎不改变液流, 塔板持液整个过程都在 MO 附近 (最低 9.90 mol)
    _, Y = run_table_5_7()
    M = Y[:, NT + 2:2 * NT + 2]
    assert M.max() <= P.M0 + 1e-12
    assert M.min() == pytest.approx(9.9014, abs=1e-3)


# ===========================================================================
# 9. 闭环与开环、稳定上限
# ===========================================================================

def test_open_loop_step_leaves_the_products_off_spec():
    # 没有 PI (R、V 固定), 同一个 z 阶跃让塔釜从 0.02 漂到 0.108, 塔顶到 0.9922
    _, Y = rk4(f, steady_state(P50), (0.0, 150.0), 0.01, args=(PO,))
    o = outputs(Y[-1], PO)
    assert o.x_L[0] == pytest.approx(0.10778, abs=1e-4)
    assert o.x_L[NT + 1] == pytest.approx(0.99222, abs=1e-4)
    assert np.max(np.abs(f(0.0, Y[-1], PO)[:2 * NT + 2])) < 1e-8


def test_explicit_euler_stability_limit():
    # 线性化后最负的特征值约 −101.7 /min, 显式 Euler 的上限 2/101.7 ≈ 0.0197 min。
    # 书上 DELTA = 0.005 是上限的四分之一
    ss = steady_state(P)
    _, Y = euler_explicit(f, steady_state(P50), (0.0, 30.0), 0.019, args=(P,))
    assert np.max(np.abs(Y[-1] - ss)) < 0.01
    with np.errstate(all="ignore"):
        _, Y = euler_explicit(f, steady_state(P50), (0.0, 30.0), 0.021, args=(P,))
    assert not np.all(np.isfinite(Y[-1])) or np.max(np.abs(Y[-1] - ss)) > 1.0