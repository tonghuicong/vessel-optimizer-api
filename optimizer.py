# 文件名: optimizer.py

import os
import math
import csv
import numpy as np
from scipy import interpolate
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA

# ==========================================================================================
# PART 0: 规范/材料/曲线计算引擎（与原版一致）
# ==========================================================================================

# ... (这部分代码与上一版完全相同，从 Q345R_DATA 到 volumes_and_mass 函数，无需改动)
# ---------------- 材料数据库（Q345R） ----------------
Q345R_DATA = {
    'yield_strength': {
        'thickness_ranges': [(3, 16), (16, 36), (36, 60), (60, 100), (100, 150), (150, 250)],
        'temperatures': [20, 100, 150, 200, 250, 300, 350, 400, 450],
        'values': [
            [345, 315, 295, 275, 250, 230, 215, 200, 190],
            [325, 295, 275, 255, 235, 215, 200, 190, 180],
            [315, 285, 260, 240, 220, 200, 185, 175, 165],
            [305, 275, 250, 225, 205, 185, 175, 165, 155],
            [285, 260, 240, 220, 200, 180, 170, 160, 150],
            [265, 245, 230, 215, 195, 175, 165, 155, 145]
        ]
    },
    'allowable_stress': {
        'thickness_ranges': [(3, 16), (16, 36), (36, 60), (60, 100), (100, 150), (150, 250)],
        'temperatures': [20, 100, 150, 200, 250, 300, 350, 400, 425, 450, 475],
        'values': [
            [189, 189, 189, 183, 167, 153, 143, 125, 93, 66, 43],
            [185, 185, 185, 170, 157, 143, 133, 125, 93, 66, 43],
            [181, 181, 173, 160, 147, 133, 123, 117, 93, 66, 43],
            [181, 181, 167, 150, 147, 133, 117, 110, 93, 66, 43],
            [178, 173, 160, 147, 133, 120, 113, 107, 93, 66, 43],
            [174, 163, 153, 143, 130, 117, 110, 103, 93, 66, 43]
        ]
    }
}


class MaterialDB:
    @staticmethod
    def get_property(thk, T, key):
        data = Q345R_DATA[key]
        tr, temps, vals = data['thickness_ranges'], np.array(data['temperatures'], float), data['values']
        if thk < tr[0][0]:
            idx = 0
        elif thk >= tr[-1][1]:
            idx = len(tr) - 1
        else:
            idx = next(i for i, (lo, hi) in enumerate(tr) if lo <= thk < hi)
        y = np.array(vals[idx], float)
        if T <= temps.min(): return float(y[0])
        if T >= temps.max(): return float(y[-1])
        return float(interpolate.interp1d(temps, y)(T))


class ExternalPressureCalculator:
    def __init__(self, ba_dir=None):
        self.Do_t_values = np.array(
            [4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000],
            float
        )
        self.A_data, self.L_Do_data = [], []
        self.temps = [30, 200, 300, 350, 450]
        self.E_values = [2.00e5, 1.86e5, 1.69e5, 1.51e5, 1.43e5]
        self.file_names = ['30.txt', '200.txt', '300.txt', '350.txt', '450.txt']
        self.all_data = []
        self.ba_dir = ba_dir or os.getenv("GB150_BA_DIR") or os.path.dirname(__file__)
        self._gen_A_curves()
        self._load_B_files_or_raise()

    def _gen_A_curves(self):
        for m in self.Do_t_values:
            nn = 1.1 * np.sqrt(m)
            n1 = np.linspace(0.05, nn, 100)
            A1 = (1.30 * m ** (-1.5)) / (n1 - 0.45 * m ** (-0.5))
            n2 = np.linspace(nn, 50, 100)
            A2 = (1.30 * m ** (-1.5)) / (nn - 0.45 * m ** (-0.5)) * np.ones_like(n2)
            ok = (A1 > 0) & np.isfinite(A1)
            self.A_data.append(np.concatenate([A1[ok], A2]))
            self.L_Do_data.append(np.concatenate([n1[ok], n2]))

    def _load_B_files_or_raise(self):
        missing = [fn for fn in self.file_names if not os.path.exists(os.path.join(self.ba_dir, fn))]
        if missing:
            raise FileNotFoundError(f"缺少 B–A 数据文件：{', '.join(missing)}（需与脚本同目录或设 GB150_BA_DIR 环境变量）")
        for fn in self.file_names:
            d = np.loadtxt(os.path.join(self.ba_dir, fn))
            order = np.argsort(d[:, 1])
            d = d[order, :]
            _, uniq_idx = np.unique(d[:, 1], return_index=True)
            d = d[np.sort(uniq_idx), :]
            self.all_data.append(d)

    def Et(self, T):
        T = float(np.clip(T, self.temps[0], self.temps[-1]))
        return float(np.interp(T, self.temps, self.E_values))

    def A_value(self, D0, de, L):
        Do_de = np.clip(D0 / float(de), 4, 1000)
        L_Do = np.clip(L / float(D0), 0.05, 50.0)
        y_lines = [
            interpolate.interp1d(ld, ad, bounds_error=False, fill_value=(ad[0], ad[-1]))(L_Do)
            for ld, ad in zip(self.L_Do_data, self.A_data)
        ]
        return float(np.interp(Do_de, self.Do_t_values, y_lines))

    def B_value(self, A, T):
        A = max(float(A), 1e-12)
        logA = np.log10(A)
        b_at_T = [np.interp(logA, np.log10(d[:, 0]), d[:, 1]) for d in self.all_data]
        return float(np.interp(T, self.temps, b_at_T))


def required_I(D0, de, As, Ls, A):
    return (D0 ** 2) * Ls * (de + As / max(Ls, 1e-9)) * A / 10.9


def B_from_p(D0, de, As, Ls, p):
    return p * D0 / (de + As / max(Ls, 1e-9))


def effective_b(D0, de):
    return 1.1 * np.sqrt(D0 * de)


def calc_rib_Is_As(rib_type, D0, de, **kw):
    b = effective_b(D0, de)
    A1 = de * b
    I1 = b * de ** 3 / 12.0
    if rib_type == 'rect':
        ds, hs = kw['ds1'], kw['hs1']
        A2 = ds * hs
        Z1 = de + hs / 2.0
        Z2 = (A1 * (de / 2.0) + A2 * Z1) / (A1 + A2)
        I2 = ds * hs ** 3 / 12.0
        Is = I1 + I2 + (Z2 - de / 2.0) ** 2 * A1 + (Z2 - Z1) ** 2 * A2
        As = A2
    elif rib_type == 'angle':
        ds1, hs1, ds2, hs2 = kw['ds1'], kw['hs1'], kw['ds2'], kw['hs2']
        A2 = ds2 * (hs2 - ds1)
        A3 = hs1 * ds1
        Z_A2 = de + (hs2 - ds1) / 2.0
        Z_A3 = de + hs2 - ds1 / 2.0
        Z1 = (A2 * Z_A2 + A3 * Z_A3) / (A2 + A3)
        Z2 = (A1 * (de / 2.0) + (A2 + A3) * Z1) / (A1 + A2 + A3)
        I2 = ds2 * (hs2 - ds1) ** 3 / 12.0
        I3 = hs1 * ds1 ** 3 / 12.0
        Is = I1 + I2 + I3 + (Z2 - de / 2.0) ** 2 * A1 + (Z2 - Z_A2) ** 2 * A2 + (Z2 - Z_A3) ** 2 * A3
        As = A2 + A3
    elif rib_type == 'tee':
        ds1, hs1, ds2, hs2 = kw['ds1'], kw['hs1'], kw['ds2'], kw['hs2']
        A2 = ds2 * (hs2 - ds1)
        A3 = 2.0 * hs1 * ds1
        Z_A2 = de + (hs2 - ds1) / 2.0
        Z_A3 = de + hs2 - ds1 / 2.0
        Z1 = (A2 * Z_A2 + A3 * Z_A3) / (A2 + A3)
        Z2 = (A1 * (de / 2.0) + (A2 + A3) * Z1) / (A1 + A2 + A3)
        I2 = ds2 * (hs2 - ds1) ** 3 / 12.0
        I3 = 2.0 * (hs1 * ds1 ** 3 / 12.0)
        Is = I1 + I2 + I3 + (Z2 - de / 2.0) ** 2 * A1 + (Z2 - Z_A2) ** 2 * A2 + (Z2 - Z_A3) ** 2 * A3
        As = A2 + A3
    else:
        raise ValueError("未知 rib_type")
    return float(Is), float(As)


def check_slenderness(rib_type, T, E, **kw):
    slist = []
    if rib_type == 'rect':
        ds, hs = kw['ds1'], kw['hs1']
        Re = MaterialDB.get_property(max(ds, 3.0), T, 'yield_strength')
        limit = 0.375 * np.sqrt(E / Re)
        val = hs / ds
        slist.append({'name': "hs/ds", 'val': val, 'lim': limit, 'ok': val <= limit})
    elif rib_type in ('angle', 'tee'):
        ds1, hs1, ds2, hs2 = kw['ds1'], kw['hs1'], kw['ds2'], kw['hs2']
        Re1 = MaterialDB.get_property(max(ds1, 3.0), T, 'yield_strength')
        Re2 = MaterialDB.get_property(max(ds2, 3.0), T, 'yield_strength')
        lim1 = 0.375 * np.sqrt(E / Re1)
        lim2 = np.sqrt(E / Re2)
        v1 = hs1 / ds1
        v2 = (hs2 - ds1) / ds2 if rib_type == 'angle' else hs2 / ds2
        slist.append({'name': "翼缘/水平肢", 'val': v1, 'lim': lim1, 'ok': v1 <= lim1})
        slist.append({'name': "腹板/竖肢", 'val': v2, 'lim': lim2, 'ok': v2 <= lim2})
    return all(s['ok'] for s in slist), slist


def check_geom_bounds(rib_type, **kw):
    if rib_type == 'rect':
        return (16 <= kw['hs1'] <= 200) and (3 <= kw['ds1'] <= 60)
    if rib_type == 'angle':
        return (kw['ds1'] == kw['ds2']) and (20 <= kw['hs2'] <= 250) and (16 <= kw['hs1'] <= 250) and (
                    3 <= kw['ds1'] <= 35)
    if rib_type == 'tee':
        return (20 <= kw['hs2'] <= 450) and (10 <= kw['hs1'] <= 150) and (3 <= kw['ds1'] <= 35) and (
                    3 <= kw['ds2'] <= 35)
    return False


def allowable_p_segment(calc, D0, de, Ls, T):
    A = calc.A_value(D0, de, Ls)
    B = calc.B_value(A, T)
    Do_de = D0 / de
    if Do_de < 20:
        Sa = MaterialDB.get_property(de, T, 'allowable_stress')
        Re = MaterialDB.get_property(de, T, 'yield_strength')
        sigma0 = min(2 * Sa, 0.9 * Re)
        p1 = (2.25 / Do_de - 0.0625) * B
        p2 = (2 * sigma0 / Do_de) * (1.0 - 1.0 / Do_de)
        return float(min(p1, p2))
    return float(B / Do_de)


def volumes_and_mass(Di, D0, L, n, As_per_ring):
    RHO_STEEL_KG_PER_MM3 = 7.85e-6
    V_shell = math.pi * L * ((D0 / 2.0) ** 2 - (Di / 2.0) ** 2)
    V_ribs_total = n * As_per_ring * math.pi * D0
    return (V_shell + V_ribs_total) * RHO_STEEL_KG_PER_MM3


# ... (以上代码无变化)

# ==========================================================================================
# PART 1: 评估函数 (与上一版一致)
# ==========================================================================================

FIXED_PARAMS = {'C1': 2.0, 'C2': 0.3}
CALCULATOR = ExternalPressureCalculator()
PENALTY_VALUE = 1e12
RIB_TYPES = ['rect', 'angle', 'tee']


def _core_evaluate(design_vars, vessel_params):
    try:
        de = float(design_vars['de'])
        n = int(round(design_vars['n']));
        n = max(1, min(999, n))
        rib_idx = int(round(design_vars['rib_type_idx']));
        rib_idx = int(np.clip(rib_idx, 0, len(RIB_TYPES) - 1))
        rib_type = RIB_TYPES[rib_idx]

        if rib_type == 'rect':
            params = {'ds1': float(design_vars['rect_ds1']), 'hs1': float(design_vars['rect_hs1'])}
        elif rib_type == 'angle':
            thk = float(design_vars['angle_thk'])
            params = {'ds1': thk, 'ds2': thk, 'hs1': float(design_vars['angle_hs1']),
                      'hs2': float(design_vars['angle_hs2'])}
        else:
            params = {'ds1': float(design_vars['tee_ds1']), 'ds2': float(design_vars['tee_ds2']),
                      'hs1': float(design_vars['tee_hs1']), 'hs2': float(design_vars['tee_hs2'])}

        if not check_geom_bounds(rib_type, **params): return {'is_valid': False}
        T = vessel_params['T']
        D0 = vessel_params['Di'] + 2.0 * (de + vessel_params.get('C1', 2.0) + vessel_params.get('C2', 0.3))
        E = CALCULATOR.Et(T)
        ok_slender, slist = check_slenderness(rib_type, T, E, **params)
        if not ok_slender: return {'is_valid': False}
        Ls = vessel_params['L'] / (n + 1.0)
        Is, As = calc_rib_Is_As(rib_type, D0, de, **params)
        p_allow_segment = allowable_p_segment(CALCULATOR, D0, de, Ls, T)
        Bv_required = B_from_p(D0, de, As, Ls, vessel_params['p'])
        A_at_T = [np.interp(Bv_required, d[:, 1], d[:, 0], left=d[0, 0], right=d[-1, 0]) for d in CALCULATOR.all_data]
        A_required = float(np.interp(T, CALCULATOR.temps, A_at_T))
        Ireq = required_I(D0, de, As, Ls, A_required)
        mass = volumes_and_mass(vessel_params['Di'], D0, vessel_params['L'], n, As)
        is_valid = (p_allow_segment >= vessel_params['p']) and (Is >= Ireq)

        return {
            'is_valid': is_valid, 'mass': mass, 'margin_I': Is / max(Ireq, 1e-12),
            'margin_p': p_allow_segment / max(vessel_params['p'], 1e-12),
            'margin_slender': min([s['lim'] / max(s['val'], 1e-12) for s in slist]) if slist else 1.5,
            'D0': D0, 'Ls': Ls, 'Do_de': (D0 / de), 'L_Do': (vessel_params['L'] / D0),
            'As': As, 'Is': Is, 'Ireq': Ireq, 'p_allow_segment': p_allow_segment,
            'de': de, 'n': n, 'rib_type': rib_type
        }
    except Exception:
        return {'is_valid': False}


def evaluate_for_optimization(design_vars, vessel_params):
    res = _core_evaluate(design_vars, vessel_params)
    return res['mass'] if res.get('is_valid', False) else PENALTY_VALUE


# ==========================================================================================
# PART 2: GA 问题定义 (已修改)
# ==========================================================================================

# === MODIFIED ===: 将 n 的范围设置为变量，例如 1 到 20 根
XL_BASE = np.array([2.0, 1.0, 0.0, 16.0, 3.0, 3.0, 16.0, 20.0, 3.0, 3.0, 10.0, 20.0], dtype=float)
XU_BASE = np.array([50.0, 20.0, 2.0, 200.0, 60.0, 35.0, 250.0, 250.0, 35.0, 35.0, 150.0, 450.0], dtype=float)


def vec_to_design_vars(x, fixed=None):
    dv = {
        'de': float(x[0]), 'n': int(round(x[1])), 'rib_type_idx': int(round(x[2])),
        'rect_hs1': float(x[3]), 'rect_ds1': float(x[4]),
        'angle_thk': float(x[5]), 'angle_hs1': float(x[6]), 'angle_hs2': float(x[7]),
        'tee_ds1': float(x[8]), 'tee_ds2': float(x[9]), 'tee_hs1': float(x[10]), 'tee_hs2': float(x[11]),
    }
    if fixed: dv.update(fixed)
    return dv


class PressureVesselProblemVec(Problem):
    def __init__(self, vessel_params, fixed=None):
        xl = XL_BASE.copy();
        xu = XU_BASE.copy()
        if fixed:
            # === MODIFIED ===: 不再固定n，只固定 rib_type_idx (如果需要的话)
            if 'rib_type_idx' in fixed: xl[2] = xu[2] = float(int(fixed['rib_type_idx']))
        super().__init__(n_var=12, n_obj=1, xl=xl, xu=xu)
        self.fixed = fixed or {}
        self.vessel_params = vessel_params

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.array([evaluate_for_optimization(vec_to_design_vars(row, self.fixed), self.vessel_params) for row in X])
        out["F"] = F.reshape(-1, 1)


def ga_early_stop_run(problem: Problem, pop_size=120, max_gen=150, seed=1, patience=25, min_delta=1e-3, warmup_gen=10,
                      verbose=False):
    algo = GA(pop_size=pop_size, eliminate_duplicates=True)
    algo.setup(problem, termination=None, seed=seed)
    best_F, best_gen, no_improve, cur_gen, seen_feasible = np.inf, -1, 0, 0, False
    while cur_gen < max_gen:
        algo.next();
        cur_gen += 1
        try:
            cur_best_F = float(algo.opt.get("F")[0])
        except (TypeError, IndexError):
            cur_best_F = algo.pop.get("F").min()
        if not seen_feasible and cur_best_F < PENALTY_VALUE:
            seen_feasible = True;
            best_F = cur_best_F;
            best_gen = cur_gen;
            no_improve = 0
        elif seen_feasible:
            if cur_best_F < best_F - min_delta:
                best_F = cur_best_F;
                best_gen = cur_gen;
                no_improve = 0
            else:
                no_improve += 1
        if seen_feasible and cur_gen >= warmup_gen and no_improve >= patience: break

    # === MODIFIED ===: 确保即使没有可行解也能正常返回，避免程序崩溃
    if not np.isfinite(algo.opt.get("F")[0]):
        return None, PENALTY_VALUE, {}

    return np.array(algo.opt.get("X")[0]), float(algo.opt.get("F")[0]), dict(best_F=best_F, best_gen=best_gen,
                                                                             last_gen=cur_gen,
                                                                             seen_feasible=seen_feasible,
                                                                             no_improve=no_improve)


# ==========================================================================================
# PART 3: API 调用的主流程 (已重构)
# ==========================================================================================

def run_api_optimization(length: float, diameter: float, temperature: float, pressure: float, pop_size=200, max_gen=250,
                         patience=30, min_delta=1e-3):
    """
    === MODIFIED ===:
    此函数现在执行一次全面的优化，同时寻找最佳的加强筋类型、数量和尺寸。
    不再需要遍历 n 或 rib_type。
    由于搜索空间变大，建议适当增加 pop_size 和 max_gen。
    """
    vessel_params = FIXED_PARAMS.copy()
    vessel_params.update({'L': length, 'Di': diameter, 'T': temperature, 'p': pressure})

    print(f"开始全局优化任务，参数: {vessel_params}")
    print(f"优化设置: pop_size={pop_size}, max_gen={max_gen}, patience={patience}")

    # 创建一个不固定任何变量的优化问题
    problem = PressureVesselProblemVec(vessel_params=vessel_params)

    X_opt, F_opt, info = ga_early_stop_run(
        problem,
        pop_size=pop_size,
        max_gen=max_gen,
        seed=1,
        patience=patience,
        min_delta=min_delta
    )

    if X_opt is None or F_opt >= PENALTY_VALUE:
        return {"status": "failed", "message": "优化算法未能找到任何可行解，请检查输入参数或尝试放宽设计约束。"}

    # 获得最优解的设计变量，并进行最终的详细评估
    best_design_vars = vec_to_design_vars(X_opt)
    final_details = _core_evaluate(best_design_vars, vessel_params)

    # 准备返回给API的结果
    best_result = {
        "rib_type": final_details.get('rib_type'),
        "stiffener_count_n": final_details.get('n'),
        "total_mass_kg": round(F_opt, 2),
        "shell_thickness_de_mm": round(final_details.get('de', 0), 2),
        "margin_strength": round(final_details.get('margin_p', 0), 3),
        "margin_stability": round(final_details.get('margin_I', 0), 3),
        "termination_reason": ("early_stop" if info.get('last_gen', max_gen) < max_gen else "max_generations"),
        "generations_run": info.get('last_gen')
    }

    return {
        "status": "success",
        "message": f"全局优化完成，找到最优解。",
        "best_solution": best_result
    }