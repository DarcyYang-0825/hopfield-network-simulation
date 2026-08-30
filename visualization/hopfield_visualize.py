"""
Hopfield 网络可视化脚本
=======================

为仓库中的 C 语言实现提供配套的 Python 可视化，算法细节与
src/hopfield_basic.c、src/hopfield_annealing.c 精确一致：

  - 异步更新规则  S[i] = +1 当 sum >= 0（与 C 版 >= 边界一致），就地更新
  - 收敛判据      整轮（25 个神经元）无变化；基础版最多 20 轮，退火版阶段一最多 30 轮
  - 退火逻辑      先标准恢复，若陷入非字母伪态则从噪声重启，
                  T=3.0 起、P(+1)=1/(1+exp(-2h/T))、每轮 T*=0.85，直到 T>0.1

生成的可视化（输出到脚本所在目录下的 output/ 文件夹）：

  gif1_pattern_recovery.gif       图案从 30% 噪声恢复到记忆的过程
  gif2_energy_descent.gif         能量函数随异步更新单调下降（逐神经元 + 逐轮两种粒度）
  gif3_annealing_recovery.gif     模拟退火：陷入伪态 -> 退火跳出 -> 恢复目标字母
  gif4_energy_landscape.gif       3D 能量景观（连续近似）+ 状态滚入能量谷动画
  fig_energy_landscape.png        静态高清 3D 能量景观图

案例：存储 A/D/S 三个字母，测试 A，30% 噪声（实验报告典型案例组）。

依赖：numpy、matplotlib、Pillow（matplotlib 动画导出 GIF 需要）；
      scipy 可选（安装后 3D 曲面会额外做高斯平滑，效果更细腻）。

运行：  python hopfield_visualize.py        （全部生成，约 5~10 分钟）
        python hopfield_visualize.py --quick （仅生成 GIF1 和 GIF2，约 1 分钟）

说明：3D 能量景观曲面为连续近似示意（软符号函数 m=tanh(v/T) 代替离散
sign 函数），谷底位置与相对深浅和严格离散计算一致。
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter

# ============ 中文字体（按平台自动寻找，找不到则退回默认字体） ============
for _fp in (r'C:\Windows\Fonts\simhei.ttf',
            r'C:\Windows\Fonts\msyh.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'):
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

# ============ 配置（与 C 代码一致） ============
N = 25
NOISE_RATE = 0.3
MAX_SWEEPS = 20

# 26 个字母图案（与 src/hopfield_basic.c 完全一致）
PATTERNS = {
    'A': [-1,-1,+1,-1,-1, -1,+1,-1,+1,-1, +1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1],
    'B': [+1,+1,+1,+1,-1, +1,-1,-1,-1,+1, +1,+1,+1,+1,-1, +1,-1,-1,-1,+1, +1,+1,+1,+1,-1],
    'C': [+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,+1,+1,+1,+1],
    'D': [+1,+1,+1,+1,-1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,-1],
    'E': [+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,+1,+1,+1,-1, +1,-1,-1,-1,-1, +1,+1,+1,+1,+1],
    'F': [+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,+1,+1,+1,-1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1],
    'G': [+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,-1,+1,+1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1],
    'H': [+1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1],
    'I': [+1,+1,+1,+1,+1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, +1,+1,+1,+1,+1],
    'J': [-1,-1,-1,-1,+1, -1,-1,-1,-1,+1, -1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1],
    'K': [+1,-1,-1,-1,+1, +1,-1,-1,+1,-1, +1,+1,+1,-1,-1, +1,-1,-1,+1,-1, +1,-1,-1,-1,+1],
    'L': [+1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,+1,+1,+1,+1],
    'M': [+1,-1,-1,-1,+1, +1,+1,-1,+1,+1, +1,-1,+1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1],
    'N': [+1,-1,-1,-1,+1, +1,+1,-1,-1,+1, +1,-1,+1,-1,+1, +1,-1,-1,+1,+1, +1,-1,-1,-1,+1],
    'O': [+1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1],
    'P': [+1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1],
    'Q': [+1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,+1,+1, +1,+1,+1,+1,+1],
    'R': [+1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1, +1,-1,-1,+1,-1, +1,-1,-1,-1,+1],
    'S': [+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,+1,+1,+1,+1, -1,-1,-1,-1,+1, +1,+1,+1,+1,+1],
    'T': [+1,+1,+1,+1,+1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1],
    'U': [+1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1],
    'V': [+1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, -1,+1,-1,+1,-1, -1,-1,+1,-1,-1],
    'W': [+1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,+1,-1,+1, +1,+1,-1,+1,+1, +1,-1,-1,-1,+1],
    'X': [+1,-1,-1,-1,+1, -1,+1,-1,+1,-1, -1,-1,+1,-1,-1, -1,+1,-1,+1,-1, +1,-1,-1,-1,+1],
    'Y': [+1,-1,-1,-1,+1, -1,+1,-1,+1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1],
    'Z': [+1,+1,+1,+1,+1, -1,-1,-1,+1,-1, -1,-1,+1,-1,-1, -1,+1,-1,-1,-1, +1,+1,+1,+1,+1],
}

# 颜色方案
C_BG = '#f5f7fa'          # 浅色背景（GIF1~3）
C_MEM = '#1a5276'         # +1 神经元
C_NOISE_HL = '#e67e22'    # 噪声翻转标记
C_UPDATE_HL = '#e74c3c'   # 当前更新的神经元
C_ACCENT = '#2980b9'
C_PHASE1 = '#2980b9'      # 阶段一: 标准异步
C_PHASE2 = '#e67e22'      # 阶段二: 退火
C_PHASE3 = '#27ae60'      # 阶段三: 低温收敛
C_FAIL = '#c0392b'

# 深色主题（3D 能量景观）
BG = '#0d0f1a'
PANEL = '#141726'
GRID_C = '#2a2e42'
TXT_C = '#e8e8f0'

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============ 算法实现（精确复现 C 版） ============
def hebb_train(letters):
    """Hebb 学习律: J = (1/N) * sum_mu(xi^mu (xi^mu)^T), 对角线置 0"""
    J = np.zeros((N, N))
    for c in letters:
        xi = np.array(PATTERNS[c], dtype=float)
        J += np.outer(xi, xi)
    J /= N
    np.fill_diagonal(J, 0.0)
    return J


def add_noise(letter, rng):
    """30% 噪声: 每个神经元以 30% 概率翻转"""
    S = np.array(PATTERNS[letter], dtype=int)
    flip = rng.random(N) < NOISE_RATE
    S[flip] = -S[flip]
    return S, flip


def energy(J, S):
    """E = -(1/2) sum_{i!=j} J_ij S_i S_j"""
    return -0.5 * S @ J @ S


def run_dynamics(J, S0):
    """
    异步更新动力学，精确复现 src/hopfield_basic.c：
    - 每个 sweep 随机打乱更新顺序
    - S[i] = +1 当 sum >= 0，否则 -1（与 C 的 >= 边界一致）
    - 就地更新：同一轮中已更新的神经元使用新值
    - 整轮无变化则收敛，最多 20 个 sweep

    随机源通过函数属性 run_dynamics.seed 指定。
    """
    S = S0.copy()
    rng = np.random.default_rng(run_dynamics.seed)

    states = [S.copy()]
    energies = [energy(J, S)]
    updated = [-1]                # 该帧更新的神经元索引（-1 表示初始帧）
    sweep_marks = [0]             # 每个 sweep 结束对应的帧索引
    sweep_energies = [energy(J, S)]

    converged = False
    for _sweep in range(MAX_SWEEPS):
        order = rng.permutation(N)
        prev = S.copy()
        for i in order:
            h = J[i] @ S
            new_val = 1 if h >= 0 else -1
            if new_val != S[i]:
                S[i] = new_val
            states.append(S.copy())
            energies.append(energy(J, S))
            updated.append(i)
        sweep_marks.append(len(states) - 1)
        sweep_energies.append(energy(J, S))
        if np.array_equal(S, prev):
            converged = True
            break

    return {
        'states': states, 'energies': energies, 'updated': updated,
        'sweep_marks': sweep_marks, 'sweep_energies': sweep_energies,
        'converged': converged, 'n_sweeps': len(sweep_energies) - 1,
        'final': S.copy(),
    }


run_dynamics.seed = 48


def matches_any_pattern(S):
    for c, p in PATTERNS.items():
        if np.array_equal(S, np.array(p)):
            return c
    return None


def run_annealing(J, S0, rng):
    """
    复现 src/hopfield_annealing.c 的 process_input 逻辑：
    阶段一  标准异步更新（最多 30 轮），若收敛到任意字母则直接返回；
    阶段二  若陷入非字母伪态：从噪声输入重启，T=3.0 起，
            P(S_i=+1) = 1/(1+exp(-2h/T))，每轮后检查是否跳到某个字母，
            T *= 0.85，直到 T <= 0.1；
    阶段三  退火结束仍未到字母态：低温标准收敛（最多 20 轮）。

    记录粒度：阶段一/三逐神经元，阶段二逐轮（updated=-2 标记）。
    """
    rec = {'states': [S0.copy()], 'energies': [energy(J, S0)],
           'updated': [-1], 'phase': [1], 'T': [None]}

    def push(S, i, ph, T=None):
        rec['states'].append(S.copy())
        rec['energies'].append(energy(J, S))
        rec['updated'].append(i)
        rec['phase'].append(ph)
        rec['T'].append(T)

    # ---- 阶段一 ----
    S = S0.copy()
    for _sweep in range(30):
        order = rng.permutation(N)
        prev = S.copy()
        for i in order:
            S[i] = 1 if (J[i] @ S) >= 0 else -1
            push(S, i, 1)
        if np.array_equal(S, prev):
            break
    if matches_any_pattern(S) is not None:
        rec['outcome'] = ('direct', matches_any_pattern(S))
        return rec

    # ---- 阶段二: 退火 ----
    S = S0.copy()
    push(S, -1, 2, 3.0)
    T = 3.0
    found = None
    while T > 0.1:
        order = rng.permutation(N)
        for i in order:
            h = J[i] @ S
            p_plus = 1.0 / (1.0 + np.exp(-2.0 * h / T))
            S[i] = 1 if rng.random() < p_plus else -1
        push(S, -2, 2, T)
        T *= 0.85
        found = matches_any_pattern(S)
        if found is not None:
            break
    if found is not None:
        rec['outcome'] = ('anneal', found)
        return rec

    # ---- 阶段三: 低温收敛 ----
    for _sweep in range(20):
        order = rng.permutation(N)
        prev = S.copy()
        for i in order:
            S[i] = 1 if (J[i] @ S) >= 0 else -1
            push(S, i, 3)
        if np.array_equal(S, prev):
            break
    rec['outcome'] = ('final', matches_any_pattern(S))
    return rec


def search_annealing_case(store_letters, test_letter, seed_range=400):
    """搜索案例：标准更新陷入非字母伪态、退火后恢复到目标字母"""
    J = hebb_train(store_letters)
    cases = []
    for seed in range(seed_range):
        rng = np.random.default_rng(seed)
        S0, _ = add_noise(test_letter, rng)
        rec = run_annealing(J, S0, np.random.default_rng(seed + 10000))
        kind, letter = rec['outcome']
        if kind != 'direct' and letter == test_letter:
            n_p2 = sum(1 for p in rec['phase'] if p == 2)
            cases.append((seed, len(rec['states']), n_p2))
    print(f"符合条件的退火案例数: {len(cases)}")
    for c in cases[:10]:
        print(f"  seed={c[0]}: 总帧数{c[1]}, 退火轮数{c[2]}")
    good = [c for c in cases if 4 <= c[2] <= 15 and c[1] < 260]
    pool = good if good else cases
    if not pool:
        return None, None
    pool.sort(key=lambda x: abs(x[2] - 8))
    seed = pool[0][0]
    rng = np.random.default_rng(seed)
    S0, noise_mask = add_noise(test_letter, rng)
    rec = run_annealing(J, S0, np.random.default_rng(seed + 10000))
    print(f"选用退火案例: seed={seed}")
    return rec, (S0, noise_mask)


# ============ 通用绘图 ============
def draw_grid(ax, S, highlight=-1, noise_mask=None, title='', dark=False):
    """绘制 5x5 状态网格"""
    c_off = PANEL if dark else 'white'
    c_on = '#e8ecff' if dark else C_MEM
    c_line = GRID_C if dark else '#cccccc'
    grid = S.reshape(5, 5)
    ax.imshow(np.where(grid > 0, 1, 0),
              cmap=matplotlib.colors.ListedColormap([c_off, c_on]), vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for i in range(6):
        ax.axhline(i - 0.5, color=c_line, lw=0.8)
        ax.axvline(i - 0.5, color=c_line, lw=0.8)
    if noise_mask is not None:
        for idx in np.where(noise_mask)[0]:
            r, c = divmod(idx, 5)
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                        edgecolor=C_NOISE_HL, lw=2.5))
    if highlight >= 0:
        r, c = divmod(highlight, 5)
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                    edgecolor=C_UPDATE_HL, lw=3))
    ax.set_title(title, fontsize=14, color=TXT_C if dark else 'black')


# ============ GIF1: 图案恢复过程 ============
def make_gif_recovery(res, noise_mask, test_letter, store_letters, path):
    states, energies = res['states'], res['energies']
    updated, sweep_marks = res['updated'], res['sweep_marks']
    HOLD = 15
    n_frames = len(states) + HOLD

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=110)
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(f'Hopfield网络联想记忆恢复过程（存储: {"".join(store_letters)}，'
                 f'测试: {test_letter}，噪声: 30%）', fontsize=16)
    ax_mem, ax_cur, ax_e = axes
    target = np.array(PATTERNS[test_letter])

    draw_grid(ax_mem, target, title=f'原始记忆 {test_letter}')

    ax_e.set_facecolor('white')
    line_fine, = ax_e.plot([], [], color=C_ACCENT, lw=1.2, label='逐神经元更新')
    line_sweep, = ax_e.plot([], [], 'o-', color='#c0392b', lw=2, ms=6, label='每轮扫描结束')
    vline = ax_e.axvline(0, color='gray', ls='--', lw=1, alpha=0.7)
    ax_e.set_xlabel('神经元更新次数', fontsize=12)
    ax_e.set_ylabel('能量 E', fontsize=12)
    ax_e.set_title('能量函数变化', fontsize=14)
    ax_e.set_xlim(0, len(states) - 1)
    emin, emax = min(energies), max(energies)
    pad = (emax - emin) * 0.12 + 1e-6
    ax_e.set_ylim(emin - pad, emax + pad)
    ax_e.legend(fontsize=10, loc='upper right')

    txt = fig.text(0.5, 0.02, '', ha='center', fontsize=13, color='#333')

    def update_frame(k):
        k = min(k, len(states) - 1)
        S = states[k]
        hl = updated[k] if k > 0 else -1
        ax_cur.clear()
        if k == 0:
            draw_grid(ax_cur, S, noise_mask=noise_mask, title='带噪声输入')
        else:
            sweep_no = sum(1 for m in sweep_marks if m < k)
            draw_grid(ax_cur, S, highlight=hl, title=f'异步更新中（第 {sweep_no} 轮）')
        line_fine.set_data(range(k + 1), energies[:k + 1])
        sw_pts = [m for m in sweep_marks if m <= k]
        line_sweep.set_data(sw_pts, [energies[m] for m in sw_pts])
        vline.set_xdata([k, k])
        same_as_target = np.array_equal(S, target)
        txt.set_text(f'更新次数: {k}   当前能量: {energies[k]:.2f}   '
                     + ('已恢复为记忆图案!' if same_as_target else ''))
        return [line_fine, line_sweep, vline, txt]

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=130, blit=False)
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"已保存: {path}  ({n_frames} 帧)")


# ============ GIF2: 能量下降曲线（改进版） ============
def make_gif_energy(res, path):
    energies = res['energies']
    sweep_marks = res['sweep_marks']
    sweep_energies = res['sweep_energies']
    n_real = len(energies)
    HOLD = 15
    n_frames = n_real + HOLD

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 8.2), dpi=110,
                                   gridspec_kw={'height_ratios': [3, 2]})
    fig.patch.set_facecolor(C_BG)
    fig.suptitle('能量函数随异步更新单调下降（存储ADS，测试A，30%噪声）', fontsize=16)

    emin, emax = min(energies), max(energies)
    pad = (emax - emin) * 0.12 + 1e-6

    # ---- 上: 逐神经元（轮次色带） ----
    ax1.set_facecolor('white')
    band_cols = ['#eaf2f8', '#fdf2e9', '#eafaf1', '#f5eef8']
    for s in range(len(sweep_marks) - 1):
        ax1.axvspan(sweep_marks[s], sweep_marks[s + 1],
                    color=band_cols[s % 4], alpha=0.7, zorder=0)
        ax1.text((sweep_marks[s] + sweep_marks[s + 1]) / 2, emax + pad * 0.35,
                 f'第{s + 1}轮', ha='center', fontsize=11, color='#555')
    ln1, = ax1.plot([], [], color=C_PHASE1, lw=2, zorder=3)
    pt1, = ax1.plot([], [], 'o', color='#e74c3c', ms=6, zorder=4)
    ax1.set_ylabel('能量 E', fontsize=12)
    ax1.set_title('粒度一：每次神经元更新后的能量（阶梯状下降）', fontsize=13)
    ax1.set_xlim(-1, n_real); ax1.set_ylim(emin - pad, emax + pad)
    ax1.grid(alpha=0.25)

    # ---- 下: 逐轮扫描 ----
    ax2.set_facecolor('white')
    ln2, = ax2.plot([], [], 'o-', color=C_FAIL, lw=2.8, ms=9, zorder=3)
    ax2.set_xlabel('扫描轮次（每轮25个神经元各更新一次）', fontsize=12)
    ax2.set_ylabel('能量 E', fontsize=12)
    ax2.set_title('粒度二：每轮扫描结束时的能量', fontsize=13)
    ax2.set_xlim(-0.3, len(sweep_marks) - 1 + 0.3)
    ax2.set_ylim(emin - pad, emax + pad)
    ax2.grid(alpha=0.25)

    txt = fig.text(0.5, 0.012, '', ha='center', fontsize=13, color='#333')

    def update_frame(k):
        k = min(k, n_real - 1)
        ln1.set_data(range(k + 1), energies[:k + 1])
        pt1.set_data([k], [energies[k]])
        done = [i for i, m in enumerate(sweep_marks) if m <= k]
        ln2.set_data(done, [sweep_energies[i] for i in done])
        for i in done[1:]:
            lbl = f'E = {sweep_energies[i]:.2f}'
            if not any(t.get_text() == lbl for t in ax2.texts):
                ax2.annotate(lbl, (i, sweep_energies[i]),
                             textcoords='offset points', xytext=(12, 6),
                             fontsize=10.5, color=C_FAIL)
                if i > 1:
                    dE = sweep_energies[i] - sweep_energies[i - 1]
                    ax2.annotate(f'ΔE = {dE:.2f}',
                                 (i - 0.5, (sweep_energies[i] + sweep_energies[i - 1]) / 2),
                                 fontsize=9.5, color='#777', ha='center')
        txt.set_text(f'神经元更新 {k} 次 · 能量 {energies[k]:.3f} · '
                     f'每次更新能量不增 → 网络最终落入能量谷（记忆态）')
        return [ln1, pt1, ln2, txt]

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=120, blit=False)
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"已保存: {path}  ({n_frames} 帧)")


# ============ GIF3: 模拟退火恢复过程 ============
def make_gif_annealing(rec, noise_mask, test_letter, store_letters, path):
    states, energies = rec['states'], rec['energies']
    updated, phase, T_hist = rec['updated'], rec['phase'], rec['T']
    n_real = len(states)
    target = np.array(PATTERNS[test_letter])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), dpi=110)
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(f'Hopfield网络模拟退火恢复（存储: {"".join(store_letters)}，'
                 f'测试: {test_letter}，噪声: 30%）', fontsize=16)
    ax_mem, ax_cur, ax_e = axes

    draw_grid(ax_mem, target, title=f'原始记忆 {test_letter}')

    ax_e.set_facecolor('white')
    idx = np.arange(n_real)
    ph = np.array(phase)
    for ph_id, col, lb in [(1, C_PHASE1, '阶段一: 标准异步更新'),
                           (2, C_PHASE2, '阶段二: 模拟退火'),
                           (3, C_PHASE3, '阶段三: 低温收敛')]:
        m = ph == ph_id
        if m.any():
            ax_e.plot(idx[m], np.array(energies)[m], 'o', ms=2.5, color=col, alpha=0.55, label=lb)
    line_now, = ax_e.plot([], [], color='#333', lw=1.6)
    vline = ax_e.axvline(0, color='gray', ls='--', lw=1, alpha=0.7)
    ax_e.set_xlabel('更新步数', fontsize=12)
    ax_e.set_ylabel('能量 E', fontsize=12)
    ax_e.set_title('能量函数变化（退火允许能量暂时升高）', fontsize=13)
    ax_e.set_xlim(0, n_real - 1)
    emin, emax = min(energies), max(energies)
    pad = (emax - emin) * 0.12 + 1e-6
    ax_e.set_ylim(emin - pad, emax + pad)
    ax_e.legend(fontsize=9, loc='upper right')

    txt = fig.text(0.5, 0.015, '', ha='center', fontsize=12.5, color='#333')
    p1_end = max(i for i, p in enumerate(phase) if p == 1)

    # 帧序列：退火每轮放慢 4 倍、陷入伪态处停顿 12 帧、末尾停留 15 帧
    seq = []
    for i in range(n_real):
        seq.append(i)
        if i == p1_end:
            seq.extend([i] * 12)
        if phase[i] == 2 and updated[i] == -2:
            seq.extend([i] * 3)
    seq.extend([n_real - 1] * 15)
    n_frames = len(seq)

    def update_frame(k):
        k = seq[min(k, len(seq) - 1)]
        S = states[k]
        hl = updated[k] if updated[k] >= 0 else -1
        ph_now = phase[k]
        ax_cur.clear()
        if k == 0:
            draw_grid(ax_cur, S, noise_mask=noise_mask, title='带噪声输入')
        else:
            if ph_now == 1:
                title = '阶段一: 标准异步更新'
            elif ph_now == 2:
                title = f'阶段二: 退火  T = {T_hist[k]:.2f}'
            else:
                title = '阶段三: 低温收敛'
            draw_grid(ax_cur, S, highlight=hl, title=title)
        line_now.set_data(range(k + 1), energies[:k + 1])
        vline.set_xdata([k, k])

        e_now = energies[k]
        if ph_now == 1:
            msg = f'阶段一 | 步数 {k} | 能量 {e_now:.2f}'
            if k == p1_end:
                msg += ' | 收敛到非字母伪态, 启动退火!'
        elif ph_now == 2:
            anneal_step = sum(1 for i in range(k + 1) if phase[i] == 2 and updated[i] == -2)
            msg = f'阶段二 | 退火第 {anneal_step} 轮 | T = {T_hist[k]:.2f} | 能量 {e_now:.2f}'
            if matches_any_pattern(S) is not None:
                msg += ' | 跳到字母态!'
        else:
            msg = f'阶段三 | 步数 {k} | 能量 {e_now:.2f}'
        if np.array_equal(S, target):
            msg += ' | 已恢复为记忆图案!'
        txt.set_text(msg)
        return [line_now, vline, txt]
    
    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=120, blit=False)
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"已保存: {path}  ({n_frames} 帧)")
# ============ 3D 能量景观（连续近似，深色主题） ============
def build_smooth_surface(J, xi1, xi2, grid_n=110, T_soft=0.32, sigma=1.6):
    """
    软连续近似曲面：状态 m = tanh((a*xi1 + b*xi2)/T_soft)，
    能量 E = -1/2 m^T J m。T_soft -> 0 时收敛到离散网络能量。
    25 维状态空间投影到与两个参考字母的重叠度平面。
    """
    a = np.linspace(-1.3, 1.3, grid_n)
    b = np.linspace(-1.3, 1.3, grid_n)
    A, B = np.meshgrid(a, b)
    V = A[..., None] * xi1 + B[..., None] * xi2
    M = np.tanh(V / T_soft)
    Z = -0.5 * np.einsum('...i,ij,...j->...', M, J, M)
    if HAS_SCIPY and sigma > 0:
        Z = gaussian_filter(Z, sigma=sigma)
    return A, B, Z


def project_traj(states, energies, xi1, xi2):
    """把轨迹投影到 (m1, m2) 重叠度平面，z 取真实能量"""
    m1 = np.array([S @ xi1 / N for S in states])
    m2 = np.array([S @ xi2 / N for S in states])
    return m1, m2, np.array(energies)


def setup_dark_axes(fig, elev=40, azim=-58):
    ax = fig.add_subplot(111, projection='3d')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANEL)
        axis.pane.set_edgecolor(GRID_C)
        axis.pane.set_alpha(0.9)
        axis._axinfo['grid'].update(color=GRID_C, linestyle='-', linewidth=0.6)
        axis.set_tick_params(colors=TXT_C)
        for lbl in axis.get_ticklabels():
            lbl.set_color(TXT_C)
    ax.tick_params(colors=TXT_C)
    ax.xaxis.label.set_color(TXT_C)
    ax.yaxis.label.set_color(TXT_C)
    ax.zaxis.label.set_color(TXT_C)
    ax.view_init(elev=elev, azim=azim)
    return ax


def letter_rgba(letter, on=(0.93, 0.96, 1.0, 1.0), off=(0.10, 0.12, 0.22, 1.0)):
    """把字母图案渲染成 5x5 RGBA 数组"""
    pat = np.array(PATTERNS[letter]).reshape(5, 5)
    rgba = np.zeros((5, 5, 4))
    for i in range(5):
        for j in range(5):
            rgba[i, j] = on if pat[i, j] > 0 else off
    return rgba


def add_valley_marker(ax, J, xi1, xi2, letter, z_hang, arrow=True):
    """字母能量谷标记：谷底亮点 + 白色箭头 + 悬挂字母图案小图"""
    xi = np.array(PATTERNS[letter], dtype=float)
    mx, my = xi @ xi1 / N, xi @ xi2 / N
    e_c = energy(J, xi)
    ax.scatter([mx], [my], [e_c], s=42, color='#ffd54f',
               edgecolor='white', linewidth=0.9, depthshade=False)
    if arrow:
        ax.quiver(mx, my, e_c + 3.2, 0, 0, -2.2,
                  color='white', lw=1.8, arrow_length_ratio=0.14, alpha=0.95)
    ax.plot([mx, mx], [my, my], [e_c - 0.3, z_hang + 0.55],
            color='#9fa8da', lw=1.1, alpha=0.85)
    size = 0.34
    xs = np.linspace(mx - size / 2, mx + size / 2, 6)
    ys = np.linspace(my - size / 2, my + size / 2, 6)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, z_hang)
    ax.plot_surface(X, Y, Z, facecolors=letter_rgba(letter),
                    shade=False, rstride=1, cstride=1)
    ax.text(mx, my, z_hang - 0.5, letter, fontsize=13, color=TXT_C,
            ha='center', fontweight='bold')


def make_zoom_inset(fig, rect, A, B, Z, xi1, xi2, letters, test_letter):
    """轨迹区域放大俯视图（2D 俯视永不被山体遮挡）"""
    axz = fig.add_axes(rect)
    axz.set_facecolor(PANEL)
    axz.contourf(A, B, Z, levels=22, cmap='magma')
    axz.set_xlim(0.38, 1.12); axz.set_ylim(-0.55, 0.35)
    axz.set_xlabel(f'$m_{test_letter}$', fontsize=9, color=TXT_C)
    axz.set_ylabel('$m_D$', fontsize=9, color=TXT_C)
    axz.tick_params(colors=TXT_C, labelsize=8)
    for s in axz.spines.values():
        s.set_color('#7f8cae')
    axz.set_title('轨迹放大俯视图', fontsize=10, color=TXT_C)
    for c in letters:
        xi = np.array(PATTERNS[c], dtype=float)
        axz.plot(xi @ xi1 / N, xi @ xi2 / N, '*', color='gold', ms=12,
                 markeredgecolor='black', markeredgewidth=0.7)
    path_bg, = axz.plot([], [], color='black', lw=5.5, alpha=0.9, solid_capstyle='round')
    path, = axz.plot([], [], color='#7ff4ff', lw=2.6, solid_capstyle='round')
    dot, = axz.plot([], [], 'o', color='#ef5350', ms=8)
    return axz, path_bg, path, dot

def make_static_landscape(res, store_letters, test_letter, path):
    """静态高清 3D 能量景观图"""
    J = hebb_train(store_letters)
    xi1 = np.array(PATTERNS[test_letter], dtype=float)
    xi2 = np.array(PATTERNS['D'], dtype=float)
    A, B, Z = build_smooth_surface(J, xi1, xi2)
    m1, m2, ez = project_traj(res['states'], res['energies'], xi1, xi2)
    z_hang = Z.min() - 2.2

    fig = plt.figure(figsize=(10.5, 8.5), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = setup_dark_axes(fig)
    ax.plot_surface(A, B, Z, cmap='magma', alpha=0.62,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True)
    for c in store_letters:
        add_valley_marker(ax, J, xi1, xi2, c, z_hang, arrow=(c == test_letter))

    # 恢复轨迹：抬离曲面 + 黑色描边 + 白色方向箭头
    dz = 0.5
    ax.plot(m1, m2, ez + dz, color='black', lw=7.0, alpha=0.9)
    ax.plot(m1, m2, ez + dz, color='#7ff4ff', lw=3.5)
    idx = list(range(5, len(m1), 9)) + [len(m1) - 1]
    for i in idx[:-1]:
        d1, d2 = m1[i + 1] - m1[i], m2[i + 1] - m2[i]
        nrm = np.hypot(d1, d2) + 1e-9
        ax.quiver(m1[i], m2[i], ez[i] + dz, d1 / nrm * 0.12, d2 / nrm * 0.12, 0,
                  color='white', lw=1.6, arrow_length_ratio=0.5, alpha=0.95)
    ax.scatter(m1[:1], m2[:1], ez[:1], color='#ffb74d', s=80,
               label='噪声输入起点', depthshade=False)
    ax.scatter(m1[-1:], m2[-1:], ez[-1:], color='#ef5350', s=80,
               label=f'收敛终点（记忆{test_letter}）', depthshade=False)

    axz, zpath_bg, zpath, zdot = make_zoom_inset(
        fig, [0.60, 0.60, 0.30, 0.30], A, B, Z, xi1, xi2, store_letters, test_letter)
    zpath_bg.set_data(m1, m2); zpath.set_data(m1, m2)
    zdot.set_data([m1[-1]], [m2[-1]])

    ax.set_xlabel(f'与{test_letter}的重叠度 $m_{test_letter}$', fontsize=12, labelpad=10)
    ax.set_ylabel('与D的重叠度 $m_D$', fontsize=12, labelpad=10)
    ax.set_zlabel('能量 E', fontsize=12, labelpad=10)
    ax.set_title(f'Hopfield网络能量景观（存储 {"".join(store_letters)}，测试 {test_letter}）\n'
                 f'连续近似示意 · 记忆存储为能量谷 · 状态沿箭头滚入最近的谷',
                 fontsize=14, color=TXT_C, pad=18)
    ax.legend(fontsize=11, loc='upper left', facecolor=PANEL,
              edgecolor=GRID_C, labelcolor=TXT_C)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"已保存: {path}")


def make_gif_landscape(res, store_letters, test_letter, path):
    """3D 能量景观 + 恢复轨迹动画"""
    J = hebb_train(store_letters)
    xi1 = np.array(PATTERNS[test_letter], dtype=float)
    xi2 = np.array(PATTERNS['D'], dtype=float)
    A, B, Z = build_smooth_surface(J, xi1, xi2, grid_n=85)
    m1, m2, ez = project_traj(res['states'], res['energies'], xi1, xi2)
    n_pts = len(m1)
    HOLD = 20
    n_frames = n_pts + HOLD
    z_hang = Z.min() - 2.2
    dz = 0.5

    fig = plt.figure(figsize=(10.5, 8.5), dpi=105)
    fig.patch.set_facecolor(BG)
    fig.suptitle(f'能量景观与恢复轨迹（存储 {"".join(store_letters)}，'
                 f'测试 {test_letter}，噪声30%）\n'
                 f'连续近似示意：状态从噪声起点滚入记忆{test_letter}的能量谷',
                 fontsize=14, color=TXT_C)
    ax = setup_dark_axes(fig)
    ax.plot_surface(A, B, Z, cmap='magma', alpha=0.62,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True)
    for c in store_letters:
        add_valley_marker(ax, J, xi1, xi2, c, z_hang, arrow=(c == test_letter))

    traj_bg, = ax.plot([], [], [], color='black', lw=7.0, alpha=0.9)
    traj, = ax.plot([], [], [], color='#7ff4ff', lw=3.5)
    pt, = ax.plot([], [], [], 'o', color='#ef5350', ms=11)

    ax_ins = fig.add_axes([0.04, 0.66, 0.15, 0.20])
    ax_ins.set_facecolor(PANEL)
    ax_ins.set_xticks([]); ax_ins.set_yticks([])
    for s in ax_ins.spines.values():
        s.set_color(GRID_C)

    axz, zpath_bg, zpath, zdot = make_zoom_inset(
        fig, [0.72, 0.56, 0.23, 0.28], A, B, Z, xi1, xi2, store_letters, test_letter)

    txt = fig.text(0.5, 0.02, '', ha='center', fontsize=12.5, color=TXT_C)

    elev0, azim0 = 40, -58

    def update_frame(k):
        k = min(k, n_pts - 1)
        traj_bg.set_data(np.asarray(m1[:k + 1]), np.asarray(m2[:k + 1]))
        traj_bg.set_3d_properties(np.asarray(ez[:k + 1] + dz))
        traj.set_data(np.asarray(m1[:k + 1]), np.asarray(m2[:k + 1]))
        traj.set_3d_properties(np.asarray(ez[:k + 1] + dz))
        pt.set_data(np.asarray([m1[k]]), np.asarray([m2[k]]))
        pt.set_3d_properties(np.asarray([ez[k] + dz]))
        ax.view_init(elev=elev0, azim=azim0 + k * 0.45)
        ax_ins.clear()
        ax_ins.set_facecolor(PANEL)
        ax_ins.set_xticks([]); ax_ins.set_yticks([])
        draw_grid(ax_ins, res['states'][k], dark=True)
        ax_ins.set_title('当前状态', fontsize=10, color=TXT_C)
        zpath_bg.set_data(np.asarray(m1[:k + 1]), np.asarray(m2[:k + 1]))
        zpath.set_data(np.asarray(m1[:k + 1]), np.asarray(m2[:k + 1]))
        zdot.set_data(np.asarray([m1[k]]), np.asarray([m2[k]]))
        txt.set_text(f'更新步数: {k}   能量: {ez[k]:.2f}   '
                     f'与{test_letter}重叠度: {m1[k]:.2f}')
        return [traj_bg, traj, pt, txt, zpath_bg, zpath, zdot]

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=130, blit=False)
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"已保存: {path}  ({n_frames} 帧)")
# ============ 主流程 ============
if __name__ == '__main__':
    QUICK = '--quick' in sys.argv
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    STORE = ['A', 'D', 'S']   # 实验报告典型案例组：存储 A/D/S，测试 A
    TEST = 'A'
    J = hebb_train(STORE)

    # ---- 基础案例：seed=48，与报告/海报一致 ----
    rng = np.random.default_rng(48)
    S0, noise_mask = add_noise(TEST, rng)
    run_dynamics.seed = 48
    res = run_dynamics(J, S0)
    assert np.array_equal(res['final'], np.array(PATTERNS[TEST])), '恢复失败，请检查种子'
    assert all(a >= b - 1e-9 for a, b in zip(res['energies'], res['energies'][1:])), '能量非单调'
    print(f"基础案例: {res['n_sweeps']} 轮收敛, 能量 {res['energies'][0]:.2f} -> {res['energies'][-1]:.2f}")

    # ---- GIF1: 图案恢复过程 ----
    make_gif_recovery(res, noise_mask, TEST, STORE,
                      os.path.join(OUTPUT_DIR, 'gif1_pattern_recovery.gif'))

    # ---- GIF2: 能量下降曲线 ----
    make_gif_energy(res, os.path.join(OUTPUT_DIR, 'gif2_energy_descent.gif'))

    if QUICK:
        print('--quick 模式: 仅生成 GIF1/GIF2, 结束')
        sys.exit(0)

    # ---- GIF3: 模拟退火恢复（自动搜索合适案例） ----
    rec, case0 = search_annealing_case(STORE, TEST)
    if rec is not None:
        _S0a, mask_a = case0
        make_gif_annealing(rec, mask_a, TEST, STORE,
                           os.path.join(OUTPUT_DIR, 'gif3_annealing_recovery.gif'))
    else:
        print('未搜索到合适退火案例, 跳过 GIF3')

    # ---- 静态 3D 能量景观图 ----
    make_static_landscape(res, STORE, TEST,
                          os.path.join(OUTPUT_DIR, 'fig_energy_landscape.png'))

    # ---- GIF4: 3D 能量景观动画 ----
    make_gif_landscape(res, STORE, TEST,
                       os.path.join(OUTPUT_DIR, 'gif4_energy_landscape.gif'))

    print(f'\n全部完成, 输出目录: {OUTPUT_DIR}')
