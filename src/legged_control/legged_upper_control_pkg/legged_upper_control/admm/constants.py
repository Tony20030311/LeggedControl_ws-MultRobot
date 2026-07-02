"""Stage 0 — mathematical gate layer for ADMM-CBF-DMPC.

Single source of truth for the frozen constants every later stage builds on:
discretized double-integrator (Ad, exact Bd), second-order HOCBF coefficients,
the a_k->p_{k+2} / CBF-constraint scalars, and the decision-vector index formulas.

Design sources (higher wins on conflict):
  - docs/spec_updates_v2.md  detail 1 (exact Bd, 3Ts^2), detail 2 (gamma sign),
                             detail 4 (N=20 index table)
  - docs/文件二_架構藍圖.md   全局設定 (index formula), C3 (HOCBF), C4.3 (Ad/Bd)

Coefficient discipline (package CLAUDE.md): coefficients that are wrong don't
crash, they make behavior subtly wrong. test_constants.py pins the documented
values; this module *derives* them from the formulas so a sign flip fails the gate.
Never hardcode integer index offsets — always the 4*N formulas below (N changes).
"""

import numpy as np

# --- locked parameters (spec_updates_v2 detail 4) ------------------------------
N = 20          # horizon steps (>= second-order lookahead lower bound 19)
TS = 0.10       # high-level planning dt [s] (10 Hz)
GAMMA1 = 0.30   # first-order HOCBF gain  (repo cbf_gamma1)
GAMMA2 = 0.30   # second-order HOCBF gain (repo cbf_gamma2)

N_X = 4         # state dim  [px, py, vx, vy]  (position first)
N_U = 2         # input dim  [ax, ay]
XI_DIM = 6 * N  # single-dog decision vector dim = 4N + 2N = 120

# --- box bounds (文件二 C4.4; respect the low-level controller limits) --------
# Per-step velocity/accel caps. Fixed spec literals -> single source here so the
# node/edge QPs never hardcode them. (test_constants.py doesn't pin these.)
MAX_VX = 0.55   # |v_x,k| <= 0.55 m/s
MAX_VY = 0.35   # |v_y,k| <= 0.35 m/s
MAX_AX = 1.0    # |a_x,k| <= 1.0 m/s^2
MAX_AY = 1.0    # |a_y,k| <= 1.0 m/s^2


# --- discretized double integrator: X(t+1) = Ad X(t) + Bd u(t) -----------------
# Ad = [[I2, Ts I2], [0, I2]]   (4x4)
# Bd = [[1/2 Ts^2 I2], [Ts I2]] (4x2)   <- EXACT Bd; forward-Euler simplification
#                                           is abolished (spec_updates_v2 detail 1)
_I2 = np.eye(2)
_Z2 = np.zeros((2, 2))
Ad = np.block([[_I2, TS * _I2],
               [_Z2, _I2]])
Bd = np.block([[0.5 * TS**2 * _I2],
               [TS * _I2]])

BD_POS_COEF = 0.5 * TS**2   # Bd position block   = 0.005
BD_VEL_COEF = TS            # Bd velocity block    = 0.1


# --- second-order (Xiong) discrete HOCBF coefficients (文件二 C3) ---------------
# h_{k+2} - (2-g1-g2) h_{k+1} + (1-g1-g2+g1 g2) h_k >= 0
# g1 g2 is POSITIVE (from (1-g1)(1-g2)); a negative sign gives 0.31, off by 37%.
COEF_HK2 = 1.0                                   # coef on h_{k+2}
COEF_HK1 = -(2.0 - GAMMA1 - GAMMA2)              # coef on h_{k+1}  = -1.4
COEF_HK = 1.0 - GAMMA1 - GAMMA2 + GAMMA1 * GAMMA2  # coef on h_k     =  0.49
# ordered [h_k, h_{k+1}, h_{k+2}] for dotting with a stacked barrier history
HOCBF_COEFS = np.array([COEF_HK, COEF_HK1, COEF_HK2])


# --- exact a_k -> p_{k+2} and linearized CBF-constraint scalars -----------------
# a_k reaches p_{k+2} by two paths: Bd into p_{k+1} (1/2 Ts^2) + Bd into v_{k+1}
# then Ad one step into p_{k+2} (Ts*Ts). Sum = 3/2 Ts^2. This equals the position
# block of Ad @ Bd (cross-checked in test_constants.py).
A_TO_P2_COEF = 1.5 * TS**2   # = 0.015
# h = ||e||^2 - d^2 linearized: grad ||e||^2 = 2e, times 3/2 Ts^2 -> 3 Ts^2.
# edge CBF row: col_a = -CBF_CONSTR_COEF * e_bar, col_b = +CBF_CONSTR_COEF * e_bar
CBF_CONSTR_COEF = 3.0 * TS**2  # = 0.03

# a_k ALSO reaches p_{k+1} via the Bd position block (1/2 Ts^2). With exact Bd
# the HOCBF three-point keeps a_k in BOTH h_{k+2} and h_{k+1}; dropping the
# h_{k+1} term under-brakes (~4 cm skirting an obstacle, verified stage 1). The
# h_{k+1} gradient coefficients (parallel to the k+2 pair above):
A_TO_P1_COEF = 0.5 * TS**2       # a_k -> p_{k+1} exact coef = 1/2 Ts^2 = 0.005
CBF_CONSTR_COEF_P1 = 2.0 * A_TO_P1_COEF  # obstacle h_{k+1} grad (2e) = Ts^2 = 0.01


# --- decision-vector index formulas (文件二 全局設定; 0-based) ------------------
# xi = [ x_1 .. x_N  (4N) ,  a_0 .. a_{N-1}  (2N) ]
# States use k = 1..N (x_0 is the measured initial condition, not a variable).
# Accels use k = 0..N-1.  ALWAYS 4*n-based — never a hardcoded 80.

def xi_dim(n=N):
    """Length of one dog's decision vector for horizon n."""
    return 6 * n


def x_index(k, n=N):
    """Start index of state x_k (k=1..n); state occupies [start:start+4]."""
    return 4 * (k - 1)


def px_index(k, n=N):
    return 4 * (k - 1)


def py_index(k, n=N):
    return 4 * (k - 1) + 1


def vx_index(k, n=N):
    return 4 * (k - 1) + 2


def vy_index(k, n=N):
    return 4 * (k - 1) + 3


def a_index(k, n=N):
    """Start index of accel a_k (k=0..n-1); accel occupies [start:start+2]."""
    return 4 * n + 2 * k


def ax_index(k, n=N):
    return 4 * n + 2 * k


def ay_index(k, n=N):
    return 4 * n + 2 * k + 1
