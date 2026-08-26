"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np
import scipy as sp
from simulation.utils import Rz, wrap_angle_pi
DOF3 = np.array([0, 1, 5])

class DPController:
    """
    Template for student DP controller.

    Students may implement any type of controller (PID, LQR, backstepping,
    ...). Only compute() is required; everything else is optional.
    """

    def __init__(self, *args, **kwargs):

        self.M3 = np.array([
            [6.007e5, 0.0, 0.0],
            [0.0, 7.067e5, -4.733e5],
            [0.0, -5.712e5, 5.456e7],
        ])
        self.D3 = np.diag([1117.6, 2.229e4, 1.95e6])

        self.A, self.B = self.build_ss_model()

        self.int_ned = np.zeros(2)  # [N, E] integral state
        self.int_psi = 0.0          # [psi] integral state

        self.Q = np.diag([
            ..., ..., ...,   # position / heading
            ..., ..., ...,   # velocities
            ..., ..., ...,   # integral states
        ])

        self.R = np.diag([
            ..., ..., ...,   # Fx, Fy, Mz
        ])

        self.K = self.build_lqr_gain()
        K_I = self.K[:, 6:9]
        if np.linalg.matrix_rank(K_I) < 3:
            raise ValueError("Integral gain matrix is singular")

        self.aw_gain = 1.0 # 1/Time constant of anti-windup back-calculation

        # Maps a BODY-wrench tracking error to the corresponding change in
        # the integral state.
        self._aw_body_map = np.linalg.solve(-K_I, np.eye(3))
        self._last_tau_d3 = np.zeros(3)
        self._has_last_tau_d = False


    def reset(self) -> None:
        """Optional: reset internal states (integrators, filters) before a run."""
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0
        self._last_tau_d3.fill(0.0)
        self._has_last_tau_d = False


    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        
        dot_eta_ref = nu_ref if nu_ref is not None else np.zeros(6)
        ddot_eta_ref = acc_ref if acc_ref is not None else np.zeros(6)


        R = Rz(eta[5])  # BODY -> NED rotation matrix

        e_eta_ned, e_eta_body, e_nu_body = self.compute_errors(
            eta,
            nu,
            eta_ref,
            dot_eta_ref,
            R
        )

        # Integrate error in NED
        self.int_ned += e_eta_ned[:2] * dt
        self.int_psi += e_eta_ned[2] * dt

        integral_ned_3 = np.array([
            self.int_ned[0],
            self.int_ned[1],
            self.int_psi,
        ])

        integral_body = R.T @ integral_ned_3

        x = np.concatenate([
            e_eta_body,
            e_nu_body,
            integral_body,
        ])

        tau_feedback = -self.K @ x

        nu_ref_body, dot_nu_ref_body = self.compute_reference_kinematics(eta_ref, dot_eta_ref, ddot_eta_ref)
        tau_ff_ref = self.M3 @ dot_nu_ref_body + self.D3 @ nu_ref_body

        tau_d = np.zeros(6)
        tau_d[DOF3] = tau_feedback + tau_ff_ref

        self._last_tau_d3[:] = tau_d[DOF3]
        self._has_last_tau_d = True

        return tau_d

    def apply_external_aw(
        self,
        tau_applied: np.ndarray,
        psi: float,
        dt: float,
    ) -> None:
        """
        Anti windup implemented with back calculation.  
        The simulator calls this after thrust allocation, so tau_applied is the actual wrench applied to the vessel.
        """
        if not self._has_last_tau_d or self.aw_gain == 0.0 or dt <= 0.0:
            return

        tau_applied3 = tau_applied[DOF3]
        wrench_error_body = tau_applied3 - self._last_tau_d3
        int_correction_body = self._aw_body_map @ wrench_error_body
        int_correction_ned = Rz(psi) @ int_correction_body

        scale = self.aw_gain * dt
        self.int_ned += scale * int_correction_ned[:2]
        self.int_psi += scale * int_correction_ned[2]


    def build_ss_model(self):
        Z = np.zeros((3, 3))
        I = np.eye(3)

        M3_inv = np.linalg.solve(self.M3, I)
        M3_inv_D3 = np.linalg.solve(self.M3, self.D3)

        A = np.block([
            [Z, I, Z],
            [Z, -M3_inv_D3, Z],
            [I, Z, Z]
        ])

        B = np.vstack([
            Z,
            M3_inv,
            Z
        ])

        return A, B

    def compute_errors(
        self,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        dot_eta_ref: np.ndarray | None,
        R: np.ndarray
    ):
        eta3 = np.asarray(eta)[DOF3]
        eta_ref3 = np.asarray(eta_ref)[DOF3]

        nu_body = np.asarray(nu)[DOF3]

        if dot_eta_ref is None:
            dot_eta_ref_ned = np.zeros(3)
        else:
            dot_eta_ref_ned = np.asarray(dot_eta_ref)[DOF3]

        # Position / heading error in NED
        e_eta_ned = eta3 - eta_ref3
        e_eta_ned = e_eta_ned.copy()
        e_eta_ned[2] = wrap_angle_pi(e_eta_ned[2])

        # Transform position error to BODY
        e_eta_body = R.T @ e_eta_ned

        # Desired NED velocity -> BODY
        nu_ref_body = R.T @ dot_eta_ref_ned

        # Velocity error in BODY
        e_nu_body = nu_body - nu_ref_body

        return e_eta_ned, e_eta_body, e_nu_body


    def build_lqr_gain(self):
        P = sp.linalg.solve_continuous_are(
                self.A,
                self.B,
                self.Q,
                self.R,
            )

        K = np.linalg.solve(
            self.R,
            self.B.T @ P,
        )

        return K

    def compute_reference_kinematics(
        self,
        eta_ref: np.ndarray,
        dot_eta_ref: np.ndarray,
        ddot_eta_ref: np.ndarray,
    ):
        eta_ref3 = np.asarray(eta_ref)[DOF3]
        dot_eta_ref3 = np.asarray(dot_eta_ref)[DOF3]
        ddot_eta_ref3 = np.asarray(ddot_eta_ref)[DOF3]

        psi_ref = eta_ref3[2]

        # BODY -> NED for the reference orientation
        R_ref = Rz(psi_ref)

        # Desired BODY velocity:
        nu_ref_body = R_ref.T @ dot_eta_ref3

        # Desired yaw rate
        r_ref = nu_ref_body[2]

        # Skew symmetric matrix
        S_r = np.array([
            [0.0,   -r_ref, 0.0],
            [r_ref,  0.0,   0.0],
            [0.0,    0.0,   0.0],
        ])

        # Desired BODY velocity derivative:
        # dot(nu_d)^b = R_d^T ddot(eta_d)^n - S(r_d) nu_d^b
        dot_nu_ref_body = R_ref.T @ ddot_eta_ref3 - S_r @ nu_ref_body

        return nu_ref_body, dot_nu_ref_body
