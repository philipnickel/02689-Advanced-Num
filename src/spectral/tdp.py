"""Time-dependent PDE solvers: time integrators and KdV equation solver."""

from __future__ import annotations

import numpy as np
from typing import Callable


# =============================================================================
# Time Integration Methods
# =============================================================================


class TimeIntegrator:
    """Base class for time integration methods."""

    def __init__(self, name: str, order: int, stages: int = 1):
        """
        Initialize time integrator.

        Parameters
        ----------
        name : str
            Name of the method
        order : int
            Order of accuracy
        stages : int
            Number of stages (for RK) or steps (for LMM)
        """
        self.name = name
        self.order = order
        self.stages = stages

    def step(self, rhs: Callable, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Take one time step.

        Parameters
        ----------
        rhs : Callable
            Right-hand side function f(u, t)
        u : np.ndarray
            Current solution
        t : float
            Current time
        dt : float
            Time step

        Returns
        -------
        np.ndarray
            Solution at next time step
        """
        raise NotImplementedError


# =============================================================================
# Runge-Kutta Methods
# =============================================================================


class RK3(TimeIntegrator):
    """3rd-order Strong Stability Preserving Runge-Kutta (SSP-RK3).

    This explicit three-stage method preserves strong stability properties,
    making it particularly suitable for hyperbolic PDEs and problems requiring
    positivity preservation. The method is 3rd-order accurate in time.

    Notes
    -----
    The SSP property ensures that the numerical solution satisfies the same
    stability bounds as forward Euler under a modified time step restriction.
    This is particularly useful for problems with steep gradients or shocks.

    References
    ----------
    Engsig-Karup, "Lecture 5: Initial Value Problems", p. 63
    """

    def __init__(self):
        super().__init__("RK3", order=3, stages=3)

    def step(self, rhs: Callable, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        u1 = u + dt * rhs(u, t)
        u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * rhs(u1, t + dt)
        u3 = (
            (1.0 / 3.0) * u
            + (2.0 / 3.0) * u2
            + (2.0 / 3.0) * dt * rhs(u2, t + 0.5 * dt)
        )
        return u3


class RK4(TimeIntegrator):
    """Classical 4th-order Runge-Kutta method (ERK4).

    The classical explicit four-stage fourth-order Runge-Kutta method.
    This is one of the most widely used explicit time integrators,
    offering a good balance between accuracy and computational cost.

    Notes
    -----
    The method evaluates the right-hand side four times per step at
    intermediate stages. It is 4th-order accurate and suitable for
    a wide range of initial value problems, though it lacks special
    stability properties like SSP methods.

    References
    ----------
    Engsig-Karup, "Lecture 5: Initial Value Problems", p. 58
    """

    def __init__(self):
        super().__init__("RK4", order=4, stages=4)

    def step(self, rhs: Callable, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = rhs(u, t)
        k2 = rhs(u + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(u + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(u + dt * k3, t + dt)
        return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def get_time_integrator(name: str, **kwargs) -> TimeIntegrator:
    """
    Retrieve a time integrator by name.

    Parameters
    ----------
    name : str
        Integrator identifier: "rk4" or "rk3"
    kwargs :
        Extra keyword arguments (currently unused, kept for API compatibility)

    Returns
    -------
    TimeIntegrator
        The requested time integrator instance
    """
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())

    if normalized == "rk4":
        return RK4()
    elif normalized == "rk3":
        return RK3()
    else:
        raise ValueError(f"Unknown time integrator '{name}'. Available: 'rk4', 'rk3'")


# =============================================================================
# KdV Equation Solver
# =============================================================================


def soliton(x: np.ndarray, t: float, c: float, x0: float = 0.0) -> np.ndarray:
    r"""
    Compute KdV soliton solution :math:`u(x,t) = \frac{c/2}{\cosh^2(\sqrt{c}/2 \cdot \xi)}`.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    t : float
        Time
    c : float
        Soliton speed parameter
    x0 : float, optional
        Initial position offset (default: 0.0)

    Returns
    -------
    np.ndarray
        Soliton amplitude at each spatial point

    Notes
    -----
    The traveling wave coordinate is :math:`\xi = x - ct - x_0`.
    """
    xi = x - c * t - x0
    return 0.5 * c / np.cosh(0.5 * np.sqrt(c) * xi) ** 2


def two_soliton_initial(
    x: np.ndarray, c1: float, x01: float, c2: float, x02: float
) -> np.ndarray:
    """
    Initial condition for two-soliton collision simulation.

    Superposition of two solitons at :math:`t=0`.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    c1 : float
        Speed parameter of first soliton
    x01 : float
        Initial position of first soliton
    c2 : float
        Speed parameter of second soliton
    x02 : float
        Initial position of second soliton

    Returns
    -------
    np.ndarray
        Initial condition u(x, 0)
    """
    return soliton(x, 0.0, c1, x01) + soliton(x, 0.0, c2, x02)


class KdVSolver:
    r"""Korteweg-de Vries equation solver for Exercises C-G using Fourier spectral methods.

    Solves the KdV equation

    .. math::

        \partial_t u + 6u \partial_x u + \partial_{xxx}u = 0

    on the domain :math:`-\infty < x < \infty`, :math:`t > 0` with periodic boundary
    conditions, using Fourier collocation for spatial discretization.
    The nonlinear term can optionally be dealiased using the 3/2-rule.

    Notes
    -----
    The Fourier spectral method provides exponential convergence for
    smooth periodic solutions. Spatial derivatives are computed in
    Fourier space via multiplication by :math:`ik` for first derivatives and
    :math:`(ik)^3` for third derivatives, where :math:`k` is the wavenumber.

    The 3/2-rule dealiasing prevents aliasing errors in the nonlinear
    convolution product :math:`u u_x` by padding the Fourier coefficients to
    3/2 times the original resolution (Exercise E).

    Used in:

    - **Exercise C**: Spatial and time discretization
    - **Exercise D**: Evolution of errors and quantities of interest
    - **Exercise E**: Dealiasing implementation
    - **Exercise F**: Soliton solutions
    - **Exercise G**: Two-soliton collisions
    """

    def __init__(self, N: int, L: float, dealias: bool = False):
        """
        Initialize the KdV solver.

        Parameters
        ----------
        N : int
            Number of Fourier modes (grid points)
        L : float
            Half-length of spatial domain [-L, L]
        dealias : bool, optional
            Apply 3/2-rule dealiasing to nonlinear term (default: False)
        """
        self.N = N
        self.L = L
        self.dealias = dealias
        self.x = np.linspace(-L, L, N, endpoint=False)
        self.dx = 2 * L / N

        # Wave numbers for Fourier spectral method
        self.k = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        self.ik = 1j * self.k
        self.ik3 = self.ik**3

    def _dealias_product(self, u_hat: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
        """
        Compute dealiased product u*v using 3/2-rule.

        The 3/2-rule pads the Fourier coefficients to 3/2*N points,
        performs multiplication in physical space, then truncates back to N points.
        This properly handles aliasing in nonlinear convolution products.

        Parameters
        ----------
        u_hat : np.ndarray
            Fourier coefficients of first function
        v_hat : np.ndarray
            Fourier coefficients of second function

        Returns
        -------
        np.ndarray
            Fourier coefficients of dealiased product u*v
        """
        N = len(u_hat)
        M = int(3 * N // 2)

        # Pad with zeros in middle of frequency space
        # [low freqs, zeros, high freqs]
        u_hat_pad = np.concatenate([u_hat[: N // 2], np.zeros(M - N), u_hat[N // 2 :]])
        v_hat_pad = np.concatenate([v_hat[: N // 2], np.zeros(M - N), v_hat[N // 2 :]])

        # Multiply in physical space (on finer grid)
        u_pad = np.fft.ifft(u_hat_pad)
        v_pad = np.fft.ifft(v_hat_pad)
        w_pad = u_pad * v_pad

        # Transform back and truncate (keep low and high freqs, discard padded region)
        w_pad_hat = np.fft.fft(w_pad)
        w_hat = (3 / 2) * np.concatenate([w_pad_hat[: N // 2], w_pad_hat[M - N // 2 :]])

        return w_hat

    def rhs(self, u: np.ndarray, t: float) -> np.ndarray:
        """
        Compute right-hand side of semi-discrete KdV equation.

        RHS = :math:`-6u u_x - u_{xxx}`

        Parameters
        ----------
        u : np.ndarray
            Solution at current time
        t : float
            Current time

        Returns
        -------
        np.ndarray
            Time derivative du/dt
        """
        # Compute derivatives in Fourier space
        u_hat = np.fft.fft(u)
        ux_hat = self.ik * u_hat
        uxxx_hat = self.ik3 * u_hat

        # Compute nonlinear term with optional dealiasing
        if self.dealias:
            # Use 3/2-rule for proper dealiasing of nonlinear product u * u_x
            nonlinear_hat = self._dealias_product(u_hat, ux_hat)
            nonlinear = np.fft.ifft(nonlinear_hat).real
        else:
            # Standard (aliased) computation
            ux = np.fft.ifft(ux_hat).real
            nonlinear = u * ux

        # Linear term (third derivative)
        uxxx = np.fft.ifft(uxxx_hat).real

        return -6 * nonlinear - uxxx

    def get_spectrum(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Fourier spectrum for spectral analysis.

        Parameters
        ----------
        u : np.ndarray
            Solution field

        Returns
        -------
        k : np.ndarray
            Wave numbers
        magnitude : np.ndarray
            Magnitude :math:`|\\hat{u}_k|` of Fourier coefficients
        phase : np.ndarray
            Phase angle of Fourier coefficients
        """
        u_hat = np.fft.fft(u)
        return self.k, np.abs(u_hat), np.angle(u_hat)

    def compute_eigenvalues(self, u_max: float) -> np.ndarray:
        """
        Compute eigenvalues of frozen-coefficient linearized KdV operator.

        For the KdV equation u_t = -6u*u_x - u_xxx, the linearization
        around a frozen coefficient u_max gives:
        L = -6*u_max*D1 - D3

        where D1 and D3 are the first and third derivative operators.
        This is useful for stability analysis.

        Parameters
        ----------
        u_max : float
            Maximum amplitude for frozen-coefficient approximation

        Returns
        -------
        np.ndarray
            Complex eigenvalues of the linearized operator
        """
        # Build differentiation matrices in Fourier space
        # For Fourier methods, D1 and D3 are diagonal in spectral space:
        # D1[k] = ik, D3[k] = (ik)^3 = -ik^3

        # Construct the frozen-coefficient operator matrix
        # In spectral space: L_hat[k] = -6*u_max*(ik) - (ik)^3
        eigvals = -6 * u_max * self.ik - self.ik3

        return eigvals

    def solve(
        self,
        u0: np.ndarray,
        t_final: float,
        dt: float,
        save_every: int = 1,
        integrator: TimeIntegrator = None,
        measure_performance: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
        """
        Solve KdV equation from t=0 to t=t_final.

        Parameters
        ----------
        u0 : np.ndarray
            Initial condition
        t_final : float
            Final time
        dt : float
            Time step
        save_every : int, optional
            Save solution every N steps, by default 1
        integrator : TimeIntegrator, optional
            Time integration method, by default RK4()
        measure_performance : bool, optional
            Measure and return performance metrics, by default False

        Returns
        -------
        t_saved : np.ndarray
            Times at which solution was saved
        u_saved : np.ndarray
            Saved solutions (shape: [n_saves, N])
        performance : dict, optional
            Performance metrics (returned if measure_performance=True):
            - 'wall_time_s': Total wall time in seconds
            - 'mean_step_time_ms': Mean time per step in milliseconds
            - 'std_step_time_ms': Standard deviation of step times
            - 'nsteps': Total number of time steps
        """
        if integrator is None:
            integrator = get_time_integrator("rk4")

        n_steps = int(np.ceil(t_final / dt))
        n_saves = n_steps // save_every + 1

        u = u0.copy()
        t = 0.0

        t_saved = np.zeros(n_saves)
        u_saved = np.zeros((n_saves, self.N))

        t_saved[0] = t
        u_saved[0] = u

        # Performance measurement
        if measure_performance:
            import time

            step_times = []

        save_idx = 1
        for step in range(n_steps):
            if measure_performance:
                t_start = time.perf_counter()

            u = integrator.step(self.rhs, u, t, dt)
            t += dt

            if measure_performance:
                step_times.append(time.perf_counter() - t_start)

            if (step + 1) % save_every == 0 and save_idx < n_saves:
                t_saved[save_idx] = t
                u_saved[save_idx] = u
                save_idx += 1

        if measure_performance:
            step_times_ms = np.array(step_times) * 1000  # Convert to ms
            performance = {
                "wall_time_s": np.sum(step_times),
                "mean_step_time_ms": np.mean(step_times_ms),
                "std_step_time_ms": np.std(step_times_ms),
                "nsteps": n_steps,
            }
            return t_saved[:save_idx], u_saved[:save_idx], performance

        return t_saved[:save_idx], u_saved[:save_idx]

    @staticmethod
    def compute_conserved_quantities(
        u: np.ndarray, dx: float
    ) -> tuple[float, float, float]:
        """
        Compute conserved quantities for KdV equation.

        Mass:     
        Momentum: 
        Energy:   

        Parameters
        ----------
        u : np.ndarray
            Solution field
        dx : float
            Grid spacing

        Returns
        -------
        M : float
            Mass
        V : float
            Momentum
        E : float
            Energy
        """
        N = len(u)
        k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        ik = 1j * k

        # Compute derivative in Fourier space
        u_hat = np.fft.fft(u)
        ux_hat = ik * u_hat
        ux = np.fft.ifft(ux_hat).real

        M = np.sum(u) * dx
        V = np.sum(u**2) * dx
        E = np.sum(0.5 * ux**2 - u**3) * dx

        return M, V, E

    @staticmethod
    def stable_dt(
        N: int,
        L: float,
        u_max: float,
        *,
        integrator_name: str = "rk4",
        dealiased: bool = False,
    ) -> float:
        """
        Compute stable time step via absolute-stability & frozen coefficients.

        Semi-discrete KdV eigenvalues (frozen u):
        :math:`\\lambda_k = ik(k^2 - 6u_{max})`, so :math:`|\\lambda_k| = |k| \\cdot |k^2 - 6u_{max}|`.
        Choose :math:`\\Delta t` so that :math:`\\Delta t |\\lambda_{max}| \\leq s(method)`
        on the imaginary axis.

        Parameters
        ----------
        N, L      : grid size and half-domain for [-L, L]
        u_max     : max :math:`|u|` expected (for 1-soliton with parameter c, use u_max = c/2)
        integrator_name : one of {"rk4", "rk3"}
        dealiased : set True if using 2/3-rule (k_max = N/3 instead of N/2)

        Returns
        -------
        float
            Suggested stable time step.
        """
        name = "".join(ch for ch in integrator_name.lower() if ch.isalnum())

        # Imag-axis crossing s(method):
        # (from stability diagrams / standard results)
        imag_axis_radius = {
            "rk4": 2.828,
            "rk3": 1.732,
        }.get(name, 2.828 if "rk4" in name else 1.732 if "rk3" in name else 0.0)

        # k_max from Fourier grid
        kmax = (np.pi / L) * (N // (3 if dealiased else 2))

        # worst-case |λ| from frozen coefficient analysis: |λ_k| = |k| * |k² - 6u_max|
        lam_max = kmax * abs(kmax**2 - 6.0 * abs(u_max))

        if lam_max == 0.0:
            return np.inf  # trivial case

        if imag_axis_radius == 0.0:
            # Warn the caller by returning an ultra-conservative estimate
            return 1e-12 / lam_max

        return imag_axis_radius / lam_max
