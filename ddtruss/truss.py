import numpy as np
from scipy.linalg import lu_factor, lu_solve


class Truss:
    """
    Simple static equilibrium solver for truss structures

    Args
    ----
    points : ndarray, shape (n_points, dim)
        Point coordinates with spatial dimension ``dim``
    lines : ndarray, shape (n_lines, 2)
        Connectivity of bars
    """

    def __init__(self, points, lines):
        self.points = np.asarray(points)
        self.lines = np.asarray(lines)
        assert self.lines.shape[1] == 2

        self.n_points, self.dim = self.points.shape
        self.n_lines = self.lines.shape[0]
        self.n_ddl = self.n_points * self.dim

        # Solving
        self.B = None  # gradient matrix
        self.K = None
        self.F = None
        self.K_lu = None

    def integrate(self, fun):
        """
        Integrate a function defined on bars

        Args
        ----
        fun : ndarray, shape (n_lines, ...)
            Function to be integrated
        """
        return np.sum(self.A * self.L * fun, axis=0)

    def solve(self, A=1, E=1, U_dict={}, F_dict={}, sig0=None, refactor_K=True):
        """
        Solve the static equilibrium problem for the truss structure

        Args
        ----
        A : float or ndarray, shape (n_lines, )
            Cross section area
        E : float
            Young's modulus
        U_dict : dict
            Prescribed displacement ``{point_id: (Ux, Uy), ...}``, use ``None`` for ``Ux`` or ``Uy`` when this component is not concerned
        F_dict : dict
            Prescribed nodal force ``{point_id: (Fx, Fy), ...}``
        sig0 : ndarray, shape (n_lines, )
            Initial stress
        refactor_K : bool
            Whether force reconstructing and performing LU factorization for the stiffness matrix

        Returns
        -------
        u : ndarray, shape (n_ddl, )
            Displacement solution
        eps : ndarray, shape (n_lines, )
            Strain
        sig : ndarray, shape (n_lines, )
            Stress
        """
        # Geometric and material properties
        try:
            assert len(A) == self.n_lines
            self.A = A
        except TypeError:
            self.A = A * np.ones(self.n_lines)
        except AssertionError:
            raise RuntimeError("Length of A must match the number of bars")
        self.E = E

        # Stiffness matrix and LU factorization
        if self.K_lu is None or refactor_K:
            self._compute_elementary_quantities()
            self._construct_K()
            self._apply_Dirichlet(U_dict, K=self.K)
            self.K_lu, self.K_piv = lu_factor(self.K, check_finite=False)

        # Right hand side
        self._apply_force(F_dict, sig0)
        self._apply_Dirichlet(U_dict, F=self.F)

        # Solve and post-processing
        u = lu_solve((self.K_lu, self.K_piv), self.F, check_finite=False)
        eps = self._strain(u)
        return u, eps, self.E * eps

    def _global_ddl_indices(self, point_id):
        return np.array([2 * point_id, 2 * point_id + 1], dtype=int)

    def _compute_elementary_quantities(self):
        if self.B is None:
            self.L = np.zeros(self.n_lines)
            self.B = np.zeros((self.n_lines, 4))

            for i, line in enumerate(self.lines):
                coords = self.points[line]
                x, y = coords[:, 0], coords[:, 1]
                self.L[i] = np.linalg.norm(coords[1] - coords[0])
                cos = (x[1] - x[0]) / self.L[i]
                sin = (y[1] - y[0]) / self.L[i]
                self.B[i] = np.array([-cos, -sin, cos, sin]) / self.L[i]

    def _construct_K(self):
        self.K = np.zeros((self.n_ddl, self.n_ddl))
        for i, line in enumerate(self.lines):
            global_indices = np.hstack(
                [self._global_ddl_indices(line[0]), self._global_ddl_indices(line[1])]
            )
            Ke = self.A[i] * self.E * np.outer(self.B[i], self.B[i]) * self.L[i]
            self.K[global_indices[:, np.newaxis], global_indices] += Ke

    def _apply_Dirichlet(self, U_dict={}, K=None, F=None):
        for point_id, U in U_dict.items():
            assert len(U) == 2
            idx = [i for i in range(len(U)) if U[i] is not None]
            if len(idx) == 0:
                continue
            U = np.asarray(U)[idx]
            global_indices = self._global_ddl_indices(point_id)[idx]
            if K is not None:
                K[global_indices, :] = 0
                K[global_indices[:, np.newaxis], global_indices] = np.eye(len(idx))
            if F is not None:
                F[global_indices] = U

    def _apply_force(self, F_dict={}, sig0=None):
        self.F = np.zeros(self.n_ddl)

        # Prescribed nodal forces
        for point_id, F in F_dict.items():
            assert len(F) == 2
            global_indices = self._global_ddl_indices(point_id)
            self.F[global_indices] += F

        # Initial stress
        if sig0 is not None:
            assert len(sig0) == self.n_lines
            self._compute_elementary_quantities()
            for i, line in enumerate(self.lines):
                global_indices = np.hstack(
                    [
                        self._global_ddl_indices(line[0]),
                        self._global_ddl_indices(line[1]),
                    ]
                )
                Fe = -self.A[i] * sig0[i] * self.B[i] * self.L[i]
                self.F[global_indices] += Fe

    def _strain(self, u):
        eps = np.zeros(self.n_lines)
        for i, line in enumerate(self.lines):
            global_indices = np.hstack(
                [self._global_ddl_indices(line[0]), self._global_ddl_indices(line[1])]
            )
            u_ = u[global_indices]
            eps[i] = self.B[i] @ u_
        return eps

    def plot(self, ax=None, u=None, eps=None, points_id=False, lines_id=False):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        if points_id:
            for i, point in enumerate(self.points):
                ax.text(point[0], point[1], f"{i:d}")

        if eps is not None:
            eps_absmax = np.abs(eps).max()
            norm = plt.cm.colors.Normalize(vmin=-eps_absmax, vmax=eps_absmax)
            colors = plt.cm.coolwarm(norm(eps))

        for i, line in enumerate(self.lines):
            coords = self.points[line]
            style = "--"
            color = "gray"
            if eps is not None and u is None:
                style = "-"
                color = colors[i]
            ax.plot(coords[:, 0], coords[:, 1], style, color=color)
            if lines_id:
                centroid = np.mean(coords, axis=0)
                ax.text(centroid[0], centroid[1], f"({i:d})")

        if u is not None:
            deformed = self.points + u.reshape((-1, 2))
            for i, line in enumerate(self.lines):
                coords = deformed[line]
                color = "C0"
                if eps is not None and u is not None:
                    color = colors[i]
                ax.plot(coords[:, 0], coords[:, 1], color=color)
