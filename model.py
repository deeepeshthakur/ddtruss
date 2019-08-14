import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import lu_factor, lu_solve


class DataDrivenTruss:
    def __init__(self, points, lines):
        self.points = np.asarray(points)
        self.lines = np.asarray(lines)
        assert self.lines.shape[1] == 2

        self.n_points, self.dim = self.points.shape
        self.n_lines = self.lines.shape[0]
        self.n_ddl = self.n_points * self.dim

    def load_material_data(self, material_data):
        """
        Load one-dimensional material data

        Args:
            material_data (ndarray, shape (n_data, 2)): ``(strain, stress)`` pairs
        """
        assert material_data.shape[1] == 2
        self.material_data = material_data
        self.n_data = material_data.shape[0]
        self.data_tree = None

    def define_problem(self, params={}, F_dict={}, U_dict={}):
        """
        Define the truss structure problem
        """
        self.A = params.get("A", 1)
        self.C = params.get("C", 1)
        self.n_iterations = params.get("n_iterations", 100)

        self.sqC = np.sqrt(self.C)
        self.F_dict = F_dict
        self.U_dict = U_dict

    def solve(self):
        self._compute_elementary_quantities()
        self._construct_K()
        self._construct_F()
        self._apply_Dirichlet(K=self.K, F=self.F)
        u = np.linalg.solve(self.K, self.F)
        eps = self._strain(u)
        return u, eps, self.C * eps

    def solve_data_driven(self):
        # Initialize local states
        idx = np.random.randint(self.n_data, size=self.n_lines)
        eps_sig_ = self.material_data[idx]

        self._compute_elementary_quantities()
        self._construct_K()
        self._apply_Dirichlet(K=self.K)
        lu, piv = lu_factor(self.K, check_finite=False)

        iteration = 0
        penalization = []
        while iteration <= self.n_iterations:
            self._construct_rhs(eps_sig_)
            self._apply_Dirichlet(F=self.F)
            self._apply_Dirichlet(F=self.F_eta)
            u = lu_solve((lu, piv), self.F, check_finite=False)
            eta = lu_solve((lu, piv), self.F_eta, check_finite=False)

            eps = self._strain(u)
            sig = eps_sig_[:, 1] + self.C * self._strain(eta)
            eps_sig = np.hstack([eps.reshape((-1, 1)), sig.reshape((-1, 1))])
            eps_sig_idx, eps_sig_, penal = self._optimal_local_states(eps_sig)
            penalization.append(penal)

            # Check for convergence
            if np.allclose(idx, eps_sig_idx):
                return u, eps_sig[:, 0], eps_sig[:, 1], penalization
            else:
                iteration += 1
                idx = eps_sig_idx.copy()
        else:
            return RuntimeError(
                f"Data-driven solver not converged after {self.n_iterations} iterations"
            )

    def plot(self, ax=None, u=None, eps=None, points_id=False, lines_id=False):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

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

    def _global_ddl_indices(self, point_id):
        return np.array([2 * point_id, 2 * point_id + 1], dtype=int)

    def _compute_elementary_quantities(self):
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
            Ke = self.A * self.C * np.outer(self.B[i], self.B[i]) * self.L[i]
            self.K[global_indices[:, np.newaxis], global_indices] += Ke

    def _construct_F(self):
        self.F = np.zeros(self.n_ddl)
        for point_id, force in self.F_dict.items():
            global_indices = self._global_ddl_indices(point_id)
            self.F[global_indices] = force

    def _construct_rhs(self, eps_sig):
        """
        Construct two RHS for data-driven problems
        """
        self.F = np.zeros(self.n_ddl)
        self.F_eta = np.zeros(self.n_ddl)
        for i, line in enumerate(self.lines):
            global_indices = np.hstack(
                [self._global_ddl_indices(line[0]), self._global_ddl_indices(line[1])]
            )
            Fe = self.A * self.C * eps_sig[i, 0] * self.B[i] * self.L[i]
            self.F[global_indices] += Fe
            Fe_eta = -self.A * eps_sig[i, 1] * self.B[i] * self.L[i]
            self.F_eta[global_indices] += Fe_eta

        for point_id, force in self.F_dict.items():
            global_indices = self._global_ddl_indices(point_id)
            self.F_eta[global_indices] += force

    def _apply_Dirichlet(self, K=None, F=None):
        for point_id, disp in self.U_dict.items():
            global_indices = self._global_ddl_indices(point_id)
            if K is not None:
                K[global_indices, :] = 0
                K[:, global_indices] = 0
                K[global_indices[:, np.newaxis], global_indices] = np.eye(2)
            if F is not None:
                F[global_indices] = 0

    def _strain(self, u):
        eps = np.zeros(self.n_lines)
        for i, line in enumerate(self.lines):
            global_indices = np.hstack(
                [self._global_ddl_indices(line[0]), self._global_ddl_indices(line[1])]
            )
            u_ = u[global_indices]
            eps[i] = self.B[i] @ u_
        return eps

    def _optimal_local_states(self, eps_sig):
        if self.data_tree is None:
            eps_sig_data = self.material_data.copy()
            eps_sig_data[:, 0] *= self.sqC
            eps_sig_data[:, 1] /= self.sqC
            self.data_tree = cKDTree(eps_sig_data)

        eps_sig[:, 0] *= self.sqC
        eps_sig[:, 1] /= self.sqC
        dist, idx = self.data_tree.query(eps_sig)
        penalization = np.sum(self.A * self.L * dist)
        eps_sig_optimal = self.material_data[idx]
        return idx, eps_sig_optimal, penalization
