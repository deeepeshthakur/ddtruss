import numpy as np
from scipy.spatial import cKDTree


class DataDrivenSolver:
    """
    Data-driven solver for truss structures

    Args
    ----
    truss :
        Object defining the truss structure
    """

    def __init__(self, truss):
        self.truss = truss

    def load_material_data(self, material_data):
        """
        Load one-dimensional material data

        Args
        ----
        material_data : ndarray, shape (n_data, 2)
            Experimentally measured ``(strain, stress)`` pairs
        """
        assert material_data.shape[1] == 2
        self.material_data = material_data
        self.n_data = material_data.shape[0]
        self.data_tree = None

    def solve(
        self,
        A=1,
        U_dict={},
        F_dict={},
        n_iterations=100,
        E_num=None,
        n_neighbors=1,
        idx=None,
        save_history=False,
    ):
        """
        Solve the static equilibrium problem for the truss structure using a
        data-driven approach

        Args
        ----
        A : float or ndarray, shape (n_lines, )
            Cross section area
        U_dict : dict
            Prescribed displacement ``{point_id: (Ux, Uy), ...}``
        F_dict : dict
            Prescribed nodal force ``{point_id: (Fx, Fy), ...}``
        n_iterations : int
            Maximimum iteration for the data-driven solver
        E_num : float
            Numerical value
        n_neighbors : int
            Number of nearest local states to look for in the material data
        idx : ndarray, shape (n_lines, )
            Initial local states
        save_history : bool
            Whether also save more iteration  history

        Returns
        -------
        u : ndarray, shape (n_ddl, )
            Displacement solution
        eps : ndarray, shape (n_lines, )
            Strain
        sig : ndarray, shape (n_lines, )
            Stress
        iter_history : ndarray
            Iteration history for objective function and if asked strain/stress
        """
        # Initialize the truss solver
        if E_num is None:
            ind = np.isclose(self.material_data[:, 0], 0)
            E_secent = self.material_data[~ind, 1] / self.material_data[~ind, 0]
            E_num = E_secent.mean()
        self.sqE = np.sqrt(E_num)
        self.iter_history = None

        # Initialize local states
        if idx is None:
            idx = np.random.randint(self.n_data, size=self.truss.n_lines)
        eps_sig_ = self.material_data[idx]
        self._iter_history(append=("idx", idx))

        # Define the zeroed Dirichlet conditions
        U_dict_0 = {}
        for key in U_dict:
            U_dict_0[key] = [0 if value is not None else None for value in U_dict[key]]

        while self._iter_history() <= n_iterations:
            # Solve the 1st problem for u driven by initial stress
            sig0 = -E_num * eps_sig_[:, 0]
            if self._iter_history() == 0:
                construct_K = True
            else:
                construct_K = False
            u, eps, _ = self.truss.solve(
                A=A, E=E_num, U_dict=U_dict, sig0=sig0, construct_K=construct_K
            )

            # Solve the 2nd problem for eta driven by initial stress and applied force
            sig0 = eps_sig_[:, 1]
            _, eps_eta, _ = self.truss.solve(
                A=A, E=E_num, U_dict=U_dict_0, F_dict=F_dict, sig0=sig0
            )
            sig = eps_sig_[:, 1] + E_num * eps_eta

            # Find the nearest material data points
            eps_sig = np.hstack([eps.reshape((-1, 1)), sig.reshape((-1, 1))])
            eps_sig_idx, eps_sig_, f_obj = self._nearest_material_points(
                eps_sig, n_neighbors=n_neighbors
            )

            # Save history
            self._iter_history(append=("idx", eps_sig_idx))
            self._iter_history(append=("f_obj", f_obj))
            if save_history:
                self._iter_history(append=("eps", eps))
                self._iter_history(append=("sig", sig))

            # Check for convergence
            if np.allclose(idx, eps_sig_idx):
                self._iter_history(finalize=True)
                return u, eps_sig[:, 0], eps_sig[:, 1], self.iter_history
            else:
                idx = eps_sig_idx.copy()
        else:
            return RuntimeError(
                f"Data-driven solver not converged after {n_iterations} iterations"
            )

    def _iter_history(self, append=None, finalize=False):
        if self.iter_history is None:
            self.iter_history = {}
            for key in ["f_obj", "idx", "eps", "sig"]:
                self.iter_history[key] = []
        if finalize:
            for key in self.iter_history:
                self.iter_history[key] = np.array(self.iter_history[key])
        elif append is not None:
            key, value = append
            self.iter_history[key].append(value)
        return len(self.iter_history["f_obj"])

    def _nearest_material_points(self, eps_sig, n_neighbors=1):
        if self.data_tree is None:
            eps_sig_data = self.material_data.copy()
            eps_sig_data[:, 0] *= self.sqE
            eps_sig_data[:, 1] /= self.sqE
            self.data_tree = cKDTree(eps_sig_data)

        # Scale the strain/stress data
        eps_sig_rescaled = eps_sig.copy()
        eps_sig_rescaled[:, 0] *= self.sqE
        eps_sig_rescaled[:, 1] /= self.sqE

        dist, idx = self.data_tree.query(eps_sig_rescaled, k=n_neighbors, n_jobs=-1)
        if n_neighbors > 1:
            dist = dist.mean(axis=1)
            k_means = self.data_tree.data[idx].mean(axis=1)
            _, idx = self.data_tree.query(k_means, k=1, n_jobs=-1)

        f_obj = self.truss.integrate(dist)
        eps_sig_optimal = self.material_data[idx]
        return idx, eps_sig_optimal, f_obj
