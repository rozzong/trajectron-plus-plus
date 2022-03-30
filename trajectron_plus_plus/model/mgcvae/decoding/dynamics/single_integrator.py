from typing import List, Mapping, Union

import torch

from ._dynamics import DynamicalModel
from ..gmm2d import GMM2D


class SingleIntegrator(DynamicalModel):

    def __init__(
            self,
            dt: float,
            limits: Mapping[str, Union[int, float]],
    ):
        self.F = None
        self.F_t = None

        super().__init__(dt, limits)

    @staticmethod
    def _attach_dim(
            t: torch.Tensor,
            n_dim_to_prepend: int = 0,
            n_dim_to_append: int = 0
    ):
        """Prepend and append dimensions to a tensor's shape.

        Parameters
        ----------
        t : torch.Tensor
            The tensor to add dimensions to.
        n_dim_to_prepend : int, default: 0
            The number of dimensions to prepend.
        n_dim_to_append : int, default: 0
            The number of dimensions to append.

        Returns
        -------
        torch.Tensor
            The tensor with added dimensions.
        """
        return t.reshape(
            torch.Size([1] * n_dim_to_prepend)
            + t.shape
            + torch.Size([1] * n_dim_to_append)
        )

    @staticmethod
    def _block_diag(
            t: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """Make a block diagonal tensor along dim=-3.

        Parameters
        ----------
        t : Union[torch.Tensor, List[torch.Tensor]]


        Returns
        -------
        torch.Tensor


        Examples
        --------
        >>> block_diag(torch.ones(4, 3, 2))
            returns a 12 x 8 matrix with blocks of 3 x 2 ones.
        """
        if isinstance(t, list):
            t = torch.cat([t_.unsqueeze(-3) for t_ in t], -3)

        n = t.shape[-3]
        size_0 = t.shape[:-3]
        size_1 = t.shape[-2:]
        t2 = t.unsqueeze(-2)
        eye = SingleIntegrator._attach_dim(
            torch.eye(n, device=t.device).unsqueeze(-2),
            t.dim() - 3,
            1
        )

        return (t2 * eye).reshape(
            size_0 + torch.Size(torch.tensor(size_1) * n)
        )

    def initialize(self, *args, **kwargs) -> None:
        self.F = torch.eye(4, dtype=torch.float32)
        self.F[0:2, 2:] = torch.eye(2, dtype=torch.float32) * self.dt
        self.F_t = self.F.transpose(-2, -1)

    def integrate_samples(self, v, x=None):
        """
        Integrates deterministic samples of velocity.

        :param v: Velocity samples
        :param x: Not used for SI.
        :return: Position samples
        """
        p_0 = self.initial_conditions["pos"].unsqueeze(1)

        return torch.cumsum(v, dim=2) * self.dt + p_0

    def integrate_distribution(self, v_dist, x=None):
        r"""
        Integrates the GMM velocity distribution to a distribution over position.
        The Kalman Equations are used.

        .. math:: \mu_{t+1} =\textbf{F} \mu_{t}

        .. math:: \mathbf{\Sigma}_{t+1}={\textbf {F}} \mathbf{\Sigma}_{t} {\textbf {F}}^{T}

        .. math::
            \textbf{F} = \left[
                            \begin{array}{cccc}
                                \sigma_x^2 & \rho_p \sigma_x \sigma_y & 0 & 0 \\
                                \rho_p \sigma_x \sigma_y & \sigma_y^2 & 0 & 0 \\
                                0 & 0 & \sigma_{v_x}^2 & \rho_v \sigma_{v_x} \sigma_{v_y} \\
                                0 & 0 & \rho_v \sigma_{v_x} \sigma_{v_y} & \sigma_{v_y}^2 \\
                            \end{array}
                        \right]_{t}

        :param v_dist: Joint GMM Distribution over velocity in x and y direction.
        :param x: Not used for SI.
        :return: Joint GMM Distribution over position in x and y direction.
        """
        p_0 = self.initial_conditions["pos"].unsqueeze(1)
        ph = v_dist.mus.shape[-3]
        sample_batch_dim = list(v_dist.mus.shape[0:2])
        pos_dist_sigma_matrix_list = []

        pos_mus = p_0[:, None] + torch.cumsum(v_dist.mus, dim=2) * self.dt

        vel_dist_sigma_matrix = v_dist.get_covariance_matrix()
        pos_dist_sigma_matrix_t = torch.zeros(
            sample_batch_dim + [v_dist.components, 2, 2],
            device=vel_dist_sigma_matrix.device
        )

        for t in range(ph):
            vel_sigma_matrix_t = vel_dist_sigma_matrix[:, :, t]
            full_sigma_matrix_t = self._block_diag(
                [pos_dist_sigma_matrix_t, vel_sigma_matrix_t]
            )
            pos_dist_sigma_matrix_t = self.F[..., :2, :].matmul(
                full_sigma_matrix_t.matmul(self.F_t)[..., :2]
            )
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t)

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)

        return GMM2D.from_log_pis_mus_cov_mats(
            v_dist.log_pis,
            pos_mus,
            pos_dist_sigma_matrix
        )
