from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

from . import dynamics
from .gmm2d import GMM2D
from ..latent.distributions import PDistribution


class MultimodalGenerativeCVAEDecoder(nn.Module):

    def __init__(
            self,
            agent_type: str,
            len_state: int,
            len_pred_state: int,
            include_robot: bool,
            x_size: int,
            z_size: int,
            rnn_dim: int,
            n_gmm_components: int,
            dynamical_model_config: Mapping[str, Any],
            len_state_robot: Optional[int] = None,
    ):
        super().__init__()

        self._include_robot = include_robot

        self.agent_type = agent_type
        self.len_state = len_state
        self.len_pred_state = len_pred_state
        self.x_size = x_size
        self.z_size = z_size
        self.rnn_dim = rnn_dim
        self.n_gmm_components = n_gmm_components
        self.dynamical_model_config = dynamical_model_config
        self.len_state_robot = len_state_robot

        self._build()

    @property
    def include_robot(self) -> bool:
        return self._include_robot

    @include_robot.setter
    def include_robot(self, value: bool):
        self._include_robot = value

    def _build(self):
        # Build the RNN
        input_size = self.len_pred_state + self.z_size + self.x_size
        if self.include_robot:
            input_size += self.len_state_robot

        self.state_action = nn.Linear(self.len_state, self.len_pred_state)
        self.rnn_cell = nn.GRUCell(input_size, self.rnn_dim)
        self.initial_h = nn.Linear(self.z_size + self.x_size, self.rnn_dim)

        # Build the GMM parameters models
        self.proj_to_gmm = nn.ModuleDict()
        self.proj_to_gmm["log_pis"] = nn.Linear(
            self.rnn_dim,
            self.n_gmm_components
        )
        self.proj_to_gmm["mus"] = nn.Linear(
            self.rnn_dim,
            self.n_gmm_components * self.len_pred_state
        )
        self.proj_to_gmm["log_sigmas"] = nn.Linear(
            self.rnn_dim,
            self.n_gmm_components * self.len_pred_state
        )
        self.proj_to_gmm["corrs"] = nn.Linear(
            self.rnn_dim,
            self.n_gmm_components
        )

        # Instantiate the dynamical model
        if self.dynamical_model_config["type"] == "Unicycle":
            self.dynamical_model_config["kwargs"]["xz_size"] = self.x_size

        self.dynamical_model = getattr(
            dynamics,
            self.dynamical_model_config["type"],
        )(**self.dynamical_model_config["kwargs"])

    def project_to_gmm_params(
            self,
            t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            parameter: self.proj_to_gmm[parameter](t)
            for parameter in self.proj_to_gmm
        }

    def forward(
            self,
            x,
            x_nr_t,
            y_r,
            x_r_t0,
            z_stacked,
            dist,
            initial_conditions,
            prediction_horizon,
            n_samples: int,
            n_components: int = 1,
            gmm_mode: bool = False
    ):
        # Check if the data is being decoded for predicting
        is_predicting = isinstance(dist, PDistribution)

        # Add the input to
        z = torch.reshape(z_stacked, (-1, self.z_size))
        zx = torch.cat([z, x.repeat(n_samples * n_components, 1)], dim=1)

        # Initialise the state
        state = self.initial_h(zx)

        log_pis, mus, log_sigmas, corrs = [], [], [], []

        # Get the first action state
        a_0 = self.state_action(x_r_t0)

        inputs = [zx, a_0.repeat(n_samples * n_components, 1)]
        if x_nr_t is not None and self.include_robot:
            inputs.append(x_nr_t.repeat(n_samples * n_components, 1))
        inputs = torch.cat(inputs, dim=1)

        for t_f in range(prediction_horizon):
            state = self.rnn_cell(inputs, state)

            # Create a Gaussian mixture model
            gmm_params = self.project_to_gmm_params(state)
            gmm = GMM2D(**gmm_params)

            a_t = gmm.mode() if is_predicting and gmm_mode else gmm.rsample()

            log_pis.append(
                dist.logits.repeat(n_samples, 1, 1)
                if n_components > 1 else
                torch.ones_like(
                    gmm_params["corrs"]
                        .reshape(n_samples, n_components, -1)
                        .permute(0, 2, 1)
                        .reshape(-1, 1)
                )
            )
            log_sigmas.append(
                gmm_params["log_sigmas"].reshape(
                    n_samples, n_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * n_components)
            )
            mus.append(
                gmm_params["mus"].reshape(
                    n_samples, n_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * n_components)
            )
            corrs.append(
                gmm_params["corrs"].reshape(
                    n_samples, n_components, -1
                ).permute(0, 2, 1).reshape(-1, n_components)
            )

            # Update inputs
            inputs = [
                zx, a_t, y_r[:, t_f].repeat(n_samples * n_components, 1)
            ] if y_r is not None and self.include_robot else [
                zx, a_t
            ]
            inputs = torch.cat(inputs, dim=1)

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        # Create a Gaussian mixture model distribution for actions
        a_dist = GMM2D(
            torch.reshape(
                log_pis,
                [n_samples, -1, prediction_horizon, n_components]
            ),
            torch.reshape(
                mus,
                [
                    n_samples,
                    -1,
                    prediction_horizon,
                    n_components * self.len_pred_state
                ]
            ),
            torch.reshape(
                log_sigmas,
                [
                    n_samples,
                    -1,
                    prediction_horizon,
                    n_components * self.len_pred_state
                ]
            ),
            torch.reshape(
                corrs,
                [n_samples, -1, prediction_horizon, n_components]
            )
        )

        # Initialize the dynamical model
        self.dynamical_model.initial_conditions = initial_conditions

        # Integrate the actions
        y_dist = self.dynamical_model.integrate_distribution(a_dist, x) if \
            self.dynamical_model_config["use_distribution"] \
            else a_dist

        # TODO: Check if we change depending of training stage
        a_sample = a_dist.mode() if gmm_mode else a_dist.rsample()
        preds = self.dynamical_model.integrate_samples(a_sample, x)

        return y_dist, preds
