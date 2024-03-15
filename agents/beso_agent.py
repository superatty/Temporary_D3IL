from functools import partial
import logging
from typing import Optional
from collections import deque

import einops
from omegaconf import DictConfig
import hydra
import torch

from agents.base_agent import BaseAgent

from agents.models.beso.agents.diffusion_agents.k_diffusion.gc_sampling import *
import agents.models.beso.agents.diffusion_agents.k_diffusion.utils as utils

# A logger for this file
log = logging.getLogger(__name__)


class BesoPolicy(nn.Module):
    def __init__(
        self,
        model: DictConfig,
        obs_encoder: DictConfig,
        n_sampling_steps: int,
        sampler_type: str,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        sigma_sample_density_type: str,
        sigma_sample_density_mean: float,
        sigma_sample_density_std: float,
        rho: float,
        noise_scheduler: str,
        visual_input: bool = False,
        device: str = "cpu",
    ):
        super(BesoPolicy, self).__init__()

        self.visual_input = visual_input
        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)
        self.model = hydra.utils.instantiate(model).to(device)

        self.n_sampling_steps = n_sampling_steps
        self.sampler_type = sampler_type

        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        self.sigma_sample_density_mean = sigma_sample_density_mean
        self.sigma_sample_density_std = sigma_sample_density_std
        
        self.rho = rho
        self.noise_scheduler = noise_scheduler
        
        self.device = device

    def get_embedding(self, inputs):

        if self.visual_input:
            agentview_image, in_hand_image, state = inputs

            B, T, C, H, W = agentview_image.size()

            agentview_image = agentview_image.view(B * T, C, H, W)
            in_hand_image = in_hand_image.view(B * T, C, H, W)
            state = state.view(B * T, -1)

            obs_dict = {
                "agentview_image": agentview_image,
                "in_hand_image": in_hand_image,
                "robot_ee_pos": state,
            }

            obs = self.obs_encoder(obs_dict)
            obs = obs.view(B, T, -1)

        else:
            obs = self.obs_encoder(inputs)

        return obs

    def get_loss(self, state, action, goal=None):
        obs = self.get_embedding(state)

        noise = torch.randn_like(action)
        sigma = self.make_sample_density()(shape=(len(action),), device=self.device)

        return self.model.loss(obs, action, goal, noise, sigma)

    def forward(self, state, goal=None):
        obs = self.get_embedding(state)

        sigmas = self.get_noise_schedule(self.n_sampling_steps, self.noise_scheduler)
        x = (
            torch.randn(
                (len(obs), 1, self.scaler.y_bounds.shape[1]),
                device=self.device,
            )
            * self.sigma_max
        )
        
        if len(self.action_context) > 0:
            previous_a = torch.cat(tuple(self.action_context), dim=1)
            x = torch.cat([previous_a, x], dim=1)
            
        x_0 = self.sample_loop(sigmas, x, obs, goal)
        
        if x_0.size()[1] > 1 and len(x_0.size()) == 3:
            x_0 = x_0[:, -1, :]
            
        x_0 = x_0.clamp_(self.min_action, self.max_action)
        self.action_context.append(x_0)
        
        return x_0

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        """
        sd_config = []

        if self.sigma_sample_density_type == "lognormal":
            loc = self.sigma_sample_density_mean
            scale = self.sigma_sample_density_std
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        elif self.sigma_sample_density_type == "loglogistic":
            loc = sd_config["loc"] if "loc" in sd_config else math.log(self.sigma_data)
            scale = sd_config["scale"] if "scale" in sd_config else 0.5
            min_value = (
                sd_config["min_value"] if "min_value" in sd_config else self.sigma_min
            )
            max_value = (
                sd_config["max_value"] if "max_value" in sd_config else self.sigma_max
            )
            return partial(
                utils.rand_log_logistic,
                loc=loc,
                scale=scale,
                min_value=min_value,
                max_value=max_value,
            )

        elif self.sigma_sample_density_type == "loguniform":
            min_value = (
                sd_config["min_value"] if "min_value" in sd_config else self.sigma_min
            )
            max_value = (
                sd_config["max_value"] if "max_value" in sd_config else self.sigma_max
            )
            return partial(
                utils.rand_log_uniform, min_value=min_value, max_value=max_value
            )
        elif self.sigma_sample_density_type == "uniform":
            return partial(
                utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max
            )

        elif self.sigma_sample_density_type == "v-diffusion":
            min_value = self.min_value if "min_value" in sd_config else self.sigma_min
            max_value = (
                sd_config["max_value"] if "max_value" in sd_config else self.sigma_max
            )
            return partial(
                utils.rand_v_diffusion,
                sigma_data=self.sigma_data,
                min_value=min_value,
                max_value=max_value,
            )
        elif self.sigma_sample_density_type == "discrete":
            sigmas = self.get_noise_schedule(self.n_sampling_steps, "exponential")
            return partial(utils.rand_discrete, values=sigmas)
        elif self.sigma_sample_density_type == "split-lognormal":
            loc = sd_config["mean"] if "mean" in sd_config else sd_config["loc"]
            scale_1 = (
                sd_config["std_1"] if "std_1" in sd_config else sd_config["scale_1"]
            )
            scale_2 = (
                sd_config["std_2"] if "std_2" in sd_config else sd_config["scale_2"]
            )
            return partial(
                utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2
            )
        else:
            raise ValueError("Unknown sample density type")

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps
        """
        if noise_schedule_type == "karras":
            return get_sigmas_karras(
                n_sampling_steps, self.sigma_min, self.sigma_max, self.rho, self.device
            )
        elif noise_schedule_type == "exponential":
            return get_sigmas_exponential(
                n_sampling_steps, self.sigma_min, self.sigma_max, self.device
            )
        elif noise_schedule_type == "vp":
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == "linear":
            return get_sigmas_linear(
                n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device
            )
        elif noise_schedule_type == "cosine_beta":
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == "ve":
            return get_sigmas_ve(
                n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device
            )
        elif noise_schedule_type == "iddpm":
            return get_iddpm_sigmas(
                n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device
            )
        raise ValueError("Unknown noise schedule type")
    
    def sample_loop(
        self,
        sigmas,
        x_t: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
    ):
        """
        Main method to generate samples depending on the chosen sampler type for rollouts
        """
        # ODE deterministic
        if self.sampler_type == "lms":
            x_0 = sample_lms(
                self.model,
                state,
                x_t,
                goal,
                sigmas,
                disable=True,
            )
        # ODE deterministic can be made stochastic by S_churn != 0
        elif self.sampler_type == "heun":
            x_0 = sample_heun(
                self.model,
                state,
                x_t,
                goal,
                sigmas,
                disable=True,
            )
        # ODE deterministic
        elif self.sampler_type == "euler":
            x_0 = sample_euler(
                self.model, state, x_t, goal, sigmas, disable=True
            )
        # SDE stochastic
        elif self.sampler_type == "ancestral":
            x_0 = sample_dpm_2_ancestral(
                self.model, state, x_t, goal, sigmas, disable=True
            )
            # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif self.sampler_type == "euler_ancestral":
            x_0 = sample_euler_ancestral(
                self.model, state, x_t, goal, sigmas, disable=True
            )
        # ODE deterministic
        elif self.sampler_type == "dpm":
            x_0 = sample_dpm_2(self.model, state, x_t, goal, disable=True)
        elif self.sampler_type == "ddim":
            x_0 = sample_ddim(
                self.model, state, x_t, goal, sigmas, disable=True
            )
        # ODE deterministic
        elif self.sampler_type == "dpm_adaptive":
            x_0 = sample_dpm_adaptive(
                self.model,
                state,
                x_t,
                goal,
                sigmas[-2].item(),
                sigmas[0].item(),
                disable=True,
            )
        # ODE deterministic
        elif self.sampler_type == "dpm_fast":
            x_0 = sample_dpm_fast(
                self.model,
                state,
                x_t,
                goal,
                sigmas[-2].item(),
                sigmas[0].item(),
                len(sigmas),
                disable=True,
            )
        # 2nd order solver
        elif self.sampler_type == "dpmpp_2s_ancestral":
            x_0 = sample_dpmpp_2s_ancestral(
                self.model, state, x_t, goal, sigmas, disable=True
            )
        elif self.sampler_type == "dpmpp_2s":
            x_0 = sample_dpmpp_2s(
                self.model, state, x_t, goal, sigmas, disable=True
            )
        # 2nd order solver
        elif self.sampler_type == "dpmpp_2m":
            x_0 = sample_dpmpp_2m(
                self.model, state, x_t, goal, sigmas, disable=True
            )
        elif self.sampler_type == "dpmpp_2m_sde":
            x_0 = sample_dpmpp_sde(
                self.model, state, x_t, goal, sigmas, disable=True
            )
        else:
            raise ValueError("desired sampler type not found!")
        return x_0

    def get_params(self):
        return self.parameters()


class BesoAgent(BaseAgent):

    def __init__(
        self,
        model: DictConfig,
        optimization: DictConfig,
        trainset: DictConfig,
        valset: DictConfig,
        train_batch_size,
        val_batch_size,
        num_workers,
        device: str,
        epoch: int,
        scale_data,
        use_ema: bool,
        lr_scheduler: DictConfig,
        decay: float,
        update_ema_every_n_steps: int,
        window_size: int,
        eval_every_n_epochs: int = 50,
    ):

        super().__init__(
            model=model,
            optimization=optimization,
            trainset=trainset,
            valset=valset,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device,
            epoch=epoch,
            scale_data=scale_data,
            eval_every_n_epochs=eval_every_n_epochs,
            use_ema=use_ema,
            decay=decay,
            update_ema_every_n_steps=update_ema_every_n_steps,
            lr_scheduler=lr_scheduler,
        )

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(
            self.device
        )
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(
            self.device
        )
        self.model.min_action = self.min_action
        self.model.max_action = self.max_action
        self.model.scaler = self.scaler

        self.eval_model_name = "eval_best_beso.pth"
        self.last_model_name = "last_beso.pth"

        # get the window size for prediction
        self.window_size = window_size

        self.bp_image_context = deque(maxlen=self.window_size)
        self.inhand_image_context = deque(maxlen=self.window_size)
        self.des_robot_pos_context = deque(maxlen=self.window_size)

        self.obs_context = deque(maxlen=self.window_size)
        # if we use DiffusionGPT we need an action context and use deques to store the actions
        self.model.action_context = deque(maxlen=self.window_size - 1)

    def reset(self):
        """Resets the context of the model."""
        self.obs_context.clear()
        self.model.action_context.clear()

        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.des_robot_pos_context.clear()

    @torch.no_grad()
    def predict(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        if_vision=False,
    ) -> torch.Tensor:
        
        if if_vision:
            bp_image, inhand_image, des_robot_pos = state

            bp_image = torch.from_numpy(bp_image).to(self.device).float().unsqueeze(0)
            inhand_image = (
                torch.from_numpy(inhand_image).to(self.device).float().unsqueeze(0)
            )
            des_robot_pos = (
                torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0)
            )

            des_robot_pos = self.scaler.scale_input(des_robot_pos)

            self.bp_image_context.append(bp_image)
            self.inhand_image_context.append(inhand_image)
            self.des_robot_pos_context.append(des_robot_pos)

            bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
            inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)
            des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)

            input_state = (bp_image_seq, inhand_image_seq, des_robot_pos_seq)

            input_state = self.model.get_embedding(input_state)

        else:
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            state = self.scaler.scale_input(state)
            if goal is not None:
                goal = self.scaler.scale_input(goal)

            self.obs_context.append(state)
            input_state = torch.stack(tuple(self.obs_context), dim=1)

        if goal is not None:
            goal = self.scaler.scale_input(goal)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()
        
        x_0 = self.model(input_state, goal)

        x_0 = x_0.clamp_(self.min_action, self.max_action)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        model_pred = self.scaler.inverse_scale_output(x_0)

        return model_pred.detach().cpu().numpy()[0]