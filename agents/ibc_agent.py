from omegaconf import DictConfig
import hydra
from typing import Optional
from einops import einops

import os
import logging

from agents.models.ibc.ebm_losses import *
from agents.base_agent import BaseAgent
from agents.models.ibc.samplers.langevin_mcmc import LangevinMCMCSampler
from agents.models.ibc.samplers.noise_sampler import NoiseSampler

# A logger for this file
log = logging.getLogger(__name__)


class IBC_Policy(nn.Module):
    def __init__(
        self,
        model: DictConfig,
        obs_encoder: DictConfig,
        sampler: DictConfig,
        loss_type: str = "info_nce",
        grad_norm_factor: float = 1,
        avrg_e_regularization: float = 0,
        kl_loss_factor: float = 0,
        visual_input: bool = False,
        device: str = "cpu",
    ):
        super(IBC_Policy, self).__init__()

        self.device = device
        self.visual_input = visual_input
        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)
        self.model = hydra.utils.instantiate(model).to(device)

        self.sampler = hydra.utils.instantiate(sampler)
        if isinstance(self.sampler, LangevinMCMCSampler) or isinstance(
            self.sampler, NoiseSampler
        ):
            self.use_grad_norm = True
        else:
            self.use_grad_norm = False

        self.loss_type = loss_type
        self.grad_norm_factor = grad_norm_factor
        self.avrg_e_regularization = avrg_e_regularization
        self.kl_loss_factor = kl_loss_factor

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

    def forward(self, state, goal=None):
        return self.sampler.infer(state, self.model, goal)

    def get_loss(self, state, action, goal=None):
        obs = self.get_embedding(state)
        
        if isinstance(self.sampler, LangevinMCMCSampler):
            negatives = self.sampler.gen_train_samples(
                obs.size(0),
                self.model,
                obs,
                goal,
                random_start_points=True,
            )
        elif isinstance(self.sampler, NoiseSampler):
            negatives = self.sampler.gen_train_samples(
                obs.size(0),
                self.model,
                obs,
                action,
                goal,
                self.steps,
            )
        else:
            negatives = self.sampler.gen_train_samples(
                obs.size(0), self.model, obs
            )

        self.model.train()
        
        actions = torch.cat([action, negatives], dim=1)
        loss = self.compute_loss(obs, actions, goal)

        return loss
    
    def compute_loss(self, state, action, goal=None):
        state = einops.rearrange(state, "b a n -> (b a) n")
        if goal is not None:
            goal = einops.rearrange(goal, "b a n -> (b a) n")

        if self.use_grad_norm:
            _, grad_norm, _ = self.sampler.compute_gradient(
                self.model, state, action, goal, False
            )
            grad_norm_loss = compute_gradient_loss(grad_norm)

        if self.loss_type == "info_nce":
            info_nce_loss, _ = compute_info_nce_loss(
                ebm=self.model,
                state=state,
                actions=action,
                device=self.device,
                avrg_e_regularization=self.avrg_e_regularization,
                goal=goal,
            )
            # add inference loss together with the gradient loss if necessary
            if self.use_grad_norm:
                loss = info_nce_loss + self.grad_norm_factor * grad_norm_loss
            else:
                loss = info_nce_loss

        elif self.loss_type == "cd":
            loss = contrastive_divergence(
                ebm=self.model,
                state=state,
                actions=action,
                avrg_e_regularization=self.avrg_e_regularization,
            )

            if self.use_grad_norm:
                loss += self.grad_norm_factor * grad_norm

        elif self.loss_type == "cd_kl":
            loss = contrastive_divergence_kl(
                ebm=self.model,
                state=state,
                actions=action,
                avrg_e_regularization=self.avrg_e_regularization,
                kl_loss_factor=self.kl_loss_factor,
            )

        elif self.loss_type == "cd_entropy":
            loss = contrastive_divergence_entropy_approx(
                ebm=self.model,
                state=state,
                actions=action,
            )
        elif self.loss_type == "autoregressive_info_nce":
            loss = compute_autoregressive_info_nce_loss(
                ebm=self.model,
                state=state,
                actions=action,
                device=self.device,
                avrg_e_regularization=self.avrg_e_regularization,
            )
        else:
            raise ValueError("Not a correct loss type! Please chose another one!")

        return loss  

    def get_params(self):
        return self.parameters()


class IBC_Agent(BaseAgent):

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
        eval_every_n_epochs,
        lr_scheduler: DictConfig,
        use_ema: bool = False,
        decay: Optional[float] = None,
        update_ema_every_n_steps: Optional[int] = None
    ):
        super().__init__(
            model=model,
            optimization=optimization,
            trainset=trainset,
            valset=valset,
            lr_scheduler=lr_scheduler,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device,
            epoch=epoch,
            scale_data=scale_data,
            eval_every_n_epochs=eval_every_n_epochs,
            use_ema=use_ema,
            decay=decay,
            update_ema_every_n_steps=update_ema_every_n_steps
        )

        self.eval_model_name = "eval_best_ibc.pth"
        self.last_model_name = "last_ibc.pth"

        self.set_bounds(self.scaler)

    def set_bounds(self, scaler):
        """
        Define the bounds for the sampler class
        """
        self.model.sampler.get_bounds(scaler)

    def predict(
        self, state: torch.Tensor, goal: Optional[torch.Tensor] = None, if_vision=False
    ) -> torch.Tensor:

        self.model.eval()

        if if_vision:
            bp_image, inhand_image, des_robot_pos = state

            bp_image = (
                torch.from_numpy(bp_image)
                .to(self.device)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )
            inhand_image = (
                torch.from_numpy(inhand_image)
                .to(self.device)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )
            des_robot_pos = (
                torch.from_numpy(des_robot_pos)
                .to(self.device)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )

            des_robot_pos = self.scaler.scale_input(des_robot_pos)

            state = (bp_image, inhand_image, des_robot_pos)

        else:
            state = (
                torch.from_numpy(state)
                .float()
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            state = self.scaler.scale_input(state)

        state = self.model.get_embedding(state)

        if goal is not None:
            goal = self.scaler.scale_input(goal)
            goal = goal.to(self.device)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        # if we use Langevin MCMC we still need the gradient therefore model.eval() is not called
        if self.model.sampler == "DerivativeFreeOptimizer":
            self.model.eval()
            out = self.model.sampler.infer(state, self.model.model, goal)
        else:
            out = self.model.sampler.infer(state, self.model.model, goal)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        out = self.scaler.inverse_scale_output(out)
        return out.detach().cpu().numpy()

    def reset(self):
        pass
