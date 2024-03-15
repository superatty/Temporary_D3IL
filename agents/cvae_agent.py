import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from typing import Optional

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class CVAE_Policy(nn.Module):
    def __init__(
        self,
        model: DictConfig,
        obs_encoder: DictConfig,
        visual_input: bool = False,
        kl_loss_factor: float = 1.0,
        device: str = "cpu",
    ):
        super(CVAE_Policy, self).__init__()

        self.visual_input = visual_input
        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)
        self.model = hydra.utils.instantiate(model).to(device)

        self.kl_loss_factor = kl_loss_factor

    def get_embedding(self, state):
        if self.visual_input:
            agentview_image, in_hand_image, state = state

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
            obs = self.obs_encoder(state)

        return obs

    def forward(self, state, goal=None):
        if goal is not None:
            state = torch.cat([state, goal], dim=-1)
            
        obs = self.get_embedding(state)
        pred = self.model.predict(obs)
        return pred

    def get_loss(self, state, action, goal=None):
        if goal is not None:
            state = torch.cat([state, goal], dim=-1)
            
        obs = self.get_embedding(state)
        pred, mean, std = self.model(obs, action)

        mse_loss = F.mse_loss(pred, action)
        kl_loss = (
            -0.5 * (1 + torch.log(std.pow(2) + 1e-8) - mean.pow(2) - std.pow(2)).mean()
        )

        return mse_loss + self.kl_loss_factor * kl_loss

    def get_params(self):
        return self.parameters()


class CVAE_Agent(BaseAgent):
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
        eval_every_n_epochs: int = 50,
        use_ema: bool = False,
        decay: Optional[float] = None,
        update_ema_every_n_steps: Optional[int] = None,
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
        )

        # Define the number of GPUs available
        num_gpus = torch.cuda.device_count()

        # Check if multiple GPUs are available and select the appropriate device
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training.")
            self.model = nn.DataParallel(self.model)

        self.eval_model_name = "eval_best_cvae.pth"
        self.last_model_name = "last_cvae.pth"

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)
        
    @torch.no_grad()
    def predict(self, state, if_vision=False) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
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

        out = self.model(state)

        out = out.clamp_(self.min_action, self.max_action)

        model_pred = self.scaler.inverse_scale_output(out)
        return model_pred.detach().cpu().numpy()[0]

    def reset(self):
        pass
