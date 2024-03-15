from collections import deque
import os
import logging
from typing import Optional

from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity

from agents.base_agent import BaseAgent

# A logger for this file
log = logging.getLogger(__name__)


class DDPM_Policy(nn.Module):
    def __init__(
        self,
        model: DictConfig,
        obs_encoder: DictConfig,
        visual_input: bool = False,
        device: str = "cpu",
    ):
        super(DDPM_Policy, self).__init__()

        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)

        self.model = hydra.utils.instantiate(model).to(device)

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
        obs = self.get_embedding(state)
        pred = self.model(obs, goal)
        return pred

    def get_loss(self, state, action, goal=None):
        obs = self.get_embedding(state)
        loss = self.model.loss(action, obs, goal)
        return loss

    def get_params(self):
        return self.parameters()


class DDPM_Agent(BaseAgent):

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
        decay: float,
        update_ema_every_n_steps: int,
        window_size: int,
        diffusion_kde: bool = False,
        diffusion_kde_samples: int = 100,
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
            update_ema_every_n_steps=update_ema_every_n_steps
        )

        # Define the bounds for the sampler class
        self.model.model.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(
            self.device
        )
        self.model.model.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(
            self.device
        )

        # Define the number of GPUs available
        num_gpus = torch.cuda.device_count()

        # Check if multiple GPUs are available and select the appropriate device
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training.")
            self.model = nn.DataParallel(self.model)

        self.eval_model_name = "eval_best_ddpm.pth"
        self.last_model_name = "last_ddpm.pth"

        self.window_size = window_size

        self.bp_image_context = deque(maxlen=self.window_size)
        self.inhand_image_context = deque(maxlen=self.window_size)
        self.des_robot_pos_context = deque(maxlen=self.window_size)

        self.obs_context = deque(maxlen=self.window_size)

        self.diffusion_kde = diffusion_kde
        self.diffusion_kde_samples = diffusion_kde_samples


    def reset(self):
        """Resets the context of the model."""
        self.obs_context.clear()
        
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

            bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=0)
            inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=0)
            des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=0)

            obs_seq = (bp_image_seq, inhand_image_seq, des_robot_pos_seq)

        else:
            obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            obs = self.scaler.scale_input(obs)
            self.obs_context.append(obs)
            obs_seq = torch.stack(tuple(self.obs_context), dim=0)  # type: ignore

        if goal is not None:
            goal = self.scaler.scale_input(goal)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        self.model.eval()

        if self.diffusion_kde:
            # Generate multiple action samples from the diffusion model as usual (these can be done in parallel).
            # Fit a simple kernel-density estimator (KDE) over all samples, and score the likelihood of each.
            # Select the action with the highest likelihood.
            # https://openreview.net/pdf?id=Pv1GPQzRrC8

            # repeat state and goal tensor before passing to the model
            state_rpt = torch.repeat_interleave(
                obs_seq, repeats=self.diffusion_kde_samples, dim=0
            )
            goal_rpt = torch.repeat_interleave(
                goal, repeats=self.diffusion_kde_samples, dim=0
            )
            # generate multiple samples
            x_0 = self.model(state_rpt, goal_rpt)
            if len(x_0.size()) == 3:
                x_0 = x_0[:, -1, :]
            # apply kde
            x_kde = x_0.detach().cpu()
            kde = KernelDensity(kernel="gaussian", bandwidth=0.4).fit(x_kde)
            kde_prob = kde.score_samples(x_kde)
            max_index = kde_prob.argmax(axis=0)
            # take prediction with the highest likelihood
            model_pred = x_0[max_index]
        else:
            # do default model evaluation
            model_pred = self.model(obs_seq, goal)
            if model_pred.size()[1] > 1 and len(model_pred.size()) == 3:
                model_pred = model_pred[:, -1, :]

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        model_pred = self.scaler.inverse_scale_output(model_pred)

        if len(model_pred.size()) == 3:
            model_pred = model_pred[0]

        return model_pred.cpu().numpy()

    def reset(self):
        """Resets the context of the model."""
        self.obs_context.clear()

        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.des_robot_pos_context.clear()
