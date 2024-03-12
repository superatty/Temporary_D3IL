import logging

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import einops
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from typing import Optional
from collections import deque

from agents.base_agent import BaseAgent
import agents.models.bet.utils as utils

log = logging.getLogger(__name__)


class BeT_Policy(nn.Module):
    def __init__(self, model: DictConfig, obs_encoder: DictConfig, visual_input: bool = False, device: str = "cpu"):
        super(BeT_Policy, self).__init__()

        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)

        self.model = hydra.utils.instantiate(model).to(device)

    def get_loss(self, inputs, latent):
        if self.visual_input:
            agentview_image, in_hand_image, state = inputs

            B, T, C, H, W = agentview_image.size()

            agentview_image = agentview_image.view(B * T, C, H, W)
            in_hand_image = in_hand_image.view(B * T, C, H, W)
            state = state.view(B * T, -1)

            obs_dict = {"agentview_image": agentview_image,
                        "in_hand_image": in_hand_image,
                        "robot_ee_pos": state}

            obs = self.obs_encoder(obs_dict)
            obs = obs.view(B, T, -1)

        else:
            obs = self.obs_encoder(inputs)

        _, loss, loss_components = self.model.get_latent_and_loss(
            obs_rep=obs,
            target_latents=latent,
            return_loss_components=True,
        )

        return loss

    def forward(self, inputs):
        # encode state and visual inputs
        # the encoder should be shared by all the baselines

        if self.visual_input:
            agentview_image, in_hand_image, state = inputs

            B, T, C, H, W = agentview_image.size()

            agentview_image = agentview_image.view(B * T, C, H, W)
            in_hand_image = in_hand_image.view(B * T, C, H, W)
            state = state.view(B * T, -1)

            obs_dict = {"agentview_image": agentview_image,
                        "in_hand_image": in_hand_image,
                        "robot_ee_pos": state}

            obs = self.obs_encoder(obs_dict)
            obs = obs.view(B, T, -1)

        else:
            obs = self.obs_encoder(inputs)

        # make prediction
        latents = self.model.generate_latents(
            obs,
            torch.ones_like(obs).mean(dim=-1),
        )

        return latents

    def get_params(self):
        return self.parameters()


class BeT_Agent(BaseAgent):
    def __init__(
            self,
            model: DictConfig,
            optimization,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            grad_norm_clip,
            window_size,
            action_ae: DictConfig,
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.state_prior_optimizer = hydra.utils.instantiate(optimization, params=self.model.parameters())

        self.eval_model_name = "eval_best_bet.pth"
        self.last_model_name = "last_bet.pth"

        self.grad_norm_clip = grad_norm_clip
        self.window_size = window_size

        self.action_ae = hydra.utils.instantiate(action_ae, _recursive_=False, num_bins=self.model.model.vocab_size).to(self.device)

        self.obs_context = deque(maxlen=self.window_size)

        self.bp_image_context = deque(maxlen=self.window_size)
        self.inhand_image_context = deque(maxlen=self.window_size)
        self.des_robot_pos_context = deque(maxlen=self.window_size)

        self.action_ae.fit_model(self.train_dataloader, self.test_dataloader, self.scaler)

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        _keys_to_save = [
            "model",
            "action_ae",
        ]
        payload = {k: self.__dict__[k] for k in _keys_to_save}

        if sv_name is None:
            file_path = os.path.join(store_path, "BeT.pth")
        else:
            file_path = os.path.join(store_path, sv_name)

        with open(file_path, "wb") as f:
            torch.save(payload, f)

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        _keys_to_save = [
            "model",
            "action_ae",
        ]

        if sv_name is None:
            file_path = os.path.join(weights_path, "BeT.pth")
        else:
            file_path = os.path.join(weights_path, sv_name)

        with open(file_path, "rb") as f:
            payload = torch.load(f, map_location=self.device)

        loaded_keys = []
        for k, v in payload.items():
            if k in _keys_to_save:
                loaded_keys.append(k)
                self.__dict__[k] = v.to(self.device)

        if len(loaded_keys) != len(_keys_to_save):
            raise ValueError(
                "Model does not contain the following keys: "
                f"{set(_keys_to_save) - set(loaded_keys)}"
            )

    def train_step(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """

        self.state_prior_optimizer.zero_grad(set_to_none=True)

        latent = self.action_ae.encode_into_latent(actions)

        loss = self.model.get_loss(inputs=state, latent=latent)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_norm_clip
        )
        self.state_prior_optimizer.step()

        return loss

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        Method for evaluating the model on one epoch of data
        """

        latent = self.action_ae.encode_into_latent(action)
        loss = self.model.get_loss(
            inputs=state,
            latent=latent,
        )

        return loss

    def predict(self, state, sample=False, if_vision=False):

        with utils.eval_mode(
            self.action_ae, self.model, no_grad=True
        ):

            if if_vision:
                bp_image, inhand_image, des_robot_pos = state

                bp_image = torch.from_numpy(bp_image).to(self.device).float().unsqueeze(0)
                inhand_image = torch.from_numpy(inhand_image).to(self.device).float().unsqueeze(0)
                des_robot_pos = torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0)

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

            latents = self.model(obs_seq)

            if type(latents) is tuple:
                latents, offsets = latents

            action_latents = (latents[:, -1:, :], offsets[:, -1:, :])

            actions = self.action_ae.decode_actions(
                latent_action_batch=action_latents,
            )

            actions = actions.clamp_(self.min_action, self.max_action)

            actions = self.scaler.inverse_scale_output(actions)

            actions = actions.cpu().numpy()

            if sample:
                sampled_action = np.random.randint(len(actions))
                actions = actions[sampled_action]
            else:
                actions = einops.rearrange(
                    actions, "batch 1 action_dim -> batch action_dim"
                )

            return actions
        
    def reset(self):
        """ Resets the context of the model."""
        self.obs_context.clear()
        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.des_robot_pos_context.clear()