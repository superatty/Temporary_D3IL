import logging

import os

from git import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from omegaconf import DictConfig
import hydra
from collections import deque

from agents.models.bet.libraries.loss_fn import FocalLoss, soft_cross_entropy
from agents.base_agent import BaseAgent
import agents.models.bet.utils as utils

log = logging.getLogger(__name__)


class BeT_MLP_Policy(nn.Module):

    def __init__(
        self,
        model: DictConfig,
        obs_encoder: DictConfig,
        action_ae: DictConfig,
        visual_input: bool = False,
        obs_dim: int = 2,
        act_dim: int = 2,
        vocab_size: int = 16,
        predict_offsets: bool = False,
        focal_loss_gamma: float = 0.0,
        offset_loss_scale: int = 1.0,
        device: str = "cpu",
    ):

        super(BeT_MLP_Policy, self).__init__()

        self.visual_input = visual_input

        self.vocab_size = vocab_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.offset_loss_scale = offset_loss_scale
        self.focal_loss_gamma = focal_loss_gamma

        self.model = hydra.utils.instantiate(
            model, _recursive_=False, output_dim=self.vocab_size * (1 + self.act_dim)
        ).to(device)
        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)
        self.action_ae = hydra.utils.instantiate(
            action_ae, _recursive_=False, num_bins=self.vocab_size
        ).to(device)

        self.predict_offsets = predict_offsets

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
        if goal is not None:
            state = torch.cat([state, goal], dim=-1)

        obs = self.get_embedding(state)
        latent = self.action_ae.encode_into_latent(action)

        if self.predict_offsets:
            target_latents, target_offsets = latent

        is_soft_target = (target_latents.shape[-1] == self.vocab_size) and (
            self.vocab_size != 1
        )

        if is_soft_target:
            target_latents = target_latents.view(-1, target_latents.size(-1))
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(-1)
            if self.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = FocalLoss(gamma=self.focal_loss_gamma)

        output = self.model(obs)

        logits = output[:, :, : self.vocab_size]
        offsets = output[:, :, self.vocab_size :]

        batch = logits.shape[0]
        seq = logits.shape[1]
        offsets = einops.rearrange(
            offsets,
            "N T (V A) -> (N T) V A",  # N = batch, T = seq
            V=self.vocab_size,
            A=self.act_dim,
        )
        # calculate (optionally soft) cross entropy and offset losses
        class_loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
        # offset loss is only calculated on the target class
        # if soft targets, argmax is considered the target class
        selected_offsets = offsets[
            torch.arange(offsets.size(0)),
            (
                target_latents.argmax(dim=-1).view(-1)
                if is_soft_target
                else target_latents.view(-1)
            ),
        ]
        offset_loss = self.offset_loss_scale * F.mse_loss(
            selected_offsets, target_offsets.view(-1, self.act_dim)
        )
        loss = offset_loss + class_loss
        logits = einops.rearrange(logits, "batch seq classes -> seq batch classes")
        offsets = einops.rearrange(
            offsets,
            "(N T) V A -> T N V A",
            # ? N, T order? Anyway does not affect loss and training (might affect visualization)
            N=batch,
            T=seq,
        )

        return loss

    def forward(self, state, goal=None):
        if goal is not None:
            state = torch.cat([state, goal], dim=-1)

        obs = self.get_embedding(state)

        output = self.model(obs)

        if self.predict_offsets:
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.act_dim,
            )
        else:
            logits = output
        probs = F.softmax(logits, dim=-1)
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        if self.predict_offsets:
            sampled_offsets = offsets[
                torch.arange(offsets.shape[0]), sampled_data.flatten()
            ].view(batch, seq, self.act_dim)

            return (sampled_data, sampled_offsets)
        else:
            return sampled_data

    def get_params(self):
        return self.parameters()


class BeT_MLP_Agent(BaseAgent):
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
        grad_norm_clip,
        window_size,
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
            grad_norm_clip=grad_norm_clip,
            eval_every_n_epochs=eval_every_n_epochs,
            use_ema=use_ema,
            decay=decay,
            update_ema_every_n_steps=update_ema_every_n_steps,
        )

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.eval_model_name = "eval_best_bet_mlp.pth"
        self.last_model_name = "last_bet_mlp.pth"

        self.window_size = window_size

        self.obs_context = deque(maxlen=self.window_size)

        self.bp_image_context = deque(maxlen=self.window_size)
        self.inhand_image_context = deque(maxlen=self.window_size)
        self.des_robot_pos_context = deque(maxlen=self.window_size)

        self.model.action_ae.fit_model(
            self.train_dataloader, self.test_dataloader, self.scaler
        )

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "action_ae": self.model.action_ae.state_dict(),
        }

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
                if k == "model":
                    self.model.load_state_dict(v)
                elif k == "action_ae":
                    self.model.action_ae.load_state_dict(v)

        if len(loaded_keys) != len(_keys_to_save):
            raise ValueError(
                "Model does not contain the following keys: "
                f"{set(_keys_to_save) - set(loaded_keys)}"
            )

    def predict(self, state, sample=False, if_vision=False):

        with utils.eval_mode(self.model.action_ae, self.model, no_grad=True):

            if if_vision:
                bp_image, inhand_image, des_robot_pos = state

                bp_image = (
                    torch.from_numpy(bp_image).to(self.device).float().unsqueeze(0)
                )
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
                des_robot_pos_seq = torch.stack(
                    tuple(self.des_robot_pos_context), dim=0
                )

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

            actions = self.model.action_ae.decode_actions(
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
        """Resets the context of the model."""
        self.obs_context.clear()

        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.des_robot_pos_context.clear()
