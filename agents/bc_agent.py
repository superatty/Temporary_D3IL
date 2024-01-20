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


class BCPolicy(nn.Module):

    def __init__(self,
                 model: DictConfig,
                 obs_encoder: DictConfig,
                 visual_input: bool = False,
                 device: str = 'cpu'):

        super(BCPolicy, self).__init__()

        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)

        self.model = hydra.utils.instantiate(model).to(device)

    def forward(self, inputs):

        # encode state and visual inputs
        # the encoder should be shared by all the baselines

        if self.visual_input:

            agentview_image, in_hand_image, state = inputs

            B, T, C, H, W = agentview_image.size()

            agentview_image = agentview_image.view(B * T, C, H, W)
            in_hand_image = in_hand_image.view(B * T, C, H, W)
            state = state.view(B * T, -1)

            # bp_imgs = einops.rearrange(bp_imgs, "B T C H W -> (B T) C H W")
            # inhand_imgs = einops.rearrange(inhand_imgs, "B T C H W -> (B T) C H W")

            obs_dict = {"agentview_image": agentview_image,
                        "in_hand_image": in_hand_image,
                        "robot_ee_pos": state}

            obs = self.obs_encoder(obs_dict)
            obs = obs.view(B, T, -1)

        else:

            obs = self.obs_encoder(inputs)

        # make prediction
        pred = self.model(obs)

        return pred

    def get_params(self):
        return self.parameters()


class BCAgent(BaseAgent):
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
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model=model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.get_params()
        )

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

    def train(self):

        if self.model.visual_input:
            self.train_vision_agent()
        else:
            self.train_agent()

    def train_agent(self):
        best_test_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch+1) % self.eval_every_n_epochs:
                test_mse = []
                for data in self.test_dataloader:
                    state, action, mask = data #[torch.squeeze(data[i]) for i in range(3)]

                    state = self.scaler.scale_input(state)
                    action = self.scaler.scale_output(action)

                    mean_mse = self.evaluate(state, action)
                    test_mse.append(mean_mse)

                    wandb.log(
                        {
                            "test_loss": mean_mse,
                        }
                    )

                avrg_test_mse = sum(test_mse) / len(test_mse)

                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))

                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')

                wandb.log(
                    {
                        "mean_test_loss": avrg_test_mse,
                    }
                )

            train_loss = []
            for data in self.train_dataloader:
                state, action, mask = data #[torch.squeeze(data[i]) for i in range(3)]

                state = self.scaler.scale_input(state)
                action = self.scaler.scale_output(action)

                batch_loss = self.train_step(state, action)

                train_loss.append(batch_loss)

                wandb.log(
                    {
                        "loss": batch_loss,
                    }
                )

            avrg_train_loss = sum(train_loss) / len(train_loss)
            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")

    def train_vision_agent(self):
        best_test_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch+1) % self.eval_every_n_epochs:

                test_mse = []
                for data in self.test_dataloader:

                    bp_imgs, inhand_imgs, obs, action, mask = data

                    # obs, action, mask = data

                    bp_imgs = bp_imgs.to(self.device)
                    inhand_imgs = inhand_imgs.to(self.device)

                    obs = self.scaler.scale_input(obs)
                    action = self.scaler.scale_output(action)

                    state = (bp_imgs, inhand_imgs, obs)

                    mean_mse = self.evaluate(state, action)
                    test_mse.append(mean_mse)

                    wandb.log(
                        {
                            "test_loss": mean_mse,
                        }
                    )

                avrg_test_mse = sum(test_mse) / len(test_mse)

                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))

                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')

                wandb.log(
                    {
                        "mean_test_loss": avrg_test_mse,
                    }
                )

            train_loss = []
            for data in self.train_dataloader:
                bp_imgs, inhand_imgs, obs, action, mask = data
                # obs, action, mask = data

                bp_imgs = bp_imgs.to(self.device)
                inhand_imgs = inhand_imgs.to(self.device)

                obs = self.scaler.scale_input(obs)
                action = self.scaler.scale_output(action)

                state = (bp_imgs, inhand_imgs, obs)

                batch_loss = self.train_step(state, action)

                train_loss.append(batch_loss)

                wandb.log(
                    {
                        "loss": batch_loss,
                    }
                )

            avrg_train_loss = sum(train_loss) / len(train_loss)
            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")

    def train_step(self, state, actions: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        if goal is not None:
            goal = self.scaler.scale_input(goal)
            out = self.model(torch.cat([state, goal], dim=-1))
        else:
            out = self.model(state)
        loss = F.mse_loss(out, actions)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, state, action: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()

        total_mse = 0.0

        if goal is not None:
            goal = self.scaler.scale_input(goal)
            out = self.model(torch.cat([state, goal], dim=-1))
        else:
            out = self.model(state)

        mse = F.mse_loss(out, action)  # , reduction="none")
        total_mse += mse.mean(dim=-1).sum().item()
        return total_mse

    @torch.no_grad()
    def predict(self, state, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self.model.eval()

        if if_vision:
            bp_image, inhand_image, des_robot_pos = state

            bp_image = torch.from_numpy(bp_image).to(self.device).float().unsqueeze(0).unsqueeze(0)
            inhand_image = torch.from_numpy(inhand_image).to(self.device).float().unsqueeze(0).unsqueeze(0)
            des_robot_pos = torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0).unsqueeze(0)

            des_robot_pos = self.scaler.scale_input(des_robot_pos)

            state = (bp_image, inhand_image, des_robot_pos)

        else:
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0).unsqueeze(0)
            state = self.scaler.scale_input(state)

        if goal is not None:
            goal = self.scaler.scale_input(goal)
            out = self.model(torch.cat([state, goal], dim=-1))
        else:
            out = self.model(state)

        out = out.clamp_(self.min_action, self.max_action)

        model_pred = self.scaler.inverse_scale_output(out)
        return model_pred.detach().cpu().numpy()[0]
    
    def reset(self):
        pass
