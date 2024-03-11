import abc
import os
import logging

import torch
from omegaconf import DictConfig
import hydra
from tqdm import tqdm

import wandb

from agents.utils.scaler import Scaler

# A logger for this file
log = logging.getLogger(__name__)


class BaseAgent(abc.ABC):

    def __init__(
        self,
        model: DictConfig,
        trainset: DictConfig,
        valset: DictConfig,
        train_batch_size: int = 512,
        val_batch_size: int = 512,
        num_workers: int = 8,
        device: str = 'cpu',
        epoch: int = 100,
        scale_data: bool = True,
        eval_every_n_epochs: int = 50
    ):

        self.model = hydra.utils.instantiate(model).to(device)

        self.trainset = hydra.utils.instantiate(trainset)
        self.valset = hydra.utils.instantiate(valset)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=10
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=10
        )

        self.eval_every_n_epochs = eval_every_n_epochs

        self.epoch = epoch

        self.device = device
        self.working_dir = os.getcwd()

        self.scaler = Scaler(self.trainset.get_all_observations(), self.trainset.get_all_actions(), scale_data, device)

        total_params = sum(p.numel() for p in self.model.get_params())

        wandb.log(
            {
                "model parameters": total_params
            }
        )

        log.info("The model has a total amount of {} parameters".format(total_params))

    def train(self):

        if self.model.visual_input:
            self.train_vision_agent()
        else:
            self.train_agent()
            
    def train_agent(self):
        """
        Main method to train the agent on the given train and test data
        """
        best_test_loss = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch+1) % self.eval_every_n_epochs:
                test_losses = []
                for data in self.test_dataloader:
                    state, action, mask = data

                    state = self.scaler.scale_input(state)
                    action = self.scaler.scale_output(action)

                    test_loss = self.evaluate(state, action)
                    test_losses.append(test_loss)

                    wandb.log(
                        {
                            "test_loss": test_loss,
                        }
                    )

                mean_test_loss = sum(test_losses) / len(test_losses)

                log.info("Epoch {}: Mean test loss is {}".format(num_epoch, mean_test_loss))

                if mean_test_loss < best_test_loss:
                    best_test_loss = mean_test_loss
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')

                wandb.log(
                    {
                        "mean_test_loss": mean_test_loss,
                    }
                )

            train_losses = []
            for data in self.train_dataloader:
                state, action, mask = data 

                state = self.scaler.scale_input(state)
                action = self.scaler.scale_output(action)

                batch_loss = self.train_step(state, action)

                train_losses.append(batch_loss)

                wandb.log(
                    {
                        "loss": batch_loss,
                    }
                )

            avg_train_loss = sum(train_losses) / len(train_losses)
            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avg_train_loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")
     
    def train_vision_agent(self):
        """
        Main method to train the vision agent on the given train and test data
        """
        train_loss = []
        for data in self.train_dataloader:
            bp_imgs, inhand_imgs, obs, action, mask = data

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

    @abc.abstractmethod
    def train_step(self, state: torch.Tensor, action: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """
        pass

    @abc.abstractmethod
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        Method for evaluating the model on one batch of data consisting of two tensors
        """
        pass

    @abc.abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        pass
    
    @abc.abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Method for resetting the agent
        """
        pass

    def get_scaler(self, scaler: Scaler):
        self.scaler = scaler

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        if sv_name is None:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        else:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)))
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        if sv_name is None:
            torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))
