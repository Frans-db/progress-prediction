import torch
from torch import nn, optim
import random
import numpy as np
from typing import Dict
import os
import wandb


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int) -> None:
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Experiment:
    def __init__(
        self,
        network: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        trainloader,
        testloader,
        train_fn,
        experiment_path: str,
        iterations: int,
        log_every: int,
        test_every: int,
        seed: int,
        result: Dict,
    ) -> None:
        self.device = get_device()
        self.network = network.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.testloader = testloader
        self.train_fn = train_fn
        self.experiment_path = experiment_path
        self.iterations = iterations
        self.log_every = log_every
        self.test_every = test_every
        self.result = result

        set_seeds(seed)
        if experiment_path:
            os.mkdir(experiment_path)

    def run(self) -> None:
        iteration = 0
        done = False
        train_result, test_result = self.result.copy(), self.result.copy()
        while not done:
            for batch in self.trainloader:
                batch_result = self.train_fn(
                    self.network, batch, self.device, optimizer=self.optimizer
                )
                self._add_result(train_result, batch_result)

                if iteration % self.log_every and iteration > 0:
                    train_result = self._log(train_result, iteration, "train")
                if iteration % self.test_every and iteration > 0:
                    for batch in self.testloader:
                        batch_result = self.train_fn(self.network, batch, self.device)
                        self._add_result(test_result, batch_result)
                    test_result = self._log(test_result, iteration, "test")
                    if self.experiment_path:
                        model_path = os.path.join(self.experiment_path, f"model_{iteration}.pth")
                        torch.save(self.network.state_dict(), model_path)
                
                iteration += 1
                if iteration > self.iterations:
                    done = True
                    break
                self.scheduler.step()

    @staticmethod
    def _add_result(result: Dict, batch_result: Dict) -> None:
        for key in result:
            if key in batch_result:
                result[key] += batch_result[key]

    @staticmethod
    def _log(result: Dict, iteration: int, prefix: str) -> Dict:
        log = {
            f"{prefix}_{key}": result[key] / result["count"]
            for key in result
            if key != "count"
        }
        log[f"{prefix}_count"] = result["count"]
        log["iteration"] = iteration
        wandb.log(log)
