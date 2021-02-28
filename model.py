import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

log = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).parents[0].resolve()
MODEL_DIR = PROJECT_DIR / 'models'


class LinearQNet(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.input_size = model_cfg.input_size
        self.hidden_size = model_cfg.hidden_size
        self.output_size = model_cfg.output_size

        self.layers = nn.Sequential(
            nn.LayerNorm(self.input_size),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, state: torch.Tensor):
        return self.layers(state)

    def save(self, name: str) -> Path:
        model_dir = MODEL_DIR / name
        model_dir.mkdir(exist_ok=True, parents=True)
        state_filename = model_dir / 'state.pth'
        torch.save(self.state_dict(), str(state_filename))
        return model_dir


class QTrainer:
    def __init__(self, model, lr: float, gamma: float) -> None:
        assert gamma >= 0 and gamma < 1
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.critereon = nn.MSELoss()

    def compute_Q(self, reward, gameover, next_pred):
        next_pred_max = next_pred.max(dim=1, keepdims=False).values
        Qnew = reward + (1 - gameover) * self.gamma * next_pred_max
        return Qnew

    def train_step(self, state, action, reward, next_state, gameover) -> None:
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        batch_size = state.shape[0]
        if state.ndim == 1:
            batch_size = 1
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            gameover = (gameover, )
        gameover = torch.tensor(gameover, dtype=torch.float32)

        selected_action = action.argmax(dim=1, keepdims=False)
        pred = self.model(state)
        next_pred = self.model(next_state)
        Qnew = self.compute_Q(reward, gameover, next_pred)
        #next_pred_max = next_pred.max(dim=1, keepdims=False).values
        #if batch_size > 1:
        #    log.info(f'R: {reward.shape} G: {gameover.shape} NPM: {next_pred_max.shape}')
        #Qnew = reward + (1 - gameover) * self.gamma * next_pred_max
        assert Qnew.shape[0] == batch_size
        target = pred.clone()
        #if batch_size > 1:
        #    log.info(f'Q: {Qnew.shape} T: {target.shape} S: {selected_action.shape}')
        target[torch.arange(batch_size), selected_action] = Qnew.squeeze()
        #for idx in range(batch_size):
        #    Qnew = reward[idx]
        #    if not gameover[idx]:
        #        Qnew = reward[idx] + self.gamma * next_pred[idx].max()
        #
        #    selected_action = action[idx].argmax().item()
        #    target[idx, int(selected_action)] = Qnew

        self.optimizer.zero_grad()
        loss = self.critereon(target, pred)
        loss.backward()
        self.optimizer.step()
