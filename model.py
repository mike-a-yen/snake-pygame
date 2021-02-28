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
        assert gamma < 1
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.critereon = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, gameover) -> None:
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        batch_size = state.shape[0]
        if state.ndim == 1:
            batch_size = 1
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            gameover = (gameover, )

        pred = self.model(state)
        next_pred = self.model(next_state)
        target = pred.clone()
        for idx in range(batch_size):
            Qnew = reward[idx]
            if not gameover[idx]:
                Qnew = reward[idx] + self.gamma * next_pred[idx].max()

            selected_action = action[idx].argmax().item()
            target[idx, int(selected_action)] = Qnew

        self.optimizer.zero_grad()
        loss = self.critereon(target, pred)
        loss.backward()
        self.optimizer.step()
