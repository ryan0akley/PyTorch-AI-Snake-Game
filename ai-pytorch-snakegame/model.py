import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import os

class Linear_QNet(nn.Module):

    def __init__ (self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): # x: tensor
        # first apply linear layer
        # use activation function (relu) on linear layer
        x = func.relu(self.linear1(x))
        x = self.linear2(x) # apply 2nd layer
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        if os.path.exists(model_folder_path):
            file_name = os.path.join(model_folder_path, file_name)
            self.load_state_dict(torch.load(file_name))

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # loss function
    
    def train_step(self, state, action, reward, next_state, game_over):
        # Since we have train short and long memory, we want train_step to handle multiple sizes
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # shape will be (n, x)

        if len(state.shape) == 1:
            # when (1, x), i.e. 1 batch, append 1-dimension with unsqueeze
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,) # convert to single value tuple

        # predict Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(game_over)): # iterate over tensors
            Q_new = reward[idx]
            if not game_over[idx]: # only calculate Q_new when not in game_over
                # Q_new = r + y * max(next_predicted Q value)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad() # empty gradient
        loss = self.criterion(target, pred) # target = Q_new, pred = Q
        loss.backward() # apply backpropagation

        self.optimizer.step()
