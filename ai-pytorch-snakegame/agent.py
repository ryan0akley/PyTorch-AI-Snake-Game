import torch
import random
import numpy as np
from collections import deque # data structure to store memory
from game import SnakeGameAI, Point
from model import Linear_QNet, QTrainer
from helper import plot

max_memory = 100000
batch_size = 1000 # amount of samples
lr = 0.001

# directions and block_size, same from game.py
up = 1
right = 2
down = 3
left = 4
block_size = 20

class Agent:

    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # parameter to control randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=max_memory) # if reached initialized max_memory, deque will auto popleft()
        self.model = Linear_QNet(11, 256, 3) # 11 states/inputs, 256 hidden size, 3 outputs since action array, i.e. [1,0,0], [0,1,0], has 3 values
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma)

    def get_state(self, game):
        """
        State array values True/False: [danger straight, danger right, danger left,
                                        direction up, direction right, direction down, direction left,
                                        food up, food right, food down, food left]
        """
        state = [False, False, False, False, False, False, False, False, False, False, False]

        head = game.snake[0]
        # points a step (block size) ahead of snake's head
        point_left = Point(head.x - block_size, head.y)
        point_right = Point(head.x + block_size, head.y)
        point_up = Point(head.x, head.y - block_size)
        point_down = Point(head.x, head.y + block_size)

        # danger straight
        if ((game.direction == up and game.is_collision(point_up))
            or (game.direction == right and game.is_collision(point_right))
            or (game.direction == down and game.is_collision(point_down))
            or (game.direction == left and game.is_collision(point_left))):
            state[0] = True

        # danger right
        if ((game.direction == up and game.is_collision(point_right))
            or (game.direction == right and game.is_collision(point_down))
            or (game.direction == down and game.is_collision(point_left))
            or (game.direction == left and game.is_collision(point_up))):
            state[1] = True
        
        # danger left
        if ((game.direction == up and game.is_collision(point_left))
            or (game.direction == right and game.is_collision(point_up))
            or (game.direction == down and game.is_collision(point_right))
            or (game.direction == left and game.is_collision(point_down))):
            state[2] = True

        state[3] = game.direction == up
        state[4] = game.direction == right
        state[5] = game.direction == down
        state[6] = game.direction == left

        state[7] = game.food.y < game.head.y
        state[8] = game.food.x > game.head.x
        state[9] = game.food.y > game.head.y
        state[10] = game.food.x < game.head.x

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # stores as one tuple
        # if reached initialized max_memory, deque will auto popleft()

    def train_long_memory(self):
        if len(self.memory) > batch_size: # when memory passed initial batch/sample size
            mini_sample = random.sample(self.memory, batch_size) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over): # train for each step
        self.trainer.train_step(state, action, reward, next_state, game_over) 

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # beginning, want random moves and explore environment. later on, want less random moves and exploit our model

        self.epsilon = 80 - self.num_games # more games ran results in smaller epsilon
        final_move = [0,0,0]

        # as epsilon becomes smaller (as more games run), this will have less chance of running.
        # when epsilon reaches negative, this will never run.
        if random.randint(0, 200) < self.epsilon: 
            move = random.randint(0, 2) # perform random move
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float) # convert state to tensor
            prediction = self.model(state0)
            # prediction is raw value, ex. [3.0, 1.6, 0.1]
            # then we take max of prediction and set it to 1, rest 0
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    if agent.model.load(): # when a saved model is loaded
        agent.num_games = 81 # don't randomize final move when get_action is called

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory (only for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train long memory, plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.num_games, 'Score', score, 'Record:', record)

            # plot scores
            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.num_games
            plot_avg_scores.append(avg_score)
            plot(plot_scores, plot_avg_scores)

if __name__ == '__main__':
    train()