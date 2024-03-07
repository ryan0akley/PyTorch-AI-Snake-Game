import pygame
import random
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# reset
# reward
# play(action) -> direction
# game_iteration (keep track of frame)
# is_collision

Point = namedtuple('Point', 'x, y')

# directions
up = 1
right = 2
down = 3
left = 4

# colours
blue = (65, 105, 225)
red = (220,20,60)
green1 = (34, 139, 34)
green2 = (0, 100, 0)
black = (0,0,0)

block_size = 20
speed = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.width = w
        self.height = h

        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()

        
        
    def _place_food(self):
        x = random.randint(0, (self.width - block_size) // block_size) * block_size 
        y = random.randint(0, (self.height - block_size) // block_size) * block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def reset(self):
        self.direction = right
        
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head, Point(self.head.x-block_size, self.head.y), Point(self.head.x-(2*block_size), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def play_step(self, action):
        self.frame_iteration += 1 # updates every step

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # check if collision OR snake AI is taking too long to eat/die
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # if food eaten, place new food. if not, move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # update ui and clock
        self._update_ui()
        self.clock.tick(speed)

        # return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, point=None):
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.width - block_size or point.x < 0 or point.y > self.height - block_size or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(black)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, green1, pygame.Rect(pt.x, pt.y, block_size, block_size))
            pygame.draw.rect(self.display, green2, pygame.Rect(pt.x+2, pt.y+2, 16, 16))
            
        pygame.draw.rect(self.display, red, pygame.Rect(self.food.x, self.food.y, block_size, block_size))
        
        text = font.render("Score: " + str(self.score), True, blue)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # action can be straight, right turn, or left turn

        clock_wise = [up, right, down, left]
        i = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]): # if action is straight
            new_direction = clock_wise[i] # no change
        elif np.array_equal(action, [0, 1, 0]): # if action is right turn
            next_i = (i + 1) % 4
            new_direction = clock_wise[next_i]
        else: # if action is left turn
            next_i = (i - 1) % 4
            new_direction = clock_wise[next_i]

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == right:
            x += block_size
        elif self.direction == left:
            x -= block_size
        elif self.direction == down:
            y += block_size
        elif self.direction == up:
            y -= block_size
            
        self.head = Point(x, y)
