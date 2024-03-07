AI self-playing snake game, using Python, PyTorch and Pygame.

Run agent.py to see results.

Required installations:
Pygame
PyTorch (torch, torchvision)
matplotlib
ipython

Reward system for the learning model:
- ate food: +10
- lost / game over: -10
- otherwise: 0

Action mapping:
- [1,0,0] go straight
- [0,1,0] right turn
- [0,0,1] left turn

State array values True/False: [danger straight, danger right, danger left,
                                direction up, direction right, direction down, direction left,
                                food up, food right, food down, food left]
