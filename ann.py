import numpy as np

class ANN:
    def __init__(self, max_wheel_speed=0.5, init_range = [-0.2, 0.2]):
        # max_wheel_speed sets the maximum wheel velocity 
        # init_range sets the min and max value of the initial weights
        self.max_wheel_speed = max_wheel_speed
        self.weights = np.random.uniform(init_range[0], init_range[1], size=(2, 13))

    def forward(self, inputs):
        z = self.weights[:, :-1] @ inputs + self.weights[:, -1]
        return self.max_wheel_speed * np.tanh(z)
