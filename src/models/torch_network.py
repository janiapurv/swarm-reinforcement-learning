import torch.nn as nn


class Actor(nn.Module):
    """Actor neural network.

    Parameters
    ----------
    nn : class
        Torch neural network class

    Returns
    -------
    array 1-D
        An array containing the action for a given state
    """
    def __init__(self, n_states, n_actions, config):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_states, 128), nn.Linear(128, 128),
                                 nn.Linear(128, 128),
                                 nn.Linear(128, n_actions))

    def forward(self, state):
        action = self.net(state)
        return action


class Critic(nn.Module):
    """A critic neural network.

    Parameters
    ----------
    nn : class
        Torch neural network class

    Returns
    -------
    float
        The value of a given state
    """
    def __init__(self, n_states, config):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_states, 128), nn.Linear(128, 128),
                                 nn.Linear(128, 128), nn.Linear(128, 128),
                                 nn.Linear(128, 1))

    def forward(self, state):
        value = self.net(state)
        return value
