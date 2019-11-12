import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * (torch.tanh(F.softplus(input)))


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
        self.net = nn.Sequential(nn.Linear(n_states, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, n_actions), nn.Sigmoid())

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
