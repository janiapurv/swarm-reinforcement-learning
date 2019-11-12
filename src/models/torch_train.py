import numpy as np

import torch
import torch.optim as optim
from .utils import visual_log


class AdvantageCritic(object):
    def __init__(self, config):
        super(AdvantageCritic, self).__init__()
        self.config = config
        self.gamma = config['network']['gamma']
        self.n_states = config['network']['n_states']
        self.n_actions = config['network']['n_actions']
        self.n_episodes = config['network']['n_episodes']
        self.learning_rate = config['network']['learning_rate']

    def get_advantage_value(self, q_val, state_values, state_rewards):
        # compute Q values
        q_vals = np.zeros_like(state_values)
        for t in reversed(range(len(state_rewards))):
            q_val = state_rewards[t] + self.gamma * q_val
            q_vals[t] = q_val

        # Update critic
        advantage = q_vals - state_values
        return advantage

    def update_network(self, optimizer, criterion):
        optimizer.zero_grad()
        criterion.backward()
        optimizer.step()
        return None

    def train(self, env, Actor, Critic):
        # Device to train the model cpu or gpu
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Computation device being used:', device)

        # Initialise the Actor and Critic networks
        actor = Actor(self.n_states, self.config).to(device)  # use GPU
        actor_optimizer = optim.Adam(actor.parameters(), lr=self.learning_rate)

        critic = Critic(self.n_states, self.config).to(device)  # use GPU
        critic_optimizer = optim.Adam(actor.parameters(),
                                      lr=self.learning_rate)

        # Visual logger
        visual_logger = visual_log('Task type classification')

        for episode in range(self.n_episodes):
            state_values = []
            state_rewards = []
            log_actions = []
            state, done = env.reset()
            while not done:
                # Evaluate actor network
                action = actor.forward(state)
                # Get the next state and reward for the action
                new_state, reward, done, _ = env.step(action)
                # Get the value from critic network
                value = critic.forward(state)

                # Store values and rewards
                state_rewards.append(reward)
                state_values.append(value)
                log_actions.append(torch.log(action))
                state = new_state

                if done:
                    q_val = critic.forward(new_state)
                    q_val = q_val.detach().numpy()
                    if episode % 10 == 0:
                        print("episode: {}, avg_reward: {}\n".format(
                            episode, np.sum(state_rewards)))
                    break

            # Convert to tensor
            q_val = torch.FloatTensor(q_val)
            state_values = torch.FloatTensor(state_values)
            state_rewards = torch.FloatTensor(state_rewards)
            log_actions = torch.stack(log_actions)

            # Peform gradient descent on Critic network
            advantage = self.get_advantage_value(q_val, state_values,
                                                 state_rewards)
            critic_criterion = 0.5 * advantage.pow(2).mean()
            self.update_network(critic_optimizer, critic_criterion)

            # Peform gradient descent on Actor network
            actor_criterion = (-log_actions * advantage).mean()
            self.update_network(actor_optimizer, actor_criterion)

            visual_logger.log(episode, [np.sum(state_rewards)])

        return None
