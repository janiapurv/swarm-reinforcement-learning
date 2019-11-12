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
        q_vals = torch.zeros_like(state_values)
        for t in reversed(range(len(state_rewards))):
            q_val = state_rewards[t] + self.gamma * q_val
            q_vals[t] = q_val

        # Update critic
        print(q_vals.shape, state_values.shape)
        advantage = q_vals - state_values
        return advantage

    def update_network(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return None

    def train(self, env, Actor, Critic):
        # Device to train the model cpu or gpu
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Computation device being used:', device)

        # Initialise the Actor and Critic networks
        actor = Actor(self.n_states, self.n_actions,
                      self.config).to(device)  # use GPU
        actor_optimizer = optim.Adam(actor.parameters(), lr=self.learning_rate)

        critic = Critic(self.n_states, self.config).to(device)  # use GPU
        critic_optimizer = optim.Adam(critic.parameters(),
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
                torch_state = torch.from_numpy(state).type(torch.float32)
                action = actor.forward(torch_state)
                # Get the next state and reward for the action
                new_state, reward, done = env.step(action.detach().numpy())
                # Get the value from critic network
                value = critic.forward(torch_state)

                # Store values and rewards
                state_rewards.append(
                    torch.tensor([reward], dtype=torch.float, device=device))
                state_values.append(
                    torch.tensor([value], dtype=torch.float, device=device))
                log_actions.append(torch.log(action))
                state = new_state

                if done:
                    torch_new_state = torch.from_numpy(new_state).type(
                        torch.float32)
                    q_val = critic.forward(torch_new_state)
                    q_val = q_val.detach().numpy()
                    if episode % 10 == 0:
                        print("episode: {}, avg_reward: {}\n".format(
                            episode, np.sum(state_rewards)))
                    break

            # Convert to tensor
            q_val = torch.FloatTensor(q_val)
            state_values = torch.stack(state_values)
            state_rewards = torch.stack(state_rewards)
            log_actions = torch.stack(log_actions)

            # Peform gradient descent on Critic network
            advantage = self.get_advantage_value(q_val, state_values,
                                                 state_rewards)
            print(advantage.shape, log_actions.shape)
            actor_loss = -torch.mean(log_actions * advantage)
            print(actor_loss)
            critic_loss = 2 * torch.mean(log_actions * advantage)
            print(critic_loss)
            self.update_network(critic_optimizer, critic_loss)

            # Peform gradient descent on Actor network
            actor_loss = -torch.mean(log_actions * advantage)
            self.update_network(actor_optimizer, actor_loss)

            visual_logger.log(episode,
                              [np.sum(state_rewards.detach().numpy())])

        return None
