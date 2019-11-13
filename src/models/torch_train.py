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
        self.eps = 0.00001

    def get_returns(self, q_val, state_values, state_rewards):
        R = q_val
        returns = []
        for step in reversed(range(len(state_rewards))):
            R = state_rewards[step] + self.gamma * R
            returns.insert(0, R)

        returns = torch.cat(returns)
        # returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns

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
                action = actor.forward(torch_state.to(device))
                # Get the next state and reward for the action
                new_state, reward, done = env.step(
                    action.detach().cpu().numpy())
                # Get the value from critic network
                value = critic.forward(torch_state.to(device))

                # Store values and rewards
                state_rewards.append(
                    torch.tensor([reward], dtype=torch.float, device=device))
                state_values.append(value)
                log_actions.append(torch.log(action))
                state = new_state

                if done == 1 or done == 2:
                    torch_new_state = torch.from_numpy(new_state).type(
                        torch.float32)
                    q_val = critic.forward(torch_new_state.to(device))
                    q_val = q_val.detach().cpu().numpy()
                    if episode % 10 == 0:
                        print("episode: {}, avg_reward: {}\n".format(
                            episode, np.sum(state_rewards)))
                    break

            # Convert to tensor
            q_val = torch.FloatTensor(q_val)
            state_values = torch.cat(state_values)
            state_rewards = torch.cat(state_rewards)
            log_actions = torch.cat(log_actions)

            # Peform gradient descent on Critic network
            returns = self.get_returns(q_val, state_values, state_rewards)

            # Actor and critic loss
            actor_loss = []
            critic_loss = []

            # Calculate advantage and loss of actor and critic for each step
            for log_action, value, r in zip(log_actions, state_values,
                                            returns):
                advantage = r.to(device) - value

                # calculate actor (policy) loss
                actor_loss.append(-log_action * advantage)

                # calculate critic (value) loss using L1 smooth loss
                critic_loss.append(advantage**2)

            # Take the mean
            critic_loss = torch.stack(critic_loss).sum()
            actor_loss = torch.stack(actor_loss).sum()

            # Update the critic and actor
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            actor_loss.backward(retain_graph=True)
            critic_loss.backward()

            actor_optimizer.step()
            critic_optimizer.step()

            visual_logger.log(episode,
                              [np.sum(state_rewards.detach().cpu().numpy())])

        return None
