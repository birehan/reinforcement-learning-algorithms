import gym
import numpy as np
import random

class GridWorldRL:
    """
    A class to implement Reinforcement Learning algorithms for Grid World environments using Gym.

    Attributes:
        env (gym.Env): Gym environment instance.
        n_states (int): Number of states in the environment.
        n_actions (int): Number of actions in the environment.
        Q (numpy.ndarray): Action-value function table.
        policy (numpy.ndarray): Optimal policy derived from RL algorithms.
        V (numpy.ndarray): State-value function table.
        action_counts (numpy.ndarray): Counts of actions taken in each state for UCB policy.

    Methods:
        value_iteration(gamma=0.9, theta=1e-6):
            Perform value iteration algorithm to find optimal policy and value function.

        policy_iteration(gamma=0.9, theta=1e-6):
            Perform policy iteration algorithm to find optimal policy and value function.

        q_learning(episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
            Perform Q-learning algorithm to learn action-value function.

        epsilon_greedy_policy(state, epsilon=0.1):
            Epsilon-greedy policy for exploration-exploitation trade-off.

        ucb_policy(state, c=1.0):
            Upper Confidence Bound (UCB) policy for exploration-exploitation trade-off.
    """

    def __init__(self, env_name='FrozenLake-v1', is_slippery=False):
        """
        Initialize GridWorldRL with a Gym environment.

        Args:
            env_name (str): Name of the Gym environment.
            is_slippery (bool): Whether the environment is slippery (Stochastic).
        """
        self.env = gym.make(env_name, is_slippery=is_slippery)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = np.zeros(self.n_states, dtype=int)
        self.V = np.zeros(self.n_states)
        self.action_counts = np.zeros((self.n_states, self.n_actions))

    def value_iteration(self, gamma=0.9, theta=1e-6):
        """
        Perform value iteration algorithm to find optimal policy and value function.

        Args:
            gamma (float): Discount factor for future rewards.
            theta (float): Convergence threshold.

        """
        while True:
            delta = 0
            for s in range(self.n_states):
                v = self.V[s]
                self.V[s] = max([sum([p * (r + gamma * self.V[s_])
                                      for p, s_, r, _ in self.env.P[s][a]])
                                 for a in range(self.n_actions)])
                delta = max(delta, abs(v - self.V[s]))
            if delta < theta:
                break
        self.policy = np.argmax([[sum([p * (r + gamma * self.V[s_])
                                       for p, s_, r, _ in self.env.P[s][a]])
                                  for a in range(self.n_actions)]
                                 for s in range(self.n_states)], axis=1)

    def policy_iteration(self, gamma=0.9, theta=1e-6):
        """
        Perform policy iteration algorithm to find optimal policy and value function.

        Args:
            gamma (float): Discount factor for future rewards.
            theta (float): Convergence threshold.

        """
        def one_step_lookahead(s, V):
            return np.array([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in self.env.P[s][a]])
                             for a in range(self.n_actions)])

        policy_stable = False
        while not policy_stable:
            while True:
                delta = 0
                for s in range(self.n_states):
                    v = self.V[s]
                    self.V[s] = sum([p * (r + gamma * self.V[s_])
                                     for p, s_, r, _ in self.env.P[s][self.policy[s]]])
                    delta = max(delta, abs(v - self.V[s]))
                if delta < theta:
                    break

            policy_stable = True
            for s in range(self.n_states):
                old_action = self.policy[s]
                self.policy[s] = np.argmax(one_step_lookahead(s, self.V))
                if old_action != self.policy[s]:
                    policy_stable = False

    def q_learning(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Perform Q-learning algorithm to learn action-value function.

        Args:
            episodes (int): Number of episodes to train.
            alpha (float): Learning rate.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Epsilon value for epsilon-greedy policy.

        """
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state])

                next_state, reward, done, _ = self.env.step(action)
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + gamma * self.Q[next_state, best_next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += alpha * td_error
                state = next_state

    def epsilon_greedy_policy(self, state, epsilon=0.1):
        """
        Epsilon-greedy policy for exploration-exploitation trade-off.

        Args:
            state (int): Current state.
            epsilon (float): Epsilon value for exploration probability.

        Returns:
            int: Action to take based on epsilon-greedy policy.

        """
        if random.uniform(0, 1) < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def ucb_policy(self, state, c=1.0):
        """
        Upper Confidence Bound (UCB) policy for exploration-exploitation trade-off.

        Args:
            state (int): Current state.
            c (float): Exploration parameter for UCB.

        Returns:
            int: Action to take based on UCB policy.

        """
        total_visits = sum(self.action_counts[state])
        ucb_values = self.Q[state] + c * np.sqrt(np.log(total_visits + 1) / (self.action_counts[state] + 1))
        return np.argmax(ucb_values)

if __name__ == "__main__":
    grid_world_rl = GridWorldRL()

    print("Running Value Iteration...")
    grid_world_rl.value_iteration()
    print("Optimal Policy (Value Iteration):")
    print(grid_world_rl.policy)

    print("Running Policy Iteration...")
    grid_world_rl.policy_iteration()
    print("Optimal Policy (Policy Iteration):")
    print(grid_world_rl.policy)

    print("Running Q-Learning...")
    grid_world_rl.q_learning()
    print("Q-Table (Q-Learning):")
    print(grid_world_rl.Q)
