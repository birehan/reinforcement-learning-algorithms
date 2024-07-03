import numpy as np
import random

class MultiArmedBanditRL:
    """
    A class to implement Reinforcement Learning algorithms for Multi-Armed Bandit problems.

    Attributes:
        K (int): Number of arms (actions).
        Q (numpy.ndarray): Action-value estimates.
        action_counts (numpy.ndarray): Number of times each action has been chosen.
        true_rewards (numpy.ndarray): True rewards for each action.

    Methods:
        value_iteration(gamma=0.9, theta=1e-6):
            Perform value iteration (not typical for bandits, included for completeness).

        policy_iteration(gamma=0.9, theta=1e-6):
            Perform policy iteration (not typical for bandits, included for completeness).

        q_learning(N=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
            Perform Q-learning algorithm to learn action-value function.

        epsilon_greedy_policy(epsilon=0.1):
            Epsilon-greedy policy for exploration-exploitation trade-off.

        ucb_policy(c=1.0):
            Upper Confidence Bound (UCB) policy for exploration-exploitation trade-off.
    """

    def __init__(self, K=10):
        """
        Initialize MultiArmedBanditRL with a specified number of arms (actions).

        Args:
            K (int): Number of arms (default: 10).
        """
        self.K = K
        self.Q = np.zeros(K)
        self.action_counts = np.zeros(K)
        self.true_rewards = np.random.normal(0, 1, K)

    def value_iteration(self, gamma=0.9, theta=1e-6):
        """
        Perform value iteration (not typical for bandits, included for completeness).

        Args:
            gamma (float): Discount factor for future rewards (default: 0.9).
            theta (float): Convergence threshold (default: 1e-6).
        """
        V = np.zeros(self.K)
        while True:
            delta = 0
            for k in range(self.K):
                v = V[k]
                V[k] = self.true_rewards[k]
                delta = max(delta, abs(v - V[k]))
            if delta < theta:
                break
        self.Q = V

    def policy_iteration(self, gamma=0.9, theta=1e-6):
        """
        Perform policy iteration (not typical for bandits, included for completeness).

        Args:
            gamma (float): Discount factor for future rewards (default: 0.9).
            theta (float): Convergence threshold (default: 1e-6).
        """
        def one_step_lookahead(V):
            return np.array([self.true_rewards[k] for k in range(self.K)])

        policy_stable = False
        V = np.zeros(self.K)
        while not policy_stable:
            while True:
                delta = 0
                for k in range(self.K):
                    v = V[k]
                    V[k] = self.true_rewards[k]
                    delta = max(delta, abs(v - V[k]))
                if delta < theta:
                    break

            policy_stable = True
            for k in range(self.K):
                old_action = np.argmax(self.Q)
                self.Q[k] = one_step_lookahead(V)[k]
                if old_action != np.argmax(self.Q):
                    policy_stable = False

    def q_learning(self, N=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Perform Q-learning algorithm to learn action-value function.

        Args:
            N (int): Number of episodes to train (default: 1000).
            alpha (float): Learning rate (default: 0.1).
            gamma (float): Discount factor for future rewards (default: 0.9).
            epsilon (float): Epsilon value for epsilon-greedy policy (default: 0.1).
        """
        for _ in range(N):
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(self.K))
            else:
                action = np.argmax(self.Q)

            reward = np.random.normal(self.true_rewards[action])
            self.action_counts[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.action_counts[action]

    def epsilon_greedy_policy(self, epsilon=0.1):
        """
        Epsilon-greedy policy for exploration-exploitation trade-off.

        Args:
            epsilon (float): Epsilon value for exploration probability (default: 0.1).

        Returns:
            int: Action to take based on epsilon-greedy policy.
        """
        if random.uniform(0, 1) < epsilon:
            return random.choice(range(self.K))
        else:
            return np.argmax(self.Q)

    def ucb_policy(self, c=1.0):
        """
        Upper Confidence Bound (UCB) policy for exploration-exploitation trade-off.

        Args:
            c (float): Exploration parameter for UCB (default: 1.0).

        Returns:
            int: Action to take based on UCB policy.
        """
        total_visits = sum(self.action_counts)
        ucb_values = self.Q + c * np.sqrt(np.log(total_visits + 1) / (self.action_counts + 1))
        return np.argmax(ucb_values)

if __name__ == "__main__":
    bandit_rl = MultiArmedBanditRL()

    print("Running Value Iteration...")
    bandit_rl.value_iteration()
    print("Q-Values (Value Iteration):")
    print(bandit_rl.Q)

    print("Running Policy Iteration...")
    bandit_rl.policy_iteration()
    print("Q-Values (Policy Iteration):")
    print(bandit_rl.Q)

    print("Running Q-Learning...")
    bandit_rl.q_learning()
    print("Q-Values (Q-Learning):")
    print(bandit_rl.Q)
