# Reinforcement Learning Algorithms for Grid World and Multi-Armed Bandit Problems

This repository contains Python implementations of several reinforcement learning algorithms applied to two distinct problem domains: Grid World environments and Multi-Armed Bandit problems.

## Grid World Problem

Grid World is a simple, yet powerful environment for testing reinforcement learning algorithms. In a Grid World, an agent navigates a grid of states to reach a goal state while avoiding obstacles and maximizing cumulative rewards. The agent can move in different directions (up, down, left, right) and receives rewards or penalties based on its actions. This environment is useful for understanding and implementing foundational reinforcement learning concepts such as value iteration, policy iteration, and Q-learning.

### Internal Implementation and Working Logic of Methods

#### 1. Value Iteration

**Internal Implementation:**
- **Initialization**: Initializes a state-value function `V` and iteratively updates it until convergence.
- **Iteration**: For each state `s`, computes the maximum expected value of all possible actions using the Bellman equation.
- **Convergence**: Checks for convergence using a threshold on the maximum change in state values.

**Mathematical Formula:**

\[ V(s) \leftarrow \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V(s')] \]

Where:
- \( V(s) \) is the value of state \( s \),
- \( \max_a \) denotes the maximum over all actions \( a \),
- \( p(s', r \mid s, a) \) is the transition probability function,
- \( r \) is the reward function,
- \( \gamma \) is the discount factor,
- \( V(s') \) is the value of the next state \( s' \).

**Implementation:**

```python
def value_iteration(self, gamma=0.9, theta=1e-6):
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
```

#### 2. Policy Iteration

**Internal Implementation:**
- **Initialization**: Initializes a policy and evaluates it iteratively until convergence.
- **Policy Evaluation**: Evaluates the state-value function under the current policy.
- **Policy Improvement**: Updates the policy to maximize expected returns based on current value estimates.

**Mathematical Formula:**

\[ V(s) \leftarrow \sum_{s', r} p(s', r \mid s, \pi(s)) [r + \gamma V(s')] \]

Where:
- \( \pi(s) \) is the policy at state \( s \).

**Implementation:**

```python
def policy_iteration(self, gamma=0.9, theta=1e-6):
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
```

#### 3. Q-Learning

**Internal Implementation:**
- **Initialization**: Initializes an action-value function `Q` and updates it based on experience.
- **Exploration vs. Exploitation**: Balances exploration (random actions) and exploitation (greedy actions based on current estimates).
- **Q-Value Update**: Updates `Q` based on the Q-learning update rule.

**Mathematical Formula:**

\[ Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] \]

Where:
- \( Q(s, a) \) is the action-value function for state \( s \) and action \( a \),
- \( \alpha \) is the learning rate,
- \( r \) is the reward received after taking action \( a \) in state \( s \),
- \( \gamma \) is the discount factor,
- \( s' \) is the next state after taking action \( a \).

**Implementation:**

```python
def q_learning(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
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
```

#### 4. Epsilon-Greedy Policy

**Internal Implementation:**
- **Policy Selection**: Selects actions using an epsilon-greedy strategy.
- **Exploration**: Randomly selects actions with a probability controlled by epsilon.
- **Exploitation**: Selects actions greedily based on current Q-value estimates.

**Implementation:**

```python
def epsilon_greedy_policy(self, state, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return self.env.action_space.sample()
    else:
        return np.argmax(self.Q[state])
```

#### 5. UCB Algorithm

**Internal Implementation:**
- **Initialization**: Initializes an action-value function `Q` and tracks action counts.
- **Exploration vs. Exploitation**: Uses Upper Confidence Bound (UCB) to balance exploration and exploitation.
- **UCB Calculation**: Computes action selection probabilities using UCB.

**Mathematical Formula:**

\[ \text{UCB}(a) = Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \]

Where:
- \( \text{UCB}(a) \) is the Upper Confidence Bound for action \( a \),
- \( Q(a) \) is the estimated value of action \( a \),
- \( c \) is a parameter controlling the balance between exploration and exploitation,
- \( t \) is the total number of time steps,
- \( N(a) \) is the number of times action \( a \) has been selected.

**Implementation:**

```python
def ucb_policy(self, state, c=1.0):
    total_visits = sum(self.action_counts[state])
    ucb_values = self.Q[state] + c * np.sqrt(np.log(total_visits + 1) / (self.action_counts[state] + 1))
    return np.argmax(ucb_values)
```

##  Multi-Armed Bandit Problem

The Multi-Armed Bandit problem is a fundamental concept in reinforcement learning and decision theory. It involves a scenario where an agent must choose between multiple options (or "arms") to maximize the total reward over a series of trials. Each arm provides a random reward from a fixed probability distribution, and the agent's goal is to find a balance between exploring new arms and exploiting the arms that have given high rewards in the past. This problem is widely used to model and solve real-world situations like clinical trials, ad placement, and A/B testing.

### Internal Implementation and Working Logic of Methods

#### 1. Value Iteration (Bandit)

**Internal Implementation:**
- **Initialization**: Initializes an action-value function `Q` and updates it until convergence.
- **Iteration**: Sets action values to their true expected rewards.

**Mathematical Formula:**

\[ Q(a) \leftarrow \mathbb{E}[r(a)] \]

Where:
- \( Q(a) \) is the action-value function for action \( a \),
- \( \mathbb{E}[r(a)] \) is the expected reward for action \( a \).

**Implementation:**

```python
def value_iteration(self, gamma=0.9, theta=1e-6):
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
```

#### 2. Policy Iteration (Bandit)

**Internal Implementation:**
- **Initialization**: Initializes an action-value function `Q` and updates policy and value function until convergence.
- **Policy Evaluation**: Evaluates the policy using the current value function.
- **Policy Improvement**: Updates the policy based on current value estimates.

**Mathematical Formula:**

\[ Q(a) \leftarrow \mathbb{E}[r(a)] \]

Where:
- \( Q(a) \) is the action-value function for action \( a \),
- \( \mathbb{E}[r(a)] \) is the expected reward for action \( a \).

**Implementation:**

```python
def policy_iteration(self,

 gamma=0.9, theta=1e-6):
    policy_stable = False
    while not policy_stable:
        while True:
            delta = 0
            for k in range(self.K):
                v = self.Q[k]
                self.Q[k] = self.true_rewards[k]
                delta = max(delta, abs(v - self.Q[k]))
            if delta < theta:
                break
        
        policy_stable = True
        for k in range(self.K):
            old_action = self.policy[k]
            self.policy[k] = np.argmax(self.Q)
            if old_action != self.policy[k]:
                policy_stable = False
```

#### 3. Q-Learning (Bandit)

**Internal Implementation:**
- **Initialization**: Initializes an action-value function `Q` and updates it based on experience.
- **Exploration vs. Exploitation**: Balances exploration and exploitation.
- **Q-Value Update**: Updates `Q` based on the Q-learning update rule.

**Mathematical Formula:**

\[ Q(a) \leftarrow Q(a) + \alpha [r + \gamma \max_{a'} Q(a') - Q(a)] \]

Where:
- \( Q(a) \) is the action-value function for action \( a \),
- \( \alpha \) is the learning rate,
- \( r \) is the reward received after taking action \( a \),
- \( \gamma \) is the discount factor.

**Implementation:**

```python
def q_learning(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    for episode in range(episodes):
        action = self.epsilon_greedy_policy(epsilon)
        reward = self.pull(action)
        best_next_action = np.argmax(self.Q)
        td_target = reward + gamma * self.Q[best_next_action]
        td_error = td_target - self.Q[action]
        self.Q[action] += alpha * td_error
```

#### 4. Epsilon-Greedy Policy (Bandit)

**Internal Implementation:**
- **Policy Selection**: Selects actions using an epsilon-greedy strategy.
- **Exploration**: Randomly selects actions with a probability controlled by epsilon.
- **Exploitation**: Selects actions greedily based on current Q-value estimates.

**Implementation:**

```python
def epsilon_greedy_policy(self, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, self.K - 1)
    else:
        return np.argmax(self.Q)
```

#### 5. UCB Algorithm (Bandit)

**Internal Implementation:**
- **Initialization**: Initializes an action-value function `Q` and tracks action counts.
- **Exploration vs. Exploitation**: Uses Upper Confidence Bound (UCB) to balance exploration and exploitation.
- **UCB Calculation**: Computes action selection probabilities using UCB.

**Mathematical Formula:**

\[ \text{UCB}(a) = Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \]

Where:
- \( \text{UCB}(a) \) is the Upper Confidence Bound for action \( a \),
- \( Q(a) \) is the estimated value of action \( a \),
- \( c \) is a parameter controlling the balance between exploration and exploitation,
- \( t \) is the total number of time steps,
- \( N(a) \) is the number of times action \( a \) has been selected.

**Implementation:**

```python
def ucb_policy(self, c=1.0):
    total_visits = sum(self.action_counts)
    ucb_values = self.Q + c * np.sqrt(np.log(total_visits + 1) / (self.action_counts + 1))
    return np.argmax(ucb_values)
```