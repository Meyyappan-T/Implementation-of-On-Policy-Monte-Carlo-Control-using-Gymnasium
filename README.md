# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
# Write your code here
# Monte Carlo Control using FrozenLake

## Student Details

**Name:** Meyyappan  
**Register Number:** 212223240086  

## Objective

Implement Monte Carlo Control using an epsilon-greedy policy to learn the optimal policy and state-value function for the FrozenLake environment.

## Complete Code

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    return np.argmax(Q[state])


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current
    epsilon-greedy policy.
    Returns a list of (state, action, reward).
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate one complete episode
    episode = generate_episode(epsilon)

    # Calculate total reward
    total_reward = sum(
        reward for _, _, reward in episode
    )

    episode_rewards.append(total_reward)

    # Initialize return
    G = 0

    # Store visited state-action pairs
    visited = set()

    # Process episode backwards
    for t in reversed(range(len(episode))):

        state, action, reward = episode[t]

        # Calculate return
        G = gamma * G + reward

        # First-visit Monte Carlo
        if (state, action) not in visited:

            visited.add((state, action))

            # Update Q-value
            Q[state, action] += alpha * (
                G - Q[state, action]
            )

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)

state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("Name: Meyyappan")
    print("Register Number: 212223240086")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


# -------------------------------------------------
# Print Final Results
# -------------------------------------------------

print("\nFinal Q-table:")

print(
    np.round(Q, 3)
)

print_value_function(state_values)

print_policy(optimal_policy)


# -------------------------------------------------
# Average Reward
# -------------------------------------------------

success_rate = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    success_rate
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)


plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "Monte Carlo Control Learning Curve"
)

plt.grid(True)

plt.show()


# -------------------------------------------------
# Close Environment
# -------------------------------------------------

env.close()


```

---

## Output


### Final Q-table:

<img width="287" height="387" alt="image" src="https://github.com/user-attachments/assets/049f59e7-6a90-4c23-b17b-93636eee6021" />



### Estimated State-Value Function:

<img width="336" height="158" alt="image" src="https://github.com/user-attachments/assets/786c5b3e-c516-4bd8-ac10-c7415bb2654e" />







### Learned Policy:
<img width="292" height="122" alt="image" src="https://github.com/user-attachments/assets/06d4d18e-d7b2-4168-be01-8ad0690e1125" />






### Average reward over last 1000 episodes: 
<img width="703" height="472" alt="image" src="https://github.com/user-attachments/assets/23d00534-0fc8-4235-b6d4-13b3b38b9911" />



---

## Result
```
The **On-Policy Monte Carlo Control** algorithm was successfully implemented using the Gymnasium `FrozenLake-v1` environment. The agent was trained for **20,000 episodes** using an epsilon-greedy policy. During training, the Q-table was updated using the returns obtained from complete episodes.

The final learned policy was obtained by selecting the action with the highest Q-value for each state using:

```python
optimal_policy = np.argmax(Q, axis=1)
---
```

## Inference

From the experiment, it can be inferred that **On-Policy Monte Carlo Control can learn an effective policy directly from experience without requiring an explicit model of the environment**.

Initially, the agent explores more because the epsilon value is high. As training progresses, epsilon decreases, allowing the agent to increasingly select actions with higher Q-values.

The final policy is obtained from the Q-table by selecting the action with the maximum Q-value:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

The learning process can be summarized as:

```text
Generate Complete Episodes
          ↓
Calculate Returns
          ↓
Update Q-values
          ↓
Improve Policy
          ↓
Repeat Training
          ↓
Extract Final Policy
```
