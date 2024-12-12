from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from autoware import FrozenLakeEnv, plot_states_actions_distribution, plot_q_values_map, EpsilonGreedy, evaluate_tabular, Params
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from pathlib import Path
sns.set_theme()
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

params = Params(
    total_episodes=1001,
    learning_rate=0.6,
    gamma=0.99,
    epsilon=0.07,
    map_size=121,
    seed=42,
    is_slippery=False,
    n_runs=1,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("output"),
)

params.savefig_folder.mkdir(parents=True, exist_ok=True)
print("Parameters: ", params)
env = FrozenLakeEnv()
params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
print("Action size: ", params.action_size)
print("State size: ",params.state_size)

class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))
    
learner = Qlearning(
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    state_size=params.state_size,
    action_size=params.action_size,
)

explorer = EpsilonGreedy(
    epsilon=params.epsilon,
)

def run_env():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):

        learner.reset_qtable()

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):

            state = env.reset(seed=params.seed)[0] 
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                all_states.append(state)
                all_actions.append(action)

                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1

                state = new_state

            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions

def postprocess(episodes, params, rewards, steps, map_size):
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st


custom_map = [
    "SFFF",
    "FHCH",
    "FFFH",
    "HFCG"
]


map_sizes = [len(custom_map)]

res_all = pd.DataFrame()
st_all = pd.DataFrame()

for map_size in map_sizes:
    env = FrozenLakeEnv(
        is_slippery=params.is_slippery,
        render_mode="rgb_array",

        # desc=generate_random_map(
        #     size=map_size, p=params.proba_frozen, seed=params.seed
        # ),
        #

        desc=custom_map
    )

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(
        params.seed
    ) 
    learner = Qlearning(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedy(
        epsilon=params.epsilon,
    )

    print(f"Map size: {map_size}x{map_size}")
    rewards, steps, episodes, qtables, all_states, all_actions = run_env()

    res, st = postprocess(episodes, params, rewards, steps, map_size)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    qtable = qtables.mean(axis=0)

    plot_states_actions_distribution(states=all_states, actions=all_actions, map_size=map_size, learningType='Q-Learning')
    plot_q_values_map(qtable, env, map_size)
    print('Plotted QT ', qtable)
    env.close()

def plot_rew(title, rew_list):
    plt.ioff()
    plt.title("Q-Learning Rewards vs. Episodes: {}".format(title))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rew_list)
    plt.legend()
    plt.show()

for i in range(len(rewards[0])):
    plot_rew(f'Run Number {i+1}', rewards[:, i])

actions = ["left", "down", "right", "up"]
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 15)

df = pd.DataFrame(qtable, columns=actions)
print("Q-Learning Q-table\n",df)

total_reward_q_learning = evaluate_tabular(env, qtable)
print("Reward Q_learning:",total_reward_q_learning)