from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from autoware import FrozenLakeEnv, plot_states_actions_distribution, qtable_directions_map, EpsilonGreedy, evaluate_tabular, Params
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

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

class SARSA:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state, new_action):
        delta = reward + self.gamma * self.qtable[new_state, new_action] - self.qtable[state, action]
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

def run_env_sarsa():
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

            action = explorer.choose_action(
                action_space=env.action_space, state=state, qtable=learner.qtable
            )

            while not done:

                all_states.append(state)
                all_actions.append(action)

                new_state, reward, terminated, truncated, info = env.step(action)

                new_action = explorer.choose_action(
                    action_space=env.action_space, state=new_state, qtable=learner.qtable
                )

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state, new_action
                )

                total_rewards += reward
                step += 1
                state = new_state
                action = new_action

            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions

def postprocess_sarsa(episodes, params, rewards, steps, map_size):
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

def plot_q_values_map_sarsa(qtable, env, map_size):
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[]
    ).set(title="Learned SARSA\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = 'q_values.png'
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

custom_map = [
    "SFFF",
    "FHCH",
    "FFFH",
    "HFCG"
]

map_sizes = [len(custom_map)]

# params._replace(total_episodes=800)
# params._replace(epsilon=0.08)
# params._replace(learning_rate=0.7)
# params._replace(gamma=0.99)
# params._replace(n_runs=1)
# params._replace(proba_frozen=0.9)
# params._replace(is_slippery=False)

res_all = pd.DataFrame()
st_all = pd.DataFrame()

for map_size in map_sizes:
    env = FrozenLakeEnv(
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        # desc=generate_random_map(
        #     size=map_size, p=params.proba_frozen, seed=params.seed
        # ),
        desc=custom_map
    )

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(
        params.seed
    )
    learner = SARSA(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedy(
        epsilon=params.epsilon,
    )

    print(f"Map size: {map_size}x{map_size}")
    rewards_sarsa, steps_sarsa, episodes_sarsa, qtables_sarsa, all_states_sarsa, all_actions_sarsa = run_env_sarsa()

    res_sarsa, st_sarsa = postprocess_sarsa(episodes_sarsa, params, rewards_sarsa, steps_sarsa, map_size)
    res_all_sarsa = pd.concat([res_all, res_sarsa])
    st_all_sarsa = pd.concat([st_all, st_sarsa])
    qtable_sarsa = qtables_sarsa.mean(axis=0)
    plot_states_actions_distribution(
        states=all_states_sarsa, actions=all_actions_sarsa, map_size=map_size, learningType='SARSA'
    )
    plot_q_values_map_sarsa(qtable_sarsa, env, map_size)

    env.close()

def plot_rew(title, rew_list):
    plt.ioff()
    plt.title("SARSA Rewards vs. Episodes: {}".format(title))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rew_list)
    plt.legend()
    plt.show()

for i in range(len(rewards_sarsa[0])):
    plot_rew(f'Run Number {i+1}', rewards_sarsa[:, i])

actions = ["left", "down", "right", "up"]
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 15)

df = pd.DataFrame(qtable_sarsa, columns=actions)
print("SARSA Q-table\n",df)

total_reward_sarsa = evaluate_tabular(env, qtable_sarsa)
print("Reward_SARSA:",total_reward_sarsa)