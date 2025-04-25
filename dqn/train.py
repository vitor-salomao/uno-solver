import torch
from gym_env import UnoEnv
from model import DQNAgent

# HYPER-PARAMETERS
NUM_EPISODES = 1500
BATCH_SIZE = 128
TARGET_UPDATE = 1000  # steps
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 3000

DISCOUNT_FACTOR = 0.90
LEARNING_RATE = 1e-4
DROPOUT = 0.1

SAVE_FILE = "uno_dqn1.pth" # "uno_dqn.pth"

play_counts, draw_counts = [], []

def main():
    env = UnoEnv()
    agent = DQNAgent(state_dim=219, action_dim=109, lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, dropout=DROPOUT)
    total_steps = 0

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()  # unpack the (obs, info) tuple
        state = torch.from_numpy(state)
        done = False
        ep_reward = 0.0
        plays = draws = 0

        while not done:
            # ε-greedy schedule
            eps = EPS_END + (EPS_START - EPS_END) * max(0, (EPS_DECAY - total_steps) / EPS_DECAY)
            action = agent.select_action(state, eps, env)
            if action < len(env.hands[0]):
                plays += 1
            else:
                draws += 1

            next_obs, reward, done, _ = env.step(action)
            next_state = torch.from_numpy(next_obs)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.optimize(BATCH_SIZE)

            state = next_state
            ep_reward += reward
            total_steps += 1

            if total_steps % TARGET_UPDATE == 0:
                agent.update_target()

        play_counts.append(plays)
        draw_counts.append(draws)

        if ep % 100 == 0:
            avg_plays = sum(play_counts[-100:]) / 100
            avg_draws = sum(draw_counts[-100:]) / 100
            print(f"Episode {ep:5d} | Steps {total_steps:7d} | EpReward {ep_reward:.2f} | Plays/Draws = {avg_plays:.1f}/{avg_draws:.1f}")

    # save weights
    torch.save(agent.policy_net.state_dict(), SAVE_FILE)

if __name__ == "__main__":
    main()
