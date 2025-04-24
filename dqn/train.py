import torch
from gym_env import UnoEnv
from model import DQNAgent

env = UnoEnv()
agent = DQNAgent(state_dim=219, action_dim=109)

num_episodes = 20000
batch_size = 128
target_update = 1000  # steps
eps_start, eps_end, eps_decay = 1.0, 0.1, 20000

def main():
    for ep in range(num_episodes):
        state = torch.from_numpy(env.reset())
        done = False
        while not done:
            eps = eps_end + (eps_start - eps_end) * \
                  max(0, (eps_decay - agent.steps_done) / eps_decay)
            action = agent.select_action(state, eps)
            next_obs, reward, done, _ = env.step(action)
            next_state = torch.from_numpy(next_obs)
            agent.buffer.push(state, action, reward, next_state, done)

            state = next_state
            agent.optimize(batch_size)

            agent.steps_done += 1
            if agent.steps_done % target_update == 0:
                agent.update_target()

        if ep % 100 == 0:
            print(f"Episode {ep} complete")

if __name__ == "__main__":
    main()
