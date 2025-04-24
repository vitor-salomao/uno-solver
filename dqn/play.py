import torch
from gym_env import UnoEnv
from model import QNetwork

NUM_GAMES = 1000
LOG_ACTIONS = False

def play_one_game(policy_path="uno_dqn.pth", render=False):
    env = UnoEnv()
    net = QNetwork(state_dim=219, action_dim=109)

    # load the trained weights
    net.load_state_dict(torch.load(policy_path))
    net.eval()   # turn off dropout/batch-norm, if any

    state, _ = env.reset()
    state = torch.from_numpy(state)

    done = False
    total_reward = 0.0

    # play until done
    while not done:
        if LOG_ACTIONS: env.render()
        with torch.no_grad():
            # greedy - pick the highest-Q legal move
            q_vals = net(state.unsqueeze(0)).squeeze()
            action = int(q_vals.argmax().item())

        # step environment
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # log actions
        if render:
            print(f"Agent played action {action}, reward={reward:.1f}")
            if "card" in info and info["card"] is not None:
                print(f"Agent played card {info["card"].value} {info["card"].color}")

        state = torch.from_numpy(next_state)

    return total_reward

if __name__ == "__main__":
    wins = 0
    for i in range(NUM_GAMES):
        r = play_one_game(render=LOG_ACTIONS)
        if r > 0:  # positive reward = good
            wins += 1
    print(f"Out of {NUM_GAMES} games, agent won {wins} times ({wins/NUM_GAMES:.2%})")