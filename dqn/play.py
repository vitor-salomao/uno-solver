import torch
from gym_env import UnoEnv
from model import QNetwork, DQNAgent

NUM_GAMES = 3000
LOG_ACTIONS = NUM_GAMES == 1
MODEL_FILE = "uno_dqn.pth"

def play_one_game(policy_path=MODEL_FILE):
    env = UnoEnv(log=LOG_ACTIONS)
    net = QNetwork(state_dim=219, action_dim=109)
    agent = DQNAgent(state_dim=219, action_dim=109)
    agent.policy_net = net  # inject your network
    net.load_state_dict(torch.load(policy_path))
    net.eval()

    state, _ = env.reset()
    state = torch.from_numpy(state)
    done = False
    total_r = 0.0
    total_wins = 0

    while not done:
        if LOG_ACTIONS:
            env.render()
        # eps=0 --> fully greedy + masked
        action = agent.select_action(state, 0.0, env)

        next_obs, reward, done, info = env.step(action)
        total_r += reward

        if "winner" in info and info["winner"] == 0:
            total_wins += 1

        if LOG_ACTIONS:
            if "bought" in info and info["bought"] is not None:
                print(f"Agent chose action {action} | bought card = {info['bought']} | reward={reward:.2f}")
            if "card" in info and info["card"] is not None:
                print(f"Agent chose action {action} | card = {info['card']} | reward={reward:.2f}")
            if "winner" in info and info["winner"] is not None:
                print(f"Player {info['winner']} wins!")
        state = torch.from_numpy(next_obs)

    return total_r, total_wins

if __name__ == "__main__":
    wins = 0
    well_played = 0
    total_rewards = 0
    for i in range(NUM_GAMES):
        rewards, win = play_one_game()
        if rewards > 0:
            well_played += 1
        wins += win
        total_rewards += rewards
        if i > 0 and i % 100 == 0:
            print(f"{i} games | Win rate: {wins/i:.2%} | Average reward: {total_rewards / i:.2f}")
    print(f"====== Final win rate: {wins/NUM_GAMES:.2%} ======")
    print(f"Well played: {well_played/NUM_GAMES:.2%} | Average reward: {total_rewards/NUM_GAMES:.2f}")