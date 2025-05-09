import argparse
import torch
import numpy as np

from gym_env import Card, UnoEnv
from model import QNetwork

COLOR_MAP = {'R':'red','G':'green','B':'blue','Y':'yellow','W':'wild'}
VALUE_MAP = {
    **{str(i):str(i) for i in range(10)},
    'skip':'skip','reverse':'reverse','draw_two':'draw_two',
    'wild':'wild','wild_draw_four':'wild_draw_four'
}

def parse_card(token: str):
    color_key = token[0]
    val_key = token[1:]
    color = COLOR_MAP[color_key]
    value = VALUE_MAP[val_key]

    from gym_env import Deck
    for c in Deck().build_full_deck():
        if c.color == color and c.value == value:
            return c
    raise ValueError(f"Unknown card token '{token}'")

def build_obs(hand_cards, top_card, opp_counts):
    vec_hand = np.zeros(108, dtype=np.float32)
    for c in hand_cards:
        vec_hand[c.index] = 1.0

    vec_top = np.zeros(108, dtype=np.float32)
    vec_top[top_card.index] = 1.0
    vec_opp = np.array(opp_counts, dtype=np.float32)

    return np.concatenate([vec_hand, vec_top, vec_opp])

def main():
    parser = argparse.ArgumentParser(
        description="Suggest the best UNO card to play with your trained DQN"
    )
    parser.add_argument(
        '--hand', required=True, nargs='+',
        help='Your cards, e.g. R5 Gskip B2'
    )
    parser.add_argument(
        '--opp-cards', required=True, nargs=3, type=int,
        help='Number of cards in opponents’ hands, in play order, e.g. 5 4 6'
    )
    parser.add_argument(
        '--top-card', required=True,
        help='Face-up card on the table, e.g. Y7'
    )
    parser.add_argument(
        '--model-path', default='uno_dqn.pth',
        help='Path to your trained model weights'
    )
    args = parser.parse_args()

    # parse
    hand_cards = [parse_card(tok) for tok in args.hand]
    top_card = parse_card(args.top_card)
    opp_counts = args.opp_cards

    # build observation vector
    obs = build_obs(hand_cards, top_card, opp_counts)
    state = torch.from_numpy(obs)

    # load net
    net = QNetwork(state_dim=219, action_dim=109)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    # set up env with state
    env = UnoEnv()
    env.hands = [hand_cards, [], [], []]
    env.top = top_card
    env.current_player = 0
    legal = env.legal_actions()

    # compute Q-values and select best legal
    with torch.no_grad():
        q = net(state.unsqueeze(0)).squeeze()
    best = max(legal, key=lambda a: q[a].item())

    # print suggestion
    if best < len(hand_cards):
        card = hand_cards[best]
        print(f"▶️ Suggestion: play → {card.color.upper()} {card.value}")
    else:
        print("▶️ Suggestion: draw a card")

if __name__ == "__main__":
    main()