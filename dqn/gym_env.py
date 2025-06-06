import random
from collections import deque, Counter
from typing import Optional, Tuple, List

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

# PARAMETERS
AGGRESSIVE_OPPONENT = True
STACK_CARDS = True
# REWARDS
REWARD_WIN = 15.0
REWARD_LOSE_FLAT = -1.0
REWARD_LOSE_MULT = 0.95   # this * number of cards left
REWARD_LEGAL = 4.0
REWARD_STACK = 6.0
REWARD_STACK_MISS = -2.0
REWARD_DRAW = -5.5
REWARD_STEP = -0.05

# card struct
class Card:
    """
    A single UNO card.
    index: unique 0–107 integer (for one‑hot encoding)
    color: 'red','green','blue','yellow', or 'wild'
    value: '0'–'9', 'skip','reverse','draw_two','wild','wild_draw_four'
    """
    def __init__(self, index: int, color: str, value: str):
        self.index = index
        self.color = color
        self.value = value

    def __repr__(self):
        return f"{self.color.upper()} {self.value}"

# deck functionality
class Deck:
    def __init__(self):
        deck_list = self.build_full_deck()
        self.cards = deque(deck_list)
        self.shuffle()

    def build_full_deck(self):
        colors = ['red','green','blue','yellow']
        values = [str(n) for n in range(1,10)]*2 + ['0']  # two copies of 1–9, one 0
        action_cards = ['skip','reverse','draw_two'] * 2  # two of each per color

        deck = []
        idx = 0
        for color in colors:
            # zeros
            deck.append(Card(idx, color, '0')); idx += 1
            # 1–9 twice
            for v in values:
                if v != '0':
                    deck.append(Card(idx, color, v)); idx += 1
            # action
            for a in action_cards:
                deck.append(Card(idx, color, a)); idx += 1
        # wild
        for _ in range(4):
            deck.append(Card(idx, 'wild', 'wild')); idx += 1
            deck.append(Card(idx, 'wild', 'wild_draw_four')); idx += 1

        assert len(deck) == 108, f"UNO deck should be 108 cards, got {len(deck)}"
        return deck

    def shuffle(self):
        deck_list = list(self.cards)
        random.shuffle(deck_list)
        # use deque for O(1) pops from left
        self.cards = deque(deck_list)

    def draw(self, n=1):
        """Draw n cards from the top of the deck; returns a list of Card."""
        drawn = []
        for _ in range(n):
            if not self.cards:
                raise RuntimeError("Deck is empty! Need to reshuffle discard pile.")
            drawn.append(self.cards.popleft())
        return drawn

    def __len__(self):
        return len(self.cards)

# uno gym environment
class UnoEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, log = False, agg_opp = False, stack = False):
        super().__init__()
        self.log = log
        self.agg_opp = agg_opp

        # state: hand (108 hot) + top card (108 hot) + 3 opp counts
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(108 + 108 + 3,), dtype=np.float32
        )
        # 108 possible plays + 1 draw action
        self.action_space = spaces.Discrete(109)

        self.seed()
        self.deck = None
        self.hands = None
        self.discard_pile = None
        self.top = None
        self.current_player = 0
        self.direction = 1

        # stacking
        self.pending_draw = 0
        self.pending_type = None
        self.stack = stack

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        if self.log: print(f"SEED: {seed}")
        return [seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self.seed(seed)
        self.deck = Deck()
        self.hands = [self.deck.draw(7) for _ in range(4)]
        self.discard_pile = deque()
        self.top = self.deck.draw(1)[0]
        self.discard_pile.append(self.top)
        self.direction = 1
        self.current_player = 0
        self.pending_draw = 0
        self.pending_type = None

        obs = self._encode_obs()
        info = {} # fill for info
        return obs, info

    # one round of a game
    def step(self, action):
        """
        Execute one turn of UNO:
          1. Current player (self.current_player) either plays a card or draws.
          2. Enforce card effects (skip, reverse, draw_two, wild, etc.).
          3. Advance to the next player.
          4. Check for end‐of‐game and assign rewards.
        """
        hand = self.hands[self.current_player]
        done, info = False, {}
        reward = 0.0
        played_card, bought_card = None, None

        # stacking +4 and +2
        if self.stack and self.pending_draw > 0:
            # the card stacks and adds to pending
            if action < len(hand) and hand[action].value == self.pending_type:
                card = hand[action]
                self._play_card(self.current_player, card)
                reward += REWARD_STACK
                add = 4 if card.value == 'wild_draw_four' else 2
                self.pending_draw += add
                if self.log: print(f"▶️ AGENT played and stacked {card}")

            # else, draw the pending amount
            else:
                self._draw_cards(self.current_player, self.pending_draw)
                if self.log: print(f"▶️ AGENT drew {self.pending_draw}")
                reward += -self.pending_draw + REWARD_STACK_MISS
                self.pending_draw = 0
                self.pending_type = None
                self._advance_to_next()

        # nothing stack related, just play normal
        else:
            if action < len(hand):
                card = hand[action]
                reward += REWARD_LEGAL
                self._play_card(self.current_player, card)
                if self.log: print(f"▶️ AGENT played {card}")
                played_card = card
            else:
                # draw action
                reward += REWARD_DRAW
                drawn = self._draw_cards(self.current_player, 1)
                bought_card = drawn[0]

                # if drawn[0] is playable, play it
                if self._is_playable(drawn[0]):
                    reward += REWARD_LEGAL
                    self._play_card(self.current_player, drawn[0])
                    played_card = drawn[0]
                    if self.log: print(f"▶️ AGENT drew and played {bought_card}")
                else:
                    if self.log: print(f"▶️ AGENT drew {bought_card}")
                    self._advance_to_next()

        reward += REWARD_STEP
        info = {"card": played_card, "bought": bought_card, "winner": None}

        # check win
        if len(self.hands[0]) == 0:
            reward += REWARD_WIN  # agent won
            info["winner"] = 0
            done = True
            return self._encode_obs(), reward, done, info

        # opponents play
        while self.current_player != 0 and not done:
            player_idx = self.current_player
            self._opponent_play(self.current_player)
            if len(self.hands[player_idx]) == 0:
                reward = REWARD_LOSE_FLAT + REWARD_LOSE_MULT * len(self.hands[0])  # opponent won
                info["winner"] = player_idx
                done = True
                break

        return self._encode_obs(), reward, done, info

    # model feature vector
    def _encode_obs(self):
        hand, top = self.hands[self.current_player], self.top
        vec_hand = np.zeros(108, dtype=np.float32)
        for c in hand:
            vec_hand[c.index] = 1.0

        vec_top = np.zeros(108, dtype=np.float32)
        vec_top[top.index] = 1.0

        # direction-based opponent card counts
        offsets = [(self.current_player + self.direction * i) % 4 for i in (1, 2, 3)]
        opp_counts = np.array([len(self.hands[p]) for p in offsets], dtype=np.float32)
        return np.concatenate([vec_hand, vec_top, opp_counts])

    def legal_actions(self) -> List[int]:
        """
        Returns a list of legal action indices:
         - for each card i in your hand, if _is_playable(card) → include i
         - always include the “draw” action (fixed index = 108)
        """
        hand = self.hands[self.current_player]
        legal = [i for i, card in enumerate(hand) if self._is_playable(card)]
        legal.append(self.action_space.n - 1)  # draw action
        return legal

    # show game state
    def render(self):
        print("\n" + "-" * 40)
        print(f"▶️ Your turn")
        print(f"Top card: {self.top}")
        for i, hand in enumerate(self.hands):
            if i == 0:
                print(f"Your hand: {hand}")
            else:
                print(f"Opponent {i}: {len(hand)} cards")

    # ============= helper functions for game loop ============= #
    def _is_playable(self, card: Card) -> bool:
        """True if color matches or value matches or card is wild."""
        top = self.top
        return (
                card.color == top.color or
                card.value == top.value or
                card.color == 'wild'
        )

    def _apply_card_effect(self, card: Card):
        """
        Handle special cards:
          - 'skip': call advance twice
          - 'reverse': flip self.direction *= -1
          - 'draw_two': next player draws 2 then skip them
          - 'wild': pick most abundant color
          - 'wild_draw_four': same + next draws 4
        """
        if card.value == 'skip':
            self._advance_to_next()
            self._advance_to_next()
        elif card.value == 'reverse':
            self.direction *= -1
            self._advance_to_next()
        elif card.value == 'draw_two':
            if self.stack:
                self.pending_draw += 2
                self.pending_type = 'draw_two'
                self._advance_to_next()
            else:
                target = (self.current_player + self.direction) % 4
                self._draw_cards(target, 2)
                self._advance_to_next()
                self._advance_to_next()
        elif card.value == 'wild':
            chosen = self._choose_color(self.hands[self.current_player])
            self.top.color = chosen
            self._advance_to_next()
        elif card.value == 'wild_draw_four':
            chosen = self._choose_color(self.hands[self.current_player])
            self.top.color = chosen
            if self.stack:
                self.pending_draw += 4
                self.pending_type = 'wild_draw_four'
                self._advance_to_next()
            else:
                target = (self.current_player + self.direction) % 4
                self._draw_cards(target, 4)
                self._advance_to_next()
                self._advance_to_next()
        else:
            self._advance_to_next()

    def _advance_to_next(self):
        """Move turn pointer by self.direction (+1 or -1 mod 4)."""
        self.current_player = (self.current_player + self.direction) % 4

    def _draw_cards(self, player_idx: int, n: int):
        """
        Draw n cards for a given player
        If deck runs out, reshuffle discard pile (except top)
        """
        # reshuffle if needed
        if len(self.deck) < n:
            # take all but the top card in discard, shuffle back into deck
            top = self.discard_pile.pop()
            reshuffle_cards = list(self.discard_pile)
            random.shuffle(reshuffle_cards)
            self.deck.cards.extend(reshuffle_cards)
            self.discard_pile = deque([top])
        cards = self.deck.draw(n)
        self.hands[player_idx].extend(cards)
        return cards

    def _choose_color(self, hand: List[Card]) -> str:
        """Heuristic: pick the color you have most of"""
        counts = Counter(c.color for c in hand if c.color != 'wild')
        if not counts:
            # only wild cards
            return random.choice(['red', 'green', 'blue', 'yellow'])
        return counts.most_common(1)[0][0]

    def _opponent_play(self, player_idx: int):
        """Simulate opponent play, either random or aggressive"""
        hand = self.hands[player_idx]
        playable = [c for c in hand if self._is_playable(c)]

        # stacking opponent - notices stack opportunities
        if self.stack and self.pending_draw > 0:
            stack_cards = [c for c in self.hands[player_idx] if c.value == self.pending_type]
            if stack_cards:
                card = random.choice(stack_cards)
                self._play_card(player_idx, card)
                if self.log: print(f"-- Player {player_idx} stacked {card}")
                return

            self._draw_cards(player_idx, self.pending_draw)
            if self.log: print(f"-- Player {player_idx} drew {self.pending_draw} cards")
            self.pending_draw = 0
            self.pending_type = None
            self._advance_to_next()
            return

        # aggressive opponent - prioritize attacking cards
        if self.agg_opp and playable:
            # priority list
            priority = ['wild_draw_four', 'draw_two', 'skip']
            # try to find best cards
            for val in priority:
                candidates = [c for c in playable if c.value == val]
                if candidates:
                    card = random.choice(candidates)
                    self._play_card(player_idx, card)
                    if self.log: print(f"-- Player {player_idx} AGGRESSIVELY played {card}")
                    return

        # regular opponent - random valid card
        if playable:
            card = random.choice(playable)
            self._play_card(player_idx, card)
            if self.log: print(f"-- Player {player_idx} played {card}")
            return

        # no playable card, draw and check
        cards = self._draw_cards(player_idx, 1)
        if self._is_playable(cards[0]):
            self._play_card(player_idx, cards[0])
            if self.log: print(f"-- Player {player_idx} bought and played {cards[0]}")
        else:
            if self.log: print(f"-- Player {player_idx} bought {hand[-1]}")
            self._advance_to_next()

    def _play_card(self, player_idx: int, card: Card):
        self.hands[player_idx].remove(card)
        self.discard_pile.append(card)
        self.top = card
        self._apply_card_effect(card)
