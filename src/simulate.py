# list of that cards that have been played
# number of cards each player has
# the order the game is going
# the cards on the player's hand
# Starter
# The cards on the player's hand
# The number of cards each player has

from typing import List
import random
import time

NUM_SIMULATIONS = 1
NUM_PLAYERS = 4
SIMPLE = True

def generate_deck() -> List[str]:
    deck = []
    types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "S", "R", "+2", "+4", "W"]
    for card in types:
        if card in ["+4", "W"]:
            for _ in range(4):
                deck.append(card)
        else:
            deck.append(f"r_{card}")
            deck.append(f"g_{card}")
            deck.append(f"b_{card}")
            deck.append(f"y_{card}")
            if card != 0:
                deck.append(f"r_{card}")
                deck.append(f"g_{card}")
                deck.append(f"b_{card}")
                deck.append(f"y_{card}")
    return deck

def pull_card(deck, start = False) -> str:
    if not deck:
        deck = generate_deck()
    index = random.randint(0, len(deck) - 1)
    if start: # no action cards to start game
        while deck[index][-2:] in ["+4", "_W", "_S", "_R", "+2"]:
            index = random.randint(0, len(deck) - 1)
    return deck.pop(index)

def buy_card(deck, cards):
    if not deck:
        deck = generate_deck()
    cards.append(deck.pop())

def play_card(deck, cards, top, randomly = True) -> str:
    valid_cards = []
    # for now, just color or wild
    card_color = top[0]
    card_type = top[-2:]
    for card in cards:
        if card in ["+4", "W"] or card_color == card[0] or card_type == card[-2:]:
            valid_cards.append(card)

    if len(valid_cards) == 0: # has to buy
        buy_card(deck, cards)
        if card_color == cards[0] or cards[0] in ["+4", "W"]: # pulled a valid card
            card = valid_cards[random.randint(0, len(valid_cards) - 1)]
            cards.remove(card)
            updated_card = f"{card_color}_{card}" if card in ["+4", "W"] else card
            return updated_card
        return f"!{cards[0]}"

    if randomly:
        card = valid_cards[random.randint(0, len(valid_cards) - 1)]
        cards.remove(card)
        updated_card = f"{card_color}_{card}" if card in ["+4", "W"] else card
        return updated_card

def apply_effects(deck, cards, top):
    # TODO: IMPLEMENT THIS
    pass

def update_player(current_player, reverse):
    if reverse:
        current_player = current_player - 1 if current_player > 0 else NUM_PLAYERS - 1
    else:
        current_player = (current_player + 1) % NUM_PLAYERS
    return current_player

def run_game(simple = False):
    # Initialize and shuffle the deck
    deck = generate_deck()

    # Deal 7 cards to each of the 4 players
    cards = [[], [], [], []]
    for i in range(7):
        cards[0].append(pull_card(deck))
        cards[1].append(pull_card(deck))
        cards[2].append(pull_card(deck))
        cards[3].append(pull_card(deck))

    gameover = False               # Flag to track when the game ends
    current_player = 0            # Start with player 0
    discard = [pull_card(deck, True)]  # Pick the starting card (non-action)
    reverse = False               # Game direction: False = clockwise, True = counterclockwise

    while not gameover:
        top = discard[-1]  # Get the top card on the discard pile

        # Handle skip cards (player loses their turn)
        if top[-1] == "S":
            print("================")
            print(f"Player {current_player + 1} was skipped!")
            current_player = update_player(current_player, reverse)

        # Display current game state
        print("================")
        print(f"Top card: {top}")
        print(f"Player {current_player + 1} has {cards[current_player]}.")

        # Attempt to play a valid card (or draw if none available)
        card_played = play_card(deck, cards[current_player], top)

        if card_played[0] != "!":  # If a valid card was played
            discard.append(card_played)  # Add it to the discard pile

            # If the card is a reverse card, toggle game direction
            if card_played[-1] == "R":
                reverse = not reverse
            else:
                # Apply card effects if needed (to be implemented)
                apply_effects(deck, cards[current_player], top)

            print(f"Player {current_player + 1} plays {card_played}")
        else:
            # Card was drawn and couldn't be played
            print(f"Player {current_player + 1} has bought {card_played[1:]}")

        # Show updated hand
        print(f"Player {current_player + 1} now has {cards[current_player]}.")

        # Check for win condition (no more cards)
        if len(cards[current_player]) == 0:
            gameover = True
            print("=*=*=*=*=*=*=*=*=*=")
            print(f"Player {current_player + 1} wins!")
            print("=*=*=*=*=*=*=*=*=*=")

        # Move to the next player based on the current direction
        current_player = update_player(current_player, reverse)


def main():
    start_time = time.time()
    for _ in range(NUM_SIMULATIONS):
        run_game(SIMPLE)
    end_time = time.time()
    print(f"\n{NUM_SIMULATIONS} games ran in {end_time - start_time} seconds.")

if __name__ == '__main__':
    main()