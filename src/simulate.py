# list of that cards that have been played
# number of cards each player has
# the order the game is going
# the cards on the player's hand
# Starter
# The cards on the player's hand
# The number of cards each player has

from typing import List
from round import Round
from collections import deque
import random
import time
import copy

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

def play_card(deck, cards: List[str], top, current_game: Round = None, randomly = True, current_player: int = 0) -> str:
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
            return [(current_game,updated_card)]
        return [(current_game, f"!{cards[0]}")]

    if randomly:
        card = valid_cards[random.randint(0, len(valid_cards) - 1)]
        cards.remove(card)
        updated_card = f"{card_color}_{card}" if card in ["+4", "W"] else card
        return [(current_game, updated_card)]
    else:
        # To ensure that we are not duplicating extra rounds
        seen_cards = []
        rounds = []
        for card in valid_cards:
            if (card in seen_cards):
                continue
            seen_cards.append(card)
            copied_all_player_cards = copy.deepcopy(current_game.cards)
            copied_all_player_cards[current_player].remove(card)
            rounds.append((Round(current_game.round, copied_all_player_cards, current_game.get_discard(), current_game.get_deck()), card))
        return rounds
        


            

def apply_effects(deck, cards, top):
    # TODO: IMPLEMENT THIS
    pass

def update_player(current_player, reverse):
    if reverse:
        current_player = current_player - 1 if current_player > 0 else NUM_PLAYERS - 1
    else:
        current_player = (current_player + 1) % NUM_PLAYERS
    return current_player


def create_game_instance(simple = False):
    # Initialize and shuffle the deck
    deck = generate_deck()

    # Deal 7 cards to each of the 4 players
    cards = [[], [], [], []]
    for i in range(7):
        cards[0].append(pull_card(deck))
        cards[1].append(pull_card(deck))
        cards[2].append(pull_card(deck))
        cards[3].append(pull_card(deck))
    # TODO: Fix Params
    return Round(0, cards, [pull_card(deck, True)], deck)
    

def run_game(games, simple = False):
    gameover = False               # Flag to track when the game ends
    current_player = 0            # Start with player 0
    reverse = False               # Game direction: False = clockwise, True = counterclockwise


    run_game_helper(games, games, simple)

def run_game_helper(games: List[Round], current_game: Round, simple = False):
    current_player = 0
    while 1:
        top = current_game.get_top_card()  # Get the top card on the discard pile
        if top[-1] == "S":
            print("================")
            print(f"Player {current_player + 1} was skipped!")
            current_player = update_player(current_player, reverse)

        # Display current game state
        print("================")
        print(f"Top card: {top}")
        print(f"Player {current_player + 1} has {current_game.get_player_cards(current_player)}.")

        # Attempt to play a valid card (or draw if none available)
        # TODO: Make Sure that Player 1 (i = 0 can play); Implement the optimal card play; (change 3 -> 1)
        if (current_player != 0):
            alternate_rounds = play_card(current_game.get_deck(),current_game.get_player_cards(current_player), top, current_game)
        else: # Only the "main player" will optimally play
            alternate_rounds = play_card(current_game.get_deck(), current_game.get_player_cards(current_player), top, current_game, False, current_player)
        
        i = 0
        for current_game, card_played in alternate_rounds:
            i += 1
            if card_played[0] != "!":  # If a valid card was played
                print(f"version {i}")
                current_game.get_discard().append(card_played)  # Add it to the discard pile

                # If the card is a reverse card, toggle game direction
                if card_played[-1] == "R":
                    reverse = not reverse
                else:
                    # Apply card effects if needed (to be implemented)
                    apply_effects(current_game.get_deck(), current_game.get_player_cards(current_player), top)

                print(f"Player {current_player + 1} plays {card_played}")
            else:
                # Card was drawn and couldn't be played
                print(f"Player {current_player + 1} has bought {card_played[1:]}")

            # Show updated hand
            print(f"Player {current_player + 1} now has {current_game.get_player_cards(current_player)}.")

            # Check for win condition (no more cards)
            if len(current_game.get_player_cards(current_player)) == 0:
                gameover = True
                print("=*=*=*=*=*=*=*=*=*=")
                print(f"Player {current_player + 1} wins!")
                print("=*=*=*=*=*=*=*=*=*=")
                return
        # TODO: Remove this return statement 
        return
        # Move to the next player based on the current direction
        # TODO: Fix functionality of reverse card -> add as a data member ?
        current_player = update_player(current_player, reverse)


def main():
    start_time = time.time()
    for _ in range(NUM_SIMULATIONS):
        # Using a deque to keep track of each new game instance
        run_game(create_game_instance(SIMPLE))
    end_time = time.time()
    print(f"\n{NUM_SIMULATIONS} games ran in {end_time - start_time} seconds.")

if __name__ == '__main__':
    main()