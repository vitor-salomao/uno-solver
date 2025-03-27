# list of that cards that have been played
# number of cards each player has
# the order the game is going
# the cards on the player's hand
# Starter
# The cards on the player's hand
# The number of cards each player has

import random

NUM_SIMULATIONS = 1000
NUM_PLAYERS = 4
SIMPLE = True

def generate_deck():
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

def pull_card(deck):
    if not deck:
        deck = generate_deck()
    index = random.randint(0, len(deck) - 1)
    return deck.pop(index)

def buy_card(deck, cards):
    if not deck:
        deck = generate_deck()
    cards.append(deck.pop())

def play_card(deck, cards, top, randomly = True):
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

def generate_train_data(simple = False):
    deck = generate_deck()
    cards = [[],[],[],[]]
    for i in range(7): # Start deck
        cards[0].append(pull_card(deck))
        cards[1].append(pull_card(deck))
        cards[2].append(pull_card(deck))
        cards[3].append(pull_card(deck))

    gameover = False
    current_player = 0
    discard = [pull_card(deck)]
    while not gameover:
        top = discard[-1]
        # TODO: IMPLEMENT CARD EFFECT
        print("================")
        print(f"Top card: {top}")
        print(f"Player {current_player} has {cards[current_player]}.")
        card_played = play_card(deck, cards[current_player], top)
        if card_played[0] != "!":
            discard.append(card_played)
            print(f"Player {current_player} plays {card_played}")
        else:
            print(f"Player {current_player} has bought {card_played[1:]}")
        print(f"Player {current_player} now has {cards[current_player]}.")
        if len(cards[current_player]) == 0:
            gameover = True
            print(f"Player {current_player} wins!")

        current_player = (current_player + 1) % NUM_PLAYERS    # update player

def main():
    generate_train_data(SIMPLE)

if __name__ == '__main__':
    main()