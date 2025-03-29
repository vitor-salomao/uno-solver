class Round:
    def __init__(self, round: int, cards: list[list[str]], discard: list[str], deck: list[str]):
        # Stores the current round of the game
        self.round = round
        # Stores the cards for all players of the current round
        self.cards = cards
        # Stores the discard pile of the current round
        self.discard = discard
        # Stores the remaining deck of the current round
        self.deck = deck
    
    def get_round(self):
        return self.round
    
    def get_player_cards(self, player: int):
        return self.cards[player]
    
    def get_top_card(self):
        return self.discard[-1]
    
    def get_discard(self):
        return self.discard
    
    def get_deck(self):
        return self.deck
    
    def remove_player_card(self, player: int, card_played: str):
        pass