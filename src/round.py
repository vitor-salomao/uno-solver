class Round:
    # TODO: Add functionality to choose the next player
    # TODO: For example, adding "reversed" data member and next player member function

    # TODO: Store data from every round so that we can identify what we played, once the game is over.
    # TODO: Only rounds that resulted in a win will use this stored data
    def __init__(self, round: int, cards: list[list[str]], discard: list[str], deck: list[str], cards_played : list[str] = [], current_player : int = 0):
        # Stores the current round of the game
        self.round = round
        # Stores the cards for all players of the current round
        self.cards = cards
        # Stores the discard pile of the current round
        self.discard = discard
        # Stores the remaining deck of the current round
        self.deck = deck
        self.cards_played = cards_played
        self.reversed = False
        # Current player
        self.current_player = current_player
        self.gameover = 0
        self.winner = None
        
    
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
    
    # def remove_player_card(self, player: int, card_played: str):
    #     pass

    def get_cards_played(self):
        return self.cards_played
    
    def apply_reverse(self):
        self.reversed = not self.reversed

    def get_current_player(self):
        return self.current_player
    
    def update_player(self, NUM_PLAYERS):
        if self.reversed:
            self.current_player = self.current_player - 1 if self.current_player > 0 else NUM_PLAYERS - 1
        else:
            self.current_player = (self.current_player + 1) % NUM_PLAYERS
        return self.current_player
    
    def over(self):
        return self.gameover

    def finish_game(self, winner):
        self.gameover = 1