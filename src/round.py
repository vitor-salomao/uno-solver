class Round:
    """
    This object stores data of some variation of a game of Uno. Each instance is created when the main player (who is trying to play optimally) has to choose a card.
    In order to find which cards are optimal, we are taking the Greedy approach, where we decide that the most optimal card to play is which ever one results in the
    least cards left over. By creating many different variations of each game, we can figure out which choices eventually lead to a loss and which ones lead to a win.
    If one variation results in the win, the data stored in that instance will be used to train the model.
    """
    # TODO: Update code in round.py and simulate.py to use decorators for readability when we have enough time
    def __init__(self, round: int, cards: list[list[str]], discard: list[str], deck: list[str], cards_played : list[str] = [], current_player : int = 0, round_data : list = []):
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

        self.round_data = round_data
        
    
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
        self.winner = winner

    def get_winner(self):
        return self.winner

    def write_round_data(self):
        """
        Should store top card, our cards, other player cards (before we actually play the card)
        Note that other information like our number of cards can be retrieved easily
        
        Other data can be added. It is so far formatted like this:
        (top_card, our_cards[], length_of_other_player_cards x3)

        This data is used to keep track of what the data was when we played our card
        """
        round_data = (self.get_top_card(), self.get_player_cards(0), len(self.get_player_cards(1)), len(self.get_player_cards(2)), len(self.get_player_cards(3)))
        self.round_data.append(round_data)

    def get_round_data(self):
        """
        Returns:
            x: Features
            y: Choices
        """
        return (self.round_data, self.cards_played)
    
