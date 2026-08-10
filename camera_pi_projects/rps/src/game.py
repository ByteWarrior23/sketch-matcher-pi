"""Rock-Paper-Scissors game logic (player move vs Pi move, score)."""
import random
from collections import deque

WIN = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


class RPSGame:
    def __init__(self, strategy="counter", history=4):
        self.strategy = strategy
        self.history = deque(maxlen=history)
        self.score = {"player": 0, "pi": 0, "draw": 0}

    def record(self, player_move):
        self.history.append(player_move)

    def pi_move(self):
        """Choose Pi's move. 'counter' beats the player's most common recent move."""
        if self.strategy != "counter" or not self.history:
            return random.choice(list(WIN.keys()))
        # most common recent player move
        counts = {}
        for m in self.history:
            counts[m] = counts.get(m, 0) + 1
        most_common = max(counts, key=counts.get)
        # what beats it: for move m, WIN[m] = move that m beats, so WIN[m] beats m
        return WIN[most_common]

    def round(self, player_move):
        """player_move in {rock,paper,scissors} or None (no gesture)."""
        if player_move is None:
            return None, None, "no gesture"
        self.record(player_move)
        pi = self.pi_move()
        if player_move == pi:
            outcome = "draw"
        elif WIN[player_move] == pi:
            outcome = "player"
        else:
            outcome = "pi"
        self.score[outcome] += 1
        return player_move, pi, outcome
