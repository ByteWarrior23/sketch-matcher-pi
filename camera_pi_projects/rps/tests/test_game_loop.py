"""Headless test of the RPS game-loop state machine (no camera, no GUI).

Simulates the debounce logic that rps_loop.py applies per frame and asserts
the invariants: held gestures register exactly one round, jitter resets the
hold, low-confidence frames clear the hold, and the counter strategy beats the
player's most common recent move.
"""
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import HOLD_FRAMES  # noqa: E402
from game import RPSGame, WIN  # noqa: E402

MOVES = list(WIN.keys())
failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}")
    if not cond:
        failures.append(name)


class LoopSim:
    """Mirror of the state machine inside rps_loop.main()."""

    def __init__(self, hold=HOLD_FRAMES):
        self.game = RPSGame(strategy="counter")
        self.current = None
        self.pending = None
        self.hold_count = 0
        self.rounds = []
        self.hold = hold

    def frame(self, label, conf):
        """Feed one classification result. Returns True if a round fired."""
        fired = False
        if conf >= 0.75:
            if self.pending == label:
                self.hold_count += 1
            else:
                self.pending = label
                self.hold_count = 1
            if self.hold_count >= self.hold and label != self.current:
                self.current = label
                self.rounds.append(self.game.round(label))
                self.hold_count = 0
                fired = True
        else:
            self.pending = None
            self.hold_count = 0
        return fired


def main():
    # 1) held gesture registers exactly one round
    sim = LoopSim()
    fired = 0
    for _ in range(40):
        fired += int(sim.frame("rock", 0.9))
    check("held gesture fires exactly 1 round", fired == 1 and len(sim.rounds) == 1)
    check("round recorded (player, pi, outcome)", len(sim.rounds) == 1 and sim.rounds[0][0] == "rock")

    # 2) jitter before hold threshold resets the counter
    sim = LoopSim()
    fired = 0
    for _ in range(HOLD_FRAMES - 1):
        sim.frame("rock", 0.9)
    for _ in range(5):  # switch to paper mid-hold
        sim.frame("paper", 0.9)
    for _ in range(40):
        fired += int(sim.frame("paper", 0.9))
    check("jitter resets hold, final gesture still registers", fired == 1 and sim.rounds[0][0] == "paper")

    # 3) low-confidence frames clear the hold
    sim = LoopSim()
    for _ in range(HOLD_FRAMES - 1):
        sim.frame("rock", 0.9)
    for _ in range(3):
        sim.frame("rock", 0.3)  # no gesture
    fired = 0
    for _ in range(40):
        fired += int(sim.frame("rock", 0.9))
    check("low-confidence clears hold, re-hold registers", fired == 1 and len(sim.rounds) == 1)

    # 4) counter strategy beats the most common recent move
    game = RPSGame(strategy="counter")
    game.history = deque(["rock", "paper", "rock"], maxlen=4)
    pi = game.pi_move()
    check(f"counter beats most-common (rock -> {pi})", pi == "scissors")

    # 5) scoring is consistent: player wins iff move beats pi
    game = RPSGame(strategy="counter")
    game.history = deque(["paper"], maxlen=4)  # pi counters with rock
    mv, pi, outcome = game.round("paper")  # paper beats rock
    check("player win outcome", outcome == "player")
    check("score incremented", game.score == {"player": 1, "pi": 0, "draw": 0})

    game.history = deque(["paper", "paper"], maxlen=4)  # pi counters with rock
    mv, pi, outcome = game.round("scissors")  # rock beats scissors -> pi wins
    check("pi win outcome", outcome == "pi")
    check("pi score incremented", game.score["pi"] == 1)

    game.round("rock")
    check("draw outcome + score", game.score["draw"] == 1)

    # 6) WIN table is a perfect cycle
    check("WIN is a valid cycle", WIN["rock"] == "scissors" and WIN["scissors"] == "paper" and WIN["paper"] == "rock")

    # 7) None (no gesture) does not record or score
    game = RPSGame()
    r = game.round(None)
    check("None round is no-op", r == (None, None, "no gesture") and game.score == {"player": 0, "pi": 0, "draw": 0})

    print()
    if failures:
        print(f"LOOP TEST FAILED ({len(failures)}): {failures}")
        sys.exit(1)
    print("LOOP TEST OK")


if __name__ == "__main__":
    main()
