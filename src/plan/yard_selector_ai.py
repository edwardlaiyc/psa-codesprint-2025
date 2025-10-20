from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional


@dataclass
class YardState:
    """Tracks dynamic congestion metrics for a single yard candidate."""

    load_estimate: float = 0.0
    assignments: int = 0
    recent_assignments: Deque[int] = field(default_factory=lambda: deque(maxlen=50))
    last_step: int = -1


class AdaptiveYardSelector:
    """
    Maintains a lightweight AI policy for yard selection that balances utilisation and congestion.

    The policy behaves like an online learner: it continuously updates an internal load estimate
    for each yard based on the assignments performed. The estimate decays over time so that the
    selector naturally forgets old work and keeps focusing on the most recent congestion footprint.

    Features
    --------
    * Soft load balancing via exponential decay – avoids starving any yard while still reacting fast.
    * Congestion penalty when a yard approaches the configured capacity (default: 700 jobs).
    * Exploration bonus for yards that have not been used recently so the policy does not get stuck.

    The selector surfaces a minimal API (`choose`) that can be reused by planners without exposing
    the internal bookkeeping logic.
    """

    def __init__(
        self,
        max_capacity: int = 700,
        decay_rate: float = 0.04,
        idle_exploration_bonus: float = 0.35,
        congestion_penalty_scale: float = 5.0,
        distance_weight: float = 0.0,
    ):
        if not 0.0 <= decay_rate < 1.0:
            raise ValueError("decay_rate must be within [0, 1).")
        if distance_weight < 0.0:
            raise ValueError("distance_weight must be non-negative.")

        self.max_capacity = max_capacity
        self.decay_rate = decay_rate
        self.idle_exploration_bonus = idle_exploration_bonus
        self.congestion_penalty_scale = congestion_penalty_scale
        self.distance_weight = distance_weight

        self._step: int = 0
        self._yard_state: Dict[str, YardState] = {}

    def _ensure_yards(self, yards: Iterable[str]) -> None:
        for yard in yards:
            if yard and yard not in self._yard_state:
                self._yard_state[yard] = YardState()

    def _advance_time(self) -> None:
        """Applies exponential decay so older assignments weigh less."""
        self._step += 1
        if self.decay_rate == 0.0:
            return

        for state in self._yard_state.values():
            if state.load_estimate <= 0.0:
                continue
            state.load_estimate *= 1.0 - self.decay_rate
            if state.load_estimate < 1e-3:
                state.load_estimate = 0.0

    def _score(
        self,
        yard: str,
        distance_lookup: Optional[Callable[[str], float]] = None,
    ) -> float:
        state = self._yard_state[yard]
        utilisation = state.load_estimate
        capacity_ratio = utilisation / float(self.max_capacity)
        congestion_penalty = max(0.0, capacity_ratio - 1.0) * (
            self.max_capacity * self.congestion_penalty_scale
        )

        if state.last_step >= 0:
            idle_steps = self._step - state.last_step
        else:
            idle_steps = self._step  # treat unseen yards as idle since start

        exploration_bonus = self.idle_exploration_bonus * idle_steps

        distance_penalty = 0.0
        if distance_lookup is not None and self.distance_weight > 0.0:
            distance_penalty = self.distance_weight * float(distance_lookup(yard))

        return utilisation + congestion_penalty + distance_penalty - exploration_bonus

    def _commit_assignment(self, yard: str) -> None:
        state = self._yard_state[yard]
        state.load_estimate += 1.0
        state.assignments += 1
        state.last_step = self._step
        state.recent_assignments.append(self._step)

    def choose(
        self,
        primary_yard: str,
        alternative_yards: Iterable[str],
        distance_lookup: Optional[Callable[[str], float]] = None,
    ) -> str:
        """
        Selects the best yard among a set of candidates using the learned congestion policy.

        Parameters
        ----------
        primary_yard : str
            The yard suggested by the job order.
        alternative_yards : Iterable[str]
            Alternative yards that are permissible fallbacks.

        Returns
        -------
        str
            The selected yard identifier.
        """
        candidates: List[str] = [
            yard for yard in [primary_yard, *alternative_yards] if yard
        ]
        if not candidates:
            raise ValueError("No yard candidates were provided.")

        self._ensure_yards(candidates)
        self._advance_time()

        chosen_yard = min(
            candidates,
            key=lambda yard: self._score(yard, distance_lookup=distance_lookup),
        )
        self._commit_assignment(chosen_yard)

        return chosen_yard

    def get_state_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Provides a serialisable view of the learned congestion profile (useful for debugging)."""
        snapshot: Dict[str, Dict[str, float]] = {}
        for yard, state in self._yard_state.items():
            snapshot[yard] = {
                "load_estimate": state.load_estimate,
                "assignments": float(state.assignments),
                "last_step": float(state.last_step),
            }
        return snapshot
