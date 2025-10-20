from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from src.constant import CONSTANT
from src.floor import Coordinate


@dataclass
class _HTState:
    load_estimate: float = 0.0
    last_assigned_step: int = -1


class AdaptiveHTSelector:
    """Adaptive HT selection balancing travel effort and utilisation."""

    def __init__(
        self,
        decay_rate: float = 0.118,
        recency_penalty: float = 5.609,
        lateral_penalty: float = 5.971,
        side_preference_penalty: float = 4.688,
        utilisation_weight: float = 3.138,
    ):
        if not 0.0 <= decay_rate < 1.0:
            raise ValueError("decay_rate must be within [0,1).")
        self.decay_rate = decay_rate
        self.recency_penalty = recency_penalty
        self.lateral_penalty = lateral_penalty
        self.side_preference_penalty = side_preference_penalty
        self.utilisation_weight = utilisation_weight
        self._step: int = 0
        self._state: Dict[str, _HTState] = {}

    def _ensure_state(self, hts: Iterable[str]) -> None:
        for ht in hts:
            if ht not in self._state:
                self._state[ht] = _HTState()

    def _advance_time(self) -> None:
        self._step += 1
        if self.decay_rate == 0.0:
            return
        for state in self._state.values():
            if state.load_estimate <= 0.0:
                continue
            state.load_estimate *= 1.0 - self.decay_rate
            if state.load_estimate < 1e-3:
                state.load_estimate = 0.0

    def _travel_cost(
        self,
        job_type: str,
        ht_coord: Coordinate,
        qc_coord: Coordinate,
        yard_coord: Coordinate,
    ) -> float:
        target = qc_coord if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE else yard_coord
        lateral_bias = 1.0 + (self.lateral_penalty if ht_coord.x > target.x else 0.0)
        manhattan = abs(ht_coord.x - target.x) + abs(ht_coord.y - target.y)
        cost = lateral_bias * manhattan

        if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:
            prefer_left = qc_coord.x >= 21
            non_preferred = (
                (prefer_left and ht_coord.x > qc_coord.x)
                or (not prefer_left and ht_coord.x < qc_coord.x)
            )
            if non_preferred:
                lateral_gap = abs(ht_coord.x - qc_coord.x) + 1
                cost += self.side_preference_penalty * lateral_gap

        return cost

    def _score(
        self,
        job_type: str,
        ht_name: str,
        get_coord: Callable[[str], Coordinate],
        qc_coord: Coordinate,
        yard_coord: Coordinate,
    ) -> float:
        state = self._state[ht_name]
        coord = get_coord(ht_name)
        travel_cost = self._travel_cost(job_type, coord, qc_coord, yard_coord)

        recency_penalty = 0.0
        if state.last_assigned_step >= 0:
            idle_steps = self._step - state.last_assigned_step
            recency_penalty = self.recency_penalty / (idle_steps + 1)

        utilisation_penalty = self.utilisation_weight * state.load_estimate
        return travel_cost + utilisation_penalty + recency_penalty

    def _commit(self, ht_name: str) -> None:
        state = self._state[ht_name]
        state.load_estimate += 1.0
        state.last_assigned_step = self._step

    def choose(
        self,
        job_type: str,
        available_hts: List[str],
        selected_hts: Iterable[str],
        get_coord: Callable[[str], Coordinate],
        qc_coord: Coordinate,
        yard_coord: Coordinate,
    ) -> Optional[str]:
        candidates = [ht for ht in available_hts if ht not in selected_hts]
        if not candidates:
            return None

        self._ensure_state(candidates)
        self._advance_time()

        best_ht = min(
            candidates,
            key=lambda ht: self._score(job_type, ht, get_coord, qc_coord, yard_coord),
        )
        self._commit(best_ht)
        return best_ht
