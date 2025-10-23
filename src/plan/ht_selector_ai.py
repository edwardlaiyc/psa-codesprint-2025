from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

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
        qc_distance_weight: float = 2.0,
        yard_distance_weight: float = 3.0,
    ):
        if not 0.0 <= decay_rate < 1.0:
            raise ValueError("decay_rate must be within [0,1).")
        if qc_distance_weight < 0.0 or yard_distance_weight < 0.0:
            raise ValueError("Distance weights must be non-negative.")
        self.decay_rate = decay_rate
        self.recency_penalty = recency_penalty
        self.lateral_penalty = lateral_penalty
        self.side_preference_penalty = side_preference_penalty
        self.utilisation_weight = utilisation_weight
        self.qc_distance_weight = qc_distance_weight
        self.yard_distance_weight = yard_distance_weight
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
        yard_coords: Sequence[Coordinate],
    ) -> float:
        # determine the closest yard among provided candidates
        closest_yard_coord: Optional[Coordinate] = None
        min_yard_distance = float("inf")
        for yard_coord in yard_coords:
            distance = abs(ht_coord.x - yard_coord.x) + abs(ht_coord.y - yard_coord.y)
            if distance < min_yard_distance:
                min_yard_distance = distance
                closest_yard_coord = yard_coord
        if min_yard_distance == float("inf"):
            min_yard_distance = 0.0
            closest_yard_coord = qc_coord

        qc_distance = abs(ht_coord.x - qc_coord.x) + abs(ht_coord.y - qc_coord.y)

        if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:
            primary_target = qc_coord
            primary_weight = self.qc_distance_weight
            secondary_distance = min_yard_distance
            secondary_weight = self.yard_distance_weight
        else:
            primary_target = closest_yard_coord
            primary_weight = self.yard_distance_weight
            secondary_distance = qc_distance
            secondary_weight = self.qc_distance_weight

        lateral_bias = 1.0 + (self.lateral_penalty if ht_coord.x > primary_target.x else 0.0)
        primary_distance = abs(ht_coord.x - primary_target.x) + abs(ht_coord.y - primary_target.y)
        cost = lateral_bias * (primary_weight * primary_distance) + secondary_weight * secondary_distance

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
        yard_coords: Sequence[Coordinate],
    ) -> float:
        state = self._state[ht_name]
        coord = get_coord(ht_name)
        travel_cost = self._travel_cost(job_type, coord, qc_coord, yard_coords)

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
        yard_coords: Sequence[Coordinate],
    ) -> Optional[str]:
        candidates = [ht for ht in available_hts if ht not in selected_hts]
        if not candidates:
            return None

        self._ensure_state(candidates)
        self._advance_time()

        best_ht = min(
            candidates,
            key=lambda ht: self._score(job_type, ht, get_coord, qc_coord, yard_coords),
        )
        self._commit(best_ht)
        return best_ht
