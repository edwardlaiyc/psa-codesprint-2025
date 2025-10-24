import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from logzero import logger

from src.constant import CONSTANT
from src.floor import Coordinate, SectorMapSnapshot
from src.job import InstructionType, Job, JobInstruction, Status
from src.operators import HT_Coordinate_View
from src.plan.job_tracker import JobTracker
from src.plan.yard_selector_ai import AdaptiveYardSelector


@dataclass
class PlannerEvent:
    seq: int
    severity: str
    message: str
    details: Optional[Dict[str, Any]] = None


class JobPlanner:
    """
    Coordinates job planning activities using HT tracker and sector map data.

    Attributes
    ----------
    ht_coord_tracker : HT_Coordinate_View
        An instance responsible for tracking the coordinates of HTs.
    sector_map_snapshot : SectorMapSnapshot
        A snapshot of the sector map representing the current state of the environment for planning.
    """

    def __init__(
        self,
        ht_coord_tracker: HT_Coordinate_View,
        sector_map_snapshot: SectorMapSnapshot,
    ):
        self.ht_coord_tracker = ht_coord_tracker
        self.sector_map_snapshot = sector_map_snapshot
        self._yard_selector_params: Dict[str, Any] = dict(
            max_capacity=5,
            decay_rate=0.039,
            idle_exploration_bonus=0.363,
            congestion_penalty_scale=0.5,
            distance_weight=0.281,
        )
        self._yard_selector_ai = AdaptiveYardSelector(**self._yard_selector_params)
        self._ht_selector_params: Dict[str, Any] = dict(
            decay_rate=0.123,
            recency_penalty=5.0,
            lateral_penalty=5.640,
            side_preference_penalty=4.290,
            utilisation_weight=3.195,
            qc_distance_weight=1.16,
            yard_distance_weight=0.131,
        )
        self._ht_selector_ai = None
        self._event_log: List[PlannerEvent] = []
        self._event_seq: int = 0
        self._event_history_limit: int = 250
        self._di_yards: Dict[str, int] = {}
        self._yard_expected_totals: Optional[Dict[str, int]] = None
        self._yard_primary_map: Dict[str, Optional[str]] = {}
        self._yard_assignment_history: Dict[str, str] = {}
        self._yard_completed_counts: Dict[str, int] = {}
        self._yard_completion_tick: int = 0

    def configure_ht_selector(self, **params: Any) -> None:
        """
        this overrides the parameters used to instantiate the adaptive HT selector.

        calling this after planning starts will reset the cached selector so the
        next selection cycle uses the updated hyper-parameters.
        """
        self._ht_selector_params = params
        self._ht_selector_ai = None
        self._emit_event(
            "info",
            "HT selector parameters updated",
            params=self._ht_selector_params,
        )

    def configure_yard_selector(self, **params: Any) -> None:
        """
        this overrides the parameters used to instantiate the adaptive yard selector.

        this immediately replaces the cached selector so future selections
        reflect the updated hyper-parameters.
        """
        self._yard_selector_params = params
        self._yard_selector_ai = AdaptiveYardSelector(**self._yard_selector_params)
        self._emit_event(
            "info",
            "Yard selector parameters updated",
            params=self._yard_selector_params,
        )

    @staticmethod
    def _normalise_yard_name(value: Optional[Any]) -> Optional[str]:
        """Convert yard identifiers to clean strings, ignoring blanks/NaNs."""
        if value is None:
            return None
        if isinstance(value, float):
            if math.isnan(float(value)):
                return None
        trimmed = str(value).strip()
        if not trimmed:
            return None
        lowered = trimmed.lower()
        if lowered in {"nan", "none", "<na>", "null"}:
            return None
        return trimmed or None

    def _initialise_yard_totals(self, job_tracker: JobTracker) -> None:
        if self._yard_expected_totals is not None:
            return

        totals: Dict[str, int] = {yard: 0 for yard in CONSTANT.YARD_FLOOR.YARD_NAMES}
        primary_map: Dict[str, Optional[str]] = {}
        for job_seq, job in job_tracker.job_sequence_map.items():
            job_info = job.get_job_info()
            yard = self._normalise_yard_name(job_info.get("yard_name"))
            if yard:
                totals.setdefault(yard, 0)
                totals[yard] += 1
            primary_map[job_seq] = yard
            for alt in job_info.get("alt_yard_names") or []:
                alt_name = self._normalise_yard_name(alt)
                if alt_name:
                    totals.setdefault(alt_name, 0)

        self._yard_primary_map = primary_map
        self._yard_expected_totals = totals
        for yard in totals:
            self._di_yards.setdefault(yard, 0)

    def _compute_yard_completed_counts(self, job_tracker: JobTracker) -> Dict[str, int]:
        counts: Dict[str, int] = {yard: 0 for yard in CONSTANT.YARD_FLOOR.YARD_NAMES}
        for job in job_tracker.job_sequence_map.values():
            info = job.get_job_info()
            if info.get("job_status") != Status.COMPLETED:
                continue
            yard = self._normalise_yard_name(
                info.get("assigned_yard_name") or info.get("yard_name")
            )
            if not yard:
                continue
            counts[yard] = counts.get(yard, 0) + 1
        return counts

    def _ensure_progress_state(self) -> Dict[str, Any]:
        progress_state = getattr(self, "_planning_progress", None)
        yard_targets = self._yard_expected_totals
        if yard_targets is None:
            yard_targets = {yard: 0 for yard in CONSTANT.YARD_FLOOR.YARD_NAMES}
            self._yard_expected_totals = yard_targets

        yard_emit_stride = 1  # update yard progress on every change

        if not isinstance(progress_state, dict):
            progress_state = {
                "total_target": 20000,
                "total_completed": 0,
                "qc_target": 2500,
                "qc_counts": {qc: 0 for qc in CONSTANT.QUAY_CRANE_FLOOR.QC_NAMES},
                "yard_targets": yard_targets,
                "emit_stride": 100,
                "last_overall_bucket": -1,
                "qc_emit_stride": 50,
                "last_qc_buckets": {},
                "yard_emit_stride": yard_emit_stride,
                "last_yard_buckets": {},
            }
            setattr(self, "_planning_progress", progress_state)
        else:
            progress_state["yard_targets"] = yard_targets
            progress_state["yard_emit_stride"] = yard_emit_stride
        return progress_state

    def _emit_progress_update(
        self,
        *,
        increment_total: bool = False,
        qc_name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        progress_state = self._ensure_progress_state()

        if increment_total:
            progress_state["total_completed"] += 1

        if qc_name is not None:
            qc_counts: Dict[str, int] = progress_state["qc_counts"]
            qc_counts[qc_name] = qc_counts.get(qc_name, 0) + 1
        else:
            qc_counts = progress_state["qc_counts"]

        raw_targets: Dict[Any, int] = progress_state["yard_targets"]
        yard_targets: Dict[str, int] = {}
        for yard, target in raw_targets.items():
            normalised = self._normalise_yard_name(yard)
            if not normalised:
                continue
            yard_targets[normalised] = int(target)
        progress_state["yard_targets"] = yard_targets

        clean_di_counts: Dict[str, int] = {}
        for yard, count in self._di_yards.items():
            normalised = self._normalise_yard_name(yard)
            if not normalised:
                continue
            clean_di_counts[normalised] = int(count)
        self._di_yards = clean_di_counts

        clean_completed: Dict[str, int] = {}
        for yard, count in (self._yard_completed_counts or {}).items():
            normalised = self._normalise_yard_name(yard)
            if not normalised:
                continue
            clean_completed[normalised] = int(count)
        self._yard_completed_counts = clean_completed

        all_yards = set(yard_targets.keys()) | set(clean_di_counts.keys()) | set(clean_completed.keys())
        yard_completed: Dict[str, int] = {}
        yard_di_counts: Dict[str, int] = {}
        for yard in all_yards:
            yard_targets.setdefault(yard, 0)
            yard_completed[yard] = clean_completed.get(yard, 0)
            yard_di_counts[yard] = clean_di_counts.get(yard, 0)

        emit_update = force

        overall_bucket = progress_state["total_completed"] // max(1, progress_state["emit_stride"])
        if increment_total and overall_bucket != progress_state["last_overall_bucket"]:
            progress_state["last_overall_bucket"] = overall_bucket
            emit_update = True

        if qc_name is not None:
            qc_bucket = qc_counts[qc_name] // max(1, progress_state["qc_emit_stride"])
            last_qc_bucket = progress_state["last_qc_buckets"].get(qc_name)
            if last_qc_bucket != qc_bucket:
                progress_state["last_qc_buckets"][qc_name] = qc_bucket
                emit_update = True
            if last_qc_bucket is None:
                emit_update = True

        yard_emit_stride = max(1, progress_state["yard_emit_stride"])
        yard_total_changes = False
        for yard, count in yard_completed.items():
            bucket = count // yard_emit_stride
            last_bucket = progress_state["last_yard_buckets"].get(yard)
            if last_bucket != bucket:
                progress_state["last_yard_buckets"][yard] = bucket
                yard_total_changes = True
        if yard_total_changes:
            emit_update = True

        if not emit_update:
            return

        def _render_bar(current: int, total: int, width: int = 30) -> str:
            safe_total = max(total, 1)
            ratio = min(max(current / safe_total, 0.0), 1.0)
            filled = min(width, int(round(ratio * width)))
            bar = "=" * filled + "-" * (width - filled)
            current_fmt = f"{current:,}"
            total_fmt = f"{total:,}" if total > 0 else "0"
            return f"[{bar}] {ratio * 100:5.1f}% ({current_fmt}/{total_fmt})"

        overall_line = f"Overall {_render_bar(progress_state['total_completed'], progress_state['total_target'])}"
        qc_lines = [
            f"{name:<8} {_render_bar(count, progress_state['qc_target'])}"
            for name, count in sorted(qc_counts.items())
        ]
        yard_keys = sorted(all_yards)
        yard_lines = [
            f"{yard:<8} {_render_bar(yard_completed.get(yard, 0), yard_targets.get(yard, 0))}  ({yard_completed.get(yard, 0):,}/{yard_targets.get(yard, 0):,} jobs | {yard_di_counts.get(yard, 0):,} DI)"
            for yard in yard_keys
        ]
        progress_block = "\n  ".join(
            [overall_line, "QC Progress:"]
            + qc_lines
            + ["Yard Progress:"]
            + yard_lines
        )
        self._emit_event(
            "info",
            f"Planner progress update\n{progress_block}",
            progress_display=progress_block,
        )

        try:
            from src.plan.progress_monitor import get_progress_monitor  # type: ignore
        except Exception:  # pragma: no cover
            return
        try:
            monitor = get_progress_monitor()
            monitor.update_progress(
                overall_completed=progress_state["total_completed"],
                overall_total=progress_state["total_target"],
                qc_counts=qc_counts,
                qc_target=progress_state["qc_target"],
                yard_completed=yard_completed,
                yard_di=yard_di_counts,
                yard_targets=yard_targets,
            )
        except Exception:
            logger.debug("Progress monitor update failed", exc_info=True)

    def _rebalance_yard_totals(
        self,
        job_seq: str,
        primary_yard: Optional[str],
        selected_yard: Optional[str],
    ) -> None:
        primary_yard = self._normalise_yard_name(primary_yard)
        selected_yard = self._normalise_yard_name(selected_yard)
        if self._yard_expected_totals is None or not selected_yard:
            if selected_yard:
                self._yard_assignment_history[job_seq] = selected_yard
            return

        totals = self._yard_expected_totals
        totals.setdefault(selected_yard, 0)
        if primary_yard:
            totals.setdefault(primary_yard, 0)

        previous_assignment = self._yard_assignment_history.get(job_seq)
        if previous_assignment == selected_yard:
            return

        should_increment_selected = False
        if previous_assignment is None:
            if primary_yard and primary_yard != selected_yard:
                totals[primary_yard] = max(0, totals.get(primary_yard, 0) - 1)
                should_increment_selected = True
            elif not primary_yard:
                should_increment_selected = True
        else:
            totals.setdefault(previous_assignment, 0)
            totals[previous_assignment] = max(0, totals.get(previous_assignment, 0) - 1)
            should_increment_selected = True

        if should_increment_selected:
            totals[selected_yard] = totals.get(selected_yard, 0) + 1

        self._yard_assignment_history[job_seq] = selected_yard

    def _update_yard_completion_snapshot(self, job_tracker: JobTracker) -> None:
        new_counts = self._compute_yard_completed_counts(job_tracker)
        if new_counts != self._yard_completed_counts:
            self._yard_completed_counts = new_counts
            self._emit_progress_update(force=True)

    def is_deadlock(self):
        return self.ht_coord_tracker.is_deadlock()

    def get_non_moving_HT(self):
        return self.ht_coord_tracker.get_non_moving_HT()

    def get_recent_events(self, limit: int = 50) -> List[PlannerEvent]:
        """Expose recent planner events so the UI can show status/alerts."""
        if limit <= 0:
            return []
        return self._event_log[-limit:].copy()

    def _emit_event(self, severity: str, message: str, **details: Any) -> None:
        """Record planner activity and surface through logger for transparency."""
        self._event_seq += 1
        event = PlannerEvent(
            seq=self._event_seq,
            severity=severity,
            message=message,
            details=details or None,
        )
        self._event_log.append(event)
        if len(self._event_log) > self._event_history_limit:
            self._event_log.pop(0)

        if severity in ("warning", "error", "critical"):
            log_fn = getattr(logger, severity, None)
            if callable(log_fn):
                log_fn("[planner] %s | %s", message, details)
            else:
                logger.info("[planner] %s | %s", message, details)

    """ YOUR TASK HERE
    Objective: modify the following functions (including input arguments as you see fit) to achieve better planning efficiency.
        select_HT():
            select HT for the job based on your self-defined logic.
        select_yard():
            select yard for the job based on your self-defined logic.
        get_path_from_buffer_to_QC():
        get_path_from_buffer_to_yard():
        get_path_from_yard_to_buffer():
        get_path_from_QC_to_buffer():
            generate an efficient path for HT to navigate between listed locations (QC, yard, buffer).        
    """

    def plan(self, job_tracker: JobTracker) -> List[Job]:
        self._emit_event("info", "Planning cycle started")
        plannable_job_seqs = job_tracker.get_plannable_job_sequences()
        self._initialise_yard_totals(job_tracker)
        self._yard_completion_tick = (self._yard_completion_tick + 1) % 10
        if self._yard_completion_tick == 0:
            self._update_yard_completion_snapshot(job_tracker)
        selected_HT_names: List[str] = []
        new_jobs: List[Job] = []

        ht_assigned_count = 0
        yard_selected_count = 0

        try:
            for job_seq in plannable_job_seqs:
                job = job_tracker.get_job(job_seq)
                job_info = job.get_job_info()
                job_type = job_info["job_type"]
                QC_name = job_info["QC_name"]
                primary_yard = self._normalise_yard_name(job_info.get("yard_name"))
                raw_alt_yards = job_info.get("alt_yard_names") or []
                alt_yard_names = [
                    yard
                    for yard in (
                        self._normalise_yard_name(alt) for alt in raw_alt_yards
                    )
                    if yard
                ]
                yard_name = primary_yard

                self._emit_event(
                    "debug",
                    "Evaluating job for dispatch",
                    job_seq=job_seq,
                    job_type=job_type,
                    qc=QC_name,
                    yard=yard_name,
                    alternatives=alt_yard_names,
                )

                HT_name = self.select_HT(
                    job_type,
                    selected_HT_names,
                    QC_name,
                    yard_name,
                    alt_yard_names,
                )

                if HT_name is None:
                    self._emit_event(
                        "warning",
                        "No available HT for job",
                        job_seq=job_seq,
                        pending=len(plannable_job_seqs) - len(new_jobs),
                    )
                    break

                selected_HT_names.append(HT_name)
                buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)
                self._emit_event(
                    "debug",
                    "HT allocated",
                    job_seq=job_seq,
                    ht=HT_name,
                    buffer_coord=(buffer_coord.x, buffer_coord.y),
                )
                ht_assigned_count += 1

                if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:
                    yard_name = self.select_yard(job_seq, yard_name, alt_yard_names, buffer_coord)
                    self._emit_event(
                        "debug",
                        "Yard selected",
                        job_seq=job_seq,
                        yard=yard_name,
                        alternatives=alt_yard_names,
                    )
                    yard_selected_count += 1
                elif yard_name:
                    yard_selected_count += 1

                job.assign_job(HT_name=HT_name, yard_name=yard_name)
                job_instructions: List[JobInstruction] = []

                if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:
                    job_instructions.append(JobInstruction(instruction_type=InstructionType.BOOK_QC))

                    buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)
                    path = self.get_path_from_buffer_to_QC(buffer_coord, QC_name)
                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path)
                    )

                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.WORK_QC, HT_name=HT_name, QC_name=QC_name)
                    )

                    path = self.get_path_from_QC_to_buffer(QC_name, buffer_coord)
                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path)
                    )

                    job_instructions.append(JobInstruction(instruction_type=InstructionType.BOOK_YARD))

                    path = self.get_path_from_buffer_to_yard(buffer_coord, yard_name)
                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path)
                    )

                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.WORK_YARD, HT_name=HT_name, yard_name=yard_name)
                    )

                    path = self.get_path_from_yard_to_buffer(yard_name, buffer_coord)
                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path)
                    )

                else:
                    job_instructions.append(JobInstruction(instruction_type=InstructionType.BOOK_YARD))

                    buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)
                    path = self.get_path_from_buffer_to_yard(buffer_coord, yard_name)
                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path)
                    )

                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.WORK_YARD, HT_name=HT_name, yard_name=yard_name)
                    )

                    path = self.get_path_from_yard_to_buffer(yard_name, buffer_coord)
                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path)
                    )

                    job_instructions.append(JobInstruction(instruction_type=InstructionType.BOOK_QC))

                    path = self.get_path_from_buffer_to_QC(buffer_coord, QC_name)
                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path)
                    )

                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.WORK_QC, HT_name=HT_name, QC_name=QC_name)
                    )

                    path = self.get_path_from_QC_to_buffer(QC_name, buffer_coord)
                    job_instructions.append(
                        JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path)
                    )

                job.set_instructions(job_instructions)
                new_jobs.append(job)
                self._emit_event(
                    "debug",
                    "Job planned successfully",
                    job_seq=job_seq,
                    ht=HT_name,
                    yard=yard_name,
                    instructions=len(job_instructions),
                )

        except Exception as exc:  # pragma: no cover - defensive logging
            self._emit_event("error", "Planning aborted due to exception", error=str(exc))
            logger.exception("Planning encountered an exception.")
            raise
        else:
            self._emit_event(
                "info",
                "Planning summary",
                planned=len(new_jobs),
                ht_assigned=ht_assigned_count,
                yards_selected=yard_selected_count,
                pending=len(plannable_job_seqs) - len(new_jobs),
            )

        return new_jobs

    # HT ASSIGNMENT LOGIC    
    def select_HT(
        self,
        job_type: str,
        selected_HT_names: List[str],
        QC_name: str,
        yard_name: Optional[str],
        alt_yard_names: List[str],
    ) -> str:
        """
        For Discharge (DI) jobs: it prefers HTs already on the correct QC side and balances workload.
        For Load (LO) jobs: it focuses only on yard congestion and HT efficiency.

        Helps avoid long detours and spreads out jobs.
        """
        available_HTs = [
            ht
            for ht in self.ht_coord_tracker.get_available_HTs()
            if ht not in selected_HT_names
        ]
        if not available_HTs:
            return None

        from src.plan.ht_selector_ai import AdaptiveHTSelector

        if self._ht_selector_ai is None:
            self._ht_selector_ai = AdaptiveHTSelector(**self._ht_selector_params)

        qc_sector = self.sector_map_snapshot.get_QC_sector(QC_name)

        yard_candidates: List[Coordinate] = []
        if yard_name:
            yard_sector = self.sector_map_snapshot.get_yard_sector(yard_name)
            if yard_sector is not None:
                yard_candidates.append(yard_sector.in_coord)
        for alt_name in alt_yard_names:
            if not alt_name:
                continue
            yard_sector = self.sector_map_snapshot.get_yard_sector(alt_name)
            if yard_sector is None:
                continue
            yard_candidates.append(yard_sector.in_coord)

        if not yard_candidates:
            yard_candidates.append(qc_sector.in_coord)

        chosen_ht = self._ht_selector_ai.choose(
            job_type=job_type,
            available_hts=available_HTs,
            selected_hts=selected_HT_names,
            get_coord=self.ht_coord_tracker.get_coordinate,
            qc_coord=qc_sector.in_coord,
            yard_coords=yard_candidates,
        )
        if not chosen_ht:
            return None

        self._emit_progress_update(increment_total=True, qc_name=QC_name)

        return chosen_ht
    
    def select_yard(
        self,
        job_seq: str,
        primary_yard: Optional[str],
        alt_yard_names: List[str],
        source_coord: Coordinate,
        capacity: int = 700,
    ) -> Optional[str]:
        primary_yard = self._normalise_yard_name(primary_yard)
        clean_alts = [
            yard for yard in (self._normalise_yard_name(alt) for alt in (alt_yard_names or [])) if yard
        ]
        alt_yard_names = clean_alts

        candidates = [primary_yard, *alt_yard_names]
        candidates = [yard for yard in candidates if yard]
        if not candidates:
            return None

        yard_targets = self._yard_expected_totals
        if yard_targets is None:
            yard_targets = {}
            self._yard_expected_totals = yard_targets
        for candidate in candidates:
            self._di_yards.setdefault(candidate, 0)
            yard_targets.setdefault(candidate, 0)

        available = [candidate for candidate in candidates if self._di_yards[candidate] < capacity]
        if not available:
            available = candidates

        def _distance_lookup(candidate: str) -> float:
            yard_sector = self.sector_map_snapshot.get_yard_sector(candidate)
            if yard_sector is None:
                return float("inf")
            yard_coord = yard_sector.in_coord
            return abs(yard_coord.x - source_coord.x) + abs(yard_coord.y - source_coord.y)

        primary = primary_yard if primary_yard in available else available[0]
        alts = [candidate for candidate in available if candidate != primary]
        selected_yard = self._yard_selector_ai.choose(primary, alts, distance_lookup=_distance_lookup)

        previous_assignment = self._yard_assignment_history.get(job_seq)
        if previous_assignment and previous_assignment != selected_yard:
            self._di_yards[previous_assignment] = max(0, self._di_yards.get(previous_assignment, 0) - 1)

        if previous_assignment != selected_yard:
            self._di_yards[selected_yard] = self._di_yards.get(selected_yard, 0) + 1

        self._rebalance_yard_totals(job_seq, primary_yard, selected_yard)
        self._emit_progress_update(force=True)

        return selected_yard



    # NAVIGATION LOGIC
    def get_path_from_buffer_to_QC(
        self, buffer_coord: Coordinate, QC_name: str
    ) -> List[Coordinate]:
        """
        Generates a path from a buffer location to a Quay Crane (QC) input coordinate.

        OLD LOGIC:
        The path follows a predefined route:
        1. Moves south to the highway left lane (y = 7).
        2. Travels west along the highway to the left boundary (x = 1).
        3. Moves north to the upper lane (y = 4).
        4. Travels east to the IN coordinate of the specified QC.

        NEW LOGIC:
        If HT is on the right of the QC,
        1. Moves south to the highway left lane (y = 7).
        2. Travels west along the highway to the left boundary (x = 1).
        3. Moves north to the upper lane (y = 4).
        4. Travels east to the IN coordinate of the specified QC.

        If HT is on the left of the QC,
        1. Moves straight up from buffer to lane 5
        2. Travels east to the below the specified QC_IN.
        3. Travel north to QC_IN.

        Args:
            buffer_coord (Coordinate): The starting coordinate in the buffer zone.
            QC_name (str): The name of the Quay Crane to which the path should lead.

        Returns:
            List[Coordinate]: A list of coordinates representing the path from the buffer to the QC.
        """
        QC_in_coord = self.sector_map_snapshot.get_QC_sector(QC_name).in_coord
        
        # HT on the right of QC
        if buffer_coord.x > QC_in_coord.x:
            # go South to take Highway Left lane (y=7)
            highway_lane_y = 7
            path = [Coordinate(buffer_coord.x, highway_lane_y)]

            # then go to the left boundary
            path.extend(
                [Coordinate(x, highway_lane_y) for x in range(buffer_coord.x - 1, 0, -1)]
            )

            # then go to upper boundary and navigate to QC_in
            up_path_x = 1
            path.extend([Coordinate(up_path_x, y) for y in range(6, 3, -1)])
            qc_travel_lane_y = 4
            path.extend(
                [Coordinate(x, qc_travel_lane_y) for x in range(2, QC_in_coord.x + 1, 1)]
            )
        
        # HT on the left of QC
        else:
            # go straight up to upper boundary
            up_path_x = buffer_coord.x
            path = []
            path.extend([Coordinate(up_path_x, 5)])
            
            # navigate to QC_in
            qc_travel_lane_y = 5
            path.extend(
                [Coordinate(x, qc_travel_lane_y) for x in range(up_path_x + 1, QC_in_coord.x + 1, 1)]
            )
            path.extend([Coordinate(QC_in_coord.x, 4) for y in range(4, 5, 1)])

        path.append(QC_in_coord)

        return path

    def get_path_from_buffer_to_yard(
        self, buffer_coord: Coordinate, yard_name: str
    ) -> List[Coordinate]:
        """
        Generates a path from a buffer location to a yard IN area's coordinate.

        OLD LOGIC:
        The path follows a specific route:
        1. Moves north to the QC travel lane (y = 5).
        2. Travels east to the right boundary of the sector (x = 42).
        3. Moves south to the Highway Left lane (y = 11).
        4. Travels west along the highway to the left boundary (x = 1).
        5. Moves south to the lower boundary (y = 12).
        6. Travels east to the IN coordinate of the specified yard.

        NEW LOGIC:
        *only move south on even-numbered lanes

        If HT to the right of yard:
        1. Go south to highway 7
        2. Move left to down_path (right above yard_in or one to the left)
        3. Move south to highway 12
        4. navigate to yard_in

        If HT to the left of yard:
        1. Move down using closest even numbered path on the left of the HT
        2. Move to the down path
        3. Travel right to the IN coordinate of the specified yard

        Args:
            buffer_coord (Coordinate): The starting coordinate in the buffer zone.
            yard_name (str): The name of the yard to which the path should lead.

        Returns:
            List[Coordinate]: A list of coordinates representing the path from the buffer to the yard.
        """
        yard_in_coord = self.sector_map_snapshot.get_yard_sector(yard_name).in_coord
     
        # HT on the right of yard
        if buffer_coord.x  >= yard_in_coord.x:
            down_path_x = yard_in_coord.x #HT will go down at yard.x
            if down_path_x % 2 != 0: # if odd change to even
                down_path_x -= 1
            # Go South to take highway lane (y=7), then go to down_path
            path = [Coordinate(buffer_coord.x, buffer_coord.y + 1)]
            highway_lane_y = 7
            path.extend(
                [Coordinate(x, highway_lane_y) for x in range(buffer_coord.x - 1, down_path_x -1, -1)]
            )

            # go down to Highway lane(12)
            path.extend([Coordinate(down_path_x, y) for y in range(8, 13, 1)])
            highway_lane_y = 12
            # navigate to yard_in
            path.extend([Coordinate(x, highway_lane_y) for x in range(down_path_x + 1, yard_in_coord.x + 1, 1)])

        # HT on the left of yard
        else:
            down_path_x = buffer_coord.x #HT will go down at current x-coord
            if down_path_x % 2 != 0: #if odd change to even
                down_path_x -= 1
            # shift down to highway 7, adjust to even number lane
            path = [Coordinate(x, 7) for x in range(buffer_coord.x, down_path_x -1, -1)]
            # go down to lane 12 using even lane
            path.extend([Coordinate(down_path_x, y) for y in range(8, 13, 1)])
            # navigate right to yard_in
            highway_lane_y = 12
            path.extend([Coordinate(x, highway_lane_y) for x in range(down_path_x + 1, yard_in_coord.x + 1, 1)])

        path.append(yard_in_coord)
        return path

    def get_path_from_yard_to_buffer(
        self, yard_name: str, buffer_coord: Coordinate
    ) -> List[Coordinate]:
        """
        Generates a path from a yard OUT area's coordinate to a buffer location.

        OLD LOGIC:
        The path follows this route:
        1. Starts at the yard OUT coordinate.
        2. Moves east along the highway lane (y = 12) towards the second-to-right boundary.
        3. Moves north to the Highway Left lane (y = 7).
        4. Travels west along the highway left lane to the target buffer coordinate.

        NEW LOGIC:
        * only move up on odd numbered lanes.

        If the HT location in the buffer is on the right or same x coordinate of the yard:
        1. Starts at the yard OUT coordinate.
        2. select the odd number vertical path on the right of the buffer
        3. move right along Lane 12 to the x coordinate of the buffer
        4. move up the selected odd number path
        5. move left along Lane 7 to the buffer if needed

        If HT location of on the left of the yard
        1. Starts at the yard OUT coordinate.
        2. select the odd number vertical path on the right of the buffer
        3. move to the right along Lane 12 to the vertical path
        4. move up the vertical path
        5. move left along lane 7 to the buffer

        Args:
            yard_name (str): The name of the yard from which the path starts.
            buffer_coord (Coordinate): The destination coordinate in the buffer zone.

        Returns:
            List[Coordinate]: A list of coordinates representing the path from the yard to the buffer.
        """
        yard_out_coord = self.sector_map_snapshot.get_yard_sector(yard_name).out_coord

        # go to Yard[OUT] first
        path = [yard_out_coord]

        # enter highway lane, go to tile second-to-right boundary
        highway_lane_y = 12
        # if HT location in the buffer is on the right or same x coordinate of the yard
        if buffer_coord.x >= yard_out_coord.x:
            # select the odd number vertical path on the right of the buffer
            up_path_x = buffer_coord.x
            if up_path_x % 2 == 0:  # if even, change to odd
                up_path_x += 1
            # move right along Lane 12 to the x coordinate of the buffer
            path.extend(
                [Coordinate(x, highway_lane_y) for x in range(yard_out_coord.x, up_path_x + 1, 1)]
            )
            # move up the selected odd number path 
            path.extend([Coordinate(up_path_x, y) for y in range(11, 6, -1)])
            
            # move left along Lane 7 to the buffer if needed
            if up_path_x > buffer_coord.x:
                path.extend([Coordinate(x, 7) for x in range(up_path_x - 1, buffer_coord.x - 1, -1)])
        
        # buffer on the left of the yard
        else:
            # select the odd number vertical path on the right of the buffer
            up_path_x = yard_out_coord.x
            if up_path_x % 2 == 0: # if even, change to odd
                up_path_x += 1

            # move to the right along Lane 12 to the vertical path
            path.extend(Coordinate(x, highway_lane_y) for x in range(yard_out_coord.x, up_path_x + 1, 1))
            # move up the vertical path
            path.extend([Coordinate(up_path_x, y) for y in range(11, 6, -1)])
            # move left along lane 7 to the buffer
            path.extend([Coordinate(x, 7) for x in range(up_path_x - 1, buffer_coord.x - 1, -1)])
        
        path.append(buffer_coord)
        return path

    def get_path_from_QC_to_buffer(
        self, QC_name: str, buffer_coord: Coordinate
    ) -> List[Coordinate]:
        """
        Generates a path from a Quay Crane (QC) OUT coordinate to a buffer location.

        OLD LOGIC:
        The path follows this route:
        1. Starts at the QC OUT coordinate.
        2. Moves south to the QC travel lane (y = 4).
        3. Travels east along the QC travel lane to the right boundary.
        4. Moves south to the Highway Left lane (y = 7).
        5. Travels west along the highway left lane to the buffer coordinate.

        NEW LOGIC:
        If buffer to the right of QC:
        1. Move south to lane 5
        2. Move east to above buffer coord

        If buffer to the left of QC:
        1. Move south to lane 4
        2. Move east to right boundary
        3. Move south to lane 7
        4. Move west to below buffer coord

        Args:
            QC_name (str): The name of the Quay Crane from which the path starts.
            buffer_coord (Coordinate): The destination coordinate in the buffer zone.

        Returns:
            List[Coordinate]: A list of coordinates representing the path from the QC to the buffer.
        """
        QC_out_coord = self.sector_map_snapshot.get_QC_sector(QC_name).out_coord

        # go to QC_out first
        path = [QC_out_coord]

        # buffer to the right of QC_out
        if buffer_coord.x >= QC_out_coord.x:
            # go down to lane 5
            down_path = QC_out_coord.x
            path.extend([Coordinate(down_path, y) for y in range(4, 6, 1)])
            # move east to above buffer coord
            path.extend([Coordinate(x, 5) for x in range(down_path + 1, buffer_coord.x + 1, 1)])
        else:            
            # go South to take QC Travel Lane
            qc_travel_lane_y = 4
            path.append(Coordinate(QC_out_coord.x, qc_travel_lane_y))
            # move all the way to right boundary
            path.extend(
                [Coordinate(x, qc_travel_lane_y) for x in range(QC_out_coord.x + 1, 43, 1)]
            )

            # go down to Highway Left lane(7), then takes left most
            down_path_x = 42
            path.extend([Coordinate(down_path_x, y) for y in range(5, 8, 1)])

            # navigate back to buffer
            highway_lane_y = 7
            path.extend(
                [Coordinate(x, highway_lane_y) for x in range(41, buffer_coord.x - 1, -1)]
            )
        path.append(buffer_coord)

        return path
