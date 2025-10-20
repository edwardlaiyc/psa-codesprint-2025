import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from logzero import logger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.simulation import Simulation


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default_bounds: Tuple[float, float]
    is_int: bool = False


PARAM_SPECS: Sequence[ParamSpec] = (
    ParamSpec("max_capacity", (600.0, 900.0), is_int=True),
    ParamSpec("decay_rate", (0.01, 0.10)),
    ParamSpec("idle_exploration_bonus", (0.15, 0.60)),
    ParamSpec("congestion_penalty_scale", (2.0, 8.0)),
)


def _run_simulation(
    params: Dict[str, float],
    max_steps: Optional[int],
    max_time: Optional[int],
) -> Dict[str, Any]:
    simulation = Simulation()
    simulation.planning_engine.job_planner.configure_yard_selector(**params)

    steps = 0
    while True:
        simulation.update()
        steps += 1
        current_time = simulation.get_current_time()
        if simulation.has_completed_all_jobs():
            return {
                "status": "ok",
                "time": current_time,
                "steps": steps,
            }
        if simulation.has_deadlock():
            return {
                "status": "deadlock",
                "time": current_time,
                "steps": steps,
            }

        if max_steps is not None and steps >= max_steps:
            return {
                "status": "timeout",
                "time": current_time,
                "steps": steps,
            }

        if max_time is not None and current_time >= max_time:
            return {
                "status": "timeout",
                "time": current_time,
                "steps": steps,
            }


def _objective(score: Dict[str, Any], penalty: float) -> float:
    if score["status"] == "ok":
        return score["time"]
    return score["time"] + penalty


def _format_params(params: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3f}" if not isinstance(v, int) else f"{k}={v:d}" for k, v in params.items())


def _write_results(output: Path, trials: Iterable[Dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(list(trials), fh, indent=2)


def _build_bounds(
    fixed_values: Dict[str, Optional[float]],
    ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for spec in PARAM_SPECS:
        fixed = fixed_values.get(spec.name)
        if fixed is not None:
            bounds[spec.name] = (float(fixed), float(fixed))
            continue

        low, high = ranges[spec.name]
        low = float(low)
        high = float(high)
        if spec.name == "decay_rate":
            high = min(high, 1.0 - 1e-6)

        if high <= low:
            raise ValueError(f"Invalid bounds for {spec.name}: [{low}, {high}]")
        bounds[spec.name] = (low, high)
    return bounds


def _sample_params(rng: random.Random, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for spec in PARAM_SPECS:
        low, high = bounds[spec.name]
        if math.isclose(low, high, abs_tol=1e-12):
            value = low
        else:
            value = rng.uniform(low, high)

        if spec.is_int:
            value = int(round(value))
            value = max(int(math.ceil(low)), min(int(math.floor(high)), value))
        params[spec.name] = value
    return params


def _params_to_unit_vector(
    params: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    vector = []
    for spec in PARAM_SPECS:
        low, high = bounds[spec.name]
        if math.isclose(low, high, abs_tol=1e-12):
            vector.append(0.0)
        else:
            span = high - low
            vector.append((params[spec.name] - low) / span)
    return np.array(vector, dtype=float)


def _unit_vector_to_params(
    vector: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for idx, spec in enumerate(PARAM_SPECS):
        low, high = bounds[spec.name]
        if math.isclose(low, high, abs_tol=1e-12):
            value = low
        else:
            value = low + vector[idx] * (high - low)

        if spec.is_int:
            value = int(round(value))
            value = max(int(math.ceil(low)), min(int(math.floor(high)), value))
        params[spec.name] = value
    return params


def _params_to_raw_vector(params: Dict[str, float]) -> np.ndarray:
    return np.array([float(params[spec.name]) for spec in PARAM_SPECS], dtype=float)


def _vector_signature(vector: Iterable[float], precision: int = 6) -> Tuple[float, ...]:
    return tuple(round(float(value), precision) for value in vector)


def _sample_candidate_matrix(
    bounds: Dict[str, Tuple[float, float]],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dims = len(PARAM_SPECS)
    samples = np.zeros((count, dims), dtype=float)
    for idx, spec in enumerate(PARAM_SPECS):
        low, high = bounds[spec.name]
        if math.isclose(low, high, abs_tol=1e-12):
            samples[:, idx] = 0.0
        else:
            samples[:, idx] = rng.random(count)
    return samples


def _fit_gp(X: np.ndarray, y: np.ndarray, seed: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
        length_scale=np.ones(X.shape[1]),
        nu=2.5,
        length_scale_bounds=(1e-3, 1e4),
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=3,
    )
    gp.fit(X, y)
    return gp


def _expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    xi: float,
) -> np.ndarray:
    improvement = best - mu - xi
    ei = np.zeros_like(mu)
    mask_indices = np.where(sigma > 1e-12)[0]
    for idx in mask_indices:
        z = improvement[idx] / sigma[idx]
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
        ei[idx] = improvement[idx] * cdf + sigma[idx] * pdf
    ei[improvement <= 0.0] = 0.0
    return ei


def _propose_next(
    gp: GaussianProcessRegressor,
    bounds: Dict[str, Tuple[float, float]],
    y_best: float,
    candidate_count: int,
    rng: np.random.Generator,
    seen_signatures: Iterable[Tuple[float, ...]],
    xi: float,
) -> Optional[np.ndarray]:
    candidates = _sample_candidate_matrix(bounds, candidate_count, rng)
    mu, sigma = gp.predict(candidates, return_std=True)
    sigma = sigma.reshape(-1)
    ei = _expected_improvement(mu, sigma, y_best, xi)

    seen_set = set(seen_signatures)
    for idx in np.argsort(-ei):
        vector = candidates[idx]
        params = _unit_vector_to_params(vector, bounds)
        signature = _vector_signature(_params_to_raw_vector(params))
        if signature not in seen_set:
            return vector
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian-optimisation tuner for AdaptiveYardSelector parameters."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15000,
        help="Optional cap on simulation steps per trial (set <=0 to disable).",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Optional cap on simulated seconds per trial. Takes effect in addition to --max-steps.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write trial results as JSON.",
    )
    parser.add_argument("--max-capacity", type=int, default=None)
    parser.add_argument("--max-capacity-range", type=int, nargs=2, default=None)
    parser.add_argument("--decay-rate", type=float, default=None)
    parser.add_argument("--decay-range", type=float, nargs=2, default=None)
    parser.add_argument("--idle-bonus", type=float, default=None)
    parser.add_argument("--idle-range", type=float, nargs=2, default=None)
    parser.add_argument("--congestion-scale", type=float, default=None)
    parser.add_argument("--congestion-range", type=float, nargs=2, default=None)
    parser.add_argument(
        "--init-random",
        type=int,
        default=4,
        help="Number of purely random trials before the Bayesian optimiser takes over.",
    )
    parser.add_argument(
        "--candidate-samples",
        type=int,
        default=512,
        help="Number of random samples used per acquisition maximisation step.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=0.05,
        help="Exploration parameter for the expected-improvement acquisition function.",
    )
    parser.add_argument(
        "--failure-penalty",
        type=float,
        default=5000.0,
        help="Penalty added to the simulated time when a trial fails (deadlock/timeout).",
    )

    args = parser.parse_args()

    logger.setLevel(logging.WARNING)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    default_ranges = {spec.name: spec.default_bounds for spec in PARAM_SPECS}
    ranges = {
        "max_capacity": tuple(args.max_capacity_range) if args.max_capacity_range else default_ranges["max_capacity"],
        "decay_rate": tuple(args.decay_range) if args.decay_range else default_ranges["decay_rate"],
        "idle_exploration_bonus": tuple(args.idle_range) if args.idle_range else default_ranges["idle_exploration_bonus"],
        "congestion_penalty_scale": tuple(args.congestion_range)
        if args.congestion_range
        else default_ranges["congestion_penalty_scale"],
    }

    fixed_values = {
        "max_capacity": args.max_capacity,
        "decay_rate": args.decay_rate,
        "idle_exploration_bonus": args.idle_bonus,
        "congestion_penalty_scale": args.congestion_scale,
    }

    bounds = _build_bounds(fixed_values, ranges)

    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None
    max_time = args.max_time

    trial_records = []
    best_record: Optional[Dict[str, Any]] = None
    X_samples: list[np.ndarray] = []
    y_samples: list[float] = []
    seen_signatures = set()

    for trial in range(1, args.trials + 1):
        if trial <= args.init_random or len(X_samples) < 2:
            params = _sample_params(rng, bounds)
        else:
            X = np.vstack(X_samples)
            y = np.array(y_samples)
            try:
                gp = _fit_gp(X, y, seed=args.seed + trial)
                proposal_unit = _propose_next(
                    gp=gp,
                    bounds=bounds,
                    y_best=float(np.min(y)),
                    candidate_count=args.candidate_samples,
                    rng=np_rng,
                    seen_signatures=seen_signatures,
                    xi=args.xi,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.error("GP fitting failed (trial %d): %s", trial, exc)
                proposal_unit = None

            if proposal_unit is None:
                params = _sample_params(rng, bounds)
            else:
                params = _unit_vector_to_params(proposal_unit, bounds)

        signature = _vector_signature(_params_to_raw_vector(params))
        if signature in seen_signatures:
            params = _sample_params(rng, bounds)
            signature = _vector_signature(_params_to_raw_vector(params))

        score = _run_simulation(params, max_steps=max_steps, max_time=max_time)
        objective = _objective(score, args.failure_penalty)
        record = {"trial": trial, "params": params, "objective": objective, **score}
        trial_records.append(record)
        seen_signatures.add(signature)
        X_samples.append(_params_to_unit_vector(params, bounds))
        y_samples.append(objective)

        status = score["status"]
        if best_record is None or objective < best_record["objective"]:
            best_record = record

        logger.warning(
            "[trial %02d/%02d] status=%s time=%.0f steps=%d objective=%.2f params=(%s)",
            trial,
            args.trials,
            status,
            score["time"],
            score["steps"],
            objective,
            _format_params(params),
        )

    if args.output:
        _write_results(args.output, trial_records)

    if best_record and best_record["status"] == "ok":
        logger.warning(
            "Best config: time=%.0f steps=%d objective=%.2f params=(%s)",
            best_record["time"],
            best_record["steps"],
            best_record["objective"],
            _format_params(best_record["params"]),
        )
    elif best_record:
        logger.warning(
            "No successful completion. Best attempt: status=%s time=%.0f objective=%.2f params=(%s)",
            best_record["status"],
            best_record["time"],
            best_record["objective"],
            _format_params(best_record["params"]),
        )

    if args.output:
        logger.warning("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
