import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from logzero import logger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.simulation import Simulation


PARAM_NAMES: Sequence[str] = [
    "decay_rate",
    "recency_penalty",
    "lateral_penalty",
    "side_preference_penalty",
    "utilisation_weight",
]

DEFAULT_RANGES: Dict[str, Tuple[float, float]] = {
    "decay_rate": (0.05, 0.18),
    "recency_penalty": (3.0, 8.0),
    "lateral_penalty": (4.0, 9.0),
    "side_preference_penalty": (1.5, 6.0),
    "utilisation_weight": (1.0, 3.5),
}


def _run_simulation(
    params: Dict[str, float],
    max_steps: Optional[int],
    max_time: Optional[int],
) -> Dict[str, Any]:
    simulation = Simulation()
    simulation.planning_engine.job_planner.configure_ht_selector(**params)

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
    return ", ".join(f"{k}={v:.3f}" for k, v in params.items())


def _write_results(output: Path, trials: Iterable[Dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(list(trials), fh, indent=2)


def _build_bounds(
    fixed_values: Dict[str, Optional[float]],
    ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for name in PARAM_NAMES:
        fixed = fixed_values.get(name)
        if fixed is not None:
            bounds[name] = (fixed, fixed)
            continue

        low, high = ranges[name]
        if name == "decay_rate":
            high = min(high, 1.0 - 1e-6)
        if high <= low:
            raise ValueError(f"Invalid bounds for {name}: [{low}, {high}]")
        bounds[name] = (low, high)
    return bounds


def _sample_params(rng: random.Random, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    params = {}
    for name in PARAM_NAMES:
        low, high = bounds[name]
        if math.isclose(low, high, abs_tol=1e-12):
            params[name] = low
        else:
            params[name] = rng.uniform(low, high)
    return params


def _params_to_vector(params: Dict[str, float]) -> np.ndarray:
    return np.array([params[name] for name in PARAM_NAMES], dtype=float)


def _vector_to_params(vector: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES, vector)}


def _vector_signature(vector: Iterable[float], precision: int = 6) -> Tuple[float, ...]:
    return tuple(round(float(value), precision) for value in vector)


def _sample_candidate_matrix(
    bounds: Sequence[Tuple[float, float]],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dims = len(bounds)
    samples = np.empty((count, dims), dtype=float)
    for dim, (low, high) in enumerate(bounds):
        if math.isclose(low, high, abs_tol=1e-12):
            samples[:, dim] = low
        else:
            samples[:, dim] = rng.uniform(low, high, size=count)
    return samples


def _fit_gp(X: np.ndarray, y: np.ndarray, seed: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X.shape[1]),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=3,
    )
    gp.fit(X, y)
    return gp


def _normal_pdf(value: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * value * value)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _propose_next(
    gp: GaussianProcessRegressor,
    bounds: Sequence[Tuple[float, float]],
    y_best: float,
    candidate_count: int,
    rng: np.random.Generator,
    seen: Iterable[Tuple[float, ...]],
    xi: float,
) -> Optional[np.ndarray]:
    candidates = _sample_candidate_matrix(bounds, candidate_count, rng)
    mu, sigma = gp.predict(candidates, return_std=True)
    sigma = sigma.reshape(-1)

    improvement = y_best - mu - xi
    ei = np.zeros_like(mu)

    mask = sigma > 1e-12
    if np.any(mask):
        Z = np.zeros_like(mu)
        Z[mask] = improvement[mask] / sigma[mask]
        for idx in np.where(mask)[0]:
            ei[idx] = improvement[idx] * _normal_cdf(Z[idx]) + sigma[idx] * _normal_pdf(Z[idx])
    ei[improvement <= 0.0] = 0.0

    seen_set = set(seen)
    for idx in np.argsort(-ei):
        vector = candidates[idx]
        if _vector_signature(vector) not in seen_set:
            return vector
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian-optimisation tuner for AdaptiveHTSelector weights."
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write trial results as JSON.",
    )
    parser.add_argument("--decay-rate", type=float, default=None)
    parser.add_argument("--decay-range", type=float, nargs=2, default=None)
    parser.add_argument("--recency-penalty", type=float, default=None)
    parser.add_argument("--recency-range", type=float, nargs=2, default=None)
    parser.add_argument("--lateral-penalty", type=float, default=None)
    parser.add_argument("--lateral-range", type=float, nargs=2, default=None)
    parser.add_argument("--side-penalty", type=float, default=None)
    parser.add_argument("--side-range", type=float, nargs=2, default=None)
    parser.add_argument("--utilisation-weight", type=float, default=None)
    parser.add_argument("--utilisation-range", type=float, nargs=2, default=None)
    parser.add_argument(
        "--init-random",
        type=int,
        default=5,
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

    ranges = {
        "decay_rate": args.decay_range or DEFAULT_RANGES["decay_rate"],
        "recency_penalty": args.recency_range or DEFAULT_RANGES["recency_penalty"],
        "lateral_penalty": args.lateral_range or DEFAULT_RANGES["lateral_penalty"],
        "side_preference_penalty": args.side_range
        or DEFAULT_RANGES["side_preference_penalty"],
        "utilisation_weight": args.utilisation_range
        or DEFAULT_RANGES["utilisation_weight"],
    }

    fixed_values = {
        "decay_rate": args.decay_rate,
        "recency_penalty": args.recency_penalty,
        "lateral_penalty": args.lateral_penalty,
        "side_preference_penalty": args.side_penalty,
        "utilisation_weight": args.utilisation_weight,
    }

    bounds = _build_bounds(fixed_values, ranges)
    bounds_array = [bounds[name] for name in PARAM_NAMES]

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
                proposal = _propose_next(
                    gp=gp,
                    bounds=bounds_array,
                    y_best=float(np.min(y)),
                    candidate_count=args.candidate_samples,
                    rng=np_rng,
                    seen=seen_signatures,
                    xi=args.xi,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.error("GP fitting failed (trial %d): %s", trial, exc)
                proposal = None

            if proposal is None:
                params = _sample_params(rng, bounds)
            else:
                params = _vector_to_params(proposal)

        signature = _vector_signature(_params_to_vector(params))
        if signature in seen_signatures:
            params = _sample_params(rng, bounds)
            signature = _vector_signature(_params_to_vector(params))

        score = _run_simulation(params, max_steps=max_steps, max_time=max_time)
        objective = _objective(score, args.failure_penalty)
        record = {"trial": trial, "params": params, "objective": objective, **score}
        trial_records.append(record)
        seen_signatures.add(signature)
        X_samples.append(_params_to_vector(params))
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
