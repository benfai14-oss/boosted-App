"""
Scenario generation utilities for climate and market simulations.

This module provides functions to create synthetic price paths and
scenario probabilities without relying on machine learning techniques.
The goal is to offer a transparent and explainable mechanism for
stress testing and sensitivity analysis in the context of climate risk.

Functions implemented here include random walk generators, geometric
Brownian motion simulators, scenario classification based on risk scores,
and utility functions to adjust probabilities across multiple scenarios.

All functions operate on pandas and numpy arrays and are designed to
be deterministic given the same random seed, ensuring reproducibility.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Iterable


def generate_random_walk(
    start: float,
    steps: int,
    scale: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a simple random walk around a starting value.

    Parameters
    ----------
    start : float
        The initial value of the series.
    steps : int
        Number of steps to simulate.
    scale : float, default 1.0
        The standard deviation of the increments.
    seed : int, default 42
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        Array of simulated values of length `steps`.

    Notes
    -----
    The random walk increments are sampled from a normal distribution
    with mean zero and the specified scale. This function can be used
    to simulate noise in agricultural yields or small perturbations in
    risk factors.
    """
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, scale, size=steps)
    return start + np.cumsum(increments)


def simulate_geometric_brownian_motion(
    start: float,
    mu: float,
    sigma: float,
    steps: int,
    dt: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a geometric Brownian motion (GBM) price path.

    Parameters
    ----------
    start : float
        Initial price.
    mu : float
        Expected return (drift) per unit time.
    sigma : float
        Volatility per unit time.
    steps : int
        Number of steps to simulate.
    dt : float, default 1.0
        Time increment between steps.
    seed : int, default 42
        Seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array of simulated price values.

    Notes
    -----
    The GBM equation is S_{t+1} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z_t).
    This implementation uses the same time increment for each step.
    """
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, np.sqrt(dt), size=steps)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * increments
    log_returns = drift + diffusion
    prices = np.empty(steps)
    prices[0] = start
    for i in range(1, steps):
        prices[i] = prices[i - 1] * np.exp(log_returns[i])
    return prices


def classify_risk_level(risk_score: float) -> str:
    """Categorise a risk score into discrete levels.

    Parameters
    ----------
    risk_score : float
        The global risk index value between 0 and 100.

    Returns
    -------
    str
        One of {"low", "moderate", "high", "extreme"}.

    Notes
    -----
    The thresholds can be adjusted to reflect different risk tolerances.
    """
    if risk_score < 30:
        return "low"
    elif risk_score < 60:
        return "moderate"
    elif risk_score < 85:
        return "high"
    else:
        return "extreme"


def assign_scenario_probabilities(
    current_risk: float,
    base_probs: Dict[str, float] = None,
) -> Dict[str, float]:
    """Allocate probabilities to scenarios based on the current risk score.

    Parameters
    ----------
    current_risk : float
        The global climate risk index (0–100).
    base_probs : dict, optional
        Baseline scenario probabilities. Keys should include "baseline",
        "bullish", "bearish", "extreme". Defaults to equal distribution
        if not provided.

    Returns
    -------
    dict
        Dictionary with updated probabilities summing to 1.

    Notes
    -----
    A higher risk score increases the probabilities of "bullish" (price up)
    or "extreme" scenarios and decreases the probability of "baseline".
    A simple logistic weighting is used to adjust the probabilities.
    """
    if base_probs is None:
        base_probs = {"baseline": 0.4, "bullish": 0.3, "bearish": 0.2, "extreme": 0.1}
    total = sum(base_probs.values())
    probs = {k: v / total for k, v in base_probs.items()}
    # Compute a weight factor from the risk score
    weight = 0.5 + 1.5 * (current_risk / 100.0)
    probs["bullish"] *= weight
    probs["extreme"] *= weight ** 2
    inverse_weight = 2.5 - weight
    probs["bearish"] *= inverse_weight
    # Normalise
    norm = sum(probs.values())
    updated = {k: v / norm for k, v in probs.items()}
    return updated


def generate_scenarios(
    start_price: float,
    risk_scores: Iterable[float],
    mu: float,
    sigma: float,
    horizon: int,
    n_paths: int = 10,
    seed: int = 42,
) -> List[Tuple[float, List[float]]]:
    """Simulate multiple price paths under varying risk conditions.

    Parameters
    ----------
    start_price : float
        The starting price for all simulations.
    risk_scores : iterable of float
        Sequence of risk scores (0–100) to use for each simulation.
    mu : float
        Expected return parameter for the GBM.
    sigma : float
        Volatility parameter for the GBM.
    horizon : int
        Number of steps to simulate.
    n_paths : int, default 10
        Number of independent paths to generate for each risk score.
    seed : int, default 42
        Base seed for reproducibility.

    Returns
    -------
    list of (float, list of float)
        A list where each tuple contains the risk score used and the
        simulated price path (a list of floats).

    Notes
    -----
    Each path uses a unique seed derived from the base seed to ensure
    independence. Volatility increases linearly with risk.
    """
    results = []
    rng = np.random.default_rng(seed)
    for risk in risk_scores:
        adjusted_sigma = sigma * (1.0 + risk / 100.0)
        for _ in range(n_paths):
            path_seed = int(rng.integers(0, 1e9))
            path = simulate_geometric_brownian_motion(
                start=start_price,
                mu=mu,
                sigma=adjusted_sigma,
                steps=horizon,
                dt=1.0,
                seed=path_seed,
            )
            results.append((risk, path.tolist()))
    return results
