"""
QxBin Cloud / ML Tier — Uncertainty-Aware Ensemble Router
Use Case: Large-scale inference routing, Monte-Carlo style model selection,
          adaptive load-balancing across specialist models, or probabilistic
          feature weighting under distribution shift.

Tier: Cloud / Server / GPU-ready (ensemble of many Binary Probability Matrices)
Uses Numba for parallel evolution. Easy to swap to CuPy / CUDA later.
By Rupesh Malpani | pikk.company | QxBin Framework
"""

import numpy as np
from numba import njit, prange
from typing import List, Dict, Optional, Tuple


@njit(parallel=True, fastmath=True)
def _evolve_ensemble(states, biases, ns, ms, blend):
    """Parallel fractional-exponent evolution of all cubit matrices."""
    n = states.shape[0]
    for i in prange(n):
        b = biases[i]
        nn = ns[i]
        mm = ms[i]
        frac = b ** nn
        tail = (1.0 - b) ** mm

        # Directed blend
        blended = (states[i] * frac + (1.0 - states[i]) * tail) * blend + states[i] * (1.0 - blend)
        total = blended.sum()
        if total > 1e-12:
            states[i] = blended / total
        else:
            states[i] = np.ones_like(blended) / blended.size
    return states


class QxBinUncertaintyRouter:
    """
    Ensemble of QxBin cubits that collectively vote on which specialist model
    (or feature group) should handle the current input under uncertainty.

    Each cubit represents a soft preference over a set of routes.
    Aggregate statistics give confidence and diversity metrics.
    """

    def __init__(
        self,
        num_cubits: int = 48,
        grid_size: int = 6,
        route_labels: Optional[List[str]] = None,
    ):
        self.num_cubits = num_cubits
        self.grid_size = grid_size
        self.states = np.random.rand(num_cubits, grid_size, grid_size).astype(np.float64)
        for i in range(num_cubits):
            s = self.states[i].sum()
            if s > 0:
                self.states[i] /= s

        self.route_labels = route_labels or [
            "vision_heavy",
            "text_light",
            "tabular",
            "hybrid",
            "fallback",
            "ensemble_vote",
        ]
        self._route_map = self._build_route_map()

    def _build_route_map(self) -> Dict[int, str]:
        n = self.grid_size * self.grid_size
        labels = self.route_labels
        return {i: labels[i % len(labels)] for i in range(n)}

    def evolve(
        self,
        uncertainty_score: float,
        n_range: Tuple[int, int] = (1, 4),
        m_range: Tuple[int, int] = (1, 3),
        blend: float = 0.6,
    ) -> np.ndarray:
        """
        Evolve the entire ensemble.
        uncertainty_score ∈ [0, 1]  (0 = confident, 1 = highly uncertain)
        Higher uncertainty spreads the probability mass (higher exploration).
        """
        # Bias distribution: higher uncertainty → more mid-range biases
        base = 0.55 + 0.3 * (1.0 - uncertainty_score)
        biases = np.clip(
            np.random.normal(base, 0.12 * (0.5 + uncertainty_score), self.num_cubits),
            0.15,
            0.92,
        )
        ns = np.random.randint(n_range[0], n_range[1] + 1, self.num_cubits)
        ms = np.random.randint(m_range[0], m_range[1] + 1, self.num_cubits)

        self.states = _evolve_ensemble(self.states, biases, ns, ms, blend)
        return self.states.mean(axis=0)

    def measure_routes(self, temperature: float = 1.0) -> Dict[str, float]:
        """
        Collapse every cubit and aggregate vote counts → soft distribution over routes.
        """
        votes = {label: 0.0 for label in self.route_labels}
        flat_all = self.states.reshape(self.num_cubits, -1)

        for i in range(self.num_cubits):
            flat = flat_all[i]
            if temperature != 1.0:
                logp = np.log(np.clip(flat, 1e-12, 1.0)) / temperature
                flat = np.exp(logp - logp.max())
                flat /= flat.sum()
            idx = int(np.random.choice(len(flat), p=flat))
            label = self._route_map[idx]
            votes[label] += 1.0

        total = sum(votes.values())
        return {k: v / total for k, v in votes.items()}

    def route(
        self,
        uncertainty_score: float,
        temperature: float = 0.9,
        steps: int = 5,
    ) -> Dict:
        """
        Full routing cycle:
        1. Evolve ensemble under the given uncertainty
        2. Measure soft votes
        3. Return primary route + confidence + diversity
        """
        for _ in range(steps):
            self.evolve(uncertainty_score)

        votes = self.measure_routes(temperature=temperature)
        primary = max(votes, key=votes.get)
        confidence = votes[primary]
        # Simple entropy as diversity proxy
        probs = np.array(list(votes.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-12))

        return {
            "primary_route": primary,
            "confidence": round(float(confidence), 4),
            "entropy": round(float(entropy), 4),
            "all_votes": {k: round(v, 4) for k, v in votes.items()},
            "uncertainty_input": uncertainty_score,
        }


# ------------------------------------------------------------------
# Demo: Adaptive model routing under distribution shift
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 62)
    print("QxBin Uncertainty Router — Cloud / ML Tier")
    print("Use case: Route inference to specialist models under uncertainty")
    print("=" * 62)

    router = QxBinUncertaintyRouter(
        num_cubits=64,
        grid_size=5,
        route_labels=[
            "vision_cnn",
            "text_transformer",
            "tabular_gbm",
            "multimodal",
            "lightweight_edge",
            "fallback_safe",
        ],
    )

    scenarios = [
        ("Clean in-distribution", 0.12),
        ("Mild domain shift", 0.38),
        ("Heavy OOD / sensor noise", 0.71),
        ("Adversarial / extreme", 0.89),
    ]

    for name, unc in scenarios:
        result = router.route(uncertainty_score=unc, temperature=0.85, steps=4)
        print(f"\nScenario: {name} (uncertainty={unc:.2f})")
        print(f"  → Primary : {result['primary_route']}")
        print(f"  → Conf    : {result['confidence']:.3f}")
        print(f"  → Entropy : {result['entropy']:.3f}")
        print(f"  → Votes   : {result['all_votes']}")

    print("\nCloud ensemble ready. Scale num_cubits + swap to CuPy for GPU.")
