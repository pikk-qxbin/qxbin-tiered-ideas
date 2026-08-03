"""
QxBin Edge / IoT Tier — Adaptive Sensor Decision Engine
Use Case: Uncertain real-world sensing (smart agriculture, environmental monitoring,
          edge robotics, Pikkstop nodes). Classic binary thresholds fail under noise;
          QxBin Binary Probability Matrices keep the "coin spinning" until evidence
          is strong enough, then collapse to action.

Tier: Edge / Embedded / Desktop (single or few cubits, low memory, pure NumPy)
By Rupesh Malpani | pikk.company | QxBin Framework
"""

import numpy as np
from typing import Tuple, Dict, Optional


class QxBinIoTDecision:
    """
    Lightweight QxBin personal cubit for IoT decision making.

    Maintains a small Binary Probability Matrix.
    Sensor readings bias the superposition.
    Measurement collapses to discrete action (irrigate / alert / idle / etc.).
    """

    def __init__(self, grid_size: int = 4, action_labels: Optional[list] = None):
        self.grid_size = grid_size
        self.state = np.random.rand(grid_size, grid_size).astype(np.float64)
        self._normalize()
        self.action_labels = action_labels or [
            "IDLE", "MONITOR", "ACT_LOW", "ACT_HIGH"
        ]
        # Map flattened indices to actions (simple spatial partitioning)
        self._action_map = self._build_action_map()

    def _normalize(self):
        s = self.state.sum()
        if s > 1e-12:
            self.state /= s
        else:
            self.state.fill(1.0 / self.state.size)

    def _build_action_map(self) -> Dict[int, str]:
        n = self.grid_size * self.grid_size
        labels = self.action_labels
        map_ = {}
        for i in range(n):
            map_[i] = labels[i % len(labels)]
        return map_

    def apply_sensor_bias(
        self,
        sensor_value: float,
        low: float = 0.0,
        high: float = 1.0,
        n: int = 2,
        m: int = 1,
        blend: float = 0.55,
    ) -> np.ndarray:
        """
        Bias the probability matrix from a normalized sensor reading.
        sensor_value in [low, high] → bias in [0, 1].
        Higher bias pushes probability mass toward 'ACT' regions via fractional powers.
        """
        # Normalize sensor to [0, 1]
        bias = np.clip((sensor_value - low) / (high - low + 1e-9), 0.0, 1.0)

        frac = bias ** n
        tail = (1.0 - bias) ** m

        # Coordinate vectors that lean according to bias
        vec = np.linspace(frac, tail, self.grid_size)
        new_matrix = np.outer(vec, vec)

        # Soft blend (superposition-like)
        self.state = (1.0 - blend) * self.state + blend * new_matrix
        self._normalize()
        return self.state

    def measure(self, temperature: float = 1.0) -> Tuple[str, int, float]:
        """
        Probabilistic collapse.
        temperature > 1 softens the distribution (more exploration);
        temperature < 1 sharpens (more exploitation).
        Returns (action_label, flat_index, confidence)
        """
        flat = self.state.flatten()
        if temperature != 1.0:
            # Softmax-style tempering
            logp = np.log(np.clip(flat, 1e-12, 1.0)) / temperature
            flat = np.exp(logp - logp.max())
            flat /= flat.sum()

        idx = int(np.random.choice(len(flat), p=flat))
        confidence = float(flat[idx])
        action = self._action_map[idx]
        return action, idx, confidence

    def decide(
        self,
        sensor_value: float,
        low: float = 0.0,
        high: float = 1.0,
        n: int = 3,
        m: int = 1,
        temperature: float = 0.85,
    ) -> Dict:
        """
        Full decision cycle: bias → measure → return structured result.
        """
        self.apply_sensor_bias(sensor_value, low=low, high=high, n=n, m=m)
        action, idx, conf = self.measure(temperature=temperature)
        return {
            "action": action,
            "confidence": round(conf, 4),
            "sensor": sensor_value,
            "matrix_mean": float(self.state.mean()),
            "matrix_max": float(self.state.max()),
            "index": idx,
        }


# ------------------------------------------------------------------
# Demo: Soil-moisture smart irrigation decision under noise
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("QxBin IoT Decision Engine — Edge Tier")
    print("Use case: Adaptive irrigation under noisy soil sensors")
    print("=" * 60)

    qx = QxBinIoTDecision(
        grid_size=4,
        action_labels=["IDLE", "MONITOR", "IRRIGATE_LIGHT", "IRRIGATE_HEAVY"],
    )

    # Simulated noisy readings over a day
    true_moisture = [0.22, 0.18, 0.35, 0.41, 0.55, 0.62, 0.48]
    noise = np.random.normal(0, 0.04, len(true_moisture))

    for i, (true, nse) in enumerate(zip(true_moisture, noise)):
        reading = np.clip(true + nse, 0.0, 1.0)
        result = qx.decide(
            sensor_value=reading,
            low=0.15,   # wilt threshold
            high=0.70,  # saturation
            n=3,
            m=1,
            temperature=0.8,
        )
        print(
            f"Step {i+1:02d} | true={true:.2f} read={reading:.2f} "
            f"→ {result['action']:<16} conf={result['confidence']:.3f} "
            f"mean_p={result['matrix_mean']:.3f}"
        )

    print("\nDone. Edge cubit ready for deployment on MCU / Raspberry Pi / Pikkstop.")
