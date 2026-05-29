# app/services/inference_curves.py
"""
Run C — Deterministic Inference Engine: Shared Saturation Curve Library.

All functions are pure math — no I/O, no DB, no side effects.
All outputs are bounded to [0.0, 1.0].
Import from here — never reimplement locally in rule handlers.

Section 6 of Run_C_Inference_Engine_Spec.md.
"""

import math


# ---------------------------------------------------------------------------
# threshold_curve
# ---------------------------------------------------------------------------

def threshold_curve(value: float | int | None, low: float, high: float) -> float:
    """
    Linear ramp from 0.0 → 1.0 between `low` and `high`. Output clamped.

    - value <= low  → 0.0
    - value >= high → 1.0
    - between       → linear interpolation

    Used for: battery_capacity, portability (inverted by caller), HBM nits.

    Examples:
        threshold_curve(3500, low=3500, high=6000) → 0.0   (floor)
        threshold_curve(5000, low=3500, high=6000) → 0.6
        threshold_curve(6000, low=3500, high=6000) → 1.0   (ceiling)

    Inversion pattern (caller side, e.g. portability — lighter is better):
        score = 1.0 - threshold_curve(weight_grams, low=155, high=215)
    """
    if value is None:
        return 0.0
    if high <= low:
        raise ValueError(f"threshold_curve: high ({high}) must be > low ({low})")
    raw = (float(value) - low) / (high - low)
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# single_sigmoid
# ---------------------------------------------------------------------------

def single_sigmoid(value: float | int | None, midpoint: float, steepness: float) -> float:
    """
    Standard logistic sigmoid — one S-curve with a single 'good enough' transition.

    Formula:  1 / (1 + exp(-steepness * (value - midpoint)))

    - midpoint:   the value where score = 0.5 (inflection point)
    - steepness:  controls how sharp the transition is.
                  Larger → steeper cliff. Typical range: 0.01 – 0.20.

    Output is naturally bounded (0, 1). Clamped to [0.0, 1.0].

    Used for: single-transition spec attributes where there is one
    meaningful 'threshold' that separates weak from good.

    Examples (midpoint=100, steepness=0.05):
        single_sigmoid(40,  100, 0.05) → ~0.047  (well below midpoint)
        single_sigmoid(100, 100, 0.05) → 0.500   (exactly at midpoint)
        single_sigmoid(160, 100, 0.05) → ~0.953  (well above midpoint)
    """
    if value is None:
        return 0.0
    try:
        raw = 1.0 / (1.0 + math.exp(-steepness * (float(value) - midpoint)))
    except OverflowError:
        # exp overflow → effectively 0 denominator → result is 0 or 1
        raw = 0.0 if steepness * (float(value) - midpoint) < 0 else 1.0
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# double_sigmoid
# ---------------------------------------------------------------------------

def double_sigmoid(
    value: float | int | None,
    m1: float,
    s1: float,
    plateau_lo: float,
    plateau_hi: float,
    m2: float,
    s2: float,
) -> float:
    """
    Two stacked logistic sigmoids with a conceptual flat plateau between them.
    Most of the score gain happens in the first sigmoid. Output bounded [0, 1].

    Architecture:
        sig1 = sigmoid(value, midpoint=m1, steepness=s1)   ← primary, steep
        sig2 = sigmoid(value, midpoint=m2, steepness=s2)   ← secondary, gentle
        score = 0.80 * sig1 + 0.20 * sig2

    The 80/20 weight split ensures most gain happens in sig1 (up to plateau_lo),
    while sig2 provides a slow additional lift from plateau_hi onward.
    plateau_lo and plateau_hi are documented for callers — they are the ranges
    where the curve is visually "most flat" between the two sigmoids.

    PPI canonical params (from spec Section 6):
        m1=265, s1=0.05   → steep rise from ~200 to ~330 ppi
        plateau_lo=330, plateau_hi=440
        m2=490, s2=0.03   → gentle rise from ~440 to ~550 ppi

    PPI score examples:
        double_sigmoid(200,  265, 0.05, 330, 440, 490, 0.03) → ~0.04  (low sharpness)
        double_sigmoid(270,  265, 0.05, 330, 440, 490, 0.03) → ~0.42  (mid-rise)
        double_sigmoid(330,  265, 0.05, 330, 440, 490, 0.03) → ~0.77  (plateau start)
        double_sigmoid(400,  265, 0.05, 330, 440, 490, 0.03) → ~0.82  (plateau zone)
        double_sigmoid(440,  265, 0.05, 330, 440, 490, 0.03) → ~0.84  (plateau end)
        double_sigmoid(550,  265, 0.05, 330, 440, 490, 0.03) → ~0.97  (very high PPI)

    Note: plateau_lo and plateau_hi are not enforced programmatically — they are
    design-intent params that callers document. The math produces natural flattening
    because sig1 is near-saturated and sig2 is not yet rising in that range.
    """
    if value is None:
        return 0.0

    sig1 = single_sigmoid(value, midpoint=m1, steepness=s1)
    sig2 = single_sigmoid(value, midpoint=m2, steepness=s2)

    # 80% first sigmoid (steep primary gain), 20% second sigmoid (gentle boost)
    raw = 0.80 * sig1 + 0.20 * sig2
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# saturating_curve
# ---------------------------------------------------------------------------

def saturating_curve(value: float | int | None, knee: float, ceiling: float) -> float:
    """
    Piecewise saturation: steep gain from 0 → knee (captures most of the score),
    then diminishing returns from knee → ceiling (slow additional gain), flat after.

    Piecewise definition:
        value <= 0       → 0.0
        0 < value < knee → linear: 0.0 → 0.75   (steep zone, 75% of score)
        knee ≤ value < ceiling → linear: 0.75 → 1.0  (slow zone, 25% of score)
        value >= ceiling → 1.0                   (saturated, clamped)

    Canonical params from spec Section 6:
        Charging wattage:   knee=67,  ceiling=120
        Touch sampling Hz:  knee=360, ceiling=720
        Refresh rate Hz:    knee=120, ceiling=165
        Screen size (in):   knee=6.5, ceiling=7.0   (Section 9, C5)

    Charging wattage examples:
        saturating_curve(18,  knee=67, ceiling=120) → ~0.20   (slow charger)
        saturating_curve(33,  knee=67, ceiling=120) → ~0.37   (moderate)
        saturating_curve(67,  knee=67, ceiling=120) → 0.75    (knee — "fast")
        saturating_curve(100, knee=67, ceiling=120) → ~0.91   (very fast)
        saturating_curve(120, knee=67, ceiling=120) → 1.00    (ceiling)
        saturating_curve(165, knee=67, ceiling=120) → 1.00    (clamped)

    Refresh rate examples:
        saturating_curve(60,  knee=120, ceiling=165) → ~0.375  (basic)
        saturating_curve(90,  knee=120, ceiling=165) → ~0.563  (standard)
        saturating_curve(120, knee=120, ceiling=165) → 0.75    (knee — "high")
        saturating_curve(144, knee=120, ceiling=165) → ~0.883  (above knee)
        saturating_curve(165, knee=120, ceiling=165) → 1.00    (ceiling)
    """
    if value is None:
        return 0.0
    if knee <= 0:
        raise ValueError(f"saturating_curve: knee ({knee}) must be > 0")
    if ceiling <= knee:
        raise ValueError(f"saturating_curve: ceiling ({ceiling}) must be > knee ({knee})")

    v = float(value)

    if v <= 0.0:
        return 0.0
    if v >= ceiling:
        return 1.0
    if v >= knee:
        # Slow gain zone: knee → ceiling maps to 0.75 → 1.0
        return 0.75 + 0.25 * (v - knee) / (ceiling - knee)
    else:
        # Steep gain zone: 0 → knee maps to 0.0 → 0.75
        return 0.75 * v / knee
