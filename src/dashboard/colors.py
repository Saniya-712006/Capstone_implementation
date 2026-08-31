"""
src/dashboard/colors.py

One function: map an attention weight in [0, 1] to an RGB color for
highlighting atoms. Uses matplotlib's "coolwarm" diverging colormap (blue =
low/scaffold, red = high/causal) rather than a hand-rolled gradient, since
it's a well-tested, perceptually-reasonable diverging map already available
wherever matplotlib is (which the dashboard needs anyway for training curves).

Deliberately uses the ABSOLUTE [0, 1] scale, not a per-molecule min-max
normalisation: normalising per molecule would always show visible contrast
even when the model's actual att_o values are barely differentiated (e.g.
early in training, when att_o sits around 0.49-0.50 for every atom) --
that would misrepresent how confident the model actually is. Showing pale,
low-contrast colors for an undertrained model is the honest picture; it
should get more vivid as training progresses and att_o genuinely spreads out.
"""

from typing import Tuple

import matplotlib

_CMAP = matplotlib.colormaps["coolwarm"]


def attention_to_rgb(weight: float) -> Tuple[float, float, float]:
    """Map an attention weight (expected range [0, 1], clamped if outside) to an (r, g, b) tuple in [0, 1] -- the format RDKit's highlightAtomColors expects."""
    w = max(0.0, min(1.0, weight))
    r, g, b, _a = _CMAP(w)
    return (r, g, b)
