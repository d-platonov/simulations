from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from dataclasses import dataclass
from typing import Final

import numpy as np

from levy_type.laws.base import JumpLaw
from levy_type.simulation.jumps import JumpSign

__all__: Final = ["JumpContext", "build_jump_contexts"]


@dataclass(slots=True)
class JumpContext:
    """Hold sign-specific jump law state and RNG."""

    sign: JumpSign
    rng: np.random.Generator
    law: JumpLaw


def build_jump_contexts(law_map: Mapping[JumpSign, JumpLaw], base_seed: int) -> tuple[JumpContext, ...]:
    """
    Return one RNG per provided sign for the configured jump laws.

    Always pass an explicit mapping, e.g. {JumpSign.POSITIVE: law_pos, JumpSign.NEGATIVE: law_neg}
    or just {JumpSign.POSITIVE: law} for one-sided processes.
    """
    contexts: list[JumpContext] = []

    for offset, (sign, sublaw) in enumerate(law_map.items()):
        if sublaw is None:
            continue
        rng = np.random.default_rng(base_seed + offset)
        contexts.append(JumpContext(sign=sign, rng=rng, law=sublaw))

    if not contexts:
        raise ValueError("Configured jump law produced no contexts.")
    return tuple(contexts)
