"""Shared task identity encoding for `aic_act_v2` (Plan D) dataset and policy.

Both `record_dataset.py` (training-time) and `RunACT.py` (inference-time) must
produce **byte-identical** task vectors for a given (target_module, port_name,
plug_type) triple. Put the one source of truth here and import it from both.

The encoding is a fixed 11-D semantic decomposition. We keep the components
separate (not a flat one-hot over the cartesian product) so the model can
generalize: e.g. "approach a SFP port" is a shared concept across all 5 NIC
mounts.

Layout (11 dims total):
- [0..6]  target_module_id  one-hot:
            nic_card_mount_0, nic_card_mount_1, nic_card_mount_2,
            nic_card_mount_3, nic_card_mount_4, sc_port_0, sc_port_1
- [7..8]  port_within_module one-hot:
            sfp_port_0, sfp_port_1   (zeros for SC trials, which have no sub-port)
- [9..10] plug_type one-hot:
            sfp, sc

If a string isn't in the known set, its slot stays zero. This is intentional:
unseen task identifiers map to "unknown" rather than crashing inference. If
you add new target modules or plug types, append to the tuples (don't reorder)
so existing checkpoints stay valid.
"""

from __future__ import annotations

import numpy as np

TARGET_MODULES: tuple[str, ...] = (
    "nic_card_mount_0",
    "nic_card_mount_1",
    "nic_card_mount_2",
    "nic_card_mount_3",
    "nic_card_mount_4",
    "sc_port_0",
    "sc_port_1",
)
PORTS_WITHIN_MODULE: tuple[str, ...] = ("sfp_port_0", "sfp_port_1")
PLUG_TYPES: tuple[str, ...] = ("sfp", "sc")

TASK_DIM: int = len(TARGET_MODULES) + len(PORTS_WITHIN_MODULE) + len(PLUG_TYPES)
assert TASK_DIM == 11


def encode_task(
    target_module_name: str,
    port_name: str,
    plug_type: str,
) -> np.ndarray:
    """Return a (TASK_DIM,) float32 vector encoding the task identity."""
    vec = np.zeros(TASK_DIM, dtype=np.float32)
    offset = 0
    if target_module_name in TARGET_MODULES:
        vec[offset + TARGET_MODULES.index(target_module_name)] = 1.0
    offset += len(TARGET_MODULES)
    if port_name in PORTS_WITHIN_MODULE:
        vec[offset + PORTS_WITHIN_MODULE.index(port_name)] = 1.0
    offset += len(PORTS_WITHIN_MODULE)
    if plug_type in PLUG_TYPES:
        vec[offset + PLUG_TYPES.index(plug_type)] = 1.0
    return vec
