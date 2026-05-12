#!/usr/bin/env python3
"""Generate a randomized aic_engine trial config for ACT data collection.

Distribution: each trial uniformly samples task-board pose, NIC/SC rail
occupancy, cable grasp jitter, and target plug type from limits drawn off
sample_config.yaml. Output is a YAML file in the same schema aic_engine
loads (see ``trials:`` section of sample_config.yaml).

Usage:
    pixi run python aic_engine/config/gen_random_trials.py \\
        --n 300 --seed 42 \\
        --out aic_engine/config/random_trials_300.yaml
"""

import argparse
import random
from pathlib import Path

import yaml


# Limits are drawn from sample_config.yaml (the canonical baseline) plus a small
# randomization range that stays within the toolkit's documented allowable rails.
TASK_BOARD_X_RANGE = (0.13, 0.18)
TASK_BOARD_Y_RANGE = (-0.22, 0.02)
TASK_BOARD_YAW_RANGE = (2.95, 3.25)  # near pi
TASK_BOARD_Z = 1.14

NIC_RAIL_TRANSLATION_RANGE = (-0.0215, 0.0234)
NIC_RAIL_YAW_RANGE = (-0.10, 0.10)
SC_RAIL_TRANSLATION_RANGE = (-0.06, 0.055)
SC_RAIL_YAW_RANGE = (-0.10, 0.10)
MOUNT_RAIL_TRANSLATION_RANGE = (-0.09425, 0.09425)

# Cable grasp jitter (canonical: gripper_offset y=0.015385, z=0.04245).
CABLE_OFFSET_X_JITTER = 0.002  # +- 2 mm
CABLE_OFFSET_Y_JITTER = 0.002
CABLE_OFFSET_Z_JITTER = 0.002
CABLE_RPY_JITTER = 0.04         # +- 0.04 rad
CABLE_OFFSET_Y_NOMINAL = 0.015385
CABLE_OFFSET_Z_NOMINAL = 0.04245
CABLE_ROLL_NOMINAL = 0.4432
CABLE_PITCH_NOMINAL = -0.4838
CABLE_YAW_NOMINAL = 1.3303

NIC_RAIL_COUNT = 5  # nic_rail_0..nic_rail_4
SC_RAIL_COUNT = 2   # sc_rail_0..sc_rail_1


def jitter(rng: random.Random, nominal: float, half_range: float) -> float:
    return nominal + rng.uniform(-half_range, half_range)


def random_rail_translation(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def build_trial_sfp(rng: random.Random) -> dict:
    """SFP-port insertion (matches sample_config trial_1 / trial_2 structure)."""
    # Pick which NIC rail receives the target NIC card.
    target_rail = rng.randrange(NIC_RAIL_COUNT)

    nic_rails = {}
    for i in range(NIC_RAIL_COUNT):
        if i == target_rail:
            nic_rails[f"nic_rail_{i}"] = {
                "entity_present": True,
                "entity_name": f"nic_card_{i}",
                "entity_pose": {
                    "translation": random_rail_translation(rng, *NIC_RAIL_TRANSLATION_RANGE),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": rng.uniform(*NIC_RAIL_YAW_RANGE),
                },
            }
        else:
            nic_rails[f"nic_rail_{i}"] = {"entity_present": False}

    # Use sc_mount_0 as a stationary distractor on sc_rail_0 (matches sample_config).
    sc_rails = {
        "sc_rail_0": {
            "entity_present": True,
            "entity_name": "sc_mount_0",
            "entity_pose": {
                "translation": random_rail_translation(rng, *SC_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": rng.uniform(*SC_RAIL_YAW_RANGE),
            },
        },
        "sc_rail_1": {"entity_present": False},
    }

    # Mount rail distractors (kept like sample_config).
    mount_rails = {
        "lc_mount_rail_0": {
            "entity_present": True,
            "entity_name": "lc_mount_0",
            "entity_pose": {
                "translation": random_rail_translation(rng, *MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
        "sfp_mount_rail_0": {
            "entity_present": True,
            "entity_name": "sfp_mount_0",
            "entity_pose": {
                "translation": random_rail_translation(rng, *MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
        "sc_mount_rail_0": {
            "entity_present": True,
            "entity_name": "sc_mount_0",
            "entity_pose": {
                "translation": random_rail_translation(rng, *MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
        "lc_mount_rail_1": {
            "entity_present": True,
            "entity_name": "lc_mount_1",
            "entity_pose": {
                "translation": random_rail_translation(rng, *MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
        "sfp_mount_rail_1": {"entity_present": False},
        "sc_mount_rail_1": {"entity_present": False},
    }

    cable = {
        "cable_0": {
            "pose": {
                "gripper_offset": {
                    "x": jitter(rng, 0.0, CABLE_OFFSET_X_JITTER),
                    "y": jitter(rng, CABLE_OFFSET_Y_NOMINAL, CABLE_OFFSET_Y_JITTER),
                    "z": jitter(rng, CABLE_OFFSET_Z_NOMINAL, CABLE_OFFSET_Z_JITTER),
                },
                "roll": jitter(rng, CABLE_ROLL_NOMINAL, CABLE_RPY_JITTER),
                "pitch": jitter(rng, CABLE_PITCH_NOMINAL, CABLE_RPY_JITTER),
                "yaw": jitter(rng, CABLE_YAW_NOMINAL, CABLE_RPY_JITTER),
            },
            "attach_cable_to_gripper": True,
            "cable_type": "sfp_sc_cable",
        },
    }

    scene = {
        "task_board": {
            "pose": {
                "x": rng.uniform(*TASK_BOARD_X_RANGE),
                "y": rng.uniform(*TASK_BOARD_Y_RANGE),
                "z": TASK_BOARD_Z,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": rng.uniform(*TASK_BOARD_YAW_RANGE),
            },
            **nic_rails,
            **sc_rails,
            **mount_rails,
        },
        "cables": cable,
    }

    # Each NIC card exposes TWO SFP ports (sfp_port_0 and sfp_port_1). Pre-Plan-D
    # data collection only ever targeted sfp_port_0, which is one of the OOD holes
    # that hurts compose trial 1 today. Coin-flip between them so the model sees
    # both. The port pose is identical scene-side; only the task target changes.
    target_port = rng.choice(("sfp_port_0", "sfp_port_1"))
    tasks = {
        "task_1": {
            "cable_type": "sfp_sc",
            "cable_name": "cable_0",
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": target_port,
            "target_module_name": f"nic_card_mount_{target_rail}",
            "time_limit": 180,
        }
    }
    return {"scene": scene, "tasks": tasks}


def build_trial_sc(rng: random.Random) -> dict:
    """SC-port insertion (matches sample_config trial_3 structure)."""
    # Random sc_rail target.
    target_sc_rail = rng.randrange(SC_RAIL_COUNT)
    sc_rails = {}
    for i in range(SC_RAIL_COUNT):
        if i == target_sc_rail:
            sc_rails[f"sc_rail_{i}"] = {
                "entity_present": True,
                "entity_name": f"sc_mount_{i}",
                "entity_pose": {
                    "translation": random_rail_translation(rng, *SC_RAIL_TRANSLATION_RANGE),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": rng.uniform(*SC_RAIL_YAW_RANGE),
                },
            }
        else:
            sc_rails[f"sc_rail_{i}"] = {"entity_present": False}

    # No NIC cards present in SC trials (matches sample_config trial_3).
    nic_rails = {f"nic_rail_{i}": {"entity_present": False} for i in range(NIC_RAIL_COUNT)}

    mount_rails = {
        "lc_mount_rail_0": {"entity_present": False},
        "sfp_mount_rail_0": {
            "entity_present": True,
            "entity_name": "sfp_mount_0",
            "entity_pose": {
                "translation": random_rail_translation(rng, *MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
        "sc_mount_rail_0": {
            "entity_present": True,
            "entity_name": "sc_mount_2",
            "entity_pose": {
                "translation": random_rail_translation(rng, *MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
        "lc_mount_rail_1": {
            "entity_present": True,
            "entity_name": "lc_mount_1",
            "entity_pose": {
                "translation": random_rail_translation(rng, *MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
        "sfp_mount_rail_1": {"entity_present": False},
        "sc_mount_rail_1": {"entity_present": False},
    }

    # SC trial uses cable_1 (sfp_sc_cable_reversed) so the SC plug is the manipulated end.
    cable_z_nominal = 0.04045
    cable = {
        "cable_1": {
            "pose": {
                "gripper_offset": {
                    "x": jitter(rng, 0.0, CABLE_OFFSET_X_JITTER),
                    "y": jitter(rng, CABLE_OFFSET_Y_NOMINAL, CABLE_OFFSET_Y_JITTER),
                    "z": jitter(rng, cable_z_nominal, CABLE_OFFSET_Z_JITTER),
                },
                "roll": jitter(rng, CABLE_ROLL_NOMINAL, CABLE_RPY_JITTER),
                "pitch": jitter(rng, CABLE_PITCH_NOMINAL, CABLE_RPY_JITTER),
                "yaw": jitter(rng, CABLE_YAW_NOMINAL, CABLE_RPY_JITTER),
            },
            "attach_cable_to_gripper": True,
            "cable_type": "sfp_sc_cable_reversed",
        },
    }

    scene = {
        "task_board": {
            "pose": {
                "x": rng.uniform(*TASK_BOARD_X_RANGE),
                "y": rng.uniform(*TASK_BOARD_Y_RANGE),
                "z": TASK_BOARD_Z,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": rng.uniform(*TASK_BOARD_YAW_RANGE),
            },
            **nic_rails,
            **sc_rails,
            **mount_rails,
        },
        "cables": cable,
    }

    tasks = {
        "task_1": {
            "cable_type": "sfp_sc",
            "cable_name": "cable_1",
            "plug_type": "sc",
            "plug_name": "sc_tip",
            "port_type": "sc",
            "port_name": "sc_port_base",
            "target_module_name": f"sc_port_{target_sc_rail}",
            "time_limit": 180,
        }
    }
    return {"scene": scene, "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sfp-fraction",
        type=float,
        default=0.67,
        help="Fraction of trials targeting SFP ports (vs SC).",
    )
    parser.add_argument("--out", type=Path, default=Path("random_trials.yaml"))
    args = parser.parse_args()

    rng = random.Random(args.seed)

    trials = {}
    for i in range(args.n):
        is_sfp = rng.random() < args.sfp_fraction
        trial = build_trial_sfp(rng) if is_sfp else build_trial_sc(rng)
        trials[f"trial_{i + 1}"] = trial

    # Match sample_config.yaml's top-level structure: scoring, task_board_limits, trials, robot.
    cfg = {
        "scoring": _default_scoring_block(),
        "task_board_limits": {
            "nic_rail": {
                "min_translation": NIC_RAIL_TRANSLATION_RANGE[0],
                "max_translation": NIC_RAIL_TRANSLATION_RANGE[1],
            },
            "sc_rail": {
                "min_translation": SC_RAIL_TRANSLATION_RANGE[0],
                "max_translation": SC_RAIL_TRANSLATION_RANGE[1],
            },
            "mount_rail": {
                "min_translation": MOUNT_RAIL_TRANSLATION_RANGE[0],
                "max_translation": MOUNT_RAIL_TRANSLATION_RANGE[1],
            },
        },
        "trials": trials,
        "robot": {
            "home_joint_positions": {
                "shoulder_pan_joint": -0.1597,
                "shoulder_lift_joint": -1.3542,
                "elbow_joint": -1.6648,
                "wrist_1_joint": -1.6933,
                "wrist_2_joint": 1.5710,
                "wrist_3_joint": 1.4110,
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {args.n} trials to {args.out}")


def _default_scoring_block() -> dict:
    """Match sample_config.yaml's scoring block verbatim."""
    return {
        "topics": [
            {"topic": {"name": "/joint_states", "type": "sensor_msgs/msg/JointState"}},
            {"topic": {"name": "/tf", "type": "tf2_msgs/msg/TFMessage"}},
            {"topic": {"name": "/tf_static", "type": "tf2_msgs/msg/TFMessage", "latched": True}},
            {"topic": {"name": "/scoring/tf", "type": "tf2_msgs/msg/TFMessage"}},
            {"topic": {"name": "/aic/gazebo/contacts/off_limit", "type": "ros_gz_interfaces/msg/Contacts"}},
            {"topic": {"name": "/fts_broadcaster/wrench", "type": "geometry_msgs/msg/WrenchStamped"}},
            {"topic": {"name": "/aic_controller/joint_commands", "type": "aic_control_interfaces/msg/JointMotionUpdate"}},
            {"topic": {"name": "/aic_controller/pose_commands", "type": "aic_control_interfaces/msg/MotionUpdate"}},
            {"topic": {"name": "/scoring/insertion_event", "type": "std_msgs/msg/String"}},
            {"topic": {"name": "/aic_controller/controller_state", "type": "aic_control_interfaces/msg/ControllerState"}},
        ]
    }


if __name__ == "__main__":
    main()
