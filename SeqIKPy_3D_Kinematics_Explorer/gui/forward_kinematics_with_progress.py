"""Forward kinematics directly from SeqIK joint-angle dictionaries.

This module skips inverse kinematics. It reproduces the stage-4 forward-
kinematics convention used by LegInvKinSeq.run_ik_and_fk.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import sys
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))
    
from seqikpy.kinematic_chain import KinematicChainSeq


LEG_NAMES: Tuple[str, ...] = ("RF", "LF", "RM", "LM", "RH", "LH")
ANGLE_SUFFIXES: Tuple[str, ...] = (
    "ThC_yaw",
    "ThC_pitch",
    "ThC_roll",
    "CTr_pitch",
    "CTr_roll",
    "FTi_pitch",
    "TiTa_pitch",
)


def _required_keys(leg_name: str) -> Tuple[str, ...]:
    return tuple(f"Angle_{leg_name}_{suffix}" for suffix in ANGLE_SUFFIXES)


def _detect_legs(joint_angles: Mapping[str, np.ndarray]) -> list[str]:
    return [
        leg
        for leg in LEG_NAMES
        if all(key in joint_angles for key in _required_keys(leg))
    ]


def _validate_angles(
    joint_angles: Mapping[str, np.ndarray], legs: Sequence[str]
) -> int:
    """Validate keys and frame counts; return the common number of frames."""
    lengths: set[int] = set()

    for leg in legs:
        missing = [key for key in _required_keys(leg) if key not in joint_angles]
        if missing:
            raise KeyError(f"Missing angle arrays for {leg}: {missing}")

        for key in _required_keys(leg):
            arr = np.asarray(joint_angles[key])
            if arr.ndim != 1:
                raise ValueError(f"{key} must be one-dimensional; got {arr.shape}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{key} contains NaN or infinite values.")
            lengths.add(arr.shape[0])

    if not lengths:
        raise ValueError("No complete leg-angle data were found.")
    if len(lengths) != 1:
        raise ValueError(f"All angle arrays must have equal length; got {lengths}.")

    return lengths.pop()


def _normalise_origins(
    origins: Optional[Mapping[str, np.ndarray]],
    legs: Sequence[str],
    n_frames: int,
) -> Dict[str, np.ndarray]:
    """Return an (N, 3) origin array for every leg.

    With origins=None, FK coordinates remain local to each ThC joint and the
    origin is [0, 0, 0].
    """
    result: Dict[str, np.ndarray] = {}

    for leg in legs:
        if origins is None or leg not in origins:
            result[leg] = np.zeros((n_frames, 3), dtype=float)
            continue

        origin = np.asarray(origins[leg], dtype=float)
        if origin.shape == (3,):
            origin = np.tile(origin, (n_frames, 1))
        elif origin.shape != (n_frames, 3):
            raise ValueError(
                f"Origin for {leg} must have shape (3,) or ({n_frames}, 3); "
                f"got {origin.shape}."
            )
        result[leg] = origin

    return result


def calculate_fk_from_seq_angles(
    joint_angles: Mapping[str, np.ndarray],
    kinematic_chain_seq: KinematicChainSeq,
    legs: Optional[Iterable[str]] = None,
    origins: Optional[Mapping[str, np.ndarray]] = None,
    anatomical_points_only: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] =   None,
    progress_every: int = 1,
) -> Dict[str, np.ndarray]:
    """Convert seven SeqIK angles per leg directly into XYZ coordinates.

    Parameters
    ----------
    joint_angles
        Dictionary with keys such as ``Angle_RF_ThC_yaw``. Angles must be in
        radians and all arrays must contain the same number of frames.
    kinematic_chain_seq
        Configured ``KinematicChainSeq`` carrying the same segment lengths and
        DOF convention that were used to produce/train the angle data.
    legs
        Legs to process. When omitted, complete legs are detected from keys.
    origins
        Optional ThC origins. Values may be constant ``(3,)`` vectors or
        time-varying ``(N, 3)`` arrays. Without origins, returned coordinates
        are local coordinates relative to [0, 0, 0].
    anatomical_points_only
        False reproduces the full 9-link output of SeqIK stage 4. True returns
        five positions: ThC origin, Coxa endpoint, Femur endpoint, Tibia
        endpoint and Claw endpoint.
    progress_callback
        Optional callable receiving ``(fraction, message)``. ``fraction`` is
        between 0.0 and 1.0. This keeps the module independent of Streamlit.
    progress_every
        Invoke the callback after this many completed frames. Increase this
        value for long sequences to reduce UI update overhead.

    Returns
    -------
    dict
        Keys follow SeqIK's convention, e.g. ``RF_leg``. Each value has shape
        ``(N, 9, 3)`` or, with anatomical_points_only=True, ``(N, 5, 3)``.
    """
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1.")

    selected_legs = list(legs) if legs is not None else _detect_legs(joint_angles)
    unknown = sorted(set(selected_legs) - set(LEG_NAMES))
    if unknown:
        raise ValueError(f"Unknown leg names: {unknown}")

    n_frames = _validate_angles(joint_angles, selected_legs)
    origin_by_leg = _normalise_origins(origins, selected_legs, n_frames)
    output: Dict[str, np.ndarray] = {}
    total_steps = len(selected_legs) * n_frames
    completed_steps = 0

    if progress_callback is not None:
        progress_callback(0.0, "Initializing Forward kinematics")

    # Stage-4 chain link positions corresponding to anatomical landmarks:
    # Base/ThC, CTr_pitch (coxa end), FTi_pitch (femur end),
    # TiTa_pitch (tibia end), Claw (tarsus end).
    anatomical_indices = np.array([0, 4, 6, 7, 8])

    for leg in selected_legs:
        frames = []

        for t in range(n_frames):
            # This call returns an ikpy.chain.Chain. The first six SeqIK
            # angles are embedded into fixed-link origin orientations.
            chain = kinematic_chain_seq.create_leg_chain(
                leg_name=leg,
                stage=4,
                angles=joint_angles,
                t=t,
            )

            link_names = [link.name for link in chain.links]
            q = np.zeros(len(chain.links), dtype=float)

            # In the sequential stage-4 chain, TiTa_pitch is the only meaningful
            # revolute DOF not already embedded as a fixed orientation.
            tita_index = link_names.index(f"{leg}_TiTa_pitch")
            q[tita_index] = float(joint_angles[f"Angle_{leg}_TiTa_pitch"][t])

            transforms = chain.forward_kinematics(q, full_kinematics=True)
            xyz = np.asarray([transform[:3, 3] for transform in transforms])
            xyz = xyz + origin_by_leg[leg][t]

            if anatomical_points_only:
                xyz = xyz[anatomical_indices]
            frames.append(xyz)

            completed_steps += 1
            if progress_callback is not None and (
                completed_steps % progress_every == 0
                or completed_steps == total_steps
            ):
                progress_callback(
                    completed_steps / total_steps,
                    f"{leg}, frame {t + 1}/{n_frames}",
                )

        output[f"{leg}_leg"] = np.stack(frames, axis=0)

    return output
