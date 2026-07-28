import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas
import plotly.graph_objects as go
import pickle
import time

#source_path=r"C:\Users\asus\Desktop\zeynep calisma\neuroscience drosophila\sequential-inverse-kinematics\examples"
source_path=r".."
if source_path not in sys.path:
    sys.path.append(source_path)

from seqikpy.utils import load_file, save_file, calculate_body_size, dict_to_nparray_pose,from_sdf
from seqikpy.alignment import AlignPose, convert_from_df3dpp_to_dict, convert_from_anipose_to_dict
from seqikpy.kinematic_chain import KinematicChainSeq, KinematicChainGeneric
from seqikpy.leg_inverse_kinematics import LegInvKinSeq, LegInvKinGeneric
from seqikpy.visualization import plot_3d_points, animate_3d_points,generate_color_map
from seqikpy.body_config import neuromechfly_body_config
from seqikpy.head_inverse_kinematics import HeadInverseKinematics

leg_joint_angle_names = [
    "ThC_yaw",
    "ThC_pitch",
    "ThC_roll",
    "CTr_pitch",
    "CTr_roll",
    "FTi_pitch",
    "TiTa_pitch",
]
legs_to_align= ["RF", "RM", "RH", "LF", "LM", "LH"]

template={
        "RF_Coxa": np.array([0.35, -0.27, 0.400]),
        "RF_Femur": np.array([0.35, -0.27, -0.025]),
        "RF_Tibia": np.array([0.35, -0.27, -0.731]),
        "RF_Tarsus": np.array([0.35, -0.27, -1.249]),
        "RF_Claw": np.array([0.35, -0.27, -1.912]),
        "LF_Coxa": np.array([0.35, 0.27, 0.400]),
        "LF_Femur": np.array([0.35, 0.27, -0.025]),
        "LF_Tibia": np.array([0.35, 0.27, -0.731]),
        "LF_Tarsus": np.array([0.35, 0.27, -1.249]),
        "LF_Claw": np.array([0.35, 0.27, -1.912]),
        "RM_Coxa": np.array([0, -0.125, 0]),
        "RM_Femur": np.array([0, -0.125, -0.182]),
        "RM_Tibia": np.array([0, -0.125, -0.965]),
        "RM_Tarsus": np.array([0, -0.125, -1.633]),
        "RM_Claw": np.array([0, -0.125, -2.328]),
        "LM_Coxa": np.array([0, 0.125, 0]),
        "LM_Femur": np.array([0, 0.125, -0.182]),
        "LM_Tibia": np.array([0, 0.125, -0.965]),
        "LM_Tarsus": np.array([0, 0.125, -1.633]),
        "LM_Claw": np.array([0, 0.125, -2.328]),
        "RH_Coxa": np.array([-0.215, -0.087, -0.073]),
        "RH_Femur": np.array([-0.215, -0.087, -0.272]),
        "RH_Tibia": np.array([-0.215, -0.087, -1.108]),
        "RH_Tarsus": np.array([-0.215, -0.087, -1.793]),
        "RH_Claw": np.array([-0.215, -0.087, -2.588]),
        "LH_Coxa": np.array([-0.215, 0.087, -0.073]),
        "LH_Femur": np.array([-0.215, 0.087, -0.272]),
        "LH_Tibia": np.array([-0.215, 0.087, -1.108]),
        "LH_Tarsus": np.array([-0.215, 0.087, -1.793]),
        "LH_Claw": np.array([-0.215, 0.087, -2.588]),
        "R_Antenna_base": np.array([1.01, -0.10, 1.41]),
        "L_Antenna_base": np.array([1.01, 0.10, 1.41]),
        "R_Antenna_edge": np.array([1.06, -0.10, 1.14]),
        "L_Antenna_edge": np.array([1.06, 0.10, 1.14]),
        "R_wing": np.array([0.08, -0.4, 1.43]),
        "L_wing": np.array([0.08, 0.4, 1.43]),
        "Neck": np.array([0.53, 0.0, 1.3]),
        "Thorax_mid": np.array([0.08, 0.0, 1.43]),
    }

def get_legs_aligned_pos(aligned_pos):
    pose_list=list(i[:2:] for i in list(aligned_pos.keys()))
    pose_leg_list=[]
    for i in pose_list:
        if i in legs_to_align:
            pose_leg_list.append(i)
    return pose_leg_list

def get_legs_LJA(LJA):
    leg_list=[]
    for i in LJA.keys():
        if i[6:8:] in legs_to_align and not i[6:8:] in leg_list:
            leg_list.append(i[6:8:])
    return leg_list

def getInitAngleRad(pose_leg_list):
    _INITIAL_ANGLES_RAD = {
        "RF": {
            # Base ThC yaw pitch CTr pitch
            "stage_1": np.array([0.0, 0.45, -0.07, -2.14]),
            # Base ThC yaw pitch roll CTr pitch CTr roll
            "stage_2": np.array([0.0, 0.45, -0.07, -0.32, -2.14, 1.4]),
            # Base ThC yaw pitch roll CTr pitch CTr roll FTi pitch
            "stage_3": np.array([0.0, 0.45, -0.07, -0.32, -2.14, -1.25, 1.48, 0.0]),
            # Base ThC yaw pitch roll CTr pitch CTr roll FTi pitch TiTa pitch
            "stage_4": np.array([0.0, 0.45, -0.07, -0.32, -2.14, -1.25, 1.48, 0.0, 0.0]),
        },
        "LF": {
            "stage_1": np.array([0.0, -0.45, -0.07, -2.14]),
            "stage_2": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.4]),
            "stage_3": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.25, 1.48, 0.0]),
            "stage_4": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
        },
        "RM": {
            "stage_1": np.array([0.0, 0.45, 0.37, -2.14]),
            "stage_2": np.array([0.0, 0.45, 0.37, -0.32, -2.14, 1.4]),
            "stage_3": np.array([0.0, 0.45, 0.37, -0.32, -2.14, -1.25, 1.48, 0.0]),
            "stage_4": np.array([0.0, 0.45, 0.37, -0.32, -2.14, -1.25, 1.48, 0.0, 0.0]),
        },
        "LM": {
            "stage_1": np.array([0.0, -0.45, 0.37, -2.14]),
            "stage_2": np.array([0.0, -0.45, 0.37, 0.32, -2.14, 1.4]),
            "stage_3": np.array([0.0, -0.45, 0.37, 0.32, -2.14, 1.25, 1.48, 0.0]),
            "stage_4": np.array([0.0, -0.45, 0.37, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
        },
        "RH": {
            "stage_1": np.array([0.0, 0.45, 0.07, -2.14]),
            "stage_2": np.array([0.0, 0.45, 0.07, -0.32, -2.14, 1.4]),
            "stage_3": np.array([0.0, 0.45, 0.07, -0.32, -2.14, -1.25, 1.48, 0.0]),
            "stage_4": np.array([0.0, 0.45, 0.07, -0.32, -2.14, -1.25, 1.48, 0.0, 0.0]),
        },
        "LH": {
            "stage_1": np.array([0.0, -0.45, 0.07, -2.14]),
            "stage_2": np.array([0.0, -0.45, 0.07, 0.32, -2.14, 1.4]),
            "stage_3": np.array([0.0, -0.45, 0.07, 0.32, -2.14, 1.25, 1.48, 0.0]),
            "stage_4": np.array([0.0, -0.45, 0.07, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
        },
    }
    pose_init_angles_rad=dict()

    for leg,angles in _INITIAL_ANGLES_RAD.items():
        if leg in pose_leg_list:
            pose_init_angles_rad[leg]=angles

    return pose_init_angles_rad

def getBounds(pose_leg_list):
    _BOUNDS_DEG = {
        "RF_ThC_yaw": (-180, 180),
        "RF_ThC_pitch": (-90, 90),
        "RF_ThC_roll": (-180, 180),
        "RF_CTr_pitch": (-180, 180),
        "RF_FTi_pitch": (-180, 180),
        "RF_CTr_roll": (-180, 180),
        "RF_TiTa_pitch": (-180, 0),
        "RM_ThC_yaw": (-50, 50),
        "RM_ThC_pitch": (-180, 180),
        "RM_ThC_roll": (-180, 0),
        "RM_CTr_pitch": (-180, 180),
        "RM_FTi_pitch": (-180, 180),
        "RM_CTr_roll": (-180, 180),
        "RM_TiTa_pitch": (-180, 0),
        "RH_ThC_yaw": (-50, 50),
        "RH_ThC_pitch": (-50, 50),
        "RH_ThC_roll": (-180, 0),
        "RH_CTr_pitch": (-180, 0),
        "RH_FTi_pitch": (-180, 180),
        "RH_CTr_roll": (-180, 180),
        "RH_TiTa_pitch": (-180, 0),
        "LF_ThC_yaw": (-180, 180),
        "LF_ThC_pitch": (-90, 90),
        "LF_ThC_roll": (-180, 180),
        "LF_CTr_pitch": (-180, 180),
        "LF_FTi_pitch": (-180, 180),
        "LF_CTr_roll": (-180, 180),
        "LF_TiTa_pitch": (-180, 0),
        "LM_ThC_yaw": (-50, 50),
        "LM_ThC_pitch": (-180, 180),
        "LM_ThC_roll": (0, 180),
        "LM_CTr_pitch": (-180, 180),
        "LM_FTi_pitch": (-180, 180),
        "LM_CTr_roll": (-180, 180),
        "LM_TiTa_pitch": (-180, 0),
        "LH_ThC_yaw": (-50, 50),
        "LH_ThC_pitch": (-50, 50),
        "LH_ThC_roll": (0, 180),
        "LH_CTr_pitch": (-180, 0),
        "LH_FTi_pitch": (-180, 180),
        "LH_CTr_roll": (-180, 180),
        "LH_TiTa_pitch": (-180, 0),
    }
    pose_bounds_deg=dict()

    for jointmov,bounds in _BOUNDS_DEG.items():
        if jointmov[:2:] in pose_leg_list:
            pose_bounds_deg[jointmov]=bounds

    return pose_bounds_deg

def getKinChain(pose_leg_list):
    body_config = neuromechfly_body_config.deepcopy()
    body_config.template=template
    body_config.initial_angles_rad = getInitAngleRad(pose_leg_list)
    body_config.set_dof_bounds_in_deg(getBounds(pose_leg_list))

    # Initialize the necessary classes
    kin_chain = KinematicChainSeq(
        bounds_dof=body_config.dof_bounds_rad,
        body_size=calculate_body_size(
            body_config.template,
            legs_list=pose_leg_list #no need to give all
        ),
        legs_list=legs_to_align,
    )
    return kin_chain

def runSeqIK_and_FK(aligned_pos):
    pose_leg_list=get_legs_aligned_pos(aligned_pos)

    body_config = neuromechfly_body_config.deepcopy()
    body_config.template=template
    body_config.initial_angles_rad = getInitAngleRad(pose_leg_list)
    body_config.set_dof_bounds_in_deg(getBounds(pose_leg_list))

    # Initialize the necessary classes
    kin_chain = KinematicChainSeq(
        bounds_dof=body_config.dof_bounds_rad,
        body_size=calculate_body_size(
            body_config.template,
            legs_list=pose_leg_list #no need to give all
        ),
        legs_list=legs_to_align,
    )

    class_seq_ik = LegInvKinSeq(
        aligned_pos=aligned_pos,
        kinematic_chain_class=kin_chain,
        initial_angles=body_config.initial_angles_rad,
    )

    leg_joint_angles, forward_kinematics = class_seq_ik.run_ik_and_fk(
        hide_progress_bar=False
    )
    return leg_joint_angles,forward_kinematics

def align_converted_dict(conv_dict):
    align = AlignPose(
        pose_data_dict=conv_dict,
        legs_list=get_legs_aligned_pos(conv_dict),
        include_claw=False,
        body_template=template,
        # if body_size is none, then the size will be
        # calculated from the template
        body_size=None,
    )
    aligned_pos = align.align_pose()
    return aligned_pos




