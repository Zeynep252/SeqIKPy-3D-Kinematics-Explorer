import sys
import io
import zipfile
from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from stqdm import stqdm

GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

import KinematicsProcesserGUI
from safe_scientific_loader import safe_load_uploaded_file,validate_scientific_data
from forward_kinematics_with_progress import calculate_fk_from_seq_angles,_detect_legs
from seqikpy.utils import load_file, save_file, calculate_body_size, dict_to_nparray_pose
from seqikpy.alignment import AlignPose, convert_from_df3dpp_to_dict, convert_from_anipose_to_dict, convert_from_df3d_to_dict
from seqikpy.kinematic_chain import KinematicChainSeq
import seqikpy.leg_inverse_kinematics as lik
from seqikpy.visualization import plot_3d_points, animate_3d_points,generate_color_map
from seqikpy.body_config import neuromechfly_body_config
from seqikpy.head_inverse_kinematics import HeadInverseKinematics
from seqikpy.leg_inverse_kinematics import LegInvKinSeq

print("seqIKPy succesfully imported.")

# Set up the constant variables
leg_joint_angle_names = [
    "ThC_yaw",
    "ThC_pitch",
    "ThC_roll",
    "CTr_pitch",
    "CTr_roll",
    "FTi_pitch",
    "TiTa_pitch",
]
legs_to_align_locomotion = ["RF", "RM", "RH", "LF", "LM", "LH"]
legs_to_align_grooming = ["RF","LF"]

keypoints=["ThC","CTr","FTi","TiTa","Claw"]

leg_ThC_origins={"RF":np.array([0.35,-0.27,0.4]),
                 "RM":np.array([0,-0.125,0]),
                 "RH":np.array([-0.215,-0.087,-0.073]),
                 "LF":np.array([0.35,0.27,0.4]),
                 "LM":np.array([0,0.125,0]),
                 "LH":np.array([-0.215,0.087,-0.073]),}

SAMPLE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "sample"
)

@st.cache_data(show_spinner=False)
def load_sample_file(file_path: str):
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Sample file not found: {path}"
        )

    return load_file(path)
  
locomotion_leg_joint_angles=load_sample_file(str(
        SAMPLE_DATA_DIR
        / "locomotion_leg_joint_angles_processed_300f.pkl"))
locomotion_forward_kinematics=load_sample_file(str(
        SAMPLE_DATA_DIR
        / "locomotion_forward_kinematics_processed_300f.pkl"))
aligned_pos_locomotion=load_sample_file(str(
        SAMPLE_DATA_DIR
        / "locomotion_aligned_pos_300f.pkl"))
grooming_leg_joint_angles=load_sample_file(str(
        SAMPLE_DATA_DIR
        / "grooming_leg_joint_angles_processed_300f.pkl"))
grooming_forward_kinematics=load_sample_file(str(
        SAMPLE_DATA_DIR
        / "grooming_forward_kinematics_processed_300f.pkl"))
aligned_pos_grooming=load_sample_file(str(
        SAMPLE_DATA_DIR
        / "grooming_aligned_pos_300f.pkl"))

print("Sample files succesfully imported")

def getOrigins(JointAngleData):
    legs=_detect_legs(JointAngleData)
    origins={}
    for i in legs:
        origins[i]=leg_ThC_origins[i]
    return origins

def df_convert(points_data_dict,line_type):
    rows=[]

    for leg_name,array in points_data_dict.items():
        if not array.shape[0]==1:
            num_frames=array.shape[0]
            num_keypoints=array.shape[1]

            for t in range(num_frames):
                for kp_idx in range(num_keypoints):
                    if num_keypoints==9:
                        if kp_idx<4:
                            kp_name=keypoints[0]
                        elif kp_idx<6:
                            kp_name=keypoints[1]
                        else:
                            kp_name=keypoints[kp_idx-4]

                    elif num_keypoints==5:
                        kp_name=keypoints[kp_idx]
                    else:
                        kp_name=f"Keypoint_{kp_idx}"

                    x = array[t, kp_idx, 0]
                    y = array[t, kp_idx, 1]
                    z = array[t, kp_idx, 2]

                    rows.append({
                        "frame":t,
                        "leg_name":leg_name,
                        "keypoint_name":kp_name,
                        "x":x,
                        "y":y,
                        "z":z,
                        "line_type":line_type,
                        "leg+line":f"{leg_name}_{line_type}"
                    })

        elif array.shape[0]==1:
            num_frames=max([array.shape[0] for leg_name,array in points_data_dict.items()])
            num_keypoints=array.shape[1]
            for t in range(num_frames):
                for kp_idx in range(num_keypoints):
                    if num_keypoints==9:
                        if kp_idx<4:
                            kp_name=keypoints[0]
                        elif kp_idx<6:
                            kp_name=keypoints[1]
                        else:
                            kp_name=keypoints[kp_idx-4]

                    elif num_keypoints==5:
                        kp_name=keypoints[kp_idx]
                    else:
                        kp_name=f"Keypoint_{kp_idx}"

                    x = array[0, kp_idx, 0]
                    y = array[0, kp_idx, 1]
                    z = array[0, kp_idx, 2]

                    rows.append({
                        "frame":t,
                        "leg_name":leg_name,
                        "keypoint_name":kp_name,
                        "x":x,
                        "y":y,
                        "z":z,
                        "line_type":line_type,
                        "leg+line":f"{leg_name}_{line_type}"
                    })

            

    return pd.DataFrame(rows), num_frames

def get_axis_range(df, column, padding_ratio=0.12):
    minimum = float(df[column].min())
    maximum = float(df[column].max())

    span = maximum - minimum
    if span == 0:
        span = 1.0

    padding = span * padding_ratio
    return [minimum - padding, maximum + padding]

def visualize(fig,poseDataSolid=None,poseDataDashed=None):
        fig.data=[]
        if poseDataSolid is not None:
            df_converted_solid, num_frames_solid=df_convert(poseDataSolid,"solid")
        if poseDataDashed is not None:
            df_converted_dash, num_frames_dash=df_convert(poseDataDashed,"dash")

        if poseDataSolid and poseDataDashed:
            df_converted= pd.concat([df_converted_solid, df_converted_dash], ignore_index=True)
            if num_frames_solid!=num_frames_dash:
                raise ValueError("Aligned pose and forward kinematics data must have the same number of frames.")
            else:
                num_frames=num_frames_solid
        elif poseDataSolid:
            df_converted=df_converted_solid
            num_frames=num_frames_solid
        elif poseDataDashed:
            df_converted=df_converted_dash
            num_frames=num_frames_dash

        
        def get_color(leg_name):
            if "RF" in leg_name: return "rgb(255, 120, 120)" 
            elif "RM" in leg_name: return "rgb(250, 40, 40)"
            elif "RH" in leg_name: return "rgb(120, 0, 0)"

            elif "LF" in leg_name: return "rgb(130, 210, 255)"
            elif "LM" in leg_name: return "rgb(0, 120, 255)"
            elif "LH" in leg_name: return "rgb(0, 45, 250)" 
            return "rgb(225,225,225)"

        init_df = df_converted[df_converted["frame"] == 0]
        for leg_name, data in init_df.groupby("leg+line"):

            hover_texts = [f"{row["leg_name"]} - {row['keypoint_name']}" for idx, row in data.iterrows()]

            fig.add_trace(go.Scatter3d(
                x=data["x"],
                y=data["y"],
                z=data["z"],
                mode="lines+markers",
                line=dict(color=get_color(leg_name), width=4,dash=data["line_type"].iloc[0]),
                marker=dict(size=3),
                name=data["leg_name"].iloc[0],
                showlegend=True,
                hovertext=hover_texts,
                hoverinfo="text+x+y+z"
            ))

        frames = []
        for t in range(num_frames):
            frame_df = df_converted[df_converted["frame"] == t]
            frame_data = []
            
            for leg_name, data in frame_df.groupby("leg+line"):
                frame_data.append(go.Scatter3d(
                    x=data["x"],
                    y=data["y"],
                    z=data["z"],

                ))
            frames.append(go.Frame(data=frame_data, name=str(t)))

        fig.frames = frames

        # Interface
        updatemenus = [dict(
            type="buttons",
            buttons=[
                dict(label="▶️ Play", method="animate",
                    args=[None, dict(frame=dict(duration=10, redraw=True), fromcurrent=True)]),
                dict(label="⏸️ Pause", method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            direction="left", pad={"r": 10, "t": 87}, x=0.1, xanchor="right", y=0, yanchor="top"
        )]

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Frame (t): ", "font": {"size": 14}},
            pad={"b": 10, "t": 50}, len=0.9, x=0.1, y=0,
            steps=[dict(label=str(t), method="animate",
                        args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate")]) for t in range(0,num_frames)]
        )]

        x_range = get_axis_range(df_converted, "x")
        y_range = get_axis_range(df_converted, "y")
        z_range = get_axis_range(df_converted, "z")

        #Layout
        fig.update_layout(
            width=800,
            height=700,
            scene=dict(
                xaxis=dict(range=x_range, title="X"),
                yaxis=dict(range=y_range, title="Y"),
                zaxis=dict(range=z_range, title="Z"),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.8),
                camera= dict(
                    up=dict(x=0, y=0, z=1),       
                    center=dict(x=0, y=0, z=0),   
                    eye=dict(x=0.75, y=0.75, z=0.7))
            
            ),
            updatemenus=updatemenus,
            sliders=sliders,
            legend=dict(x=1.05, y=0.8)
        )

st.set_page_config(page_title="Drosophila melanogaster Kinematics Explorer", layout="wide")
st.title("*Drosophila melanogaster* Kinematics Explorer")

tab_visualize, tab_process, tab_help = st.tabs(
    [
        "3D Visualization",
        "Process Data",
        "Help & Data Formats",
    ]
)

@st.fragment
def alignment():
    st.markdown("SeqIKPy 3D visuailization tool uses aligned pose data. If you have anipose, df3d, df3dpp or converted dictionary data, you can align them here before visualizing.")
    raw_pose_selection = st.selectbox(
        "Select your raw data type to align",
        options=[
            "DeepFly3D (df3d)",
            "DeepFly3D Post Processing (df3dpp)",
            "Anipose",
            "Converted dictionary"
        ])
    raw_pose_upload=st.file_uploader(
                label=f"Load your {raw_pose_selection} data to align", 
                type=["pkl","h5","hdf","npy"])
    if raw_pose_upload is not None:
        with st.spinner("Confirming data safety..."):
            pose_safety_load=safe_load_uploaded_file(raw_pose_upload)
        if not raw_pose_selection=="Converted dictionary":
            with st.spinner("Converting your pose into dictionary... please wait"):
                if raw_pose_selection=="DeepFly3D (df3d)":
                    converted_dict= convert_from_df3d_to_dict(pose_safety_load,pts2align=neuromechfly_body_config.points_to_align)
                    #not tested yet
                elif raw_pose_selection=="DeepFly3D Post Processing (df3dpp)":
                    converted_dict= convert_from_df3dpp_to_dict(pose_safety_load)
                elif raw_pose_selection=="Anipose":
                    converted_dict= convert_from_anipose_to_dict(pose_safety_load,pts2align=neuromechfly_body_config.points_to_align)
        else:
            converted_dict=pose_safety_load
        with st.spinner("Aligning your pose... please wait"):
            aligned_pose=KinematicsProcesserGUI.align_converted_dict(converted_dict)
        st.success("Pose Aligned succesfully")

        pkl_aligned_data = pickle.dumps(
            aligned_pose,
            protocol=pickle.HIGHEST_PROTOCOL)

        upload_file_name = Path(raw_pose_upload.name).stem

        st.download_button(
            label="Download your aligned pose data as .pkl",
            data=pkl_aligned_data,
            file_name=f"{upload_file_name}-Aligned.pkl",
            mime="application/octet-stream",
            on_click="ignore",
            key="download_align_result")
               
@st.fragment
def crop():  
    st.markdown("3D Visualization of data with too many frames can slow down your device. If your data consists of 1000+ frames, you can consider cropping your data here.")    
    
    aligned_pose_upload_edit=st.file_uploader(
                label=f"Load your aligned data to edit", 
                type=["pkl","h5","hdf","npy"])
    if aligned_pose_upload_edit is not None:
        with st.spinner("Confirming data safety..."):
            safe_load_pose=safe_load_uploaded_file(aligned_pose_upload_edit)
        with st.spinner("Validating data content..."):
            pose_validate=validate_scientific_data(safe_load_pose)
        st.success("Data succesfully loaded")
        pose_to_edit=dict(pose_validate)
        try:
            num_frames_original=pose_to_edit[list(pose_to_edit.keys())[0]].shape[0]
            st.markdown(f"Crop {aligned_pose_upload_edit.name} of {num_frames_original} frames")
            edit_area=st.empty()
            num_left,num_right=edit_area.columns([1,1])
            min_frame=num_left.number_input("From:",min_value=0,max_value=num_frames_original-1,step=1,value=0)
            max_frame=num_right.number_input("To:",min_value=0,max_value=num_frames_original-1,step=1,value=num_frames_original-1)
            
            if not ((max_frame<min_frame) or (min_frame==0 and max_frame==num_frames_original-1)):
                if st.button(f"**Crop Pose Data**"):
                    with st.spinner("Editing your data..."):
                        if max_frame>min_frame:
                            cropped_pose=dict()
                            for limb, data in pose_to_edit.items():
                                cropped_pose[limb]=data[min_frame:max_frame:]
                        elif max_frame==min_frame:
                            cropped_pose=dict()
                            for limb, data in pose_to_edit.items():
                                cropped_pose[limb]=data[min_frame]
                    st.markdown("Pose data cropped.")
                    cropped_data_download = pickle.dumps(
                        cropped_pose,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )

                    st.download_button(
                        label="Download cropped data as .pkl",
                        data=cropped_data_download,
                        file_name=f"{aligned_pose_upload_edit.name}_{min_frame}-{max_frame}.pkl",
                        mime="application/octet-stream",
                        on_click="ignore",
                        key="download_crop_result"
                    )
        except Exception as e:
            st.error(f"File cannot be processed: {e}\n\nPlease try again.")

@st.fragment
def IK_FK():
    uploaded_file= st.file_uploader(
                    label="Load your aligned pose data to process", 
                    type=["pkl","h5","hdf","npy"])
    if uploaded_file:
        with st.spinner("Confirming data safety..."):
            poseDataLoad=safe_load_uploaded_file(uploaded_file)
        with st.spinner("Validating data content..."):
            poseDataValidate, summary = validate_scientific_data(poseDataLoad,data_type="pose",max_frames=20000,max_keypoints=100)
        st.success("Pose data loaded successfully.")
        poseData3D=poseDataValidate
        progress_area=st.empty()
        if progress_area.button("Run Kinematics"):
            progress_area.empty()
            progress_area.markdown("**Running inverse and forward kinematics**\n\nPlease wait")
            def streamlit_trange(
                *args,
                desc=None,
                disable=False,
                **kwargs,):
                iterable = range(*args)
                total = len(iterable)
        
                if disable:
                    yield from iterable
                    return
        
                progress_bar = progress_area.progress(0,text=desc,)
        
                for index, value in enumerate(iterable):
                    yield value
        
                    progress = (index + 1) / total

                    progress_bar.progress(
                        progress,
                        text=("**Running inverse and forward kinematics...**\n\n"
                            f"{desc} — "
                            f"%{int(progress * 100)}"
                        ),
                    )
            lik.trange = streamlit_trange
            leg_joint_angles,FKData= KinematicsProcesserGUI.runSeqIK_and_FK(poseData3D)
            progress_area.empty()

            st.success("Kinematics Process Completed.")
            pkl_joint_angles_data = pickle.dumps(leg_joint_angles)
            pkl_FK_data = pickle.dumps(FKData)

            base_name = Path(uploaded_file.name).stem

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(
                zip_buffer,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as zip_file:
                zip_file.writestr(
                    f"{base_name}_forward_kinematics.pkl",
                    pkl_FK_data,
                )

                zip_file.writestr(
                    f"{base_name}_leg_joint_angles.pkl",
                    pkl_joint_angles_data,
                )
            st.download_button(
                label="Download Joint Angles & Forward Kinematics Data as ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"{base_name}-Kinematics-Output.zip",
                mime="application/zip",
                on_click="ignore",
                key="download_kinematics_results",
            )

@st.fragment
def FK():
    uploaded_file_lja= st.file_uploader(label=f"Load your leg joint angles data to run forward kinematics", type=["pkl","h5","hdf","npy"])              
    if uploaded_file_lja:
        with st.spinner("Confirming data safety..."):
            AngleDataLoad=safe_load_uploaded_file(uploaded_file_lja)
        with st.spinner("Validating data content..."):
            AngleDataValidate, summary = validate_scientific_data(AngleDataLoad,data_type="joint_angles",max_frames=20000)
        st.success("Data succesfully loaded.")
        AngleData=AngleDataValidate
        process_space=st.empty()
        if process_space.button("Run Forward Kinematics"):
            info_space=st.empty()
            info_space.info(f"Running Forward kinematics on your data. Please wait.")
            process_space.empty()
            with process_space:
                
                origins=getOrigins(AngleData)

                progress_bar = st.progress(0)

                def update_progress(fraction: float, message: str) -> None:
                    percentage = int(fraction * 100)

                    progress_bar.progress(
                        percentage,
                        text=f"%{percentage} — {message}",
                    )

                kin_chain=KinematicsProcesserGUI.getKinChain(_detect_legs(AngleData))
                origins=getOrigins(AngleData)

                fk_result = calculate_fk_from_seq_angles(
                    joint_angles=AngleData,
                    kinematic_chain_seq=kin_chain,
                    origins=origins,
                    anatomical_points_only=True,
                    progress_callback=update_progress,
                    progress_every=1,
                )
            process_space.empty()
            info_space.empty()
            st.success("Forward Kinematics completed.")

            base_name=Path(uploaded_file_lja.name).stem

            pkl_FK_data = pickle.dumps(
                fk_result,
                protocol=pickle.HIGHEST_PROTOCOL
            )

            st.download_button(
                label="Download forward kinematics output as .pkl",
                data=pkl_FK_data,
                file_name=f"{base_name}- Forward Kinematics.pkl",
                mime="application/octet-stream",
                on_click="ignore",
                key="download_FK_result"
            )

@st.fragment
def LJA_View(LJA):
    leg_choice=st.selectbox("Leg to display joint angles:",options=KinematicsProcesserGUI.get_legs_LJA(LJA))
    graph = go.Figure()
    num_frames=LJA[list(LJA.keys())[0]].shape[0]
    for angle,values in LJA.items():
        if leg_choice in angle:
            graph.add_trace(go.Scatter(x=np.arange(num_frames), y=values, mode='lines', name=angle[9::]))
    graph.update_layout(
        title=f"{leg_choice} Leg Joint Angles",
        xaxis_title="Frame",
        yaxis_title="Angle(rad)",
        template="plotly_dark",  # Koyu tema ("plotly", "plotly_white" da seçebilirsiniz)
        hovermode="x unified"    # Fare ile üzerine gelindiğinde tüm değerleri aynı anda gösterir
    )
    st.plotly_chart(graph, use_container_width=True)

@st.fragment
def Grid3dVisualizer():

    selection = st.selectbox(
        "Select the pose to visualize",
        options=[
            "Locomotion - Forward Kinematics Sample",
            "Grooming - Forward Kinematics Sample",
            "Visualize my uploaded pose data"
        ])
    
    fig=go.Figure()
    
    if selection == "Locomotion - Forward Kinematics Sample":
        poseData3D= locomotion_forward_kinematics
        AlignedPose3D=aligned_pos_locomotion
        st.info("Visualizing Locomotion data sample  from SeqIKPy forward kinematics\n\n____ Solid lines: Aligned pose\n\n------- Dashed lines: SeqIKPy Forward Kinematics")
        visualize(fig,AlignedPose3D,poseData3D)
        st.plotly_chart(fig, use_container_width=True)
        LJA_View(locomotion_leg_joint_angles)

    elif selection == "Grooming - Forward Kinematics Sample":
        poseData3D= grooming_forward_kinematics
        AlignedPose3D=aligned_pos_grooming
        st.info("Visualizing Grooming data sample from SeqIKPy forward kinematics\n\n____ Solid lines: Aligned pose\n\n------- Dashed lines: SeqIKPy Forward Kinematics")
        visualize(fig,AlignedPose3D,poseData3D)
        st.plotly_chart(fig, use_container_width=True)
        LJA_View(grooming_leg_joint_angles)

    elif selection== "Visualize my uploaded pose data":
        upload_area=st.empty()
        with upload_area:
            aligned_pose_upload_area, forward_kinematics_upload_area, lja_upload_area=st.columns([1,1,1])
            with aligned_pose_upload_area:
                uploaded_file_AL= st.file_uploader(
                    label=f"Load your aligned pose data here (Solid lines)", 
                    type=["pkl","h5","hdf","npy"])
            with forward_kinematics_upload_area:            
                uploaded_file_FK= st.file_uploader(
                    label=f"Load your forward kinematics data here (Dashed lines)", 
                    type=["pkl","h5","hdf","npy"])
            with lja_upload_area:
                uploaded_file_LJA= st.file_uploader(
                    label=f"Load your leg joint angles data here", 
                    type=["pkl","h5","hdf","npy"])
        
        if any([uploaded_file_AL,uploaded_file_FK,uploaded_file_LJA]):
            if st.button("Visualize Data"):
                try:
                    with st.spinner("Confirming data safety..."):
                        if uploaded_file_AL:
                            poseDataLoad_aligned=safe_load_uploaded_file(uploaded_file_AL)
                        if uploaded_file_FK:
                            poseDataLoad_fk=safe_load_uploaded_file(uploaded_file_FK)
                        if uploaded_file_LJA:
                            AngleDataLoad=safe_load_uploaded_file(uploaded_file_LJA)

                    with st.spinner("Validating data content..."):
                        if uploaded_file_AL:
                            poseDataValidate_aligned, summary_aligned = validate_scientific_data(poseDataLoad_aligned,data_type="pose",max_frames=20000,max_keypoints=100)
                            poseData3D_aligned=poseDataValidate_aligned
                        if uploaded_file_FK:
                            poseDataValidate_fk, summary_fk = validate_scientific_data(poseDataLoad_fk,data_type="forward_kinematics",max_frames=20000,max_keypoints=100)
                            poseData3D_fk=poseDataValidate_fk
                        if uploaded_file_LJA:
                            AngleDataValidate, summary = validate_scientific_data(AngleDataLoad,data_type="joint_angles",max_frames=20000)
                            AngleData=AngleDataValidate

                    st.success("Data succesfully loaded.")                       
                    st.info("Visualizing your data")
                    if uploaded_file_AL and uploaded_file_FK:
                        visualize(fig,poseData3D_aligned,poseData3D_fk)
                        st.plotly_chart(fig, use_container_width=True)
                    elif uploaded_file_AL:
                        visualize(fig,poseDataSolid=poseData3D_aligned)
                        st.plotly_chart(fig, use_container_width=True)
                    elif uploaded_file_FK:
                        visualize(fig,poseDataDashed=poseData3D_fk)
                        st.plotly_chart(fig, use_container_width=True)

                    if uploaded_file_LJA:
                        LJA_View(AngleData)
                        
                except Exception as e:
                    st.error(f"❌ Cannot visualize data: {e}")

@st.fragment
def ShowInfo():
    st.markdown("""
**Raw Pose:** the original pose data before processing and aligning. To visualize raw pose data, such as df3d, df3dpp or anipose, it must first be converted into a Python dictionary and then be aligned. 
\n\n**Aligned pose:** Pose data with a kinematic configuration that has been transformed and optimized to ensure spatial consistency with target reference frames and temporal continuity across the motion sequence.  It is kept as a Python dictionary, just like Forward kinematics data, leg joint angles data and converted dictionaries. For example:
`{“RF_leg”: np.array([[[ 2.14, -0.33, 0.56) with the shape of (frames, keypoints, 3(x-y-z))`
\n\n**Converted dictionary:** a Python dictionary that keeps raw pose data. To align a pose, run kinematics and visualize, raw pose data must be converted into a dictionary.
\n\n**Forward Kinematics (FK):** The mathematical process of determining the absolute position and orientation (pose) of fly model’s end-effector (or the tip of a kinematic chain) in 3D space, based on a given set of joint angles.
\n\n**Inverse Kinematics (IK):** the process of determining the required joint angles or displacement parameters of a fly model’s kinematic chain to place its end-effector (tip) at a specific desired position and orientation (pose) in 3D space.
\n\n**Leg joint angles data:** A Python dictionary that keeps joint angle values of te fly model. For example:
`{“Angle_RF_ThC_yaw”:np.array([0.31,  0.28,… (with the lengt of number of frames)`
                """)
    
with tab_visualize:
    left_space,right_space=st.columns([3,2])
    with left_space:
        Grid3dVisualizer()

with tab_process:
    left_space,right_space=st.columns([3,2])
    with left_space:
        with st.expander("Align Raw Pose Data"):
            try:
                alignment()
            except Exception as e:
                st.error(f"Cannot process data:{e}")
        with st.expander("Crop Your Aligned Pose Data"):
            try:
                crop()
            except Exception as e:
                st.error(f"Failed to process data: {e}")
        with st.expander("Run Inverse and Forward Kinematics on Aligned Pose Data"):
            try:
                IK_FK()
            except Exception as e:
                st.error(f"Cannot process data:{e}")
        with st.expander("Run Forward Kinematics on Leg Joint Angles Data"):
            try:
                FK()
            except Exception as e:
                st.error(f"Cannot process leg joint angles data:{e}")

with tab_help:
    left_space,right_space=st.columns([3,2])
    with left_space:
        ShowInfo()

