import argparse
import json
import numpy as np
import pandas as pd
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from google.protobuf.timestamp_pb2 import Timestamp
from pathlib import Path
from mcap_protobuf.writer import Writer
from urchin import URDF

from itertools import chain

def rot_matrix_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=float)

def convert(data_root: Path, chunk: str, episode: str) -> None:
    urdf = URDF.load("./fr3_franka_hand/fr3_franka_hand.urdf")
    data_frame = pd.read_parquet(f'{data_root}/data/chunk-{chunk}/episode_{episode}.parquet', engine="pyarrow")
    
    modality = json.load(open(f'{data_root}/meta/modality.json'))
    state_desc = modality["state"]
    
    joint_pos_idx_start = state_desc["joint_position"]["start"]
    joint_pos_idx_end = state_desc["joint_position"]["end"]
    # the arm joints' names in the order they are in observation.state (see below)
    joint_names = [f"fr3_joint{i}" for i in range(1, 8)]

    gripper_pos_idx_start = state_desc["gripper_qpos"]["start"]
    gripper_pos_idx_end = state_desc["gripper_qpos"]["end"]
    # the gripper joints' names in the order they are in observations.state (see below)
    gripper_names = ["fr3_finger_joint1", "fr3_finger_joint2"]

    joint_vel_idx_start = state_desc["joint_velocity"]["start"]
    joint_vel_idx_end = state_desc["joint_velocity"]["end"]
    
    with open(f"{data_root.name}-{chunk}-{episode}-state.mcap", "wb") as stream, Writer(stream) as writer:
        for _, row in data_frame.iterrows():
            sec_whole, sec_dec = divmod(row["timestamp"], 1)
            sec = int(sec_whole)
            nsec = int(sec_dec * 1_000_000_000)

            timestamp_ns = int(row["timestamp"] * 1_000_000_000)

            state = row["observation.state"]
            
            joint_pos_values = state[joint_pos_idx_start:joint_pos_idx_end]

            gripper_pos_values = state[gripper_pos_idx_start:gripper_pos_idx_end]

            # dict(zip(list1, list2)) uses list1 as keys and list2 as vals
            # chain(zip(...), zip(...)) is used to create a single iterable
            joint_pos_dict = dict(chain(zip(joint_names, joint_pos_values), 
                                        zip(gripper_names, gripper_pos_values)))

            fk_poses = urdf.link_fk(cfg=joint_pos_dict)

            transforms = []
            transforms.append(
                FrameTransform(
                    timestamp       = Timestamp(seconds=sec, nanos=nsec),
                    parent_frame_id = "world",
                    child_frame_id  = "base",
                    translation     = Vector3(x=0.0, y=0.0, z=0.0),
                    rotation        = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )
            )

            for joint in urdf.joints:
                T_parent = fk_poses[urdf.link_map[joint.parent]]
                T_child = fk_poses[urdf.link_map[joint.child]]
                T_local = np.linalg.inv(T_parent) @ T_child
                trans = T_local[:3, 3]
                quat = rot_matrix_to_quat(T_local[:3, :3])
                transforms.append(
                    FrameTransform(
                        timestamp       = Timestamp(seconds=sec, nanos=nsec),
                        parent_frame_id = joint.parent,
                        child_frame_id  = joint.child,
                        translation     = Vector3(x=trans[0], y=trans[1], z=trans[2]),
                        rotation        = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
                    )
                )

                writer.write_message(
                    topic        = "/tf",
                    message      = FrameTransforms(transforms=transforms),
                    log_time     = timestamp_ns,
                    publish_time = timestamp_ns
                )

            joint_vel_values = state[joint_vel_idx_start:joint_vel_idx_end]
            for i,v in enumerate(joint_vel_values):
                writer.write_message(
                    topic        = f"/arm/joint_velocities/joint{i}",
                    message      = Vector3(x=0, y=0, z=v),
                    log_time     = timestamp_ns,
                    publish_time = timestamp_ns
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, help="root directory of dataset")
    parser.add_argument("chunk", help="chunk number")
    parser.add_argument("episode", help="episode number")
    args = parser.parse_args()
    convert(args.data_root, args.chunk, args.episode)