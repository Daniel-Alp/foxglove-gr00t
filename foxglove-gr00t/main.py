import json
import foxglove
import numpy as np
import os
import pandas as pd

from urchin import URDF
from foxglove.schemas import FrameTransform, FrameTransforms, Vector3, Quaternion, Timestamp

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

def convert(data_root: str, chunk: str, episode: str) -> None:
    urdf = URDF.load("./panda/panda.urdf")
    data_frame = pd.read_parquet(f'{data_root}/data/chunk-{chunk}/episode_{episode}.parquet', engine="pyarrow")
    
    modality = json.load(open(f'{data_root}/meta/modality.json'))
    state_desc = modality["state"]
    joint_pos_idxs = state_desc["joint_position"]
    joint_pos_idx_start = joint_pos_idxs["start"]
    joint_pos_idx_end   = joint_pos_idxs["end"]

    joint_pos_dict = {}
    
    with foxglove.open_mcap(f"{os.path.basename(data_root)}-{chunk}-{episode}-tf.mcap"):
        for _, row in data_frame.iterrows():
            sec_whole, sec_dec = divmod(row["timestamp"], 1)

            sec = int(sec_whole)
            nsec = int(sec_dec * 1_000_000_000)

            state = row["observation.state"]
            for i, joint in enumerate(urdf.joints):
                if i >= joint_pos_idx_end - joint_pos_idx_start:
                    continue
                joint_pos_dict[joint.name] = state[joint_pos_idx_start + i]
            fk_poses = urdf.link_fk(cfg=joint_pos_dict)

            transforms = []
            transforms.append(
                FrameTransform(
                    timestamp       = Timestamp(sec=sec, nsec=nsec),
                    parent_frame_id = "world",
                    child_frame_id  = "base",
                    translation     = Vector3(x=0.0, y=0.0, z=0.0),
                    rotation        = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )
            )

            for joint in urdf.joints:
                T_parent = fk_poses[urdf.link_map[joint.parent]]
                T_child = fk_poses[urdf.link_map[joint.child]]
                # Local transform from parent->child
                T_local = np.linalg.inv(T_parent) @ T_child
                trans = T_local[:3, 3]
                quat = rot_matrix_to_quat(T_local[:3, :3])
                transforms.append(
                    FrameTransform(
                        timestamp       = Timestamp(sec=sec, nsec=nsec),
                        parent_frame_id = joint.parent,
                        child_frame_id  = joint.child,
                        translation     = Vector3(x=float(trans[0]), y=float(trans[1]), z=float(trans[2])),
                        rotation        = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
                    )
                )

            foxglove.log(
                topic    = "/tf",
                message  = FrameTransforms(transforms=transforms)
            )

if __name__ == "__main__":
    data_root = "/home/alp/single_panda_gripper-TurnOffSinkFaucet"
    chunk = "000"
    episode = "000000"
    convert(data_root, chunk, episode)