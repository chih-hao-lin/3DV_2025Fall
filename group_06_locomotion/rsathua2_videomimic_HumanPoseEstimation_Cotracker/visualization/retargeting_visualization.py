# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import viser
import tyro
import pickle
import trimesh
import torch
import smplx
import numpy as onp
import cv2
import time
import h5py
import os.path as osp
import yourdfpy

from pathlib import Path
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.spatial.transform import Rotation as R
import viser_camera_util


# Visualize the pointcloud from megasam/monst3r
# Visualize the SMPL mesh
# Visualize the retargeted poses of the robot
# Visualize the background mesh

def load_dict_from_hdf5(h5file, path="/"):
    """f
    Recursively load a nested dictionary from an HDF5 file.
    
    Args:
        h5file: An open h5py.File object.
        path: The current path in the HDF5 file.
    
    Returns:
        A nested dictionary with the data.
    """
    result = {}
    for key in h5file[path].keys():
        key_path = f"{path}{key}"
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path + "/")
        else:
            result[key] = h5file[key_path][:]
    
    # Load attributes (scalars stored as attributes)
    for attr_key, attr_value in h5file.attrs.items():
        if attr_key.startswith(path):
            result[attr_key[len(path):]] = attr_value

    return result

def add_distance_measurement(server: viser.ViserServer) -> None:
    control0 = server.scene.add_transform_controls(
        "/controls/0",
        position=(1, 0, 0),
        scale=0.5,
        visible=False,
    )
    control1 = server.scene.add_transform_controls(
        "/controls/1",
        position=(1, 0, 0),
        scale=0.5,
        visible=False,
    )
    segments = server.scene.add_line_segments(
        "/controls/line",
        onp.array([control0.position, control1.position])[None, :, :],
        colors=(255, 0, 0),
        visible=False,
    )

    show_distance_tool = server.gui.add_checkbox("Show Distance Tool", False)
    distance_text = server.gui.add_text("Distance", initial_value="Distance: 0")

    @show_distance_tool.on_update
    def _(_) -> None:
        control0.visible = show_distance_tool.value
        control1.visible = show_distance_tool.value
        segments.visible = show_distance_tool.value

    def update_distance():
        distance = onp.linalg.norm(control0.position - control1.position)
        distance_text.value = f"Distance: {distance:.2f}"
        segments.points = onp.array([control0.position, control1.position])[None, :, :]

    control0.on_update(lambda _: update_distance())
    control1.on_update(lambda _: update_distance())

def main(
    megahunter_path: Path = None,
    postprocessed_dir: Path = None,
    robot_name: str = "g1",
    person_id: int = 1,
    bg_pc_downsample_factor: int = 4,
    confidence_threshold: float = 0.0,
    gender: str = 'male'
):
    """
    megahunter_path: contains the megahunter data which includes the pointcloud and the SMPL mesh
    postprocessed_dir: contains the postprocessed data which includes the retargeted poses of the robot and the background mesh
    robot_name: the name of the robot
    person_id: the id of the person in the megahunter data; Only supports one person for now
    bg_pc_downsample_factor: the factor to downsample the background pointcloud
    """
    if megahunter_path is None and postprocessed_dir is None:
        raise ValueError("At least one of megahunter_path or postprocessed_dir must be provided\n You should either visualize human motino or robot motion!!")

    if megahunter_path is None:
        # try to retrieve the megahunter path from the postprocessed dir
        megahunter_path = postprocessed_dir / '..' / '..' / 'output_smpl_and_points' / f'{postprocessed_dir.name}.h5'
        if not megahunter_path.exists():
            megahunter_path = None
            print(f"[Warning] The megahunter path {megahunter_path} does not exist")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Load the megahunter pickle file
    if megahunter_path is not None:
        with h5py.File(megahunter_path, 'r') as f:
            megahunter_data = load_dict_from_hdf5(f)

        world_env = megahunter_data['our_pred_world_cameras_and_structure']
        human_params_in_world = megahunter_data['our_pred_humans_smplx_params']

        person_id = list(human_params_in_world.keys())[0]

        # Extract the smpl data
        smpl_batch_layer = smplx.create(model_path = './assets/body_models', model_type = 'smpl', gender = gender, num_betas = 10, batch_size = len(human_params_in_world[person_id]['body_pose'])).to(device)

        num_frames = human_params_in_world[person_id]['body_pose'].shape[0]
        smpl_betas = torch.from_numpy(human_params_in_world[person_id]['betas']).to(device)
        if smpl_betas.ndim == 1:
            smpl_betas = smpl_betas.repeat(num_frames, 1)

        smpl_output_batch = smpl_batch_layer(body_pose=torch.from_numpy(human_params_in_world[person_id]['body_pose']).to(device), betas=smpl_betas, global_orient=torch.from_numpy(human_params_in_world[person_id]['global_orient']).to(device), pose2rot=False) #, left_hand_pose=human_params_in_world['left_hand_pose'], right_hand_pose=human_params_in_world['right_hand_pose'])
        smpl_joints = smpl_output_batch['joints']
        smpl_root_joint = smpl_joints[:, 0:1, :] # (len(frame_names), 1, 3)
        smpl_verts = smpl_output_batch['vertices'] - smpl_root_joint + torch.from_numpy(human_params_in_world[person_id]['root_transl']).to(device)

        # Need to apply the world rotation to the smpl joints and vertices
        smpl_joints3d = smpl_joints.detach().cpu().numpy() - smpl_root_joint.detach().cpu().numpy() + human_params_in_world[person_id]['root_transl']    
        smpl_verts = smpl_verts.detach().cpu().numpy()

        # maybe later
        # smpl_joints3d = smpl_joints3d @ world_rotation.T
        # smpl_verts = smpl_verts @ world_rotation.T

        # Refeactor the pointmaps and cameras to a numpy array
        bg_pt3ds = []
        fg_pt3ds = []
        bg_colors = []
        fg_colors = []
        bg_confs = []
        fg_confs = []
        cam2worlds = []
        kernel_size = 20
        kernel = onp.ones((kernel_size, kernel_size),onp.uint8)
        frame_names = megahunter_data['person_frame_info_list'][person_id]
        for frame_name in frame_names:
            frame_name = frame_name.item()
            pt3d = world_env[frame_name]['pts3d'].reshape(-1, 3)
            # pt3d = pt3d[::point_downsample_factor, :]
            conf = world_env[frame_name]['conf'].reshape(-1)
            # conf = conf[::point_downsample_factor]
            colors = world_env[frame_name]['rgbimg'].reshape(-1, 3)
            # colors = colors[::point_downsample_factor, :]
            # depths = world_env[frame_name]['depths']
            cam2world = world_env[frame_name]['cam2world']
            dynamic_msk = world_env[frame_name]['dynamic_msk'].astype(onp.uint8)
            dynamic_msk = cv2.dilate(dynamic_msk, kernel, iterations=1).flatten() > 0
            # dynamic_msk = dynamic_msk[::point_downsample_factor]

            bg_mask = ~dynamic_msk  
            fg_mask = dynamic_msk 
            bg_pt3ds.append(pt3d[bg_mask][::bg_pc_downsample_factor, :])
            bg_colors.append(colors[bg_mask][::bg_pc_downsample_factor, :])
            bg_confs.append(conf[bg_mask][::bg_pc_downsample_factor])

            fg_pt3ds.append(pt3d[fg_mask])
            fg_colors.append(colors[fg_mask])
            fg_confs.append(conf[fg_mask])
            cam2worlds.append(cam2world)

        try:
            assert len(smpl_joints3d) == len(bg_pt3ds) == len(bg_colors) == len(cam2worlds)
        except:
            print("[Warning] The number of frames in the world environment does not match the number of frames in the SMPL data. Trimming the world environment to match the SMPL data.")
            # trim referring to smpl_joints3d
            bg_pt3ds = bg_pt3ds[-len(smpl_joints3d):]
            bg_colors = bg_colors[-len(smpl_joints3d):]
            bg_confs = bg_confs[-len(smpl_joints3d):]
            fg_pt3ds = fg_pt3ds[-len(smpl_joints3d):]
            fg_colors = fg_colors[-len(smpl_joints3d):]
            fg_confs = fg_confs[-len(smpl_joints3d):]
            cam2worlds = cam2worlds[-len(smpl_joints3d):]
    else:
        num_frames = -1
        smpl_joints3d = None
        smpl_verts = None
        world_env = None

    if postprocessed_dir is not None:
        # Load the postprocessed data
        rotated_keypoints = postprocessed_dir / 'gravity_calibrated_keypoints.h5'
        with h5py.File(rotated_keypoints, 'r') as f:
            rotated_keypoints = load_dict_from_hdf5(f)
        
        # Load the background mesh
        background_mesh = postprocessed_dir / 'background_mesh.obj'
        background_mesh = trimesh.load(background_mesh)

        # Load the retargeted poses - check for both single and multi-person formats
        retargeted_poses_path = postprocessed_dir / f'retarget_poses_{robot_name}.h5'
        retargeted_poses_multiperson_path = postprocessed_dir / f'retarget_poses_{robot_name}_multiperson.h5'
        
        retargeted_poses = None
        is_multiperson = False
        
        # Try loading multi-person file first
        if retargeted_poses_multiperson_path.exists():
            try:
                with h5py.File(retargeted_poses_multiperson_path, 'r') as f:
                    retargeted_poses = load_dict_from_hdf5(f)
                    is_multiperson = True
                    print(f"Loaded multi-person retargeted poses from {retargeted_poses_multiperson_path}")
            except Exception as e:
                print(f"Failed to load multi-person file: {e}")
        
        # Fall back to single person file
        if retargeted_poses is None and retargeted_poses_path.exists():
            try:
                with h5py.File(retargeted_poses_path, 'r') as f:
                    retargeted_poses = load_dict_from_hdf5(f)
                    is_multiperson = False
                    print(f"Loaded single-person retargeted poses from {retargeted_poses_path}")
            except Exception as e:
                print(f"Failed to load single-person file: {e}")
        
        if retargeted_poses is None:
            raise ValueError(f"No retargeted poses file found")
        # "joint_names": actuated_joint_names,
        # "joints": onp.zeros((num_timesteps, actuated_num_joints)),
        # "root_quat": onp.zeros((num_timesteps, 4)),
        # "root_pos": onp.zeros((num_timesteps, 3)),
        # "link_names": link_names,
        # "link_pos": onp.zeros((num_timesteps, len(link_names), 3)),
        # "link_quat": onp.zeros((num_timesteps, len(link_names), 4)),
        # "contacts": 
        #     "left_foot": onp.zeros((num_timesteps,), dtype=bool),
        #     "right_foot": onp.zeros((num_timesteps,), dtype=bool)

        # Determine number of frames based on whether it's multi-person or single-person
        if is_multiperson:
            # For multi-person, get max frames across all persons
            person_ids = list(retargeted_poses["persons"].keys())
            num_frames = max(retargeted_poses["persons"][pid]["joints"].shape[0] for pid in person_ids)
            print(f"Multi-person mode: {len(person_ids)} persons detected")
        else:
            # Single person format
            if num_frames != -1 and not is_multiperson and num_frames != retargeted_poses["joints"].shape[0]:
                raise ValueError(f"The number of frames in the retargeted poses ({retargeted_poses['joints'].shape[0]}) does not match the number of frames in the postprocessed data ({num_frames})")
            elif num_frames == -1:
                num_frames = retargeted_poses["joints"].shape[0]

        # Apply the world rotation to the smpl joints and vertices and pointmaps and cameras
        # extract the world rotation from the rotated_keypoints
        world_rotation = rotated_keypoints['world_rotation']
        # Apply the world rotation to the smpl joints and vertices
        if smpl_joints3d is not None and smpl_verts is not None and world_env is not None:
            if not is_multiperson:
                assert len(smpl_joints3d) == len(retargeted_poses["joints"])
            smpl_joints3d = smpl_joints3d @ world_rotation.T
            smpl_verts = smpl_verts @ world_rotation.T

            bg_pt3ds = [pt3d @ world_rotation.T for pt3d in bg_pt3ds]
            fg_pt3ds = [pt3d @ world_rotation.T for pt3d in fg_pt3ds]

            world_rotation_trans = onp.eye(4)
            world_rotation_trans[:3, :3] = world_rotation # double check if this is correct
            cam2worlds = [world_rotation_trans @ cam2world for cam2world in cam2worlds]

    # Ok, let's start visualizing
    # Start viser server.
    server = viser.ViserServer(port=8081)

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=15
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)
    
    

    # Visualize the background mesh
    if background_mesh is not None:
        # Add background mesh to viser
        bg_mesh_handle =server.scene.add_mesh_simple(
            name="/bg_mesh",
            vertices=background_mesh.vertices,
            faces=background_mesh.faces,
            color=(200, 200, 200),  # Light gray color
            opacity=1.0,
            material="standard",
            flat_shading=False,
            side="double",  # Render both sides of the mesh
            visible=True
        )

    # Visualize the background pointcloud and camera
    # this is visualizing static components
    if world_env is not None:
        bg_pt3ds = onp.concatenate(bg_pt3ds, axis=0)
        bg_colors = onp.concatenate(bg_colors, axis=0)
        bg_confs = onp.concatenate(bg_confs, axis=0)
        # cam2worlds = onp.concatenate(cam2worlds, axis=0)

        bg_conf_mask = bg_confs >= confidence_threshold

        # Visualize the background pointcloud
        bg_pt3ds_handle = server.scene.add_point_cloud(
            name="/bg_pt3ds",
            points=bg_pt3ds[bg_conf_mask],
            colors=bg_colors[bg_conf_mask],
            point_size=0.005,
        )

    # Visualize the robot(s)
    if retargeted_poses is not None:
        # Load URDF.
        if robot_name == "h1":
            raise ValueError("H1 is not supported yet")
        elif robot_name == "g1":
            # urdf = load_robot_description("g1_description")
            urdf_path = osp.join(osp.dirname(__file__), "../assets/robot_asset/g1/g1_29dof_anneal_23dof_foot.urdf")
            urdf = yourdfpy.URDF.load(urdf_path)
        else:
            raise ValueError(f"Robot {robot_name} is not supported yet")


        # Create robot frames and urdf visers for all persons
        robot_frames = {}
        urdf_visers = {}
        
        if is_multiperson:
            for person_id in person_ids:
                robot_frames[person_id] = server.scene.add_frame(f"/robot_{person_id}", axes_length=0.2, show_axes=False)
                urdf_visers[person_id] = ViserUrdf(
                    server,
                    urdf_or_path=urdf,
                    root_node_name=f"/robot_{person_id}",
                )
        else:
            # Single robot for backward compatibility
            robot_frames["single"] = server.scene.add_frame("/robot", axes_length=0.2, show_axes=False)
            urdf_visers["single"] = ViserUrdf(
                server,
                urdf_or_path=urdf,
                root_node_name="/robot",
            )

        # Update root frame(s).
        def update_robot_cfg(t: int):
            if is_multiperson:
                # Update multiple robots
                for idx, person_id in enumerate(person_ids):
                    person_data = retargeted_poses["persons"][person_id]
                    
                    if t >= person_data["joints"].shape[0]:
                        # Hide robot if timestep exceeds this person's data
                        robot_frames[person_id].visible = False
                        continue
                    
                    if not gui_show_robot.value:
                        robot_frames[person_id].visible = False
                        for joint_frame in urdf_visers[person_id]._joint_frames:
                            joint_frame.visible = False
                        for mesh_node in urdf_visers[person_id]._meshes:
                            mesh_node.visible = False
                    else:
                        robot_frames[person_id].visible = True
                        for joint_frame in urdf_visers[person_id]._joint_frames:
                            joint_frame.visible = True
                        for mesh_node in urdf_visers[person_id]._meshes:
                            mesh_node.visible = True
                        
                        T_world_robot_xyzw = person_data["root_quat"][t] # xyzw
                        T_world_robot_xyz = person_data["root_pos"][t] # xyz
                        T_world_robot_wxyz = onp.concatenate([T_world_robot_xyzw[3:], T_world_robot_xyzw[:3]])
                        
                        robot_frames[person_id].wxyz = onp.array(T_world_robot_wxyz)
                        robot_frames[person_id].position = onp.array(T_world_robot_xyz)
                        
                        # Update joints
                        joints = onp.array(person_data["joints"][t])
                        joints[8] = 0.0  # TEMP
                        urdf_visers[person_id].update_cfg(joints)
            else:
                # Single robot update
                if not gui_show_robot.value:
                    robot_frames["single"].wxyz = onp.array([1, 0, 0, 0])
                    robot_frames["single"].position = onp.array([0, 0, 0])
                    for joint_frame in urdf_visers["single"]._joint_frames:
                        joint_frame.visible = False
                    for mesh_node in urdf_visers["single"]._meshes:
                        mesh_node.visible = False
                    return
                else:
                    for joint_frame in urdf_visers["single"]._joint_frames:
                        joint_frame.visible = True
                    for mesh_node in urdf_visers["single"]._meshes:
                        mesh_node.visible = True

                    T_world_robot_xyzw = retargeted_poses["root_quat"][t] # xyzw
                    T_world_robot_xyz = retargeted_poses["root_pos"][t] # xyz
                    T_world_robot_wxyz = onp.concatenate([T_world_robot_xyzw[3:], T_world_robot_xyzw[:3]])

                    robot_frames["single"].wxyz = onp.array(T_world_robot_wxyz)
                    robot_frames["single"].position = onp.array(T_world_robot_xyz)

                    # Update joints.
                    # TEMP
                    retargeted_poses["joints"][t][8] = 0.0
                    urdf_visers["single"].update_cfg(onp.array(retargeted_poses["joints"][t]))

    # Visualize the camera and human pointcloud
    if world_env is not None:
        # Visualize the camera
        # rot_180 = onp.eye(3)
        # rot_180[1, 1] = -1
        # rot_180[2, 2] = -1

        vfov_rad_list = []
        aspect_list = []
        rgbimg_list = []
        quat_list = []
        trans_list = []
        for frame_name, camera in zip(world_env.keys(), cam2worlds):
            # camera[:3, :3] = rot_180 @ camera[:3, :3] 
            # camera[:3, 3] = camera[:3, 3] @ rot_180
            
            # rotation matrix to quaternion
            quat = R.from_matrix(camera[:3, :3]).as_quat()
            # xyzw to wxyz
            quat = onp.concatenate([quat[3:], quat[:3]])
            # translation vector
            trans = camera[:3, 3]

            # add camera frustum
            rgbimg = world_env[frame_name]['rgbimg']
            rgbimg = rgbimg[::bg_pc_downsample_factor, ::bg_pc_downsample_factor, :]
            K = world_env[frame_name]['intrinsic']
            # fov_rad = 2 * np.arctan(intrinsics_K[0, 2] / intrinsics_K[0, 0])DA
            assert K.shape == (3, 3)
            vfov_rad = 2 * onp.arctan(K[1, 2] / K[1, 1])
            aspect = rgbimg.shape[1] / rgbimg.shape[0]

            vfov_rad_list.append(vfov_rad)
            aspect_list.append(aspect)
            rgbimg_list.append(rgbimg)
            quat_list.append(quat)
            trans_list.append(trans)

        camera_frustums = []
        def update_camera_frustum(t: int):
            if len(camera_frustums) <= t:
                camera_frustm = server.scene.add_camera_frustum(
                    f"/cameras/{t}",
                    vfov_rad_list[t],
                    aspect_list[t],
                    scale=0.1, #gui_frustum_scale.value,
                    line_width=1.0, #gui_line_width.value,
                    color=(255, 127, 14), #gui_frustum_ours_color.value,
                    wxyz=quat_list[t],
                    position=trans_list[t],
                    image=rgbimg_list[t],
                )
                camera_frustums.append(camera_frustm)
            else:
                for camera_frustm in camera_frustums:
                    camera_frustm.visible = False
                if 'gui_play_camera_to_follow' in locals() and not gui_play_camera_to_follow.value:
                    camera_frustums[t].visible = True

        # Visualize the human pointcloud
        human_pts3d_handle = []
        def update_human_pt3ds(t: int):
            fg_conf_mask = fg_confs[t] >= confidence_threshold
            if len(human_pts3d_handle) <= t:
                human_pt3ds_handle = server.scene.add_point_cloud(
                    f"/human_pt3ds/{t}",
                    points=fg_pt3ds[t][fg_conf_mask],
                    colors=fg_colors[t][fg_conf_mask],
                    point_size=0.005, #gui_point_size.value,
                )
                human_pts3d_handle.append(human_pt3ds_handle)
            else:
                for human_pt3ds_handle in human_pts3d_handle:
                    human_pt3ds_handle.visible = False
            
                if 'gui_show_fg_pt3ds' in locals() and gui_show_fg_pt3ds.value:
                    human_pts3d_handle[t].visible = True
        
    # Visualize the smpl mesh
    if smpl_verts is not None:
        smpl_mesh_handle_list = []
        def update_smpl_mesh(t: int):
            if len(smpl_mesh_handle_list) <= t:
                smpl_mesh_handle = server.scene.add_mesh_simple(
                    f"/smpl_mesh/{t}",
                    vertices=smpl_verts[t],
                    faces=smpl_batch_layer.faces,
                    flat_shading=False,
                    wireframe=False,
                    color=(255, 215, 0),
                    visible=True,
                )
                smpl_mesh_handle_list.append(smpl_mesh_handle)
            else:
                for smpl_mesh_handle in smpl_mesh_handle_list:
                    smpl_mesh_handle.visible = False
                
                if 'gui_show_smpl_mesh' in locals() and gui_show_smpl_mesh.value:
                    smpl_mesh_handle_list[t].visible = True
    
    # Visualize the joints
    if smpl_joints3d is not None:
        smpl_joints3d_handle_list = []
        def update_smpl_joints(t: int):
            if len(smpl_joints3d_handle_list) <= t:
                smpl_joints3d_handle = server.scene.add_point_cloud(
                    f"/smpl_joints3d/{t}",
                    points=smpl_joints3d[t],
                    colors=onp.array([[128, 0, 128]] * smpl_joints3d[t].shape[0]),
                    point_size=0.03,
                    point_shape="circle",
                )
                smpl_joints3d_handle_list.append(smpl_joints3d_handle)
            else:
                for smpl_joints3d_handle in smpl_joints3d_handle_list:
                    smpl_joints3d_handle.visible = False
                
                if 'gui_show_smpl_joints' in locals() and gui_show_smpl_joints.value:
                    smpl_joints3d_handle_list[t].visible = True
    

    # Initialize camera follow functions
    stop_camera_follow = None
    resume_camera_follow = None
    update_camera_follow_called = False
    
    def setup_camera_follow():
        nonlocal stop_camera_follow, resume_camera_follow, update_camera_follow_called
        if update_camera_follow_called:
            return
        update_camera_follow_called = True
        
        for camera_frustm in camera_frustums:
            camera_frustm.visible = False

        target_positions = onp.array([smpl_mesh_handle_list[i].vertices.mean(axis=0) for i in range(num_frames)])
    
        # Calculate average FOV from camera intrinsics if available
        if world_env is not None:
            fov_degrees_list = []
            
            for frame_name in world_env.keys():
                K = world_env[frame_name]['intrinsic']
                # Calculate vertical FOV from intrinsics
                vfov_rad = 2 * onp.arctan(K[1, 2] / K[1, 1])
                vfov_degrees = onp.degrees(vfov_rad)
                fov_degrees_list.append(vfov_degrees)
            avg_fov = onp.mean(fov_degrees_list)
        else:
            avg_fov = 45.0  # Default FOV
        
        # Set up camera to follow the target position
        stop_camera_follow, resume_camera_follow = viser_camera_util.setup_camera_follow(
            server=server,
            slider=gui_timestep,
            target_positions=target_positions,
            camera_positions=trans_list,
            camera_wxyz=quat_list,
            fov=avg_fov
        )

    # Update the scene
    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        # with server.atomic():
        #     frame_nodes[current_timestep].visible = True
        #     frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep

        with server.atomic():
            if retargeted_poses is not None:
                update_robot_cfg(current_timestep)

            if background_mesh is not None and 'gui_show_bg_mesh' in locals() and gui_show_bg_mesh.value:
                bg_mesh_handle.visible = True
            elif background_mesh is not None:
                bg_mesh_handle.visible = False

            if world_env is not None and 'gui_show_bg_pt3ds' in locals() and gui_show_bg_pt3ds.value:
                bg_pt3ds_handle.visible = True
            elif world_env is not None:
                bg_pt3ds_handle.visible = False

            if world_env is not None:
                update_camera_frustum(current_timestep)
                update_human_pt3ds(current_timestep)
            
            if smpl_joints3d is not None and smpl_verts is not None:
                update_smpl_mesh(current_timestep)
                update_smpl_joints(current_timestep)
        
            if len(smpl_mesh_handle_list) == num_frames or len(smpl_joints3d_handle_list) == num_frames:
                setup_camera_follow()

        server.flush()  # Optional!

    # Initialize GUI controls after all scene elements are created
    with server.gui.add_folder("Scene Elements"):
        if background_mesh is not None:
            gui_show_bg_mesh = server.gui.add_checkbox("Show Bg Mesh", False)
        if world_env is not None:
            gui_show_bg_pt3ds = server.gui.add_checkbox("Show Bg Pointcloud", True)
            gui_show_fg_pt3ds = server.gui.add_checkbox("Show Human Pointcloud", True)
        if retargeted_poses is not None:
            robot_label = "Show Robot(s)" if is_multiperson else "Show Robot"
            gui_show_robot = server.gui.add_checkbox(robot_label, False)
    
    with server.gui.add_folder("Human Visualization"):
        if smpl_verts is not None:
            gui_show_smpl_mesh = server.gui.add_checkbox("Show Smpl Mesh", False)
        if smpl_joints3d is not None:
            gui_show_smpl_joints = server.gui.add_checkbox("Show Smpl Joints", False)
    
    with server.gui.add_folder("Camera Controls"):
        if world_env is not None and smpl_verts is not None:
            gui_play_camera_to_follow = server.gui.add_checkbox("Play Camera to Follow", initial_value=False)
            @gui_play_camera_to_follow.on_update
            def _(_) -> None:
                if stop_camera_follow is not None and resume_camera_follow is not None:
                    if gui_play_camera_to_follow.value:
                        resume_camera_follow()
                    else:
                        stop_camera_follow()
        else:
            gui_play_camera_to_follow = None
    
    add_distance_measurement(server)
    
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


    # Visualize the robot
if __name__ == "__main__":
    tyro.cli(main)
