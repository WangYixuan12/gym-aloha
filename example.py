import imageio
import numpy as np
from gym_aloha.env import AlohaEnv
import transforms3d

env = AlohaEnv(task="transfer_cube")
observation, info = env.reset()
frames = []
wrist_frames = []

for _ in range(100):
    # action = env.action_space.sample()
    euler = np.array([0.0, np.pi / 4.0, 0.0])
    quat = transforms3d.euler.euler2quat(euler[0], euler[1], euler[2])
    min_sample = np.array([0.0, 0.0, 0.1, 0.8, 0.8, 0.8, 0.8, -0.8, -0.05, 0.1, 0.0, 0.0, 0.0, 1.0])
    max_sample = np.array([0.1, 0.0, 0.2, 1.0, 1.0, 1.0, 1.0, -0.7, 0.05, 0.2, 0.0, 0.0, 0.0, 1.0])
    action = np.random.uniform(min_sample, max_sample)
    # action[3:6] = quat[1:]
    # action[6:7] = quat[0]
    action[3:7] = quat
    observation, reward, terminated, truncated, info = env.step(action)
    render_dict = env.render(["top", "left_wrist", "right_wrist", "front_close"])
    gripper_pos = env._env.physics.data.body('vx300s_left/gripper_link').xpos
    gripper_quat = env._env.physics.data.body('vx300s_left/gripper_link').xquat
    wrist_pos = env._env.physics.data.camera('left_wrist').xpos
    wrist_mat = env._env.physics.data.camera('left_wrist').xmat
    wrist_pose = np.eye(4)
    gripper_pose = np.eye(4)
    wrist_pose[:3, :3] = wrist_mat.reshape(3,3)
    wrist_pose[:3, 3] = wrist_pos
    gripper_pose[:3, 3] = gripper_pos
    gripper_pose[:3, :3] = transforms3d.quaternions.quat2mat(gripper_quat)
    gripper_t_wrist = np.linalg.inv(gripper_pose) @ wrist_pose
    print(gripper_t_wrist)
    print("gripper euler:", transforms3d.euler.mat2euler(gripper_t_wrist[:3, :3]))
    frames.append(render_dict["front_close"])
    # frames.append(render_dict["top"])
    wrist_frames.append(render_dict["left_wrist"])

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
imageio.mimsave("example_wrist.mp4", np.stack(wrist_frames), fps=25)
