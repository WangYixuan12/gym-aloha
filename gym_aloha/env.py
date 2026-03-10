import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces
import transforms3d

from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask, PushTTask
from gym_aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose, sample_pusht_pose


class AlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        # self.curr_vel = np.zeros(14)
        # self.k_p, self.k_v = 100, 20  # PD control
        # self.dt = 1/30.
        # self.acc_lim = 10

        self._env = self._make_env_task(self.task)

        if self.obs_type == "state":
            raise NotImplementedError()
            self.observation_space = spaces.Box(
                low=np.array([0] * len(JOINTS)),  # ???
                high=np.array([255] * len(JOINTS)),  # ???
                dtype=np.float64,
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "top": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)

    def render(self, camera_ids=["top"]):
        return self._render(visualize=True, camera_ids=camera_ids)

    def _render(self, visualize=False, camera_ids=["top"]):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        render_dict = {}
        for camera_id in camera_ids:
            render_dict[camera_id] = self._env.physics.render(height=height, width=width, camera_id=camera_id)
        return render_dict

    def get_cam_intrinsic(self, camera_id: str, img_shape: tuple[int, int]) -> np.ndarray:
        """
        Returns 3x3 intrinsic matrix K for an MJCF-defined camera.
        Focal is derived from MuJoCo's vertical FOV at the requested resolution.
        """
        physics = self._env.physics
        m = physics.model
        height, width = img_shape

        # camera numeric id
        cam_id = mujoco.mj_name2id(m._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)

        # MuJoCo stores vertical FOV in degrees
        fovy_deg = float(m.cam_fovy[cam_id])
        fovy = np.deg2rad(fovy_deg)

        # standard pinhole: fy from vertical FOV; fx from aspect
        fy = 0.5 * height / np.tan(0.5 * fovy)
        fovx = 2.0 * np.arctan((width / height) * np.tan(0.5 * fovy))
        fx = 0.5 * width / np.tan(0.5 * fovx)

        # principal point at image center (MuJoCo uses square pixels; centered frustum)
        cx, cy = (width - 1) * 0.5, (height - 1) * 0.5

        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return K

    # === New: camera extrinsics in OpenCV convention ===
    def get_cam_extrinsic(self, camera_id: str) -> np.ndarray:
        """
        Returns (R, t) where X_cam = R * X_world + t, using OpenCV camera axes:
          +x right, +y down, +z forward.
        """
        physics = self._env.physics
        m, d = physics.model, physics.data

        # camera numeric id and attachment body
        cam_id = mujoco.mj_name2id(m._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        body_id = int(m.cam_bodyid[cam_id])

        # camera pose relative to its body (model coordinates)
        t_b_c = np.array(m.cam_pos[cam_id], dtype=np.float64)     # (3,)
        q_b_c = np.array(m.cam_quat[cam_id], dtype=np.float64)    # (w,x,y,z)
        R_b_c = transforms3d.quaternions.quat2mat(q_b_c)

        # body world pose at the current step (data "x*" = global/world frame)
        R_w_b = np.array(d.xmat[body_id]).reshape(3, 3)
        t_w_b = np.array(d.xpos[body_id])

        # compose: camera pose in world (MuJoCo camera axes)
        R_w_c_mj = R_w_b @ R_b_c
        t_w_c    = t_w_b + R_w_b @ t_b_c

        # world->camera extrinsics in MuJoCo camera axes
        R_c_w_mj = R_w_c_mj.T
        t_c_w_mj = -R_c_w_mj @ t_w_c

        # Convert to OpenCV axes: (x,y,z)_cv = ( +x, -y, -z )_mj
        A = np.diag([1.0, -1.0, -1.0])             # axis flip matrix
        R_c_w_cv = A @ R_c_w_mj
        t_c_w_cv = A @ t_c_w_mj

        pose = np.eye(4)
        pose[:3, :3] = R_c_w_cv
        pose[:3, 3] = t_c_w_cv

        return pose

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = ASSETS_DIR / "aloha" / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif task_name == "insertion":
            xml_path = ASSETS_DIR / "aloha" / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "pusht":
            xml_path = ASSETS_DIR / "aloha" / "bimanual_viperx_pusht.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = PushTTask(0)
        elif task_name == "end_effector_transfer_cube":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif task_name == "end_effector_insertion":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {}
            for k in raw_obs["images"]:
                obs[k] = raw_obs["images"][k].copy()
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {"top": raw_obs["images"]["top"].copy()},
                "agent_pos": raw_obs["qpos"],
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        elif self.task == "pusht":
            BOX_POSE[0] = sample_pusht_pose(seed)  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?
        
        # DEBUG ONLY: add noise to the action
        # if hasattr(self, "plotter"):
        #     if self.iter // 10 == 0 or self.iter % 10 == 1:
        #         action[8] += 0.5
        
        # obs = self._env.task.get_observation(self._env.physics)  # noqa
        # self.curr_pos = obs["qpos"].copy()
        # acceleration = self.k_p * (action - self.curr_pos) + self.k_v * (
        #     np.zeros(14) - self.curr_vel
        # )
        # acceleration = np.clip(acceleration, -self.acc_lim, self.acc_lim)
        # self.curr_vel += acceleration * self.dt
        # pid_action = self.curr_pos + self.curr_vel * self.dt
        
        # ### DEBUG ONLY
        # if not hasattr(self, "plotter"):
        #     self.plotter = RealTimePlotter(
        #         title="PD control",
        #         window_size=300,
        #         num_lines=2,
        #         y_max=1,
        #         y_min=-1,
        #         legends=["target", "pid"],
        #     )
        #     self.iter = 0
        # self.plotter.append(
        #     np.array([self.iter]),
        #     np.stack([action, pid_action], axis=1)[8:9],
        # )
        # self.iter += 1
        # ### END OF DEBUG ONLY
                
        _, reward, _, raw_obs = self._env.step(action)

        # TODO(rcadene): add an enum
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
