import mujoco
import numpy as np
import os
import gym
from gym import spaces
import mujoco_py
import xml.etree.ElementTree as ET
import math
import random
from scipy.spatial.transform import Rotation as R

# Convert the observation of the environment into the observation space with its limits

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class AIconGrapper(gym.Env):
    """
    This environment can be used to train an RL agent to control a UR5e robot 
    arm to reach a target position while minimizing the torque used
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = 'human',
                 simulation_frames=4
                 ):

        # Init configurations
        self.simulation_frames = simulation_frames
        self.render_modes = ['human']

        # limites del target a alcanzar ****** we need to make randomized this part
        self.target_bounds = np.array(
            ((-0.20, 0.20), (-0.15, 0.15)), dtype=object)
        
        # Init model
        model_filename = "UR5gripper_v3.xml"
        current_dir = os.path.dirname(__file__)
        fullpath = os.path.join(current_dir, model_filename)
        if not os.path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        # Init simulation
        self.robot = mujoco.MjModel.from_xml_path(fullpath)
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None  # Initialize the viewer attribute
        model = mujoco_py.load_model_from_path(fullpath)
        sim1 = mujoco_py.MjSim(model)
        self.num_actuators = self.sim.model.nu
        self.steps = 0
        self.jvel = []
        self.phase = 0
        self.has_grasp=False
        self.init_qpos = sim1.data.qpos  
        print('self.init_qpos',self.init_qpos)
        self.init_qvel =  sim1.data.qvel 
        print('self.init_qvel',self.init_qvel)
        startindex, endindex = self.sim.model.get_joint_qpos_addr('box_1_joint')
        free_joint_qpos_data = self.sim.data.qpos[startindex:endindex]
        free_position = free_joint_qpos_data[:3]
        print("free_position",free_position)
        free_orientation = free_joint_qpos_data[3:]
        print("free_orientation",free_orientation)
        self.num_actuators = self.sim.model.nu
        self.qpos_bounds = np.array(
            ((-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)), dtype=object)  # rango de articulaciones

        # config action space
        bounds = self.robot.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        observation = self.get_observation() 

        self.observation_space = convert_observation_to_space(observation)
        Test_body = self.sim.model.body_name2id('box_3')
        body_inertia = self.sim.model.body_inertia[Test_body]
        print('body_inertia',body_inertia)
        
        Test_geom = self.sim.model.geom_name2id('brick_geom3')
        geom_size = self.sim.model.geom_size[Test_geom]
        print('geom_size',geom_size)
        self.reset()
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        if mode == 'human':
            self.viewer.render()
        elif mode == 'rgb_array':
            # Get the data from the viewer's render function for offscreen rendering
            # This might require setting up an offscreen window; refer to mujoco_py documentation
            data = self.viewer.read_pixels(width=800, height=600, depth=False)
            # Return RGB array suitable for video
            return data
        else:
            raise ValueError('Unsupported render mode: {}'.format(mode))

    def close(self):
        if self.viewer is not None:
            # Destroy the viewer if it exists
            self.viewer = None
            mujoco_py.cymj.MjViewer.finish()
    
    
    def reset(self):
        self.steps=0
        self.grasp = False
        self.has_grasp=False
        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = self.init_qvel
        self.phase = 0
        joint_velocities_array = np.array(self.jvel)
        mean_joint_velocities = np.mean(joint_velocities_array, axis=0)
        self.sim.forward()

        return self.get_observation(), mean_joint_velocities

    def step(self, action):

        # Init variables
        self.done = False
        reward = 0

        action = np.clip(action, self.action_space.low, self.action_space.high)

        while np.any(np.isinf(action)):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        self.steps = self.steps +1

        observation = self.get_observation()
        reward = self.compute_reward(observation, action)
        
        body1_IDt = self.sim.data.get_body_xvelp("shoulder_link")
        body1_velocities = np.array(body1_IDt)
        vel1 = np.linalg.norm(body1_velocities)
        body2_IDt = self.sim.data.get_body_xvelp("upper_arm_link")
        body2_velocities = np.array(body2_IDt)
        vel2 = np.linalg.norm(body2_velocities)
        body3_IDt = self.sim.data.get_body_xvelp("forearm_link")
        body3_velocities = np.array(body3_IDt)
        vel3 = np.linalg.norm(body3_velocities)
        body4_IDt = self.sim.data.get_body_xvelp("wrist_1_link")
        body4_velocities = np.array(body4_IDt)
        vel4 = np.linalg.norm(body4_velocities)
        body5_ID = self.sim.data.get_body_xvelp("wrist_2_link")
        body5_velocities = np.array(body5_ID)
        vel5 = np.linalg.norm(body5_velocities)
        body6_ID = self.sim.data.get_body_xvelp("wrist_3_link")
        body6_velocities = np.array(body6_ID)
        vel6 = np.linalg.norm(body6_velocities)
        body7_ID = self.sim.data.get_body_xvelp("ee_link")
        body7_velocities = np.array(body7_ID)
        vel7 = np.linalg.norm(body7_velocities)
        
        body_vel = [vel1, vel2, vel3, vel4, vel5, vel6, vel7]
        self.jvel.append(body_vel)
        self.do_simulation(action, self.simulation_frames)

        info = self.get_info()
        if self.render_mode == "human":
            self.render()

        
        return observation, reward, self.done, info
        

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )




    def get_observation(self):
        # 1. End-Effector Position (3D)
        left_finger = self.sim.data.get_body_xpos("left_inner_finger")
        right_finger = self.sim.data.get_body_xpos("right_inner_finger")
        gripper_position = (left_finger + right_finger) / 2
        
        # 2. Target Brick Position (3D)
        # (Assuming box_1 is the active brick for now. Change 'box_1' to 'box_2' for phase 2)
        target_position = self.sim.data.get_geom_xpos('box_1')
        
        # 3. System Joint Positions (20D)
        # 13 Dimensions for the Robot (Husky 5 + UR5e 6 + Gripper 2)
        # Since the robot is included first in the XML, its qpos indices are 0 to 13
        robot_qpos = self.sim.data.qpos[:13].copy().astype(np.float32) 
        
        # 7 Dimensions for the Active Brick (3 Position + 4 Quaternion)
        startindex_qpos, endindex_qpos = self.sim.model.get_joint_qpos_addr('box_1_joint')
        active_brick_qpos = self.sim.data.qpos[startindex_qpos:endindex_qpos].copy().astype(np.float32)
        
        joints_position = np.concatenate((robot_qpos, active_brick_qpos)) # Exact 20D
        
        # 4. System Joint Velocities (19D)
        # 13 Dimensions for Robot velocities
        robot_qvel = self.sim.data.qvel[:13].copy().astype(np.float32)
        
        # 6 Dimensions for Active Brick velocities (3 Linear + 3 Angular)
        startindex_qvel, endindex_qvel = self.sim.model.get_joint_qvel_addr('box_1_joint')
        active_brick_qvel = self.sim.data.qvel[startindex_qvel:endindex_qvel].copy().astype(np.float32)
        
        joints_velocity = np.concatenate((robot_qvel, active_brick_qvel)) # Exact 19D
        
        # Total Observation: 3 (EE) + 3 (Target) + 20 (Qpos) + 19 (Qvel) = 45D
        observation = np.concatenate(
            (gripper_position, target_position, joints_position, joints_velocity)
        )

        return observation
    def change_position(self):
        """
        This function resets the target position randomly.
        """
        self.goal = np.random.rand(2) * (self.target_bounds[:, 1] -
                                         self.target_bounds[:, 0]
                                         ) + self.target_bounds[:, 0]
        return self.goal
 

    def do_simulation(self, ctrl, n_frames):
        """
        This function allows applying control on n simulation frames. These 
        steps are different from the agent's steps. The simulation frames are 
        simulation steps using a contget_observation()rol action.        
        """

        
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError(
                "dimensión  de las acción no concuerda con el controlador")

        
        ctrl = np.clip(ctrl, self.action_space.low, self.action_space.high)

        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            self.sim.step()

    def compute_reward(self, state, action):
        """
        This function compute the reward
        """
        reward = 0
        target_position = np.array([state[3], state[4], state[5]])
        gripper_velp = self.sim.data.get_body_xvelp("ee_link")
        gripper_velr = self.sim.data.get_body_xvelr("ee_link")
        gripper_vel = np.concatenate((np.array(gripper_velp), np.array(gripper_velr)))
        vel = np.sum(abs(gripper_vel)) / 6
        ee_index = self.sim.model.body_name2id('ee_link')
        ee_mass = self.sim.model.body_mass[ee_index]
        ee_inertia = self.sim.model.body_inertia[ee_index]
        ee_force = self.sim.data.cfrc_ext[ee_index][:3]
        ee_torque = self.sim.data.cfrc_ext[ee_index][3:]
        ee_linear_acceleration = np.sum(abs(ee_force / ee_mass))/3
        ee_angular_acceleration = np.sum(abs(ee_torque / ee_inertia))/3

        #target_position = self.sim.data.get_geom_xpos('box_1')
        self.cube_geom_id= self.sim.model.geom_name2id('box_1')
        left_knuckle_id= self.sim.model.geom_name2id('left_inner_knuckle_geom')
        left_finger_id= self.sim.model.geom_name2id('left_inner_finger_geom')
        right_knuckle_id= self.sim.model.geom_name2id('right_inner_knuckle_geom')
        right_finger_id= self.sim.model.geom_name2id('right_inner_finger_geom')
        floor = self.sim.model.geom_name2id("floor")
        #self.obstacle = self.sim.model.body_name2id("cuboid")
        ee_index = self.sim.model.body_name2id('ee_link')
        #ee_link_id = self.sim.model.geom_name2id('ee_link')
        wrist_3_link_id = self.sim.model.body_name2id('wrist_3_link')
        wrist_2_link_id = self.sim.model.body_name2id('wrist_2_link')
        wrist_1_link_id = self.sim.model.body_name2id('wrist_1_link')
        forearm_link_id = self.sim.model.body_name2id('forearm_link')
        
        #Husky's body id
        Husky_base_id = self.sim.model.body_name2id('base')
        
        
        id_1= self.sim.model.geom_name2id('1')
        id_2= self.sim.model.geom_name2id('2')
        id_3= self.sim.model.geom_name2id('3')
        id_4= self.sim.model.geom_name2id('4')
        id_5= self.sim.model.geom_name2id('5')
        id_6= self.sim.model.geom_name2id('6')
        id_7= self.sim.model.geom_name2id('7')
        id_8= self.sim.model.geom_name2id('8')
        id_9= self.sim.model.geom_name2id('9')
        id_10= self.sim.model.geom_name2id('10')
        id_11= self.sim.model.geom_name2id('11')
        id_12= self.sim.model.geom_name2id('12')
        id_13= self.sim.model.geom_name2id('13')
        id_14= self.sim.model.geom_name2id('14')
        id_15= self.sim.model.geom_name2id('15')
        id_16= self.sim.model.geom_name2id('16')
        id_17= self.sim.model.geom_name2id('17')
        id_18= self.sim.model.geom_name2id('18')

               
        wall61 = self.sim.model.geom_name2id('wall6_1')
        wall62 = self.sim.model.geom_name2id('wall6_2')
        wall63 = self.sim.model.geom_name2id('wall6_3')
        wall64 = self.sim.model.geom_name2id('wall6_4')
        wall65 = self.sim.model.geom_name2id('wall6_5')
        wall66 = self.sim.model.geom_name2id('wall6_6')
        wall67 = self.sim.model.geom_name2id('wall6_7')
        wall68 = self.sim.model.geom_name2id('wall6_8')
        wall69 = self.sim.model.geom_name2id('wall6_9')
        wall610 = self.sim.model.geom_name2id('wall6_10')
        
        wall51 = self.sim.model.geom_name2id('wall5_1')
        wall52 = self.sim.model.geom_name2id('wall5_2')
        wall53 = self.sim.model.geom_name2id('wall5_3')
        wall54 = self.sim.model.geom_name2id('wall5_4')
        wall55 = self.sim.model.geom_name2id('wall5_5')
        wall56 = self.sim.model.geom_name2id('wall5_6')
        wall57 = self.sim.model.geom_name2id('wall5_7')
        wall58 = self.sim.model.geom_name2id('wall5_8')
        wall59 = self.sim.model.geom_name2id('wall5_9')
        
        wall41 = self.sim.model.geom_name2id('wall4_1')
        wall42 = self.sim.model.geom_name2id('wall4_2')
        wall43 = self.sim.model.geom_name2id('wall4_3')
        wall44 = self.sim.model.geom_name2id('wall4_4')
        wall45 = self.sim.model.geom_name2id('wall4_5')
        wall46 = self.sim.model.geom_name2id('wall4_6')
        wall47 = self.sim.model.geom_name2id('wall4_7')
        wall48 = self.sim.model.geom_name2id('wall4_8')
        wall49 = self.sim.model.geom_name2id('wall4_9')
        wall410 = self.sim.model.geom_name2id('wall4_10')
        
        wall31 = self.sim.model.geom_name2id('wall3_1')
        wall32 = self.sim.model.geom_name2id('wall3_2')
        wall33 = self.sim.model.geom_name2id('wall3_3')
        wall34 = self.sim.model.geom_name2id('wall3_4')
        wall35 = self.sim.model.geom_name2id('wall3_5')
        wall36 = self.sim.model.geom_name2id('wall3_6')
        wall37 = self.sim.model.geom_name2id('wall3_7')
        wall38 = self.sim.model.geom_name2id('wall3_8')
        wall39 = self.sim.model.geom_name2id('wall3_9')
        
        wall = [wall61, wall62, wall63, wall64, wall65, wall66, wall67, wall68, wall69, wall610, wall51, wall52, wall53, wall54, wall55, wall56, wall57, wall58, wall59, wall41, wall42, wall43, wall44, wall45, wall46, wall47, wall48, wall49, wall410, wall31, wall32, wall33, wall34, wall35, wall36, wall37, wall38, wall39]
        

        wrist_3_link_id = self.sim.model.body_name2id('wrist_3_link')
        
        arm = [forearm_link_id, wrist_1_link_id, wrist_2_link_id, wrist_3_link_id, ee_index, right_finger_id, right_knuckle_id, left_finger_id, left_knuckle_id]
        self.l_finger_geom_ids = [ left_finger_id, left_knuckle_id ]
        self.r_finger_geom_ids = [ right_finger_id, right_knuckle_id]

        RobotList=[id_1, id_2, id_3, id_4, id_5 ,id_6 ,id_7 ,id_8 ,id_9 ,id_10 ,id_11 ,id_12 ,id_13 ,id_14 ,id_15 ,id_16 ,id_17 ,id_18]
        collisionList= [floor,Husky_base_id] + RobotList
        
        touch_left_finger = False
        touch_right_finger = False

        L_Finger = self.sim.data.get_body_xpos("left_inner_finger").astype(np.float32)
        R_Finger = self.sim.data.get_body_xpos('right_inner_finger').astype(np.float32)
        gripper_position = [(L_Finger[0] +  R_Finger[0])/2, (L_Finger[1] +  R_Finger[1])/2, (L_Finger[2] +  R_Finger[2])/2 ]
        target_subposition = np.array([target_position[0], target_position[2]])
        gripper_subposition = np.array([gripper_position[0], gripper_position[2]])
        wrist_1_joint_index = self.sim.model.get_joint_qpos_addr('wrist_1_joint')
        wrist_1_joint_angle = self.sim.data.qpos[wrist_1_joint_index]
        wrist_2_joint_index = self.sim.model.get_joint_qpos_addr('wrist_2_joint')
        wrist_2_joint_angle = self.sim.data.qpos[wrist_2_joint_index]
        wrist_3_joint_index = self.sim.model.get_joint_qpos_addr('wrist_3_joint')
        wrist_3_joint_angle = self.sim.data.qpos[wrist_3_joint_index]
        
        # Desired angle (90 degrees)
        desired_start_angle =  0 #np.pi / 2  90 degrees in radians
        # Compute angle deviation (considering wrap-around)
        angle_difference1 = (wrist_1_joint_angle - desired_start_angle + np.pi) % (2 * np.pi) - np.pi
        angle_deviation1 = np.abs(angle_difference1)        
        angle_difference2 = (wrist_2_joint_angle - desired_start_angle + np.pi) % (2 * np.pi) - np.pi
        angle_deviation2 = np.abs(angle_difference2)
        angle_difference3 = (wrist_3_joint_angle - desired_start_angle + np.pi) % (2 * np.pi) - np.pi
        angle_deviation3 = np.abs(angle_difference3)
        # Compute the penalty
        wrist_1_penalty = angle_deviation1 ** 2
        wrist_2_penalty = angle_deviation2 ** 2
        wrist_3_penalty = angle_deviation3 ** 2
        # Scaling factor
        scaling_factor = 10
        distance_norm = np.linalg.norm(
            target_subposition - gripper_subposition).astype(np.float32)

        
        Ydistance = abs(gripper_position[1]- target_position[1])
        Ydistance_reward = 0.5 * (1 - (np.tanh(6.0 * Ydistance)))
        
        distance_reward = 2 * (1 - (np.tanh( 3 *distance_norm)))

        
        
        reward += 0.1 * (- vel  - ee_linear_acceleration   - ee_angular_acceleration)




        	
        
        collision = False
        for i in range (self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in arm and c.geom2 in collisionList:
                collision = True
                self.done = True
                return -10
            if c.geom1 in collisionList and c.geom2 in arm:
                collision = True
                self.done = True
                return -10
        	
        if collision == True:
            reward -= 10
        for i in range (self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 == self.cube_geom_id and c.geom2 in RobotList:
                self.done = True
                return -50
            if c.geom1 in RobotList and c.geom2 == self.cube_geom_id:
                self.done = True
                return -50
        		
        		
        for i in range (self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 == self.cube_geom_id and c.geom2 == floor:
                self.done = True
                if self.phase==0 or self.phase==1:
                    return -50
                if self.phase==2 or self.phase==3:
                    return -100
            if c.geom1 == floor and c.geom2 == self.cube_geom_id:
                self.done = True
                if self.phase==0 or self.phase==1:
                    return -50
                if self.phase==2 or self.phase==3:
                    return -100

            
            

        self.has_grasp=False
        for i in range (self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.cube_geom_id:
                touch_left_finger = True
            if c.geom1 == self.cube_geom_id and c.geom2 in self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.cube_geom_id:
                touch_right_finger = True
            if c.geom1 == self.cube_geom_id and c.geom2 in self.r_finger_geom_ids:
                touch_right_finger = True
        			
        self.has_grasp = touch_left_finger and touch_right_finger
        

        if (distance_norm < 0.08 and Ydistance < 0.08):
            self.phase = 1
        if self.phase==0:
            reward +=  0.5 * (distance_reward + Ydistance_reward)
            reward -= 0.1
            reward -= (scaling_factor) * wrist_1_penalty
            reward -= (scaling_factor) * wrist_2_penalty
            reward -= (scaling_factor) * wrist_3_penalty

        if self.phase == 1:
            reward +=  0.5 * (distance_reward + Ydistance_reward) -0.1
            reward -= (scaling_factor) * wrist_1_penalty
            reward -= (scaling_factor) * wrist_2_penalty
            reward -= (scaling_factor) * wrist_3_penalty

            if self.has_grasp:
                self.phase = 2
                if self.grasp == False:
                    reward += 300
                    self.grasp = True

        
        if self.phase == 2:
            reward += 1
            reward -= (scaling_factor) * wrist_1_penalty
            reward -= (scaling_factor) * wrist_2_penalty
            reward += -10 * abs(action[-2])
            cube_touch_wall = False
            for i in range (self.sim.data.ncon):
                c = self.sim.data.contact[i]
                if c.geom1 in wall and c.geom2 == self.cube_geom_id:
                    cube_touch_wall = True
                if c.geom1 == self.cube_geom_id and c.geom2 in wall:
                    cube_touch_wall = True
            if cube_touch_wall:
                self.done = True
                reward -= 100

            distance_norm2 = np.linalg.norm( (np.array([-0.33, -4.6, 0.5]) - target_position)).astype(np.float32)
            distance_reward2 = (30 * (1 - (np.tanh( 3*distance_norm2))) ) #- (5 * distance_norm2)

            reward += distance_reward2
            if distance_norm2 < 0.03:
                reward += 6000
                self.phase = 3
        
        if self.phase == 3:
            reward -= (scaling_factor) * wrist_1_penalty
            reward -= (scaling_factor) * wrist_2_penalty
            reward += -2000 * abs(action[-2])
            target_obj_pos = np.linalg.norm(np.array([-0.33, -4.6, 0.5]) - target_position).astype(np.float32)
            distance_reward3 = 1 - (np.tanh( target_obj_pos))
            reward += 1000 * distance_reward3 
            self.done = True
       
        return reward
        	

    def get_info(self):
        """
        This function returns info

        ### description
            --"gripper_position": xyz position of the end effector.
            --"target_position": target position.
            --"j_position": position of the joints.
            --"j_velocity": velocity of the joints.
            --"dist": distance between the end effector and the goal.
        """

        gripper_position = self.sim.data.get_body_xpos("ee_link").astype(np.float32)
        target_position = self.sim.data.get_body_xpos("box_1").astype(np.float32)


        info = {
            'gripper_position': gripper_position,
            'target_position': target_position,
            'dist': np.linalg.norm(target_position - gripper_position).astype(np.float32),
            'observation': self.get_observation(),
            'j_position': self.sim.data.qpos.flat.copy().astype(np.float32),
            'j_velocity': self.sim.data.qvel.flat.copy().astype(np.float32),
        }

        return info
