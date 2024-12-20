from argparse import ArgumentParser
import airsimdroneracinglab as airsim
from airsimdroneracinglab import Vector3r
from scipy.interpolate import interp1d
import cv2
import threading
import time
import utils
import numpy as np
import math
import os
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
import numpy as np
import cvxpy as cp
from time import sleep
import threading
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
from airsimdroneracinglab.types import Pose, Quaternionr
from collections import namedtuple
"""
class Slerp:
    def __init__(self, times, quaternions):
        self.times = times
        self.rotations = R.from_quat(quaternions)
    
    def interpolate(self, t_fine):
        return R.slerp(t_fine, self.times, self.rotations).as_quat()

"""
# drone_name should match the name in ~/Document/AirSim/settings.json
class BaselineRacer(object):
    def __init__(
        self,
        drone_name="drone_1",
        viz_traj=True,
        viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0],
        viz_image_cv2=True,
    ):
        

        self.kp = 0.38
        self.ki = 0.009
        self.kd = 0.009
        self.previous_error = 0
        self.integral = 0
        self.error_history = deque()  


        self.yawerror_history = []
        self.yawkp = 1.1
        self.yawki = 0.1
        self.yawkd = 0.09
        self.yawprevious_error = 0
        self.yawintegral = 0
        self.yawerror_history = deque() 
        




        self.drone_state = None


        self.max_vx = 10
        self.max_vy = 10
        self.max_vz = 10
        self.max_ax = 5  # Maximum acceleration in x
        self.max_ay = 5  # Maximum acceleration in y
        self.max_az = 5  # Maximum acceleration in z
        self.timefactor = 1

        self.Z_velocity = 0

        self.z_min = -10
        self.duration = 10
        self.drivetrain_type = airsim.DrivetrainType.MaxDegreeOfFreedom # airsim.DrivetrainType.ForwardOnly  #airsim.DrivetrainType.MaxDegreeOfFreedom
        self.yaw_rate = 30
        self.control_frequency = 1  # Control loop frequency in Hz

        self.x_smooth = None
        self.y_smooth = None
        self.z_smooth = None
        self.smooth_orientations = None

  
        self.time_horizon = 30
        self.x = cp.Variable((4, self.time_horizon + 1))  # State variable
        self.u = cp.Variable((2, self.time_horizon))     # Control variable
         
        self.dt = 1.0 / self.control_frequency  # Time step for control updates
        
        # Define cost and constraints in MPC
        self.Q = np.diag([100,100,3,3])     #np.diag([60,60,5,5])  # Weighting on states (position and velocity)
        self.R = np.diag([1,1])       #np.eye(2)  # Weighting on control inputs (velocity and yaw rate)
        
        self.running = False
        self.control_thread = None

        self.errorx = []
        self.errory = []
        self.errorz = []
        self.pos = []

        self.pausecontrol =  False
        self.controltime =0






        self.current_gate_index = 0
        self.drone_name = drone_name
        self.gate_poses_ground_truth = None
        self.viz_image_cv2 = viz_image_cv2
        self.viz_traj = viz_traj
        self.viz_traj_color_rgba = viz_traj_color_rgba

        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        # we need two airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
        # so we poll images in a thread using one airsim MultirotorClient object
        # and use another airsim MultirotorClient for querying state commands
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = airsim.MultirotorClient()
        self.airsim_client_odom.confirmConnection()
        self.level_name = None

        self.image_callback_thread = threading.Thread(
            target=self.repeat_timer_image_callback, args=(self.image_callback, 0.03)
        )
        self.odometry_callback_thread = threading.Thread(
            target=self.repeat_timer_odometry_callback,
            args=(self.odometry_callback, 0.02),
        )
        self.is_image_thread_active = False
        self.is_odometry_thread_active = False

        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = (
            10  # see https://github.com/microsoft/AirSim-Drone-Racing-Lab/issues/38
        )

    # loads desired level
    def load_level(self, level_name, sleep_sec=2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection()  # failsafe
        time.sleep(sleep_sec)  # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=3):
        self.airsim_client.simStartRace(tier)

    # Resets a current race: moves players to start positions, timer and penalties reset
    def reset_race(self):
        self.airsim_client.simResetRace()

    # arms drone, enable APIs, set default traj tracker gains
    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        # set default values for trajectory tracker gains
        traj_tracker_gains = airsim.TrajectoryTrackerGains(
            kp_cross_track=5.0,
            kd_cross_track=0.0,
            kp_vel_cross_track=3.0,
            kd_vel_cross_track=0.0,
            kp_along_track=0.4,
            kd_along_track=0.0,
            kp_vel_along_track=0.04,
            kd_vel_along_track=0.0,
            kp_z_track=2.0,
            kd_z_track=0.0,
            kp_vel_z=0.4,
            kd_vel_z=0.0,
            kp_yaw=3.0,
            kd_yaw=0.1,
        )

        self.airsim_client.setTrajectoryTrackerGains(
            traj_tracker_gains, vehicle_name=self.drone_name
        )
        time.sleep(0.2)

    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height=1.0):
        start_position = self.airsim_client.simGetVehiclePose(
            vehicle_name=self.drone_name
        ).position
        takeoff_waypoint = airsim.Vector3r(
            start_position.x_val,
            start_position.y_val,
            start_position.z_val - takeoff_height,
        )

        self.airsim_client.moveOnSplineAsync(
            [takeoff_waypoint],
            vel_max=15.0,
            acc_max=5.0,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        ).join()

    # stores gate ground truth poses as a list of airsim.Pose() objects in self.gate_poses_ground_truth
    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [
            int(gate_name.split("_")[0][4:]) for gate_name in gate_names_sorted_bad
        ]
        gate_indices_correct = sorted(
            range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k]
        )
        gate_names_sorted = [
            gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct
        ]
        self.gate_poses_ground_truth = []
        for gate_name in gate_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            counter = 0
            while (
                math.isnan(curr_pose.position.x_val)
                or math.isnan(curr_pose.position.y_val)
                or math.isnan(curr_pose.position.z_val)
            ) and (counter < self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS):
                print(f"DEBUG: {gate_name} position is nan, retrying...")
                counter += 1
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(
                curr_pose.position.x_val
            ), f"ERROR: {gate_name} curr_pose.position.x_val is still {curr_pose.position.x_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.y_val
            ), f"ERROR: {gate_name} curr_pose.position.y_val is still {curr_pose.position.y_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.z_val
            ), f"ERROR: {gate_name} curr_pose.position.z_val is still {curr_pose.position.z_val} after {counter} trials"
            self.gate_poses_ground_truth.append(curr_pose)

    # this is utility function to get a velocity constraint which can be passed to moveOnSplineVelConstraints()
    # the "scale" parameter scales the gate facing vector accordingly, thereby dictating the speed of the velocity constraint
    def get_gate_facing_vector_from_quaternion(self, airsim_quat, scale=1.0):
        import numpy as np

        # convert gate quaternion to rotation matrix.
        # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        q = np.array(
            [
                airsim_quat.w_val,
                airsim_quat.x_val,
                airsim_quat.y_val,
                airsim_quat.z_val,
            ],
            dtype=np.float64,
        )
        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return airsim.Vector3r(0.0, 1.0, 0.0)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        rotation_matrix = np.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ]
        )
        gate_facing_vector = rotation_matrix[:, 1]
        return airsim.Vector3r(
            scale * gate_facing_vector[0],
            scale * gate_facing_vector[1],
            scale * gate_facing_vector[2],
        )

    def fly_through_all_gates_one_by_one_with_moveOnSpline(self):
        if self.level_name == "Building99_Hard":
            vel_max = 5.0
            acc_max = 2.0

        if self.level_name in [
            "Soccer_Field_Medium",
            "Soccer_Field_Easy",
            "ZhangJiaJie_Medium",
        ]:
            vel_max = 10.0
            acc_max = 5.0

        return self.airsim_client.moveOnSplineAsync(
            [gate_pose.position],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def fly_through_all_gates_at_once_with_moveOnSpline(self):
        if self.level_name in [
            "Soccer_Field_Medium",
            "Soccer_Field_Easy",
            "ZhangJiaJie_Medium",
            "Qualifier_Tier_1",
            "Qualifier_Tier_2",
            "Qualifier_Tier_3",
            "Final_Tier_1",
            "Final_Tier_2",
            "Final_Tier_3",
        ]:
            vel_max = 30.0
            acc_max = 15.0

        if self.level_name == "Building99_Hard":
            vel_max = 4.0
            acc_max = 1.0

        return self.airsim_client.moveOnSplineAsync(
            [gate_pose.position for gate_pose in self.gate_poses_ground_truth],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def fly_through_all_gates_one_by_one_with_moveOnSplineVelConstraints(self):
        add_velocity_constraint = True
        add_acceleration_constraint = False

        if self.level_name in ["Soccer_Field_Medium", "Soccer_Field_Easy"]:
            vel_max = 15.0
            acc_max = 3.0
            speed_through_gate = 2.5

        if self.level_name == "ZhangJiaJie_Medium":
            vel_max = 10.0
            acc_max = 3.0
            speed_through_gate = 1.0

        if self.level_name == "Building99_Hard":
            vel_max = 2.0
            acc_max = 0.5
            speed_through_gate = 0.5
            add_velocity_constraint = False

        # scale param scales the gate facing vector by desired speed.
        return self.airsim_client.moveOnSplineVelConstraintsAsync(
            [gate_pose.position],
            [
                self.get_gate_facing_vector_from_quaternion(
                    gate_pose.orientation, scale=speed_through_gate
                )
            ],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=add_velocity_constraint,
            add_acceleration_constraint=add_acceleration_constraint,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def fly_through_all_gates_at_once_with_moveOnSplineVelConstraints(self):
        if self.level_name in [
            "Soccer_Field_Easy",
            "Soccer_Field_Medium",
            "ZhangJiaJie_Medium",
        ]:
            vel_max = 15.0
            acc_max = 7.5
            speed_through_gate = 2.5

        if self.level_name == "Building99_Hard":
            vel_max = 5.0
            acc_max = 2.0
            speed_through_gate = 1.0

        return self.airsim_client.moveOnSplineVelConstraintsAsync(
            [gate_pose.position for gate_pose in self.gate_poses_ground_truth],
            [
                self.get_gate_facing_vector_from_quaternion(
                    gate_pose.orientation, scale=speed_through_gate
                )
                for gate_pose in self.gate_poses_ground_truth
            ],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=True,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def get_camera_info(self):
        # Connect to the AirSim simulator
        camera_name = "fpv_cam"
        drone_name = self.drone_name
        
        # Get camera information
        try:
            camera_info = self.airsim_client.simGetCameraInfo("fpv_cam", vehicle_name=self.drone_name)
            #print(f"Camera Info for '{camera_name}' on '{drone_name}':")
            #print(f"  Position: {camera_info.pose.position}")
            #print(f"  Orientation: {camera_info.pose.orientation}")
            #print(camera_info)
            #print(f"  Projection Matrix: {camera_info.proj_mat}")
            self.pmatrix = camera_info.proj_mat
        except Exception as e:
            print(f"Error retrieving camera info: {e}")

    

    def calgateError(self):
        dist = np.sqrt((self.gatex - self.calculated_gatex)**2 + (self.gatey - self.calculated_gatey)**2 + (self.gatez - self.calculated_gatez)**2)
        print("gate error", dist)
        if dist > 2.5 and self.current_gate_index > 6:

            print(self.gate_poses_ground_truth[self.current_gate_index].position.x_val , self.calculated_gatex)
            self.gate_poses_ground_truth[self.current_gate_index].position.x_val = self.calculated_gatex
            self.gate_poses_ground_truth[self.current_gate_index].position.y_val = self.calculated_gatey
            #self.gate_poses_ground_truth[self.current_gate_index].position.z_val = self.calculated_gatez
            print("recaliberating")

            #self.stop_control_loop()
            #self.airsim_client.moveByVelocityAsync( 0, 0, 0, 0.1, vehicle_name="drone_1")
        
            #self.generate_spline(self.gate_poses_ground_truth[self.current_gate_index  :])
            #time.sleep(1)
            #self.start_control_loop()


    def image_to_world(self, u, v, depth):

        # Extract projection matrix parameters (fx, fy, cx, cy)
        # AirSim proj_mat is typically [fx, 0, cx, 0; 0, fy, cy, 0; 0, 0, 1, 0]
        # It's returned as a flat array of 4 floats per row (12 values total), so we reconstruct it
        # Note: The exact format may vary. Adjust indexing if needed.
        
        




        fx = 160  # Focal length in x direction
        fy = 160  # Focal length in y direction
        cx = 160  # Principal point in x direction
        cy = 120  # principal point (center of image)

        # Step 1: Convert from pixel + depth to camera coordinates
        # Assuming a pinhole camera model:
        # X_cam = (u - cx)*Z/fx
        # Y_cam = (v - cy)*Z/fy
        # Z_cam = Z
        X_cam = (u - cx) * depth / fx
        Y_cam = (v - cy) * depth / fy
        Z_cam = depth

        #print(X_cam,Y_cam,Z_cam,"inside")

        # Now we have the point in camera coordinates (X_cam, Y_cam, Z_cam)

        # Step 2: Transform camera coordinates to world coordinates
        # camera_info.pose gives position and orientation of the camera in world frame
        cam_position = self.drone_state.kinematics_estimated.position
        cam_orientation = self.drone_state.kinematics_estimated.orientation

        # Convert camera quaternion to rotation matrix
        r = R.from_quat([cam_orientation.x_val, cam_orientation.y_val, cam_orientation.z_val, cam_orientation.w_val])
        rotation_matrix = r.as_matrix()

        # The camera's coordinate system in AirSim:
        # By default, the camera looks along -Z, with X to the right and Y down. 
        # Check coordinate frame assumptions. If needed, rotate the point accordingly.
        # If the camera projection is standard pinhole, we assume:
        # X_cam: right, Y_cam: down, Z_cam: forward (negative in AirSim by default for front camera)
        # You might need to adjust signs based on how AirSim defines the camera direction.
        # For a standard pinhole model, if depth is positive going forward along camera's viewing direction:
        # AirSim front camera looks along +X in drone body frame or -Z in the Unreal engine frame depending on version.
        # Adjust if needed. For now, we assume the camera frame is: forward = Z_cam, right = X_cam, down = Y_cam.
        # If you find the forward direction mismatched, you may need to permute or invert some axes.

        # Transform point from camera frame to world frame:
        # p_world = R_world_cam * p_cam + t_world_cam
        p_cam = np.array([X_cam, Y_cam, Z_cam])
        p_world = rotation_matrix.dot(p_cam) + np.array([cam_position.x_val, cam_position.y_val, cam_position.z_val])

        # Return as AirSim Vector3r
        self.calculated_gatex = p_world[0]
        self.calculated_gatey = p_world[1]
        self.calculated_gatez = p_world[2]
        self.calgateError()
        return [p_world[0], p_world[1], p_world[2]]

    





    def check_proximity_and_invoke_vision(self):
        """
        Check if the drone is within 2 meters of the current target gate.
        If yes, call the vision function.
        """
        if self.gate_poses_ground_truth is None or self.current_gate_index >= len(self.gate_poses_ground_truth):
            return  # No gates or no current gate index to process

        # Get current gate position
        current_gate_pose = self.gate_poses_ground_truth[self.current_gate_index]
        gx, gy, gz = (current_gate_pose.position.x_val, 
                      current_gate_pose.position.y_val, 
                      current_gate_pose.position.z_val)

        # Get current drone position (make sure self.drone_state is not None)
        if self.drone_state is None:
            return
        dx, dy, dz = (self.drone_state.kinematics_estimated.position.x_val, 
                      self.drone_state.kinematics_estimated.position.y_val, 
                      self.drone_state.kinematics_estimated.position.z_val)

        # Compute distance to the gate
        dist = np.sqrt((gx - dx)**2 + (gy - dy)**2 + (gz - dz)**2)

        # Check threshold
        if dist < 5:
            # Call the vision function
            #print("gate within range")
            self.current_gate_index += 1
            print(f"Passed gate {self.current_gate_index - 1}, now heading to gate {self.current_gate_index}")
            #self.vision_function(current_gate_pose)
            # Optionally, you can advance to the next gate after processing
            # self.current_gate_index += 1
            self.imagekey = True
            self.start_image_callback_thread()
            self.gatex = gx
            self.gatey = gy
            self.gatez = gz
            print("given gate",gx, gy, gz)
            #self.calgateError()

       
            

            
        
            #self.stop_image_callback_thread()



            
    def generate_spline(self, gate_poses):
        # Get gate positions and orientations
        
        gate_positions = [gate_pose.position for gate_pose in gate_poses]
        gate_orientations = [gate_pose.orientation for gate_pose in gate_poses]
        lengates = len(gate_positions) + 1
        # Get the drone's initial position 
        # This could be done right after takeoff or here. If you've started odometry callback,
        # self.drone_state should be available. Otherwise, you can directly get from simGetVehiclePose.
        if self.drone_state is not None:
            drone_initial_pos = self.drone_state.kinematics_estimated.position
        else:
            drone_pose = self.airsim_client.simGetVehiclePose(vehicle_name=self.drone_name)
            drone_initial_pos = drone_pose.position

        # Convert the gate positions into a NumPy array
        positions = np.array([(pos.x_val, pos.y_val, pos.z_val) for pos in gate_positions])

        # Prepend the drone's initial position at the start
        start_pos = np.array([drone_initial_pos.x_val, drone_initial_pos.y_val, drone_initial_pos.z_val])
        positions = np.vstack((start_pos, positions))

        # Extract orientations
        orientations = np.array([(ori.x_val, ori.y_val, ori.z_val, ori.w_val) for ori in gate_orientations])

        # Since we added an initial position that is not associated with a gate orientation,
        # we need to handle orientation carefully. We can assume the initial orientation is the 
        # drone's current orientation or simply repeat the first gate's orientation for now.
        # A more robust approach might be needed depending on application requirements.
        if self.drone_state is not None:
            drone_orientation = self.drone_state.kinematics_estimated.orientation
            initial_quat = (drone_orientation.x_val, drone_orientation.y_val, drone_orientation.z_val, drone_orientation.w_val)
        else:
            # fallback: use the first gate orientation if drone orientation is not available
            initial_quat = orientations[0]

        # Prepend the initial orientation
        orientations = np.vstack((initial_quat, orientations))

        # Extract x, y, z arrays
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        # Parameter t along the trajectory
        t = np.arange(len(x))

        # Choose spline parameters based on level_name (as you did before)
        if args.level_name == "Soccer_Field_Easy":
            self.lsteps = int(40 /self.totalinitial_gates * lengates )
            self.kp = 1.1
            self.ki = 0.1
            self.kd = 0.001
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)
            t_fine = np.linspace(t[0], t[-1], self.lsteps)
            self.timefactor = 2
            self.sleeptimefactor = 10

        elif args.level_name == "Soccer_Field_Medium":

            self.yawkp = 1.1
            self.yawki = 0.1
            self.yawkd = 0.001
            self.kp = 1.1
            self.ki = 0.1
            self.kd = 0.001
            self.lsteps = int(400 /self.totalinitial_gates * lengates )
            self.timefactor = 1
            self.sleeptimefactor = 3
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)
            t_fine = np.linspace(t[0], t[-1], self.lsteps)

        elif  args.level_name == "Qualifier_Tier_2":

            self.yawkp = 1.7
            self.yawki = 0.2
            self.yawkd = 0.1
            self.kp = 1.1
            self.ki = 0.1
            self.kd = 0.09
            self.lsteps = int(250 /self.totalinitial_gates * lengates )
            self.timefactor = 1
            self.sleeptimefactor = 2
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)
            t_fine = np.linspace(t[0], t[-1], self.lsteps)

        elif  args.level_name == "Qualifier_Tier_1":
            self.kp = 1.1
            self.ki = 0.1
            self.kd = 0.09
            self.lsteps = int(170 /self.totalinitial_gates * lengates )
            self.timefactor = 1
            self.sleeptimefactor = 3
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)
            t_fine = np.linspace(t[0], t[-1], self.lsteps)

        elif  args.level_name == "ZhangJiaJie_Medium":
            self.kp = 1.1
            self.ki = 0.1
            self.kd = 0.09
            self.lsteps = int(200 /self.totalinitial_gates * lengates )
            self.timefactor = 1
            self.sleeptimefactor = 3
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)
            t_fine = np.linspace(t[0], t[-1], self.lsteps)

        else :
            self.kp = 1.1
            self.ki = 0.1
            self.kd = 0.09
            self.lsteps = int(170 /self.totalinitial_gates * lengates )
            self.timefactor = 1
            self.sleeptimefactor = 3
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)
            t_fine = np.linspace(t[0], t[-1], self.lsteps)

        # Compute the smoothed paths
        self.x_smooth = spline_x(t_fine)
        self.y_smooth = spline_y(t_fine)
        self.z_smooth = spline_z(t_fine)

        self.dx_smooth = spline_x(t_fine, 1)
        self.dy_smooth = spline_y(t_fine, 1)
        self.dz_smooth = spline_z(t_fine, 1)

        # Generate smooth orientations using Slerp
        rotations = R.from_quat(orientations)
        slerp = Slerp(t, rotations)
        self.smooth_orientations = slerp(t_fine)

        euler_angles = self.smooth_orientations.as_euler('xyz', degrees=True)
        self.roll_smooth = euler_angles[:, 0]
        self.pitch_smooth = euler_angles[:, 1]
        self.yaw_smooth = euler_angles[:, 2]

        # Plot the spline in the environment
        points = [Vector3r(x_val, y_val, z_val) for x_val, y_val, z_val in zip(self.x_smooth, self.y_smooth, self.z_smooth)]
        self.airsim_client.simPlotLineList(points, is_persistent=True)



    def plotgraphs(self, gate_poses):
        self.pitch_smooth
        gate_positions = [gate_pose.position for gate_pose in gate_poses]
        gate_orientations = [gate_pose.orientation for gate_pose in gate_poses]

        positions = np.array([(pos.x_val, pos.y_val, pos.z_val) for pos in gate_positions])
        orientations = np.array([(ori.x_val, ori.y_val, ori.z_val, ori.w_val) for ori in gate_orientations])

        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        time = range(len(self.errorx)) 

        axes[0].plot(time, self.errorx, label='Error X', color='r')
        axes[0].set_ylabel('Error X')
        axes[0].legend(loc='upper right')
        axes[0].grid(True)

        # Plot errory
        axes[1].plot(time, self.errory, label='Error Y', color='r')
        axes[1].set_ylabel('Error Y')
        axes[1].legend(loc='upper right')
        axes[1].grid(True)

        # Plot errorz
        axes[2].plot(time, self.errorz, label='Error Z', color='r')
        axes[2].set_ylabel('Error Z')
        axes[2].legend(loc='upper right')
        axes[2].grid(True)

        # Set the x-axis label for the bottom plot
        axes[2].set_xlabel('Time')

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.savefig("1.png")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pos_x = [p[0] for p in self.pos]
        pos_y = [p[1] for p in self.pos]
        pos_z = [p[2] for p in self.pos]
        
        # Plot the spline trajectory
        ax.plot(self.x_smooth, self.y_smooth, self.z_smooth, label='Spline Trajectory', color='blue', linewidth=2)

        # Plot the actual trajectory (path followed)
        ax.plot(pos_x, pos_y, pos_z, label='Actual Trajectory', color='orange', linewidth=2, linestyle='--')

        # Plot the start and stop positions for the spline trajectory
        ax.scatter(self.x_smooth[0], self.y_smooth[0], self.z_smooth[0], color='green', s=100, label='Start Position')  # Start position
        ax.scatter(self.x_smooth[-1], self.y_smooth[-1], self.z_smooth[-1], color='red', s=100, label='Stop Position')  # Stop position

        # Plot gate positions (if defined)
        try:
            ax.scatter(x, y, z, color='purple', s=50, label='Gate Positions', alpha=0.8)
        except AttributeError:
            print("Gate positions not defined, skipping gate visualization.")

        # Add labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Display the plot
        #plt.show()
        plt.savefig("2.png")

    def image_callback(self):
        # get uncompressed FPV cam image

        MIN_DEPTH_METERS = 0
        MAX_DEPTH_METERS = 100
        THRESHOLD_METERS =6
        RED_CHANNEL_THRESHOLD = 60
        GREEN_CHANNEL_THRESHOLD = 60
        BLUE_CHANNEL_THRESHOLD =  240

        while self.imagekey:
            responses = self.airsim_client.simGetImages([
                airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False),  # Color image
                airsim.ImageRequest("fpv_cam", airsim.ImageType.DepthPerspective, True, False),  # Depth image
            ])

            if not responses or len(responses) < 2:
                print("Failed to get images. Please check AirSim setup.")
                break

            # Process color image
            color_response = responses[0]
            color_img = np.frombuffer(color_response.image_data_uint8, dtype=np.uint8)
            color_img = color_img.reshape((color_response.height, color_response.width, 3))
            #color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            # Process depth image
            depth_response = responses[1]
            depth_img_in_meters = airsim.list_to_2d_float_array(
                depth_response.image_data_float, depth_response.width, depth_response.height
            )
            mask = (depth_img_in_meters < THRESHOLD_METERS).astype('uint8') * 255  # Binary mask: 255 for True, 0 for False



            # Convert depth to 8-bit for visualization
            
            masked_image = cv2.bitwise_and(color_img, color_img, mask=mask)


            non_zero_coords = np.column_stack(np.where(mask > 0))
            if len(non_zero_coords) > 0:
                mid_point = np.mean(non_zero_coords, axis=0).astype(int)  # Midpoint (y, x)
                average_depth = np.mean(depth_img_in_meters[mask > 0])   # Average depth (z)

                # Draw midpoint on the masked image for visualization
                cv2.circle(masked_image, (mid_point[1], mid_point[0]), 5, (0, 0, 255), -1)
                #print(f"camera cordinates: ({mid_point[1]}, {mid_point[0]}), Average Depth: {average_depth:.2f} meters")
                print("calculated world cordinates of gate",self.image_to_world( mid_point[1], mid_point[0], average_depth))
            else:
                print("No points within threshold.")

            self.depth_img_in_meters = depth_img_in_meters
            self.color_img = color_img
            self.maskimage =  masked_image

            # Display the color image, depth visualization, and mask
            #cv2.imshow("Color Image", color_img)
            #cv2.imshow("Depth Visualization (Colormap)", colormap)
            #cv2.imshow("Mask (Depth < 4m)", mask)
            #cv2.imshow("Masked Color Image (Depth < 4m)", self.maskimage)
            cv2.imwrite("maskimage.png",self.maskimage)
            # Wait for 1 ms and break on 'q' key press
            time.sleep(0.01)
            #cv2.destroyAllWindows()
            self.imagekey = False
            

        # Close all OpenCV windows
        

    

        """
        tmp_dir = os.path.join("/home/mkg7/AirSim-1.5.0-linux/AirSim-Drone-Racing-Lab/baselines", "airsim_drone")  #tempfile.gettempdir()

        #print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise
        
        for idx, response in enumerate(responses):

            #filename = os.path.join(tmp_dir, str(idx))

            if response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                #airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            elif response.compress: #png format
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                #airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else: #uncompressed array
                #print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                #img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
                #img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
                #print(img_rgb.shape)
                #cv2.imshow("img_rgb", img_rgb)
                #cv2.waitKey(1)
    
        """

    def odometry_callback(self):
        # get uncompressed fpv cam image
        drone_state = self.airsim_client_odom.getMultirotorState()
        # in world frame:
        position = drone_state.kinematics_estimated.position
        orientation = drone_state.kinematics_estimated.orientation
        linear_velocity = drone_state.kinematics_estimated.linear_velocity
        angular_velocity = drone_state.kinematics_estimated.angular_velocity

        #print(position, orientation, linear_velocity, angular_velocity)

        self.drone_state = drone_state
        time.sleep(0.001)  # Adjust sleep as needed

    # call task() method every "period" seconds.
    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_odometry_callback(self, task, period):
        while self.is_odometry_thread_active:
            task()
            time.sleep(period)

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("Stopped image callback thread.")

    def start_odometry_callback_thread(self):
        if not self.is_odometry_thread_active:
            self.is_odometry_thread_active = True
            self.odometry_callback_thread.start()
            print("Started odometry callback thread")

    def stop_odometry_callback_thread(self):
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")

    """

    def build_mpc_problem(self, x_current, current_time):
        cost = 0
        constraints = []

        constraints.append(self.x[:, 0] == x_current)  # Set initial state



        for t in range(self.time_horizon):
            # Get desired state from spline
            spline_index = min(int(current_time + t), len(self.x_smooth) - 1)
            x_desired = np.array([
                self.x_smooth[spline_index],
                self.y_smooth[spline_index],
                self.z_smooth[spline_index],
                0,  # Desired velocity x (you may want to compute this from the spline)
                0   # Desired velocity y (you may want to compute this from the spline)
            ])

            # Cost for position and velocity tracking
            cost += cp.quad_form(self.x[:, t] - x_desired, self.Q)
            
            # Cost for control input (minimizing velocity and yaw)
            cost += cp.quad_form(self.u[:, t], self.R)
            
            # Dynamics constraint
            if t < self.time_horizon - 1:
                constraints.append(self.x[:3, t+1] == self.x[:3, t] + self.dt * self.x[3:6, t])
                constraints.append(self.x[3:6, t+1] == self.x[3:6, t] + self.dt * self.u[:, t])
            # Velocity limits (vx, vy, vz)
            constraints.append(cp.abs(self.u[0, t]) <= self.max_vx)
            constraints.append(cp.abs(self.u[1, t]) <= self.max_vy)
            constraints.append(cp.abs(self.u[]t), constraints)
        return problem
    """

    def zCompute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        
        self.error_history.append(error * dt)
        if len(self.error_history) > 8:
            self.error_history.popleft()  # Remove oldest error if size exceeds 10
        
        i_term = self.ki * sum(self.error_history)

        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Update previous error
        self.previous_error = error
        
        return output

    def yawCompute(self, setpoint, measured_value, dt):
        error = (setpoint - measured_value +180 )%360-180
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        
        self.yawerror_history.append(error * dt)
        if len(self.yawerror_history) > 8:
            self.yawerror_history.popleft()  # Remove oldest error if size exceeds 10
        
        i_term = self.yawki * sum(self.error_history)

        
        # Derivative term
        derivative = (error - self.yawprevious_error) / dt
        d_term = self.yawkd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Update previous error
        self.yawprevious_error = error
        
        return output
    


    def build_mpc_problem(self, x_current, current_time):
        cost = 0
        constraints = []
        constraints.append(self.x[:, 0] == x_current[:4])

        for t in range(self.time_horizon):
            spline_index = min(int(current_time + t), len(self.x_smooth) - 1)
            x_desired = np.array([
                self.x_smooth[spline_index],
                self.y_smooth[spline_index],
                self.dx_smooth[spline_index],
                self.dy_smooth[spline_index]
            ])

            cost += cp.quad_form(self.x[:, t] - x_desired, self.Q)
            cost += cp.quad_form(self.u[:, t], self.R)

            # Dynamics constraints
            if t < self.time_horizon - 1:
                constraints.append(self.x[:2, t+1] == self.x[:2, t] + self.dt * self.x[2:4, t])
                constraints.append(self.x[2:4, t+1] == self.x[2:4, t] + self.dt * self.u[:, t])

            # Velocity limits
            constraints.append(cp.abs(self.x[2, t]) <= self.max_vx)
            constraints.append(cp.abs(self.x[3, t]) <= self.max_vy)

            #constraints.append(cp.abs(self.x[5, t]) <= self.max_vz)

            # Acceleration limits
            constraints.append(cp.abs(self.u[0, t]) <= self.max_ax)
            constraints.append(cp.abs(self.u[1, t]) <= self.max_ay)
            #constraints.append(cp.abs(self.u[2, t]) <= self.max_az)

            # Altitude constraint
            #constraints.append(self.x[2, t] >= self.z_min)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        return problem

    def solve_mpc(self, x_current, current_time):
        problem = self.build_mpc_problem(x_current, current_time)
        problem.solve()
        optimal_controls = self.u[:, 0].value

        return optimal_controls

    """
    def move_drone(self, vx, vy, z_min, duration, yaw_mode):

        self.airsim_client.moveByVelocityZAsync(vx, vy, z_min, duration, self.drivetrain_type,yaw_mode, "drone_1")
        # Wait for the command to complete
        time.sleep(duration)

    """
    """
    def control_loop(self):
        time.sleep(1)
        current_time = 0
        #print(self.drone_state)
        while self.running and self.drone_state != None :
            # Get current state from AirSim
            #drone_state = self.airsim_client.getMultirotorState()
            #drone_state = self.airsim_client_odom.getMultirotorState()
            #drone_state = self.drone_state
            #print(drone_state,"hi")
            position = self.drone_state.kinematics_estimated.position
            linear_velocity = self.drone_state.kinematics_estimated.linear_velocity
            x_current = np.array([
                position.x_val,
                position.y_val,
                position.z_val,
                linear_velocity.x_val,
                linear_velocity.y_val
            ])

            # Solve the MPC problem to get the optimal control
            optimal_controls = self.solve_mpc(x_current, current_time)
            print(optimal_controls )
            if optimal_controls is not None:
                # Extract optimal control commands
                vx, vy, vz = optimal_controls[0], optimal_controls[1], optimal_controls[2]
                
                # Get desired yaw from spline
                spline_index = min(int(current_time), len(self.smooth_orientations) - 1)
                desired_orientation = self.smooth_orientations[spline_index]
                desired_yaw = desired_orientation.as_euler('xyz')[2]
                
                # YawMode: Face the direction of travel
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=np.degrees(desired_yaw))
                
                # Apply the optimal controls to the drone
                self.move_drone(vx, vy,  self.z_min, self.dt, yaw_mode)
                
                # Update current time
            else:
                print("Warning: MPC solver failed to find optimal controls")

            current_time += self.dt
            
            # Optionally, implement a delay to maintain control frequency
            time.sleep(self.dt)
    """

    def control_loop(self):
        #time.sleep(2)
        current_time = 0 #self.controltime
        while self.is_control_running and self.drone_state is not None:
            # Get current state from drone
            position = self.drone_state.kinematics_estimated.position
            orri = self.drone_state.kinematics_estimated.orientation
            linear_velocity = self.drone_state.kinematics_estimated.linear_velocity
            x_current = np.array([
                position.x_val,
                position.y_val,
                #position.z_val,
                linear_velocity.x_val,
                linear_velocity.y_val,
                #linear_velocity.z_val
            ])
            r = R.from_quat([orri.x_val, orri.y_val, orri.z_val, orri.w_val])

            # Convert the rotation to Euler angles (roll, pitch, yaw)
            euler_angles = r.as_euler('xyz', degrees=False)

            # Extract yaw (z-axis rotation)
            self.currentyaw = euler_angles[2]

            #new_orientation = Quaternionr(orri.x_val,   orri.y_val, math.sin(self.yaw_smooth[int(current_time)] / 2),   math.cos(self.yaw_smooth[int(current_time)] / 2)  )
            #new_pose = Pose(position, new_orientation)

            #self.airsim_client.simSetVehiclePose(new_pose, True, self.drone_name)

            #print("current", x_current)
            # Solve the MPC problem to get the optimal control
            optimal_controls = self.solve_mpc(x_current, current_time)
            #print("control", optimal_controls)
            if optimal_controls is not None:
                # Extract optimal control commands (accelerations)
                ax, ay = optimal_controls

                # Compute desired velocities
                desired_velocity = x_current[2:4] + self.dt * optimal_controls

                #print("desired", desired_velocity)
                # YawMode: Face the direction of travel (optional)
                
                # Apply the optimal controls to the drone
                #print("targets", self.x_smooth[int(current_time)],self.y_smooth[int(current_time)])
                #print("z info ", self.z_smooth[int(current_time)-2],position.z_val)
                #print(current_time)
                if current_time < self.lsteps:
                    if args.level_name == "Qualifier_Tier_2":
                        setvalue = self.yaw_smooth[int(current_time)] 
                    else:
                        setvalue = self.yaw_smooth[int(current_time)] +90

                    setvalue = (setvalue + 180) % 360 - 180

                    self.currentyaw= math.degrees(self.currentyaw)

                    # Ensure the value is within the range of -180 to 180
                    self.currentyaw = (self.currentyaw + 180) % 360 - 180
                    self.yawval = self.yawCompute(setvalue,self.currentyaw,self.dt/self.sleeptimefactor)
                    self.yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=self.yawval)  # Adjust yaw as needed
                    #print(setvalue, self.currentyaw ,self.yawval )
                    #print("z info ", self.z_smooth[int(current_time)],position.z_val,self.Z_velocity)
                    self.Z_velocity = self.zCompute(self.z_smooth[int(current_time)],position.z_val,self.dt/self.sleeptimefactor)
                    self.errorz.append(self.z_smooth[int(current_time)]- position.z_val)
                    self.errory.append(self.y_smooth[int(current_time)]- position.y_val)
                    self.errorx.append(self.x_smooth[int(current_time)]- position.x_val)
                    self.pos.append([position.x_val,position.y_val,position.z_val])
                    #print(self.z_smooth[int(current_time)], position.z_val,self.y_smooth[int(current_time)], position.y_val,self.x_smooth[int(current_time)],position.x_val)
                    if self.Z_velocity > 15:
                        self.Z_velocity = 15
                        print("Z vel max reached")
                    elif self.Z_velocity < -15:
                        self.Z_velocity = -15
                        print("Z vel min reached")
                    #print("Z velocity",self.Z_velocity-2)
                    #print("z info ", self.z_smooth[current_time],position.z_val)
                    self.move_drone(
                        desired_velocity[0],
                        desired_velocity[1],
                        self.Z_velocity,
                        self.dt,
                        self.yaw_mode
                    )
                else:
                    self.move_drone(
                        x_current[2],
                        x_current[3],
                        0,
                        self.dt,
                        self.yaw_mode
                    )
                    time.sleep(2)
                    self.move_drone(
                        0,
                        0,
                        0,
                        self.dt,
                        self.yaw_mode
                    )
                    self.plotgraphs(self.gate_poses_ground_truth)
            else:
                print("Warning: MPC solver failed to find optimal controls")


            self.check_proximity_and_invoke_vision()
            while self.pausecontrol:
                pass
            #current_orientation = self.smooth_orientations[int(current_time)]
            #pitch, roll, yaw = current_orientation.as_euler('xyz', degrees=True)
            #print(pitch, roll, yaw, int(current_time),self.smooth_orientations)
            # Move the drone using the calculated pitch, roll, yaw, and position
            #print(self.pitch_smooth[int(current_time)], self.roll_smooth[int(current_time)], position.z_val, self.yaw_smooth[int(current_time)])

            
            #self.airsim_client.moveByAngleZAsync(90, 0, position.z_val, self.yaw_smooth[int(current_time)], 0.1, self.drone_name)
            if args.level_name == "Soccer_Field_Easy":
                
                current_time += self.dt*2
                self.sleeptimefactor = 1
                time.sleep(self.dt/self.sleeptimefactor )

            elif args.level_name == "Soccer_Field_Medium":

                current_time += self.dt
            
                time.sleep(self.dt/self.sleeptimefactor )
            else:
                current_time += self.dt
                self.sleeptimefactor 
                time.sleep(self.dt/self.sleeptimefactor )

            self.controltime  = current_time

            
            #time.sleep(self.dt/2)

    def move_drone(self, vx, vy, vz, duration, yaw_mode):
        self.airsim_client.moveByVelocityAsync( vx, vy, vz, duration,self.drivetrain_type,yaw_mode, vehicle_name="drone_1")
        #print("move command")


    def start_control_loop(self):
        """
        Starts the control loop in a separate thread.
        
        Parameters:
        - x_current: Current state of the drone
        - x_desired: Desired state [x, y, z, vx, vy]
        """
        self.is_control_running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()
        if self.control_thread:
            self.control_thread.join()

    def stop_control_loop(self):
        """
        Stops the control loop gracefully.
        """
        self.is_control_running = False  # Set running to False to stop the loop
        if self.control_thread.is_alive():
            self.control_thread.join() 


    def cal_gate_len(self,gate_poses):
        gate_positions = [gate_pose.position for gate_pose in gate_poses]
        self.totalinitial_gates = len(gate_positions) + 1

    
    """
    def follow_spline(self, x_smooth, y_smooth, z_smooth, smooth_orientations, control_frequency=10):
        # Loop over the generated spline and move the drone along the path
        for i in range(len(x_smooth)):
            target_position = airsim.Vector3r(x_smooth[i], y_smooth[i], z_smooth[i])
            target_orientation = smooth_orientations[i]
            target_quat = airsim.Quaternionr(*target_orientation.as_quat())  # Unpack the quaternion

            # Set the vehicle pose to the target position and orientation
            self.client.simSetVehiclePose(airsim.Pose(target_position, target_quat), ignore_collisions=True)

            # Simulate smooth motion with sleep (adjust time step to match control frequency)
            time.sleep(1.0 / control_frequency)

    """

def main(args):
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    baseline_racer = BaselineRacer(
        drone_name="drone_1",
        viz_traj=args.viz_traj,
        viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0],
        viz_image_cv2=args.viz_image_cv2,
    )
    baseline_racer.load_level(args.level_name)
    if args.level_name == "Qualifier_Tier_1":
        args.race_tier = 1
    if args.level_name == "Qualifier_Tier_2":
        args.race_tier = 2
    if args.level_name == "Qualifier_Tier_3":
        args.race_tier = 3

    baseline_racer.start_race(args.race_tier)
    baseline_racer.initialize_drone()
    baseline_racer.takeoff_with_moveOnSpline()
    
    baseline_racer.get_ground_truth_gate_poses()
    baseline_racer.get_camera_info()
    #baseline_racer.start_image_callback_thread()
    baseline_racer.start_odometry_callback_thread()
    baseline_racer.cal_gate_len(baseline_racer.gate_poses_ground_truth)
    #print(baseline_racer.gate_poses_ground_truth)
    #baseline_racer.generateSpline() self.gate_poses_ground_truth
    baseline_racer.generate_spline(baseline_racer.gate_poses_ground_truth)
    baseline_racer.start_control_loop()
    time.sleep(2000)  # Run for 30 seconds
    baseline_racer.stop_control_loop()


    """
    if args.planning_baseline_type == "all_gates_at_once":
        if args.planning_and_control_api == "moveOnSpline":
            baseline_racer.fly_through_all_gates_at_once_with_moveOnSpline().join()
        if args.planning_and_control_api == "moveOnSplineVelConstraints":
            baseline_racer.fly_through_all_gates_at_once_with_moveOnSplineVelConstraints().join()

    if args.planning_baseline_type == "all_gates_one_by_one":
        if args.planning_and_control_api == "moveOnSpline":
            baseline_racer.fly_through_all_gates_one_by_one_with_moveOnSpline().join()
        if args.planning_and_control_api == "moveOnSplineVelConstraints":
            baseline_racer.fly_through_all_gates_one_by_one_with_moveOnSplineVelConstraints().join()

    """

    # Comment out the following if you observe the python script exiting prematurely, and resetting the race
    #baseline_racer.stop_image_callback_thread()
    #baseline_racer.stop_odometry_callback_thread()
    #baseline_racer.reset_race()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--level_name",
        type=str,
        choices=[
            "Soccer_Field_Easy",
            "Soccer_Field_Medium",
            "ZhangJiaJie_Medium",
            "Building99_Hard",
            "Qualifier_Tier_1",
            "Qualifier_Tier_2",
            "Qualifier_Tier_3",
            "Final_Tier_1",
            "Final_Tier_2",
            "Final_Tier_3",
        ],
        default="ZhangJiaJie_Medium",
    )
    parser.add_argument(
        "--planning_baseline_type",
        type=str,
        choices=["all_gates_at_once", "all_gates_one_by_one"],
        default="all_gates_at_once",
    )
    parser.add_argument(
        "--planning_and_control_api",
        type=str,
        choices=["moveOnSpline", "moveOnSplineVelConstraints"],
        default="moveOnSpline",
    )
    parser.add_argument(
        "--enable_viz_traj", dest="viz_traj", action="store_true", default=True
    )
    parser.add_argument(
        "--enable_viz_image_cv2",
        dest="viz_image_cv2",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--vision_controller",
        type=bool,
        choices=[True, False],
        default=False,
    )
    parser.add_argument("--race_tier", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    main(args)
