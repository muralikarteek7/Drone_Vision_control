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
        



        self.drone_state = None


        self.max_vx = 10
        self.max_vy = 10
        self.max_vz = 10
        self.max_ax = 5  # Maximum acceleration in x
        self.max_ay = 5  # Maximum acceleration in y
        self.max_az = 5  # Maximum acceleration in z


        self.Z_velocity = 0

        self.z_min = -10
        self.duration = 10
        self.drivetrain_type = airsim.DrivetrainType.ForwardOnly  #airsim.DrivetrainType.MaxDegreeOfFreedom
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
        #self.Q = np.diag([100,100,3,3])     #np.diag([60,60,5,5])  # Weighting on states (position and velocity)
        self.Q = np.diag([60,60,5,5])
        self.R = np.diag([1,1])       #np.eye(2)  # Weighting on control inputs (velocity and yaw rate)
        
        self.running = False
        self.control_thread = None






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


    def image_callback(self):

        camera_name="fpv_cam"
        requests = [
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, pixels_as_float=True),
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True),
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, compress=False),  # uncompressed RGB
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, compress=True)    # compressed RGB (PNG)
        ]

        responses = self.airsim_client.simGetImages(requests)
        save_dir =  "/home/mkg7/nanosam/assets"
        os.makedirs(save_dir, exist_ok=True)
        """
        for idx, response in enumerate(responses):
            # Check if the response matches the target dimensions
            if response.width == 640 and response.height == 480:
                filename = os.path.join(save_dir, f"image_{idx}")
                
                if response.pixels_as_float:  # Depth images
                    print(f"Saving depth image {idx} with resolution {response.width}x{response.height}")
                    airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
                elif response.compress:  # Compressed RGB image (PNG)
                    print(f"Saving compressed RGB image {idx} with resolution {response.width}x{response.height}")
                    airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
                else:  # Uncompressed RGB image
                    print(f"Saving uncompressed RGB image {idx} with resolution {response.width}x{response.height}")
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # Get numpy array
                    img_rgb = img1d.reshape(response.height, response.width, 3)       # Reshape array to H x W x 3
                    cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb)         # Save image
                    cv2.imshow("img_rgb", img_rgb)
                    cv2.waitKey(1)
            else:
                print(f"Image {idx} has unexpected resolution: {response.width}x{response.height}")
        """



    def generate_spline(self, gate_poses):
        gate_positions = [gate_pose.position for gate_pose in gate_poses]
        gate_orientations = [gate_pose.orientation for gate_pose in gate_poses]
        # print(gate_positions)
        positions = np.array([(pos.x_val, pos.y_val, pos.z_val) for pos in gate_positions])
        orientations = np.array([(ori.x_val, ori.y_val, ori.z_val, ori.w_val) for ori in gate_orientations])
        print(positions)
        x_gate = []
        y_gate = [] 
        z_gate = []  
        for pos in positions:
            print("hi")
            x_gate.append(pos[0])
            y_gate.append(pos[1])
            z_gate.append(pos[2])
        # print(x,y,z)
        # print(x)
        x,y,z = x_gate, y_gate, z_gate
        t = np.arange(len(x_gate))
        k = 2


        if args.level_name == "Soccer_Field_Easy":
            self.lsteps = 50
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)
            t_fine = np.linspace(t[0], t[-1], self.lsteps)
            #self.lsteps = self.lsteps +1
            #t_fine_extended = np.append(t_fine, t[-1] + self.lsteps) 

        elif args.level_name == "Soccer_Field_Medium":

            self.kp = 1.7 #1.5
            self.ki = 0.1#0.09
            self.kd = 0.09#0.05
            self.lsteps = 230
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)
            t_fine = np.linspace(t[0], t[-1], self.lsteps) #130
            #self.lsteps = self.lsteps +1
            #t_fine_extended = np.append(t_fine, t[-1] + self.lsteps) 
        else:
            self.kp = 1.7 #1.5
            self.ki = 0.1#0.09
            self.kd = 0.09#0.05
            self.lsteps = 150
            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)
            #spline_x = BSpline(t, x, k)
            #spline_y = BSpline(t, y, k)
            #spline_z = BSpline(t, z, k)
            t_fine = np.linspace(t[0], t[-1], self.lsteps)
            #self.lsteps = self.lsteps +1
            #t_fine_extended = np.append(t_fine, t[-1] + self.lsteps*5) 


        #t_fine = np.linspace(t[0], t[-1], 50)  #50 for soccer easy

        
        self.x_smooth = spline_x(t_fine)
        self.y_smooth = spline_y(t_fine)
        self.z_smooth = spline_z(t_fine)

        self.dx_smooth = spline_x(t_fine, 1)
        self.dy_smooth = spline_y(t_fine, 1)
        self.dz_smooth = spline_z(t_fine, 1)

        rotations = R.from_quat(orientations)
        slerp = Slerp(t, rotations)
        self.smooth_orientations = slerp(t_fine)

        points = []
        for x, y, z in zip(self.x_smooth, self.y_smooth, self.z_smooth):
            points.append(Vector3r(x, y, z))

        # Plot the spline
        self.airsim_client.simPlotLineList(points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the spline trajectory
        ax.plot(self.x_smooth, self.y_smooth, self.z_smooth, label='Spline Trajectory', color='blue')
        
        # Plot the start and stop positions with different colors
        ax.scatter(self.x_smooth[0], self.y_smooth[0], self.z_smooth[0], color='green', s=100, label='Stop Position')  # Start position
        ax.scatter(self.x_smooth[-1], self.y_smooth[-1], self.z_smooth[-1], color='red', s=100, label='Start Position')   # Stop position
        
        # Plot all the gate positions
        print(x)
        ax.scatter(x_gate, y_gate, z_gate, label='Gate Positions', s=10, color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        # ax.show()
        # plt.show()
        plt.savefig('spline_for_medium.png')
        plt.close()

        # # Print a sample of positions and orientations
        # for pos, ori in zip(zip(self.x_smooth, self.y_smooth, self.z_smooth), smooth_orientations.as_quat()):
        #     print(f"Position: {pos}, Orientation (Quaternion): {ori}")

        # for i in range(len(t_fine)):
        #     target_position = airsim.Vector3r(x_smooth[i], y_smooth[i], z_smooth[i])
        #     target_orientation = smooth_orientations[i]
        #     target_quat = airsim.Quaternionr(*target_orientation.as_quat())  # Unpack the quaternion
    
        #     # Move the drone to the target position and orientation
        #     self.airsim_client.simSetVehiclePose(airsim.Pose(target_position, target_quat), ignore_collisions=True)
        #     time.sleep(0.1)  # Delay to simulate smooth motion
        
            
    

        #return x_smooth, y_smooth, z_smooth, smooth_orientations
    """
    def image_callback(self):
        # get uncompressed fpv cam image

        requests = [airsim.ImageRequest("0", airsim.ImageType.DepthVis), airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True),airsim.ImageRequest("1", airsim.ImageType.Scene),airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        responses = self.airsim_client_images.simGetImages(requests)

        tmp_dir = os.path.join("/home/mkg7/AirSim-1.5.0-linux/AirSim-Drone-Racing-Lab/baselines", "airsim_drone")  #tempfile.gettempdir()

        print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise

        for idx, response in enumerate(responses):

            filename = os.path.join(tmp_dir, str(idx))

            if response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            elif response.compress: #png format
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else: #uncompressed array
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
                print(img_rgb.shape)
                cv2.imshow("img_rgb", img_rgb)
                cv2.waitKey(1)
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
        time.sleep(0.02)  # Adjust sleep as needed

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
        if len(self.error_history) > 4:
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
        time.sleep(3)
        current_time = 0
        while self.running and self.drone_state is not None:
            # Get current state from drone
            position = self.drone_state.kinematics_estimated.position
            linear_velocity = self.drone_state.kinematics_estimated.linear_velocity
            x_current = np.array([
                position.x_val,
                position.y_val,
                #position.z_val,
                linear_velocity.x_val,
                linear_velocity.y_val,
                #linear_velocity.z_val
            ])
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
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)  # Adjust yaw as needed

                # Apply the optimal controls to the drone
                #print("targets", self.x_smooth[int(current_time)],self.y_smooth[int(current_time)])
                #print("z info ", self.z_smooth[int(current_time)-2],position.z_val)
                #print(current_time)
                if current_time < self.lsteps:
                    print("z info ", self.z_smooth[int(current_time)],position.z_val,self.Z_velocity)
                    self.Z_velocity = self.zCompute(self.z_smooth[int(current_time)],position.z_val,self.dt)
                    if self.Z_velocity > 10:
                        self.Z_velocity = 10
                        print("Z vel max reached")
                    elif self.Z_velocity < -10:
                        self.Z_velocity = -10
                        print("Z vel min reached")
                    #print("Z velocity",self.Z_velocity-2)
                #print("z info ", self.z_smooth[current_time],position.z_val)
                self.move_drone(
                    desired_velocity[0],
                    desired_velocity[1],
                    self.Z_velocity,
                    self.dt,
                    yaw_mode
                )
            else:
                print("Warning: MPC solver failed to find optimal controls")

            
            if args.level_name == "Soccer_Field_Easy":
                current_time += self.dt

            elif args.level_name == "Soccer_Field_Medium":
                current_time += self.dt*1.35
            else:
                current_time += self.dt

            current_time += self.dt
            time.sleep(self.dt)

    def move_drone(self, vx, vy, vz, duration, yaw_mode):
        self.airsim_client.moveByVelocityAsync( vx, vy, vz, duration, vehicle_name="drone_1")
        print("move command")


    def start_control_loop(self):
        """
        Starts the control loop in a separate thread.
        
        Parameters:
        - x_current: Current state of the drone
        - x_desired: Desired state [x, y, z, vx, vy]
        """
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

    def stop_control_loop(self):
        """Stop the control loop thread."""
        self.running = False
        if self.control_thread:
            self.control_thread.join()

    
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
    #baseline_racer.start_image_callback_thread()
    baseline_racer.start_odometry_callback_thread()
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
    parser.add_argument("--race_tier", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    main(args)