
---
## AirSim Drone Racing Lab: Overview
<img src="https://github.com/muralikarteek7/Drone_Vision_control/blob/main/images/race.gif?raw=true" width="400"> 

ADRL is a framework for drone racing research, built on [Microsoft AirSim](https://github.com/Microsoft/Airsim).   
We used our framework to host a simulation-based drone racing competition at NeurIPS 2019, [Game of Drones](https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing). 

Currently, ADRL allows you to focus on three core research directions pertinent to autonomous drone racing -  perception, trajectory planning and control, and head-tp-head competition with a single competitor drone. 

## Race Tiers

<img src="https://github.com/muralikarteek7/Drone_Vision_control/blob/main/images/tier_1.gif?raw=true" width="400"> <img src="https://github.com/muralikarteek7/Drone_Vision_control/blob/main/images/tier_2.gif?raw=true" width="400"> 

1. **Airsim Setup**: 

    [AirSim-Drone-Racing-Lab Repository](https://github.com/microsoft/AirSim-Drone-Racing-Lab.git)


2. **Nanosam Setup**: 

    [Nanosam Repository](https://github.com/NVIDIA-AI-IOT/nanosam)


2. **State base controller**: 

    MPC for XY Plane Control

        Model Predictive Control (MPC) is used to control the drone’s position in the XY plane. MPC is an optimal control method that uses a model of the system to predict future states and solve an optimization problem to determine the best control inputs. It can handle constraints and nonlinearities in the system.

            XY Plane Control: In this system, MPC is applied to control the drone's position along the X and Y axes, aiming to move the drone to a desired waypoint.

            Implementation:
                The MPC algorithm calculates control inputs (like velocity or acceleration) by solving an optimization problem over a prediction horizon.
                The control inputs are adjusted at each time step based on the current state (position, velocity) of the drone.

            Tuning: MPC parameters (such as horizon length, control horizon, and weighting factors) are tuned to optimize the drone’s trajectory and performance.

    PID for Altitude Control

    Altitude control is another critical aspect of the drone's flight. The PID controller is used for controlling the drone's altitude (height above the ground). The goal is to maintain a stable altitude, or to smoothly transition to a new altitude, based on a target value.

        Altitude Control: Similar to yaw control, altitude control uses a PID controller to manage the drone’s altitude.

        Implementation:
            The altitude error is the difference between the target altitude and the current altitude.
            The PID controller adjusts the drone's vertical velocity to reduce this error over time.

    Combined Control System

    In your project, the State-Based Controller combines both MPC for the XY plane and PID for yaw and altitude.
        
   
3. **Install Dependencies**: Provides steps to install Python dependencies via `pip` using the `requirements.txt` file.

4. **Setting Up AirSim**: Detailed instructions on setting up the **AirSim** simulator on the user's system. It includes specific steps for both **Windows** and **Linux/Mac** environments.

5. **Running the Controller**: Instructions on how to run the drone controller once the setup is complete.

6. **Usage**: Brief explanation of how both the **vision-based** and **state-based controllers** work, along with an example of combining them.

7. **Test the Controller**: How to test various components of the project using **pytest**.

8. **Troubleshooting**: Common issues with setup and their solutions.

9. **Contributions**: Guidelines on how others can contribute to your project.

---

This README will guide users through every step of setting up your project and give them clear instructions on how to get started with AirSim, the controllers, and testing. Make sure to update the links and project-specific details as needed.
