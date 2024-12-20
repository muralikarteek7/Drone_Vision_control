
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

    In this project, we use a **State-Based Controller** that combines **Model Predictive Control (MPC)** for the **XY plane** and **PID control** for **Yaw** and **Altitude**.

    #### **MPC for XY Plane Control**
    The **Model Predictive Control (MPC)** is applied to control the drone’s position in the **XY plane**. MPC optimizes the control inputs over a prediction horizon, enabling the drone to follow the desired path while handling constraints effectively.

    #### **PID for Yaw and Altitude Control**
    - **Yaw Control**: A **PID controller** is used to control the yaw angle of the drone to ensure the desired orientation is maintained.
    - **Altitude Control**: Another **PID controller** manages the altitude, ensuring the drone stays at or moves toward the target height.

    Both control methods work together to ensure the drone can move to the desired position, maintain stable altitude, and adjust its orientation in flight.

   
3. **Install Dependencies**: Provides steps to install Python dependencies via `pip` using the `requirements.txt` file.

4. **Setting Up AirSim**: Detailed instructions on setting up the **AirSim** simulator on the user's system. It includes specific steps for both **Windows** and **Linux/Mac** environments.

5. **Running the Controller**: Instructions on how to run the drone controller once the setup is complete.

6. **Usage**: Brief explanation of how both the **vision-based** and **state-based controllers** work, along with an example of combining them.

7. **Test the Controller**: How to test various components of the project using **pytest**.

8. **Troubleshooting**: Common issues with setup and their solutions.

9. **Contributions**: Guidelines on how others can contribute to your project.

---

This README will guide users through every step of setting up your project and give them clear instructions on how to get started with AirSim, the controllers, and testing. Make sure to update the links and project-specific details as needed.
