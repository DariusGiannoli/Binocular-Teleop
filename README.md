# Binocular Teleoperation for Dexterous Humanoid Manipulation

A robust, markerless teleoperation pipeline designed to control a highly articulated robotic hand in physics-based simulation using a single binocular stereo camera. 

Unlike standard "angle-mapping" teleop systems, this project utilizes **epipolar geometry** for true 3D spatial tracking and **optimization-based kinematic retargeting** to map human hand vectors to robotic joint limits. The ultimate goal is to generate high-fidelity, jitter-free `(State, Action)` datasets for training Imitation Learning (IL) models, such as Diffusion Policies, for autonomous robotic manipulation.

---

## 1. Executive Summary
This project presents a low-cost, high-fidelity infrastructure for collecting expert demonstration datasets for humanoid robotics. Utilizing a Stereolabs ZED Camera and computer vision heuristics, the system captures human hand movements and translates them into precise, 1:1 kinematic control signals for the open-source LEAP Hand in MuJoCo. By bridging the anatomical mismatch between humans and robots through Inverse Kinematics (IK), the system provides research-grade deterministic control suitable for advanced machine learning pipelines and zero-shot Sim-to-Real transfer.

## 2. Core Objectives
* **Precision Teleoperation:** Achieve millimeter-accurate, 6 Degrees of Freedom (6-DoF) spatial tracking and continuous finger articulation without the need for expensive motion-capture suits or data gloves.
* **Kinematic Retargeting:** Map the anatomical structure of a human hand to the mechanical constraints of the LEAP hand using optimization-based IK to prevent self-collisions and ensure accurate, natural grasping.
* **High-Quality Data Generation:** Establish a synchronized, deterministic control loop to record clean, jitter-free episodic datasets suitable for Imitation Learning frameworks.
* **Sim-to-Real Readiness:** Develop the control architecture entirely within MuJoCo using official hardware XML models, ensuring policies trained in simulation transfer directly to physical hardware.

---

## 3. System Architecture

### A. Perception & Vision Layer
* **Hardware:** A ZED Binocular Stereo Camera captures the human operator's workspace.
* **Feature Extraction:** MediaPipe extracts 21 2D landmarks from the operator’s hand in both the left and right camera frames simultaneously.
* **Depth Triangulation:** Epipolar Geometry matches corresponding landmarks between frames, triangulating true 3D spatial coordinates (X, Y, Z in meters) in real-time.
* **Signal Processing:** A One Euro Filter (or Exponential Moving Average) is applied to raw landmark data to eliminate high-frequency camera jitter, ensuring actuators receive perfectly smooth target trajectories.

### B. Kinematic Retargeting Layer
* **1:1 Spatial Mapping:** The Pinhole Camera Model un-projects pixel coordinates into true physical workspace coordinates (moving the human hand exactly 10 cm moves the robot exactly 10 cm).
* **6-DoF Wrist Pose:** Spatial vectors between the human wrist, index knuckle, and pinky knuckle compute a mathematically robust Quaternion (Pitch, Yaw, Roll) to drive the robot's orientation.
* **Finger Articulation via IK:** Instead of naive joint-angle copying, the system solves for the optimal motor radians required for the LEAP hand's 16 actuators to reach the operator's 3D fingertip coordinates.

### C. Simulation & Control Layer
* **Physics Engine:** Built in MuJoCo, simulating rigid body dynamics, joint limits, and friction constraints to mimic real-world manipulation.
* **Mocap Proxy Control:** A virtual "mocap" body pulls the base of the LEAP hand through space using stable spring-damper equality constraints (welds), preventing physics explosions from aggressive human movements.
* **Clutched Teleoperation:** An ergonomic clutch mechanism allows the user to decouple their hand from the robot, reposition, and re-engage—enabling infinitely scalable workspace control.

### D. Machine Learning & Dataset Pipeline
* **Asynchronous Engine:** The vision processing (~30Hz) and the physics engine (50Hz) are decoupled to prevent simulation lag.
* **Episodic Recording:** A data logger captures the front-facing camera frame (pixels), current joint states (`qpos`), and human-commanded actions (`ctrl`) at a fixed frequency, saving them into `.hdf5` or `.zarr` formats.

---

## 4. Hardware & Software Requirements

* **Camera:** Stereolabs ZED Camera (ZED Mini / ZED 2 / ZED 2i)
* **Robot Model:** [DeepMind MuJoCo Menagerie - LEAP Hand](https://github.com/google-deepmind/mujoco_menagerie)
* **Environment:** MuJoCo Physics Simulator
* **Dependencies:** Python 3.10+, OpenCV, MediaPipe, ZED SDK, NumPy

---

## 5. Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/binocular-teleop.git](https://github.com/yourusername/binocular-teleop.git)
   cd binocular-teleop