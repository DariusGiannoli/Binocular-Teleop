import os
import tempfile
import time
import math
import numpy as np
import mujoco
from mujoco import viewer

# Import your modules
from camera import ZEDCamera
from detectors import StereoHandTracker
import geometry

# --- XML DEFINITION ---
xml_string = """
<mujoco>
  <include file="leap_hand/right_hand.xml"/>
  
  <statistic center="0 0 0.2" extent="0.5"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <worldbody>
    <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 1.0" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 1" rgba="0.2 0.3 0.4 1" pos="0 0 0"/>
    
    <body name="hand_proxy" mocap="true" pos="0 0 0.3">
        <geom type="box" size=".02 .02 .02" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
    </body>
    
    <body name="target_box" pos="0.1 0 0.05">
        <joint type="free"/>
        <geom type="box" size=".02 .02 .02" rgba="0 1 0 1" mass="0.05"/>
    </body>
  </worldbody>

  <equality>
    <weld body1="hand_proxy" body2="palm"/>
  </equality>
</mujoco>
"""

# Global references
zed = None
tracker = None
mocap_id = None
thumb_actuators = []
index_actuators = []

def update_physics(model, data):
    """
    Runs every physics step.
    Processes vision and moves the floating LEAP Hand + controls the fingers.
    """
    global zed, tracker, mocap_id, thumb_actuators, index_actuators
    
    if zed:
        frame_l, frame_r = zed.get_frames()
        if frame_l is not None:
            res_l, res_r = tracker.process(frame_l, frame_r)
            
            if res_l.multi_hand_landmarks and res_r.multi_hand_landmarks:
                hand_l = res_l.multi_hand_landmarks[0]
                hand_r = res_r.multi_hand_landmarks[0]
                
                # --- 1. POSITION TRACKING (Knuckle X,Y,Z) ---
                kp_l = hand_l.landmark[5]
                kp_r = hand_r.landmark[5]
                h, w, _ = frame_l.shape
                
                px_l, py_l = int(kp_l.x * w), int(kp_l.y * h)
                px_r = int(kp_r.x * w)
                
                z_cm = geometry.triangulate_depth(px_l, px_r)
                
                if z_cm:
                    z_m = z_cm / 100.0
                    x_norm = (kp_l.x - 0.5)
                    y_norm = (kp_l.y - 0.5)
                    
                    # Adjust mapping for floating hand workspace
                    sim_x = x_norm * 1.0 
                    sim_y = (1.0 - z_m) + 0.1 
                    sim_z = -y_norm * 1.0 + 0.3 
                    
                    data.mocap_pos[mocap_id] = [sim_x, sim_y, sim_z]

                # --- 2. GRIPPER TRACKING (Pinch gesture) ---
                thumb_tip = hand_l.landmark[4]
                index_tip = hand_l.landmark[8]
                
                # Euclidean distance between Thumb and Index on camera plane
                pinch_dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                
                # If distance is small enough (pinching), set target joint to 1.0 radian (curled)
                target_pos = 1.0 if pinch_dist < 0.05 else 0.0
                
                # Apply control signal to ONLY the thumb and index finger joints
                for act_idx in thumb_actuators + index_actuators:
                    data.ctrl[act_idx] = target_pos

def main():
    global zed, tracker, mocap_id, thumb_actuators, index_actuators
    
    zed = ZEDCamera(camera_id=1) # Try 0 if 1 fails
    tracker = StereoHandTracker()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    leap_hand_dir = os.path.join(script_dir, 'leap_hand')
    xml_for_load = xml_string.replace(
        '<include file="leap_hand/right_hand.xml"/>',
        '<include file="right_hand.xml"/>'
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=leap_hand_dir, delete=False) as f:
        f.write(xml_for_load)
        tmp_path = f.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    data = mujoco.MjData(model)
    mocap_id = model.body("hand_proxy").mocapid[0]

    # --- ACTUATOR DISCOVERY ---
    # Automatically scan the model and find the actuator indices for the thumb and index fingers.
    # This prevents errors if DeepMind changes the motor order in the XML file!
    for i in range(model.nu):
        # We wrap in try/except just in case some actuators have no name
        try:
            name = model.actuator(i).name.lower()
            if 'thumb' in name:
                thumb_actuators.append(i)
            elif 'index' in name:
                index_actuators.append(i)
        except AttributeError:
            continue

    print("Starting Simulation...")
    print(f"Discovered {len(thumb_actuators)} Thumb motors and {len(index_actuators)} Index motors.")
    print("Move your hand to move the LEAP Hand in 3D space.")
    print("Pinch your physical thumb and index to grasp the green box.")
    print("Press ESC in the viewer window to quit.")

    # Launch the viewer (Blocking)
    viewer.launch(model, data, loader=None)

    zed.close()

if __name__ == "__main__":
    main()