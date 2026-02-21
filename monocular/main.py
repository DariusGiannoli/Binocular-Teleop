from camera import MonocularCamera
from detector import HandDetector
from retargeting import landmarks_to_joints
from sim import LeapSimulator

XML_PATH = "../leap_hand/scene_right.xml"

def main():
    camera = MonocularCamera(camera_id=0)
    detector = HandDetector()
    sim = LeapSimulator(XML_PATH)
    sim.launch_viewer()

    print("Running — close the MuJoCo window to quit.")

    while sim.viewer.is_running():
        frame = camera.get_frame()
        if frame is None:
            continue

        results = detector.process(frame)
        landmarks = detector.get_landmarks(results)

        if landmarks is not None:
            joints = landmarks_to_joints(landmarks)
            sim.set_actuators(joints)

        sim.step()

    camera.close()
    sim.close()

if __name__ == "__main__":
    main()
