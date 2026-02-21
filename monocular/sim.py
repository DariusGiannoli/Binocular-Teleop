import mujoco
import mujoco.viewer

class LeapSimulator:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def set_actuators(self, angles):
        for i, angle in enumerate(angles):
            if i < self.model.nu:
                self.data.ctrl[i] = angle

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
