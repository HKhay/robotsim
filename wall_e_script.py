from robots import *
import time
from coppeliasim_zmqremoteapi_client import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

client = RemoteAPIClient()
sim = client.require("sim")

# HANDLES FOR ACTUATORS AND SENSORS
robot = Robot_OS(sim, DeviceNames.ROBOT_OS)

top_image_sensor = ImageSensor(sim, DeviceNames.TOP_IMAGE_SENSOR_OS)
small_image_sensor = ImageSensor(sim, DeviceNames.SMALL_IMAGE_SENSOR_OS)

left_motor = Motor(sim, DeviceNames.MOTOR_LEFT_OS, Direction.CLOCKWISE)
right_motor = Motor(sim, DeviceNames.MOTOR_RIGHT_OS, Direction.CLOCKWISE)

# HELPER FUNCTION
def show_image(image):
    plt.imshow(image)
    plt.show()

def get_battery():
    signal = robot.get_string_signal("battery")
    if signal is None:
        return 0.0
    return float(signal.decode("utf-8") if isinstance(signal, bytes) else signal)
# Starts coppeliasim simulation if not done already
sim.startSimulation()

time.sleep(0.5)

# MAIN CONTROL LOOP
while True:
    print(get_battery())
    left_motor.run(3)
    right_motor.run(-3)
    robot.compress()