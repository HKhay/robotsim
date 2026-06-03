import time
import random
import numpy as np

from robots import *
from coppeliasim_zmqremoteapi_client import *


client = RemoteAPIClient()
sim = client.require("sim")


# --------------------
# ROBOT SETUP
# --------------------

robot = Robot_OS(sim, DeviceNames.ROBOT_OS)

top_image_sensor = ImageSensor(sim, DeviceNames.TOP_IMAGE_SENSOR_OS)
small_image_sensor = ImageSensor(sim, DeviceNames.SMALL_IMAGE_SENSOR_OS)

left_motor = Motor(sim, DeviceNames.MOTOR_LEFT_OS, Direction.CLOCKWISE)
right_motor = Motor(sim, DeviceNames.MOTOR_RIGHT_OS, Direction.CLOCKWISE)


# --------------------
# BASIC MOVEMENT
# --------------------

def move_forward(speed=1.5):
    left_motor.run(speed)
    right_motor.run(speed)


def move_backward(speed=1.5):
    left_motor.run(-speed)
    right_motor.run(-speed)


def spin(direction=1, speed=2):
    """
    direction = 1  -> spin one way
    direction = -1 -> spin the other way
    """
    left_motor.run(speed * direction)
    right_motor.run(-speed * direction)


def stop():
    left_motor.run(0)
    right_motor.run(0)


# --------------------
# SENSOR HELPERS
# --------------------

def get_sonar():
    """
    Manual sonar reading.
    This avoids the binary signal issue from robot.get_sonar_sensor().
    """
    try:
        sonar_handle = sim.getObject('/dr12/dr12_sonar')
        result = sim.readProximitySensor(sonar_handle)

        if result[0] > 0:
            return result[1]

        return -1
    except:
        return -1


def get_center_rgb(sensor):
    sensor._update_image()
    img = sensor.image

    h, w = img.shape[0], img.shape[1]
    cy, cx = h // 2, w // 2

    region = img[
        cy - h // 6: cy + h // 6,
        cx - w // 6: cx + w // 6
    ]

    r = float(np.mean(region[:, :, 0])) / 255 * 100
    g = float(np.mean(region[:, :, 1])) / 255 * 100
    b = float(np.mean(region[:, :, 2])) / 255 * 100

    return r, g, b


# --------------------
# CONTAINER DETECTION
# --------------------
def detect_cube_front():
    r, g, b = get_center_rgb(small_image_sensor)

    # Green plant cube
    seeing_plant = (
        g > 35 and
        g > r * 1.4 and
        g > b * 1.4
    )

    # Brown/red trash cube
    # Your logs show trash/brown around:
    # r=40-55, g=17-30, b=0-20
    seeing_trash = (
        r > 30 and
        g < 35 and
        b < 25 and
        r > g * 1.4 and
        r > b * 1.8
    )

    if seeing_plant:
        return "Plant"

    if seeing_trash:
        return "Trash"

    return None

def detect_container_front():
    r, g, b = get_center_rgb(small_image_sensor)

    seeing_blue = (
        b > 45 and
        b > r * 1.8 and
        b > g * 1.1
    )

    if seeing_blue:
        return "Blue"

    return None



def emergency_avoid_container():
    container = detect_container_front()
    dist = get_sonar()
    r, g, b = get_center_rgb(small_image_sensor)

    print(
        f"RGB r={r:.1f}, g={g:.1f}, b={b:.1f} | "
        f"container={container} | sonar={dist}"
    )

    if container is not None:
        print(f"DANGER: {container} container detected. Backing away.")

        stop()
        time.sleep(0.2)

        move_backward(2)
        time.sleep(1.0)

        spin(random.choice([-1, 1]), 2.5)
        time.sleep(0.8)

        stop()
        time.sleep(0.2)

        return True

    return False


# --------------------
# SAFE WANDER ONLY
# --------------------

def safe_wander():
    # For now, do NOT back away from sonar objects.
    # We are only testing container avoidance.
    move_forward(1.0)
    time.sleep(0.4)

    if random.random() < 0.2:
        spin(random.choice([-1, 1]), 1.5)
        time.sleep(0.25)


# --------------------
# START SIMULATION
# --------------------

sim.startSimulation()
time.sleep(2)

print("Starting clean safety test...")


# --------------------
# MAIN LOOP
# --------------------

while True:
    if emergency_avoid_container():
        continue

    safe_wander()