from robots import *
from coppeliasim_zmqremoteapi_client import *
import time
import random
import numpy as np


# ============================================================
# CONNECT TO COPPELIASIM
# ============================================================

client = RemoteAPIClient()
sim = client.require("sim")


# ============================================================
# SETUP
# ============================================================

robot = Robot_OS(sim, DeviceNames.ROBOT_OS)

top_image_sensor = ImageSensor(sim, DeviceNames.TOP_IMAGE_SENSOR_OS)
small_image_sensor = ImageSensor(sim, DeviceNames.SMALL_IMAGE_SENSOR_OS)

left_motor = Motor(sim, DeviceNames.MOTOR_LEFT_OS, Direction.CLOCKWISE)
right_motor = Motor(sim, DeviceNames.MOTOR_RIGHT_OS, Direction.CLOCKWISE)


# ============================================================
# SETTINGS
# ============================================================

DEBUG = True

FORWARD_SPEED = 1.4
APPROACH_SPEED = 1.1
SLOW_SPEED = 0.7
TURN_SPEED = 1.0
PUSH_SPEED = 0.9
BACK_SPEED = -1.2

WALL_DISTANCE = 0.18

OBJECT_MIN_PIXELS = 10
OBJECT_CLOSE_PIXELS = 90

GOAL_MIN_PIXELS_TOP = 40
GOAL_MIN_PIXELS_FRONT = 120

CENTER_TOLERANCE = 0.18
STEP_TIME = 0.15


# ============================================================
# BASIC MOVEMENT
# ============================================================

def debug(message):
    if DEBUG:
        print(message)


def clamp(value, low, high):
    return max(low, min(high, value))


def drive(forward, turn=0.0):
    """
    forward > 0 = move forward
    forward < 0 = move backward
    turn > 0    = turn right
    turn < 0    = turn left
    """

    left_speed = clamp(forward + turn, -2.5, 2.5)
    right_speed = clamp(forward - turn, -2.5, 2.5)

    left_motor.run(left_speed)
    right_motor.run(right_speed)


def stop():
    left_motor.run(0)
    right_motor.run(0)


def back_away():
    drive(BACK_SPEED, 0)
    time.sleep(0.6)

    drive(0, random.choice([-1, 1]) * TURN_SPEED)
    time.sleep(0.6)

    stop()


# ============================================================
# SENSOR HELPERS
# ============================================================

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

    except Exception:
        return -1


def get_bumper_force():
    try:
        bumper_handle = sim.getObject('/dr12/dr12_bumperForceSensor_')
        result = sim.readForceSensor(bumper_handle)

        if result[0] > 0:
            force = result[1]
            return abs(force[0]) + abs(force[1]) + abs(force[2])

        return 0.0

    except Exception:
        return 0.0


def get_camera_image(sensor):
    sensor._update_image()
    return sensor.image


# ============================================================
# PIXEL CLASSIFIERS
# ============================================================

def is_plant_pixel(r, g, b):
    """
    Green cube / plant.
    """

    return (
        g > 80 and
        g > r * 1.25 and
        g > b * 1.25
    )


def is_trash_pixel(r, g, b):
    """
    Brown/orange/red-brown trash cube.
    This is deliberately NOT too strict because the trash cube can look orange-brown.
    """

    return (
        r > 70 and
        g > 20 and
        g < 120 and
        b < 90 and
        r > g * 1.10 and
        r > b * 1.35
    )


def is_blue_goal_pixel(r, g, b):
    """
    Blue plant collector.
    """

    return (
        b > 90 and
        g > 70 and
        r < 90 and
        b > r * 1.4 and
        g > r * 1.2
    )


def is_red_goal_pixel(r, g, b):
    """
    Red trash container.
    Much stricter than trash detection so the brown cube is not treated as the container.
    """

    return (
        r > 150 and
        g < 90 and
        b < 90 and
        r > g * 1.6 and
        r > b * 1.6
    )


# ============================================================
# BLOB DETECTION
# ============================================================

def detect_blob(sensor, classifiers, min_pixels=10, lower_camera=True):
    """
    Finds the largest matching colour blob.

    Returns:
    {
        "label": "Plant" / "Trash" / "Blue" / "Red",
        "pixels": number of matching pixels,
        "offset": -1 left, 0 center, +1 right,
        "bottom": how low the blob is in the image
    }

    or None.
    """

    img = get_camera_image(sensor)

    height = img.shape[0]
    width = img.shape[1]

    if lower_camera:
        y_start = int(height * 0.25)
        y_end = int(height * 0.95)
    else:
        y_start = int(height * 0.05)
        y_end = int(height * 0.95)

    x_start = int(width * 0.03)
    x_end = int(width * 0.97)

    best_label = None
    best_pixels = []
    best_bottom = 0

    for label, classifier in classifiers:
        pixels = []
        bottom = 0

        for y in range(y_start, y_end, 3):
            for x in range(x_start, x_end, 3):
                r = img[y, x, 0]
                g = img[y, x, 1]
                b = img[y, x, 2]

                if classifier(r, g, b):
                    pixels.append(x)
                    bottom = max(bottom, y)

        if len(pixels) > len(best_pixels):
            best_label = label
            best_pixels = pixels
            best_bottom = bottom

    if best_label is None or len(best_pixels) < min_pixels:
        return None

    average_x = sum(best_pixels) / len(best_pixels)
    offset = (average_x - width / 2) / (width / 2)
    bottom_fraction = best_bottom / height

    return {
        "label": best_label,
        "pixels": len(best_pixels),
        "offset": offset,
        "bottom": bottom_fraction
    }


def detect_object():
    """
    Detects only plant or trash.
    Does NOT detect compressed/dark objects, because that caused false positives.
    """

    obj = detect_blob(
        small_image_sensor,
        [
            ("Plant", is_plant_pixel),
            ("Trash", is_trash_pixel)
        ],
        min_pixels=OBJECT_MIN_PIXELS,
        lower_camera=True
    )

    if obj is not None:
        debug(
            f"Object: {obj['label']} | pixels={obj['pixels']} | "
            f"offset={obj['offset']:.2f} | bottom={obj['bottom']:.2f}"
        )

    return obj


def detect_top_goal():
    goal = detect_blob(
        top_image_sensor,
        [
            ("Blue", is_blue_goal_pixel),
            ("Red", is_red_goal_pixel)
        ],
        min_pixels=GOAL_MIN_PIXELS_TOP,
        lower_camera=False
    )

    if goal is not None:
        debug(
            f"Top goal: {goal['label']} | pixels={goal['pixels']} | "
            f"offset={goal['offset']:.2f}"
        )

    return goal


def detect_front_goal():
    goal = detect_blob(
        small_image_sensor,
        [
            ("Blue", is_blue_goal_pixel),
            ("Red", is_red_goal_pixel)
        ],
        min_pixels=GOAL_MIN_PIXELS_FRONT,
        lower_camera=True
    )

    if goal is not None:
        debug(
            f"Front goal: {goal['label']} | pixels={goal['pixels']} | "
            f"offset={goal['offset']:.2f}"
        )

    return goal


# ============================================================
# TASK HELPERS
# ============================================================

def target_goal_for_object(object_type):
    if object_type == "Plant":
        return "Blue"

    if object_type == "Trash":
        return "Red"

    return None


def compress_trash():
    debug("ACTION: Compress trash")

    stop()
    time.sleep(0.2)

    try:
        robot.compress()
    except Exception:
        try:
            sim.setInt32Signal("compress", 1)
        except Exception:
            pass

    time.sleep(0.8)
    stop()


# ============================================================
# SUBSUMPTION BEHAVIOURS
# ============================================================

def avoid_container_when_not_pushing():
    """
    High priority safety behaviour.
    If the robot sees a red/blue container in front while not pushing,
    it backs away.
    """

    front_goal = detect_front_goal()

    if front_goal is not None:
        debug(f"SAFETY: avoiding {front_goal['label']} container")
        back_away()
        return True

    return False


def avoid_wall_if_needed():
    """
    Wall avoidance only happens if:
    - sonar sees something very close
    - the camera does NOT see an object

    This prevents the robot from backing away from cubes.
    """

    distance = get_sonar()

    if distance == -1 or distance > WALL_DISTANCE:
        return False

    obj = detect_object()

    if obj is not None:
        debug("Close object is a cube, not a wall")
        return False

    debug("SAFETY: avoiding wall/unknown obstacle")
    back_away()
    return True


def wander():
    """
    Lowest priority behaviour.
    Used only when no object is visible.
    """

    debug("BEHAVIOUR: Wander")

    if random.random() < 0.20:
        drive(0, random.choice([-1, 1]) * TURN_SPEED)
    else:
        drive(FORWARD_SPEED, random.choice([-0.15, 0.15]))

    time.sleep(STEP_TIME)


def approach_object(obj):
    """
    Moves toward the detected cube.
    """

    offset = obj["offset"]
    pixels = obj["pixels"]
    bottom = obj["bottom"]

    turn = clamp(offset * 0.9, -0.5, 0.5)

    if pixels > 60 or bottom > 0.70:
        speed = SLOW_SPEED
    else:
        speed = APPROACH_SPEED

    debug(f"BEHAVIOUR: Approach {obj['label']}")

    drive(speed, turn)
    time.sleep(STEP_TIME)


def object_close_enough(obj):
    """
    Decides when the cube is close enough to begin goal-directed pushing.
    """

    if obj is None:
        return False

    centred = abs(obj["offset"]) < CENTER_TOLERANCE
    visually_close = obj["pixels"] >= OBJECT_CLOSE_PIXELS or obj["bottom"] > 0.78
    sonar_close = get_sonar() != -1 and get_sonar() < 0.12

    return centred and (visually_close or sonar_close)


def scan_for_goal(target_goal):
    """
    Rotates slowly until the correct goal is visible in the top camera.
    """

    debug(f"BEHAVIOUR: Scan for {target_goal} goal")

    top_goal = detect_top_goal()

    if top_goal is not None and top_goal["label"] == target_goal:
        return top_goal

    drive(0.15, 0.45)
    time.sleep(STEP_TIME)

    return None


def push_to_goal(object_type, target_goal):
    """
    Goal-directed pushing.
    Uses the top camera to steer toward the correct goal.
    Stops and backs away when the correct container is directly in front.
    """

    debug(f"BEHAVIOUR: Push {object_type} to {target_goal}")

    # Delivery condition: correct goal appears in the front camera
    front_goal = detect_front_goal()

    if front_goal is not None and front_goal["label"] == target_goal:
        debug(f"DELIVERED: {object_type} to {target_goal}")

        # Small final push
        drive(0.7, clamp(front_goal["offset"] * 0.3, -0.2, 0.2))
        time.sleep(0.5)

        # Immediately reverse so the robot does not fall into the container
        drive(BACK_SPEED, 0)
        time.sleep(0.9)

        drive(0, random.choice([-1, 1]) * TURN_SPEED)
        time.sleep(0.6)

        stop()
        return True

    # Steer toward target using top camera
    top_goal = detect_top_goal()

    if top_goal is not None and top_goal["label"] == target_goal:
        turn = clamp(top_goal["offset"] * 0.8, -0.45, 0.45)
        drive(PUSH_SPEED, turn)
        time.sleep(STEP_TIME)
        return False

    # If target not visible, rotate slowly while keeping slight forward pressure
    drive(0.25, 0.45)
    time.sleep(STEP_TIME)
    return False


# ============================================================
# CONTROLLER STATE
# ============================================================

state = "SEARCH"
current_object = None
target_goal = None


def set_state(new_state):
    global state
    debug(f"STATE: {state} -> {new_state}")
    state = new_state


# ============================================================
# START SIMULATION
# ============================================================

sim.startSimulation()
time.sleep(2)

print("Simulation object sorting controller started.")
print("Architecture: Subsumption + simple state machine")
print("Green = Plant -> Blue")
print("Brown/Orange = Trash -> Compress -> Red")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # PRIORITY 0: If not pushing, avoid containers
    # --------------------------------------------------------

    if state in ["SEARCH", "APPROACH"]:
        if avoid_container_when_not_pushing():
            continue

    # --------------------------------------------------------
    # PRIORITY 1: Wall avoidance
    # --------------------------------------------------------

    if state in ["SEARCH", "APPROACH"]:
        if avoid_wall_if_needed():
            continue

    # --------------------------------------------------------
    # STATE: SEARCH
    # --------------------------------------------------------

    if state == "SEARCH":
        obj = detect_object()

        if obj is not None:
            current_object = obj["label"]
            target_goal = target_goal_for_object(current_object)

            debug(f"FOUND: {current_object}, target={target_goal}")
            set_state("APPROACH")
            continue

        wander()
        continue

    # --------------------------------------------------------
    # STATE: APPROACH
    # --------------------------------------------------------

    if state == "APPROACH":
        obj = detect_object()

        if obj is None:
            debug("Lost object, returning to search")
            current_object = None
            target_goal = None
            set_state("SEARCH")
            continue

        if object_close_enough(obj):
            debug(f"{current_object} close enough")

            if current_object == "Trash":
                compress_trash()

            set_state("SCAN_GOAL")
            continue

        approach_object(obj)
        continue

    # --------------------------------------------------------
    # STATE: SCAN_GOAL
    # --------------------------------------------------------

    if state == "SCAN_GOAL":
        goal = scan_for_goal(target_goal)

        if goal is not None:
            debug(f"Target goal found: {target_goal}")
            set_state("PUSH")
            continue

        continue

    # --------------------------------------------------------
    # STATE: PUSH
    # --------------------------------------------------------

    if state == "PUSH":
        delivered = push_to_goal(current_object, target_goal)

        if delivered:
            current_object = None
            target_goal = None
            set_state("SEARCH")

        continue