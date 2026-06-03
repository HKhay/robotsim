from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math


# ============================================================
# CONNECT TO COPPELIASIM
# ============================================================

client = RemoteAPIClient()
sim = client.require("sim")


# ============================================================
# GET OBJECT HANDLES
# ============================================================

robot = sim.getObject("/dr12")

left_motor = sim.getObject("/dr12_leftJoint_")
right_motor = sim.getObject("/dr12_rightJoint_")

bumper_sensor = sim.getObject("/dr12_bumperForceSensor_")
sonar = sim.getObject("/dr12_sonar")
small_camera = sim.getObject("/dr12_small_camera")

trash_can = sim.getObject("/Trash_Can")
plant_collector = sim.getObject("/Plant_Collector")


# ============================================================
# OPTIONAL DROP POINTS
# ============================================================
# Recommended:
# In CoppeliaSim, create two dummies:
# Add > Dummy
# Rename them:
# Plant_Drop_Point
# Trash_Drop_Point
#
# Put them in FRONT of the bins, not inside the bins.

try:
    plant_drop_point = sim.getObject("/Plant_Drop_Point")
    print("Plant_Drop_Point found.")
except Exception:
    plant_drop_point = plant_collector
    print("Plant_Drop_Point not found. Using Plant_Collector.")

try:
    trash_drop_point = sim.getObject("/Trash_Drop_Point")
    print("Trash_Drop_Point found.")
except Exception:
    trash_drop_point = trash_can
    print("Trash_Drop_Point not found. Using Trash_Can.")


# ============================================================
# SETTINGS
# ============================================================

# If robot moves backward instead of forward, change to -1
MOTOR_SIGN = 1

FORWARD_SPEED = 2.0
SLOW_SPEED = 1.0
TURN_SPEED = 1.5
BACKWARD_SPEED = -1.2

# Smaller distance avoids panic behavior
SAFE_DISTANCE = 0.35
COLLECT_DISTANCE = 0.25

# If obstacle is extremely close, reverse first
DANGER_DISTANCE = 0.20

BUMPER_FORCE_LIMIT = 5.0

CAMERA_CENTER_TOLERANCE = 0.12
TARGET_REACHED_DISTANCE = 0.35

DEBUG_CAMERA = True
DEBUG_BEHAVIOR = True


# ============================================================
# ROBOT STATE
# ============================================================

carrying_object = False
carried_type = None

# carried_type values:
# "plant"
# "trash"
# "compressed"


# ============================================================
# BASIC MOTOR FUNCTIONS
# ============================================================

def set_motors(left_speed, right_speed):
    sim.setJointTargetVelocity(left_motor, MOTOR_SIGN * left_speed)
    sim.setJointTargetVelocity(right_motor, MOTOR_SIGN * right_speed)


def stop_robot():
    set_motors(0, 0)


def move_forward(speed=FORWARD_SPEED):
    set_motors(speed, speed)


def move_backward():
    set_motors(BACKWARD_SPEED, BACKWARD_SPEED)


def turn_left(speed=TURN_SPEED):
    set_motors(-speed, speed)


def turn_right(speed=TURN_SPEED):
    set_motors(speed, -speed)


def print_behavior(message):
    if DEBUG_BEHAVIOR:
        print(message)


# ============================================================
# SENSOR FUNCTIONS
# ============================================================

def read_sonar():
    try:
        result = sim.readProximitySensor(sonar)

        detected = result[0]
        distance = result[1]

        if detected > 0:
            return True, distance

        return False, None

    except Exception as e:
        print("Sonar error:", e)
        return False, None


def read_bumper():
    try:
        result = sim.readForceSensor(bumper_sensor)
        state = result[0]

        if state > 0:
            force_vector = result[1]

            force_size = math.sqrt(
                force_vector[0] ** 2 +
                force_vector[1] ** 2 +
                force_vector[2] ** 2
            )

            if force_size > BUMPER_FORCE_LIMIT:
                return True

        return False

    except Exception as e:
        print("Bumper error:", e)
        return False


# ============================================================
# CAMERA FUNCTIONS
# ============================================================

def get_camera_image():
    """
    Reads image from the small camera.
    Different CoppeliaSim versions may use different function names.
    """

    try:
        image, resolution = sim.getVisionSensorImg(small_camera)
        width = resolution[0]
        height = resolution[1]
        return image, width, height

    except Exception:
        try:
            image, resolution = sim.getVisionSensorImage(small_camera)
            width = resolution[0]
            height = resolution[1]
            return image, width, height

        except Exception as e:
            print("Camera error:", e)
            return None, 0, 0


def get_pixel_rgb(image, index):
    r = image[index]
    g = image[index + 1]
    b = image[index + 2]

    if isinstance(r, str):
        r = ord(r)
        g = ord(g)
        b = ord(b)

    return r, g, b


def detect_colored_object():
    """
    Detects:
    Green  = plant
    Brown  = trash
    Black  = compressed trash

    Returns:
    object_type, offset, confidence
    """

    image, width, height = get_camera_image()

    if image is None or width == 0 or height == 0:
        return None, 0, 0

    plant_pixels = []
    trash_pixels = []
    compressed_pixels = []

    # Scan central/lower camera area
    y_start = int(height * 0.30)
    y_end = int(height * 0.90)

    x_start = int(width * 0.05)
    x_end = int(width * 0.95)

    for y in range(y_start, y_end, 3):
        for x in range(x_start, x_end, 3):
            index = 3 * (y * width + x)

            if index + 2 >= len(image):
                continue

            r, g, b = get_pixel_rgb(image, index)

            # Green object = plant
            if g > 80 and g > r * 1.25 and g > b * 1.25:
                plant_pixels.append(x)

            # Black/dark object = compressed trash
            elif r < 45 and g < 45 and b < 45:
                compressed_pixels.append(x)

            # Brown object = trash
            elif r > 70 and g > 25 and b < 90 and r > b * 1.3 and r >= g:
                trash_pixels.append(x)

    candidates = {
        "plant": plant_pixels,
        "trash": trash_pixels,
        "compressed": compressed_pixels
    }

    object_type = None
    best_pixels = []

    for candidate_type, pixels in candidates.items():
        if len(pixels) > len(best_pixels):
            object_type = candidate_type
            best_pixels = pixels

    confidence = len(best_pixels)

    if confidence < 20:
        return None, 0, confidence

    average_x = sum(best_pixels) / confidence
    center_x = width / 2

    offset = (average_x - center_x) / center_x

    if DEBUG_CAMERA:
        print(f"Camera: {object_type}, offset={offset:.2f}, pixels={confidence}")

    return object_type, offset, confidence


# ============================================================
# SMART OBSTACLE AVOIDANCE
# ============================================================

def scan_free_space():
    """
    The robot turns slightly left and right,
    reads sonar distance, then chooses the more open side.
    """

    print_behavior("BEHAVIOR: Scan free space")

    stop_robot()
    time.sleep(0.1)

    # Scan left
    turn_left(SLOW_SPEED)
    time.sleep(0.35)

    stop_robot()
    time.sleep(0.1)

    left_detected, left_distance = read_sonar()

    if left_distance is None:
        left_distance = 2.0

    # Scan right
    turn_right(SLOW_SPEED)
    time.sleep(0.70)

    stop_robot()
    time.sleep(0.1)

    right_detected, right_distance = read_sonar()

    if right_distance is None:
        right_distance = 2.0

    # Return approximately to center
    turn_left(SLOW_SPEED)
    time.sleep(0.35)

    stop_robot()
    time.sleep(0.1)

    print(f"Scan result: left={left_distance:.2f}, right={right_distance:.2f}")

    if left_distance > right_distance:
        return "left"

    return "right"


def smart_avoid_obstacle():
    """
    Smarter obstacle avoidance:
    1. Stop
    2. Reverse only if very close
    3. Scan left and right
    4. Choose more open side
    5. Move forward into the gap
    """

    print_behavior("BEHAVIOR: Smart avoid obstacle")

    stop_robot()
    time.sleep(0.1)

    sonar_detected, sonar_distance = read_sonar()

    if sonar_detected and sonar_distance is not None and sonar_distance < DANGER_DISTANCE:
        move_backward()
        time.sleep(0.45)

        stop_robot()
        time.sleep(0.1)

    best_direction = scan_free_space()

    if best_direction == "left":
        print("Choosing left free space")
        turn_left(SLOW_SPEED)
        time.sleep(0.45)

    else:
        print("Choosing right free space")
        turn_right(SLOW_SPEED)
        time.sleep(0.45)

    # Move forward carefully into the gap
    move_forward(SLOW_SPEED)
    time.sleep(0.70)

    stop_robot()
    time.sleep(0.1)


def bumper_escape():
    """
    Used when robot physically touches something.
    Stronger than normal sonar avoidance.
    """

    print_behavior("BEHAVIOR: Bumper escape")

    stop_robot()
    time.sleep(0.1)

    move_backward()
    time.sleep(0.7)

    best_direction = scan_free_space()

    if best_direction == "left":
        turn_left(SLOW_SPEED)
    else:
        turn_right(SLOW_SPEED)

    time.sleep(0.7)

    move_forward(SLOW_SPEED)
    time.sleep(0.5)

    stop_robot()


# ============================================================
# EMERGENCY ESCAPE FROM BINS
# ============================================================

def escape_if_inside_bin():
    """
    Prevents robot from getting stuck inside the collector/bin.
    Adjust x values based on your scene if needed.
    """

    robot_pos = sim.getObjectPosition(robot, -1)
    x = robot_pos[0]

    # Plant collector side
    if x < -2.75:
        print("Emergency: robot too deep in Plant Collector area.")

        move_backward()
        time.sleep(1.2)

        turn_right(SLOW_SPEED)
        time.sleep(0.8)

        stop_robot()
        return True

    # Trash can side
    if x > 2.75:
        print("Emergency: robot too deep in Trash Can area.")

        move_backward()
        time.sleep(1.2)

        turn_left(SLOW_SPEED)
        time.sleep(0.8)

        stop_robot()
        return True

    return False


# ============================================================
# OBJECT ACTION FUNCTIONS
# ============================================================

def compress_object():
    print_behavior("BEHAVIOR: Compress object")

    stop_robot()
    time.sleep(0.2)

    # This signal is used by your Spawner script
    sim.setInt32Signal("compress", 1)

    time.sleep(0.8)
    stop_robot()


def approach_object(object_type, offset):
    """
    Align robot with object using camera offset.
    """

    print_behavior(f"BEHAVIOR: Approach {object_type}")

    if offset < -CAMERA_CENTER_TOLERANCE:
        turn_left(SLOW_SPEED)

    elif offset > CAMERA_CENTER_TOLERANCE:
        turn_right(SLOW_SPEED)

    else:
        move_forward(SLOW_SPEED)


def collect_object(object_type):
    global carrying_object
    global carried_type

    print_behavior(f"BEHAVIOR: Collect {object_type}")

    stop_robot()
    time.sleep(0.2)

    if object_type == "trash":
        compress_object()
        carrying_object = True
        carried_type = "compressed"

    elif object_type == "plant":
        print("Plant detected. Do not compress plant.")
        carrying_object = True
        carried_type = "plant"

    elif object_type == "compressed":
        carrying_object = True
        carried_type = "compressed"

    move_backward()
    time.sleep(0.4)

    stop_robot()


def search_for_object():
    """
    Slow wandering search behavior.
    """

    print_behavior("BEHAVIOR: Search")

    move_forward(SLOW_SPEED)
    time.sleep(0.20)

    detected, distance = read_sonar()

    # If no obstacle in front, make small turning movement
    # This helps the robot explore instead of going straight forever
    if not detected:
        turn_left(SLOW_SPEED)
        time.sleep(0.20)


# ============================================================
# NAVIGATION FUNCTIONS
# ============================================================

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle


def go_to_target(target_handle):
    """
    Simple navigation toward target using robot position and yaw.
    """

    robot_pos = sim.getObjectPosition(robot, -1)
    target_pos = sim.getObjectPosition(target_handle, -1)

    robot_ori = sim.getObjectOrientation(robot, -1)
    robot_yaw = robot_ori[2]

    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]

    distance = math.sqrt(dx * dx + dy * dy)

    if distance < TARGET_REACHED_DISTANCE:
        stop_robot()
        return True

    target_angle = math.atan2(dy, dx)
    angle_error = normalize_angle(target_angle - robot_yaw)

    if angle_error > 0.25:
        turn_left(SLOW_SPEED)

    elif angle_error < -0.25:
        turn_right(SLOW_SPEED)

    else:
        move_forward(FORWARD_SPEED)

    return False


def deliver_object():
    global carrying_object
    global carried_type

    print_behavior(f"BEHAVIOR: Deliver {carried_type}")

    if carried_type == "plant":
        target = plant_drop_point
    else:
        target = trash_drop_point

    reached = go_to_target(target)

    if reached:
        print("Drop point reached. Pushing object into container.")

        stop_robot()
        time.sleep(0.2)

        # Push object slightly into bin
        move_forward(SLOW_SPEED)
        time.sleep(0.5)

        # Reverse immediately to avoid falling inside
        move_backward()
        time.sleep(1.0)

        # Turn away from bin
        if carried_type == "plant":
            turn_right(SLOW_SPEED)
        else:
            turn_left(SLOW_SPEED)

        time.sleep(0.8)

        stop_robot()

        carrying_object = False
        carried_type = None


# ============================================================
# MAIN SUBSUMPTION CONTROLLER
# ============================================================

def run_controller():
    global carrying_object
    global carried_type

    print("Starting CoppeliaSim simulation...")
    sim.startSimulation()
    time.sleep(1)

    print("Smart Object Sorting Controller started.")
    print("Press CTRL + C to stop.")

    try:
        while True:
            bumper_hit = read_bumper()
            sonar_detected, sonar_distance = read_sonar()
            object_type, object_offset, confidence = detect_colored_object()

            # Emergency check: prevents robot from getting stuck in bin
            if escape_if_inside_bin():
                continue

            # ==================================================
            # PRIORITY 1: BUMPER COLLISION
            # Physical contact has highest priority
            # ==================================================
            if bumper_hit:
                bumper_escape()

            # ==================================================
            # PRIORITY 2: OBJECT CLOSE ENOUGH TO COLLECT
            # If object is near and camera identifies it
            # ==================================================
            elif (
                sonar_detected
                and sonar_distance is not None
                and sonar_distance < COLLECT_DISTANCE
                and object_type is not None
            ):
                collect_object(object_type)

            # ==================================================
            # PRIORITY 3: UNKNOWN OBSTACLE / WALL
            # If obstacle is close but not recognized as object
            # ==================================================
            elif (
                sonar_detected
                and sonar_distance is not None
                and sonar_distance < SAFE_DISTANCE
                and object_type is None
            ):
                # If there is still some distance, move slowly through the gap
                if sonar_distance > 0.28:
                    print_behavior("BEHAVIOR: Careful forward through gap")
                    move_forward(SLOW_SPEED)
                    time.sleep(0.25)
                else:
                    smart_avoid_obstacle()

            # ==================================================
            # PRIORITY 4: DELIVER OBJECT
            # ==================================================
            elif carrying_object:
                deliver_object()

            # ==================================================
            # PRIORITY 5: OBJECT VISIBLE
            # ==================================================
            elif object_type is not None:
                approach_object(object_type, object_offset)

            # ==================================================
            # PRIORITY 6: SEARCH
            # Lowest priority
            # ==================================================
            else:
                search_for_object()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Controller stopped by user.")

    finally:
        stop_robot()
        time.sleep(0.2)
        sim.stopSimulation()
        print("Simulation stopped.")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    run_controller()