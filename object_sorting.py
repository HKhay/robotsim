from robots import *
import random
import time
from coppeliasim_zmqremoteapi_client import *


client = RemoteAPIClient()
sim = client.require("sim")

# Handles supplied by the starter files.
robot = Robot_OS(sim, DeviceNames.ROBOT_OS)
top_image_sensor = ImageSensor(sim, DeviceNames.TOP_IMAGE_SENSOR_OS)
small_image_sensor = ImageSensor(sim, DeviceNames.SMALL_IMAGE_SENSOR_OS)
small_camera = small_image_sensor._handle

left_motor = Motor(sim, DeviceNames.MOTOR_LEFT_OS, Direction.CLOCKWISE)
right_motor = Motor(sim, DeviceNames.MOTOR_RIGHT_OS, Direction.CLOCKWISE)


# -------------------------- Controller settings --------------------------

DEBUG = True
CODE_VERSION = "object_sorting_subsumption_v21_lock_must_see_object"

# Battery/recharge.
RECHARGE_THRESHOLD = 0.25
CHARGED_ENOUGH = 0.55

# The motor commands are wheel angular-speed style commands, not metres/sec.
MAX_WHEEL_SPEED = 2.6
WANDER_SPEED = 1.25
APPROACH_SPEED = 1.05
CLOSE_APPROACH_SPEED = 0.75
PUSH_SPEED = 0.65
BACKUP_SPEED = -1.4
SCAN_HOLD_SPEED = 0.11

# Camera steering. Positive image offset should steer toward that side.
STEERING_SIGN = 1.0
OBJECT_TURN_GAIN = 0.95
CONTAINER_TURN_GAIN = 0.85
APPROACH_TURN_LIMIT = 0.55
GOAL_CENTER_TOLERANCE = 0.26
GOAL_SCAN_TURN = 0.36
GRAB_CENTER_TOLERANCE = 0.07
READY_CENTER_TOLERANCE = 0.08
READY_STABLE_FRAMES = 3
OBJECT_LOCK_SECONDS = 7.0

# Vision thresholds. The images are low resolution, so small pixel counts matter.
MIN_OBJECT_PIXELS = 6
MIN_TARGET_CONTAINER_PIXELS = 22
MIN_DANGER_CONTAINER_PIXELS = 140
MIN_DELIVERY_CONTAINER_PIXELS = 85
MIN_TOP_TARGET_CONTAINER_PIXELS = 180
MIN_TOP_SCAN_CONTAINER_PIXELS = 55
MIN_PUSH_OBJECT_PIXELS = 175
MAX_PUSH_OBJECT_PIXELS = 310
MIN_PUSH_OBJECT_BOTTOM = 0.82
OBJECT_SLIP_CENTER_TOLERANCE = 0.24
OBJECT_SLIP_MIN_PIXELS = 125

# Cubes should appear as compact blobs. Large edge-to-edge brown/red areas are
# usually a container wall or background, not the object we want to chase.
MAX_SEARCH_OBJECT_PIXELS = 180
MAX_OBJECT_WIDTH_FRAC = 0.72
MAX_OBJECT_HEIGHT_FRAC = 0.72
EDGE_BLOB_PIXELS = 70

# Contact and state timing.
CONTACT_DISTANCE = 0.065
CONTACT_FORCE = 7.5
WALL_DISTANCE = 0.30
CLOSE_OBJECT_PIXELS = 90
LOST_LIMIT = 6
STEP_TIME = 0.16


# -------------------------- Basic utilities --------------------------

_handle_cache = {}


def debug(message):
    if DEBUG:
        print(message)


def clamp(value, low, high):
    return max(low, min(high, value))


def get_handle(path):
    if path not in _handle_cache:
        _handle_cache[path] = sim.getObject(path)
    return _handle_cache[path]


def drive(forward, turn=0.0):
    """
    Differential-drive kinematics in the same simple style as the pyARTE Husky
    examples: forward speed plus angular steering becomes left/right wheels.
    """
    turn = turn * STEERING_SIGN
    left_speed = clamp(forward + turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    right_speed = clamp(forward - turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    left_motor.run(left_speed)
    right_motor.run(right_speed)


def rotate(turn_speed):
    drive(0.0, turn_speed)


def stop():
    left_motor.run(0)
    right_motor.run(0)


def timed_drive(forward, turn, duration):
    drive(forward, turn)
    time.sleep(duration)
    stop()


def random_turn_direction():
    return random.choice([-1.0, 1.0])


# -------------------------- Sensors --------------------------

def get_battery():
    signal = robot.get_string_signal("battery")
    if signal is None:
        return 0.0
    if isinstance(signal, bytes):
        signal = signal.decode("utf-8")
    return float(signal)


def get_sonar():
    try:
        result = sim.readProximitySensor(get_handle("/dr12/dr12_sonar"))
        if result[0] > 0:
            return result[1]
    except Exception:
        pass
    return -1


def get_bumper_force():
    try:
        result = sim.readForceSensor(get_handle("/dr12/dr12_bumperForceSensor_"))
        if result[0] > 0:
            force = result[1]
            return abs(force[0]) + abs(force[1]) + abs(force[2])
    except Exception:
        pass
    return 0.0


def touching_something():
    distance = get_sonar()
    force = get_bumper_force()
    # The force sensor often sits around 4-5 even with no cube contact, so keep
    # this threshold high and let camera evidence decide most object handling.
    touching = force > CONTACT_FORCE
    return touching, distance, force


def camera_image(camera_handle):
    image, resolution = sim.getVisionSensorImg(camera_handle)
    return image, resolution[0], resolution[1]


def read_front_center_rgb():
    image, width, height = camera_image(small_camera)
    total_r = total_g = total_b = count = 0

    for y in range(int(height * 0.38), int(height * 0.64), 4):
        for x in range(int(width * 0.36), int(width * 0.64), 4):
            idx = 3 * (y * width + x)
            if idx + 2 < len(image):
                total_r += image[idx]
                total_g += image[idx + 1]
                total_b += image[idx + 2]
                count += 1

    if count == 0:
        return None
    return total_r / count, total_g / count, total_b / count


def read_top_goal_colour():
    try:
        top_image_sensor._update_image()
        return classify_goal_colour(*top_image_sensor.rgb())
    except Exception:
        return None


# -------------------------- Colour rules --------------------------

def classify_goal_colour(r, g, b):
    if is_yellow_pixel(r, g, b):
        return "Yellow"
    if is_blue_container_pixel(r, g, b):
        return "Blue"
    if is_red_container_pixel(r, g, b):
        return "Red"
    return None


def is_yellow_pixel(r, g, b):
    return r > 120 and g > 115 and b < 95 and abs(r - g) < 70


def is_plant_pixel(r, g, b):
    return g > 60 and g > r * 1.12 and g > b * 1.08 and g - r > 25


def is_trash_pixel(r, g, b):
    return (
        55 < r < 190
        and 18 < g < 135
        and b < 90
        and r > g * 1.08
        and g > b * 1.05
        and r - b > 30
        and classify_goal_colour(r, g, b) is None
    )


def is_red_container_pixel(r, g, b):
    return r > 175 and g < 105 and b < 105 and r > g * 1.8 and r > b * 1.8


def is_blue_container_pixel(r, g, b):
    return r < 100 and g > 90 and b > 90 and b > r * 1.5 and g > r * 1.5


def target_container_for(colour):
    if colour == "Plant":
        return "Blue"
    if colour == "Trash":
        return "Red"
    return None


def object_ready_for_push(seen):
    if seen is None:
        return False
    centred = abs(seen["offset"]) <= READY_CENTER_TOLERANCE
    large_enough = seen["pixels"] >= MIN_PUSH_OBJECT_PIXELS
    not_overfilled = seen["pixels"] <= MAX_PUSH_OBJECT_PIXELS
    low_in_view = seen["bottom"] >= MIN_PUSH_OBJECT_BOTTOM
    return centred and large_enough and not_overfilled and low_in_view


def object_controlled_for_push(seen, touching):
    return object_ready_for_push(seen)


def object_visible_in_gripper(seen, colour):
    if seen is None or seen["colour"] != colour:
        return False
    centred_enough = abs(seen["offset"]) <= 0.55
    close_enough = seen["pixels"] >= 70 and seen["bottom"] >= 0.78
    return centred_enough and close_enough


def object_slipping_while_scanning(seen, colour):
    if seen is None or seen["colour"] != colour:
        return False
    too_far_to_side = abs(seen["offset"]) > OBJECT_SLIP_CENTER_TOLERANCE
    not_deep_enough = seen["pixels"] < OBJECT_SLIP_MIN_PIXELS
    return seen["bottom"] >= 0.80 and (too_far_to_side or not_deep_enough)


def search_object_blob_is_plausible(blob):
    """Reject container/background blobs while the robot is choosing a cube."""
    close_centred_cube = blob["bottom"] > 0.82 and abs(blob["offset"]) < 0.55

    if (
        close_centred_cube
        and blob["width_frac"] <= 0.86
        and blob["height_frac"] <= 0.86
    ):
        return True

    if blob["width_frac"] > MAX_OBJECT_WIDTH_FRAC:
        return False
    if blob["height_frac"] > MAX_OBJECT_HEIGHT_FRAC:
        return False
    if blob["pixels"] > MAX_SEARCH_OBJECT_PIXELS and not close_centred_cube:
        return False
    if blob["touches_edge"] and blob["pixels"] > EDGE_BLOB_PIXELS and not close_centred_cube:
        return False
    return True


def object_selection_priority(blob):
    """
    Prefer the nearest directly visible object instead of averaging/chasing the
    largest coloured thing in view.
    """
    closeness = 4.0 * blob["bottom"]
    compact_size = min(blob["pixels"], 85) / 85.0
    centre_factor = 1.0 - min(abs(blob["offset"]), 1.0) * 0.28
    return (closeness + compact_size) * centre_factor


# -------------------------- Vision blob detection --------------------------

def scan_blobs(camera_handle, classifiers, min_pixels, y_min=0.08, y_max=0.96,
               x_min=0.03, x_max=0.97, step=3, prefer_bottom=True):
    """
    Finds connected colour blobs. This prevents the robot from averaging several
    cubes together and aiming between them.
    """
    image, width, height = camera_image(camera_handle)
    x_values = list(range(int(width * x_min), int(width * x_max), step))
    y_values = list(range(int(height * y_min), int(height * y_max), step))
    pixels_by_label = {label: {} for label, _ in classifiers}

    for gy, y in enumerate(y_values):
        for gx, x in enumerate(x_values):
            idx = 3 * (y * width + x)
            if idx + 2 >= len(image):
                continue

            r = image[idx]
            g = image[idx + 1]
            b = image[idx + 2]

            for label, predicate in classifiers:
                if predicate(r, g, b):
                    y_weight = 1.0 + (5.0 * (y / height) ** 2 if prefer_bottom else 0.0)
                    pixels_by_label[label][(gx, gy)] = (x, y, r, g, b, y_weight)
                    break

    blobs = []
    for label, pixels in pixels_by_label.items():
        seen = set()
        for start in pixels:
            if start in seen:
                continue

            stack = [start]
            seen.add(start)
            count = 0
            score = 0.0
            sum_x = sum_y = 0.0
            sum_r = sum_g = sum_b = 0.0
            min_x_seen = width
            max_x_seen = 0
            min_y_seen = height
            max_y = 0

            while stack:
                point = stack.pop()
                x, y, r, g, b, weight = pixels[point]
                count += 1
                score += weight
                sum_x += x * weight
                sum_y += y * weight
                sum_r += r * weight
                sum_g += g * weight
                sum_b += b * weight
                min_x_seen = min(min_x_seen, x)
                max_x_seen = max(max_x_seen, x)
                min_y_seen = min(min_y_seen, y)
                max_y = max(max_y, y)

                px, py = point
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbour = (px + dx, py + dy)
                        if neighbour in pixels and neighbour not in seen:
                            seen.add(neighbour)
                            stack.append(neighbour)

            if count < min_pixels or score <= 0:
                continue

            offset = (sum_x / score / width - 0.5) * 2.0
            vertical = sum_y / score / height
            bottom = max_y / height
            width_frac = (max_x_seen - min_x_seen + step) / width
            height_frac = (max_y - min_y_seen + step) / height
            left_edge = int(width * x_min) + step
            right_edge = int(width * x_max) - step
            top_edge = int(height * y_min) + step
            bottom_edge = int(height * y_max) - step
            touches_edge = (
                min_x_seen <= left_edge
                or max_x_seen >= right_edge
                or min_y_seen <= top_edge
                or max_y >= bottom_edge
            )
            priority = count * (0.6 + bottom) / (1.0 + abs(offset) * 0.12)

            blobs.append({
                "label": label,
                "pixels": count,
                "offset": offset,
                "vertical": vertical,
                "bottom": bottom,
                "width_frac": width_frac,
                "height_frac": height_frac,
                "touches_edge": touches_edge,
                "priority": priority,
                "rgb": (sum_r / score, sum_g / score, sum_b / score),
            })

    blobs.sort(key=lambda item: item["priority"], reverse=True)
    return blobs


def detect_object(preferred_colour=None, strict_search=True):
    classifiers = [
        ("Plant", is_plant_pixel),
        ("Trash", is_trash_pixel),
    ]
    blobs = scan_blobs(
        small_camera,
        classifiers,
        MIN_OBJECT_PIXELS,
        y_min=0.16,
        y_max=0.95,
        x_min=0.04,
        x_max=0.96,
        step=3,
        prefer_bottom=True,
    )

    if preferred_colour is not None:
        preferred = [blob for blob in blobs if blob["label"] == preferred_colour]
        if preferred:
            blobs = preferred
        else:
            return None

    if strict_search:
        plausible = [blob for blob in blobs if search_object_blob_is_plausible(blob)]
        if plausible:
            blobs = plausible
        else:
            return None

    for blob in blobs:
        blob["priority"] = object_selection_priority(blob)
    blobs.sort(key=lambda item: item["priority"], reverse=True)

    if not blobs:
        return None

    best = blobs[0]
    r, g, b = best["rgb"]
    debug(
        f"Detected {best['label']}: pixels={best['pixels']} "
        f"offset={best['offset']:.2f} "
        f"bottom={best['bottom']:.2f} rgb=({r:.1f},{g:.1f},{b:.1f})"
    )
    return {
        "colour": best["label"],
        "pixels": best["pixels"],
        "offset": best["offset"],
        "bottom": best["bottom"],
    }


def detect_container(camera_handle, min_pixels, source):
    classifiers = [
        ("Red", is_red_container_pixel),
        ("Blue", is_blue_container_pixel),
    ]
    blobs = scan_blobs(
        camera_handle,
        classifiers,
        min_pixels,
        y_min=0.05,
        y_max=0.97,
        x_min=0.02,
        x_max=0.98,
        step=3,
        prefer_bottom=False,
    )

    if not blobs:
        return None

    best = blobs[0]
    return {
        "goal": best["label"],
        "pixels": best["pixels"],
        "offset": best["offset"],
        "source": source,
    }


def detect_front_container(min_pixels=MIN_DANGER_CONTAINER_PIXELS):
    detail = detect_container(small_camera, min_pixels, "front")
    if detail is not None:
        debug(f"Front container: {detail['goal']} pixels={detail['pixels']}")
    return detail


def detect_top_container(min_pixels=MIN_TARGET_CONTAINER_PIXELS):
    detail = detect_container(top_image_sensor._handle, min_pixels, "top")
    if detail is not None:
        debug(
            f"Top container: {detail['goal']} pixels={detail['pixels']} "
            f"offset={detail['offset']:.2f}"
        )
    return detail


def find_target_container(goal, allow_top=False, top_min_pixels=MIN_TOP_TARGET_CONTAINER_PIXELS):
    front = detect_front_container(MIN_TARGET_CONTAINER_PIXELS)
    if front is not None and front["goal"] == goal:
        return front

    if allow_top:
        top = detect_top_container(top_min_pixels)
        if top is not None and top["goal"] == goal:
            return top

    return None


def front_or_top_yellow():
    rgb = read_front_center_rgb()
    if rgb is not None and classify_goal_colour(*rgb) == "Yellow":
        return True
    return read_top_goal_colour() == "Yellow"


# -------------------------- Subsumption behaviours --------------------------

def back_away(turn_direction=None):
    if turn_direction is None:
        turn_direction = random_turn_direction()
    timed_drive(BACKUP_SPEED, 0.0, 0.45)
    timed_drive(0.0, 1.2 * turn_direction, 0.55)


def leave_container_area(goal=None):
    debug(f"Leaving {goal or 'container'} area")
    timed_drive(-1.5, 0.0, 0.75)
    timed_drive(0.0, random_turn_direction() * 1.2, 0.55)


def wall_avoidance_needed():
    distance = get_sonar()
    if distance == -1 or distance >= WALL_DISTANCE:
        return False
    seen = detect_object()
    if seen is not None:
        return False
    return True
    #return distance != -1 and distance < WALL_DISTANCE


def avoid_obstacles_if_needed(allowed_goal=None, carrying=False):
    top_goal = read_top_goal_colour()
    front = detect_front_container(MIN_DANGER_CONTAINER_PIXELS)

    if front is not None and front["goal"] in ("Red", "Blue"):
        if carrying and front["goal"] == allowed_goal:
            return None
        leave_container_area(front["goal"])
        return f"Escaping {front['goal']} container"

    if top_goal in ("Red", "Blue"):
        if carrying and top_goal == allowed_goal:
            return None
        leave_container_area(top_goal)
        return f"Escaping {top_goal} container area"

    if top_goal == "Yellow" and not carrying and get_battery() >= RECHARGE_THRESHOLD:
        timed_drive(0.9, random_turn_direction() * 0.8, 0.45)
        return "Leaving charger area"

    if not carrying and wall_avoidance_needed():
        back_away()
        return "Avoiding wall"

    return None


def recharge_behaviour():
    battery = get_battery()
    if battery >= RECHARGE_THRESHOLD:
        return None

    if read_top_goal_colour() == "Yellow":
        stop()
        if battery >= CHARGED_ENOUGH:
            return "Battery charged enough"
        return "Charging"

    if front_or_top_yellow():
        drive(0.75, 0.0)
        time.sleep(STEP_TIME)
        return "Moving to charger"

    rotate(0.85)
    time.sleep(STEP_TIME)
    return "Looking for charger"


# -------------------------- FSM on top of the behaviours --------------------------

class SortingController:
    def __init__(self):
        self.state = "SEARCH"
        self.object_colour = None
        self.target_goal = None
        self.lost_steps = 0
        self.search_turn = random_turn_direction()
        self.state_start = time.time()
        self.push_stall = 0
        self.goal_scan_steps = 0
        self.last_turn = 0.0
        self.object_lock_until = 0.0
        self.ready_frames = 0

    def set_state(self, state, colour=None):
        if state != self.state:
            debug(f"State: {self.state} -> {state}")
        self.state = state
        self.object_colour = colour
        self.target_goal = target_container_for(colour)
        self.lost_steps = 0
        self.goal_scan_steps = 0
        self.last_turn = 0.0
        self.ready_frames = 0
        self.state_start = time.time()
        if state == "SEARCH":
            self.object_lock_until = 0.0
        if random.random() < 0.5:
            self.search_turn = -self.search_turn

    def lock_object(self):
        self.object_lock_until = time.time() + OBJECT_LOCK_SECONDS

    def object_locked(self):
        return time.time() < self.object_lock_until

    def update_ready_frames(self, seen):
        if object_ready_for_push(seen):
            self.ready_frames += 1
        else:
            self.ready_frames = 0
        return self.ready_frames >= READY_STABLE_FRAMES

    def smooth_drive(self, forward, turn, blend=0.35):
        self.last_turn = (1.0 - blend) * self.last_turn + blend * turn
        drive(forward, self.last_turn)

    def reset_to_search(self):
        self.set_state("SEARCH", None)

    def deliver_if_possible(self):
        front = detect_front_container(MIN_DELIVERY_CONTAINER_PIXELS)

        if front is not None and front["goal"] == self.target_goal and abs(front["offset"]) < 0.48:
            final_turn = clamp(front["offset"] * 0.35, -0.25, 0.25)
            timed_drive(0.70, final_turn, 0.55)
            leave_container_area(front["goal"])
            self.reset_to_search()
            return f"Delivered {self.object_colour} to {front['goal']} container"

        if front is not None and front["goal"] != self.target_goal:
            leave_container_area(front["goal"])
            self.reset_to_search()
            return f"Wrong container for {self.object_colour}; backing away"

        return None

    def wander_step(self):
        if random.random() < 0.18:
            self.search_turn = random_turn_direction()
            rotate(0.75 * self.search_turn)
        else:
            drive(WANDER_SPEED, 0.10 * self.search_turn)
        time.sleep(STEP_TIME)
        return "Wandering"

    def approach_step(self, seen):
        if seen is None:
            self.lost_steps += 1
            rotate(0.75 * self.search_turn)
            time.sleep(STEP_TIME)
            if self.lost_steps >= LOST_LIMIT:
                self.reset_to_search()
                return "Lost object"
            return f"Searching for {self.object_colour}"

        self.lost_steps = 0
        offset = seen["offset"]
        pixels = seen["pixels"]
        touching, distance, force = touching_something()
        close_by_camera = (
            pixels >= CLOSE_OBJECT_PIXELS
            and seen["bottom"] > 0.76
            and abs(offset) < 0.25
        )

        grab_ready = self.update_ready_frames(seen)

        if grab_ready:
            self.lock_object()
            self.set_state("SCAN_GOAL", self.object_colour)
            stop()
            time.sleep(STEP_TIME)
            return f"{self.object_colour} centred between arms; scanning for {self.target_goal}"

        if touching:
            debug(
                f"Contact ignored until {self.object_colour} is centred/large: "
                f"pixels={pixels} offset={offset:.2f} "
                f"bottom={seen['bottom']:.2f} force={force:.2f}"
            )

        if close_by_camera:
            turn = clamp(offset * 0.75, -0.28, 0.28)
            if abs(offset) > GRAB_CENTER_TOLERANCE:
                self.smooth_drive(0.0, turn, 0.45)
                time.sleep(STEP_TIME)
                return f"Fine-aligning {self.object_colour} between arms"

            self.smooth_drive(0.08, turn, 0.35)
            time.sleep(STEP_TIME)
            return f"Creeping onto centred {self.object_colour} ({self.ready_frames}/{READY_STABLE_FRAMES})"

        turn = clamp(offset * OBJECT_TURN_GAIN, -APPROACH_TURN_LIMIT, APPROACH_TURN_LIMIT)
        if abs(offset) > 0.65:
            speed = 0.45
        elif pixels > 70 or seen["bottom"] > 0.62:
            speed = 0.45
        else:
            speed = APPROACH_SPEED
        self.smooth_drive(speed, turn, 0.35)
        time.sleep(STEP_TIME)
        if abs(offset) > 0.26:
            return f"Steering toward {self.object_colour}"
        return f"Approaching {self.object_colour}"

    def scan_goal_step(self, seen):
        touching, distance, force = touching_something()
        same_object_seen = seen is not None and seen["colour"] == self.object_colour

        if same_object_seen and object_slipping_while_scanning(seen, self.object_colour):
            turn = clamp(seen["offset"] * 0.55, -0.24, 0.24)
            self.smooth_drive(SCAN_HOLD_SPEED, turn, 0.35)
            time.sleep(STEP_TIME)
            return f"Holding {self.object_colour}; re-centering before scan"

        if self.object_locked():
            if object_visible_in_gripper(seen, self.object_colour):
                self.lock_object()
                self.lost_steps = 0
            elif same_object_seen:
                self.object_lock_until = 0.0
                self.set_state("APPROACH", self.object_colour)
                return f"{self.object_colour} slipped sideways; aligning again"
            else:
                self.lost_steps += 1
                stop()
                time.sleep(STEP_TIME)
                if self.lost_steps >= 2:
                    self.object_lock_until = 0.0
                    self.set_state("APPROACH", self.object_colour)
                    return f"Lost hold of {self.object_colour}; approaching again"
                return f"Checking hold on {self.object_colour}"
        elif touching and object_visible_in_gripper(seen, self.object_colour):
            self.lost_steps = 0
        elif not same_object_seen:
            self.lost_steps += 1
            rotate(0.45 * self.search_turn)
            time.sleep(STEP_TIME)
            if self.lost_steps >= LOST_LIMIT:
                self.reset_to_search()
                return f"Lost {self.object_colour} while scanning"
            return f"Scanning for {self.target_goal}; keeping {self.object_colour}"
        else:
            self.lost_steps = 0

        if not self.object_locked() and not object_controlled_for_push(seen, touching):
            self.set_state("APPROACH", self.object_colour)
            return f"Realigning with {self.object_colour} before scanning"

        front = detect_front_container(MIN_TARGET_CONTAINER_PIXELS)
        if front is not None and front["goal"] == self.target_goal:
            self.goal_scan_steps = 0
            offset = front["offset"]
            if abs(offset) <= GOAL_CENTER_TOLERANCE:
                self.set_state("PUSH", self.object_colour)
                return f"{self.target_goal} container in front; pushing {self.object_colour}"

            turn = clamp(offset * CONTAINER_TURN_GAIN, -GOAL_SCAN_TURN, GOAL_SCAN_TURN)
            self.smooth_drive(SCAN_HOLD_SPEED, turn, 0.30)
            time.sleep(STEP_TIME)
            return f"Front-aligning with {self.target_goal} container"

        top = detect_top_container(MIN_TOP_SCAN_CONTAINER_PIXELS)
        if top is not None and top["goal"] == self.target_goal:
            self.goal_scan_steps = 0
            offset = top["offset"]
            if abs(offset) < 0.10:
                turn = GOAL_SCAN_TURN * 0.35 * self.search_turn
            else:
                turn = clamp(offset * CONTAINER_TURN_GAIN, -GOAL_SCAN_TURN, GOAL_SCAN_TURN)
            self.smooth_drive(SCAN_HOLD_SPEED, turn, 0.35)
            time.sleep(STEP_TIME)
            return f"Top-guided scan for {self.target_goal} container"

        if front is not None and front["goal"] != self.target_goal:
            self.goal_scan_steps += 1
            turn = GOAL_SCAN_TURN * (-1 if front["offset"] > 0 else 1)
            self.smooth_drive(SCAN_HOLD_SPEED * 0.8, turn, 0.35)
            time.sleep(STEP_TIME)
            return f"Turning away from wrong {front['goal']} container"

        self.goal_scan_steps += 1
        if self.goal_scan_steps % 18 == 0:
            self.search_turn = -self.search_turn
        self.smooth_drive(SCAN_HOLD_SPEED, (GOAL_SCAN_TURN * 0.55) * self.search_turn, 0.35)
        time.sleep(STEP_TIME)
        return f"Scanning for {self.target_goal} container"

    def push_step(self, seen):
        touching, distance, force = touching_something()
        visible_in_gripper = object_visible_in_gripper(seen, self.object_colour)
        if visible_in_gripper:
            self.lock_object()
        pushing_real_object = visible_in_gripper and (
            self.object_locked()
            or object_controlled_for_push(seen, touching)
            or touching
        )
        if not pushing_real_object:
            if seen is not None and seen["colour"] == self.object_colour:
                self.set_state("APPROACH", self.object_colour)
                return f"Not pushing {self.object_colour}; aligning again"

            self.lost_steps += 1
            rotate(0.55 * self.search_turn)
            time.sleep(STEP_TIME)
            if self.lost_steps >= LOST_LIMIT:
                self.reset_to_search()
                return f"Lost {self.object_colour}"
            return f"Looking for {self.object_colour}"

        delivered = self.deliver_if_possible()
        if delivered is not None:
            return delivered

        obstacle_action = avoid_obstacles_if_needed(
            allowed_goal=self.target_goal,
            carrying=True,
        )
        if obstacle_action is not None:
            self.reset_to_search()
            return obstacle_action

        if force > 8.0:
            self.push_stall = getattr(self, "push_stall", 0) + 1
        else:
            self.push_stall = 0

        if self.push_stall > 8:
            debug("Cube wedged against wall - backing off to re-angle")
            timed_drive(BACKUP_SPEED, 0.0, 0.6)
            timed_drive(0.0, random_turn_direction() * 1.3, 0.6)
            self.push_stall = 0
            self.set_state("APPROACH", self.object_colour)
            return "RE-approaching from new angle"

        target = find_target_container(self.target_goal, allow_top=False)
        if target is None:
            stop()
            if visible_in_gripper:
                self.lock_object()
                self.set_state("SCAN_GOAL", self.object_colour)
                return f"Lost sight of {self.target_goal}; scanning again"
            self.object_lock_until = 0.0
            self.set_state("APPROACH", self.object_colour)
            return f"Lost sight of {self.target_goal} and {self.object_colour}; aligning again"

        if touching:
            self.lost_steps = 0
            object_turn = 0.0
        elif seen is not None and seen["colour"] == self.object_colour:
            self.lost_steps = 0
            object_turn = clamp(seen["offset"] * 0.75, -0.35, 0.35)
        else:
            self.lost_steps += 1
            object_turn = 0.18 * self.search_turn

        target_turn = clamp(target["offset"] * CONTAINER_TURN_GAIN, -0.35, 0.35)
        turn = clamp((0.75 * object_turn) + (0.25 * target_turn), -0.45, 0.45)
        speed = PUSH_SPEED if target["source"] == "front" else PUSH_SPEED * 0.85

        self.smooth_drive(speed, turn, 0.25)
        time.sleep(STEP_TIME)

        if self.lost_steps >= LOST_LIMIT * 2:
            self.reset_to_search()
            return f"Lost {self.object_colour} while pushing"

        return f"Pushing {self.object_colour} toward {self.target_goal}"

    def step(self):
        recharge_action = recharge_behaviour()
        if recharge_action is not None:
            return recharge_action

        if self.state == "SEARCH":
            obstacle_action = avoid_obstacles_if_needed()
            if obstacle_action is not None:
                return obstacle_action

        preferred = self.object_colour if self.state in ("APPROACH", "SCAN_GOAL", "PUSH") else None
        strict_vision = self.state in ("SEARCH", "APPROACH", "SCAN_GOAL")
        seen = detect_object(preferred, strict_search=strict_vision)

        if self.state == "SEARCH":
            if seen is not None:
                self.set_state("APPROACH", seen["colour"])
                return self.approach_step(seen)
            return self.wander_step()

        if self.state == "APPROACH":
            return self.approach_step(seen)

        if self.state == "SCAN_GOAL":
            return self.scan_goal_step(seen)

        if self.state == "PUSH":
            return self.push_step(seen)

        self.reset_to_search()
        return "Resetting controller"


# -------------------------- Main loop --------------------------

sim.startSimulation()
time.sleep(2)

battery = 0.0
while battery <= 0.0:
    try:
        battery = get_battery()
    except Exception:
        battery = 0.0
    time.sleep(0.2)

controller = SortingController()
print("Ready! Battery:", battery)
print("Controller:", CODE_VERSION)

while True:
    action = controller.step()
    debug(action)
    debug("-" * 40)
    time.sleep(0.05)
