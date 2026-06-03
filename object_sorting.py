from robots import *
import random
import time
from coppeliasim_zmqremoteapi_client import *


client = RemoteAPIClient()
sim = client.require("sim")

robot = Robot_OS(sim, DeviceNames.ROBOT_OS)
top_image_sensor = ImageSensor(sim, DeviceNames.TOP_IMAGE_SENSOR_OS)
small_image_sensor = ImageSensor(sim, DeviceNames.SMALL_IMAGE_SENSOR_OS)
small_camera = small_image_sensor._handle

left_motor = Motor(sim, DeviceNames.MOTOR_LEFT_OS, Direction.CLOCKWISE)
right_motor = Motor(sim, DeviceNames.MOTOR_RIGHT_OS, Direction.CLOCKWISE)


DEBUG = True
CODE_VERSION = "object_sorting_basic_subsumption_v3_strict_search"

MAX_WHEEL_SPEED = 2.6
SEARCH_SPEED = 1.1
APPROACH_SPEED = 1.0
CLOSE_SPEED = 0.28
PUSH_SPEED = 0.65
BACK_SPEED = -1.4
STEP_TIME = 0.16

LOW_BATTERY = 0.25
WALL_DISTANCE = 0.28
OBJECT_READY_PIXELS = 100
OBJECT_READY_BOTTOM = 0.70
OBJECT_READY_OFFSET = 0.20
MIN_OBJECT_PIXELS = 6
MIN_SEARCH_OBJECT_PIXELS = 20
MIN_SEARCH_WIDTH_FRAC = 0.05
MIN_SEARCH_HEIGHT_FRAC = 0.05
MIN_SEARCH_BOTTOM = 0.42
MIN_CONTAINER_PIXELS = 24
DELIVERY_CONTAINER_PIXELS = 95


def debug(message):
    if DEBUG:
        print(message)


def clamp(value, low, high):
    return max(low, min(high, value))


def drive(forward, turn=0.0):
    left_speed = clamp(forward + turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    right_speed = clamp(forward - turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    left_motor.run(left_speed)
    right_motor.run(right_speed)


def stop():
    left_motor.run(0)
    right_motor.run(0)


def timed_drive(forward, turn, duration):
    drive(forward, turn)
    time.sleep(duration)
    stop()


def get_battery():
    signal = robot.get_string_signal("battery")
    if signal is None:
        return 1.0
    if isinstance(signal, bytes):
        signal = signal.decode("utf-8")
    return float(signal)


_handle_cache = {}


def get_handle(path):
    if path not in _handle_cache:
        _handle_cache[path] = sim.getObject(path)
    return _handle_cache[path]


def get_sonar():
    try:
        result = sim.readProximitySensor(get_handle("/dr12/dr12_sonar"))
        if result[0] > 0:
            return result[1]
    except Exception:
        pass
    return -1


def get_force():
    try:
        result = sim.readForceSensor(get_handle("/dr12/dr12_bumperForceSensor_"))
        if result[0] > 0:
            force = result[1]
            return abs(force[0]) + abs(force[1]) + abs(force[2])
    except Exception:
        pass
    return 0.0


def camera_image(camera_handle):
    image, resolution = sim.getVisionSensorImg(camera_handle)
    return image, resolution[0], resolution[1]


# -------------------------- Colour Rules --------------------------

def is_yellow(r, g, b):
    return r > 120 and g > 115 and b < 95 and abs(r - g) < 75


def is_red_container(r, g, b):
    return r > 175 and g < 105 and b < 105 and r > g * 1.8 and r > b * 1.8


def is_blue_container(r, g, b):
    return r < 100 and g > 90 and b > 90 and b > r * 1.4 and g > r * 1.4


def is_goal_colour(r, g, b):
    return is_yellow(r, g, b) or is_red_container(r, g, b) or is_blue_container(r, g, b)


def is_plant(r, g, b):
    if is_goal_colour(r, g, b):
        return False
    return g > 45 and g > b * 1.25 and (g > r * 0.85 or g - r > 18)


def is_trash(r, g, b):
    if is_goal_colour(r, g, b) or is_plant(r, g, b):
        return False
    return (
        60 < r < 190
        and 18 < g < 135
        and b < 80
        and r > g * 1.12
        and g > b * 1.12
        and r - g > 12
        and r - b > 32
    )


def target_for(colour):
    if colour == "Plant":
        return "Blue"
    if colour == "Trash":
        return "Red"
    return None


# -------------------------- Vision --------------------------

def scan_blobs(camera_handle, tests, min_pixels, bottom_weight=True):
    image, width, height = camera_image(camera_handle)
    step = 3
    x_min = int(width * 0.04)
    x_max = int(width * 0.96)
    y_min = int(height * 0.12)
    y_max = int(height * 0.95)
    pixels_by_label = {label: {} for label, _ in tests}

    for gy, y in enumerate(range(y_min, y_max, step)):
        for gx, x in enumerate(range(x_min, x_max, step)):
            idx = 3 * (y * width + x)
            if idx + 2 >= len(image):
                continue
            r = image[idx]
            g = image[idx + 1]
            b = image[idx + 2]
            for label, test in tests:
                if test(r, g, b):
                    weight = 1.0 + (4.0 * (y / height) ** 2 if bottom_weight else 0.0)
                    pixels_by_label[label][(gx, gy)] = (x, y, r, g, b, weight)
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
            sum_x = sum_r = sum_g = sum_b = 0.0
            min_x = width
            max_x = 0
            min_y_seen = height
            max_y_seen = 0

            while stack:
                point = stack.pop()
                x, y, r, g, b, weight = pixels[point]
                count += 1
                score += weight
                sum_x += x * weight
                sum_r += r * weight
                sum_g += g * weight
                sum_b += b * weight
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y_seen = min(min_y_seen, y)
                max_y_seen = max(max_y_seen, y)

                px, py = point
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbour = (px + dx, py + dy)
                    if neighbour in pixels and neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)

            if count < min_pixels or score == 0:
                continue

            offset = (sum_x / score / width - 0.5) * 2.0
            bottom = max_y_seen / height
            width_frac = (max_x - min_x + step) / width
            height_frac = (max_y_seen - min_y_seen + step) / height
            priority = (bottom * 3.0) + min(count, 120) / 60.0 - abs(offset) * 0.25

            blobs.append({
                "colour": label,
                "goal": label,
                "pixels": count,
                "offset": offset,
                "bottom": bottom,
                "priority": priority,
                "width_frac": width_frac,
                "height_frac": height_frac,
                "rgb": (sum_r / score, sum_g / score, sum_b / score),
            })

    return blobs


def choose_object(blobs):
    if not blobs:
        return None

    for plant in [b for b in blobs if b["colour"] == "Plant"]:
        for trash in [b for b in blobs if b["colour"] == "Trash"]:
            same_place = abs(plant["offset"] - trash["offset"]) < 0.35
            same_depth = abs(plant["bottom"] - trash["bottom"]) < 0.30
            if same_place and same_depth:
                plant["priority"] = max(plant["priority"], trash["priority"] + 2.0)

    plausible = []
    for blob in blobs:
        close_cube = blob["bottom"] > 0.72
        compact = blob["width_frac"] < 0.85 and blob["height_frac"] < 0.85
        if compact or close_cube:
            plausible.append(blob)

    if plausible:
        blobs = plausible
    blobs.sort(key=lambda item: item["priority"], reverse=True)
    return blobs[0]


def reliable_search_object(obj):
    return (
        obj["pixels"] >= MIN_SEARCH_OBJECT_PIXELS
        and obj["width_frac"] >= MIN_SEARCH_WIDTH_FRAC
        and obj["height_frac"] >= MIN_SEARCH_HEIGHT_FRAC
        and obj["bottom"] >= MIN_SEARCH_BOTTOM
        and obj["width_frac"] < 0.70
        and obj["height_frac"] < 0.82
    )


def detect_object(strict=True):
    blobs = scan_blobs(
        small_camera,
        [("Plant", is_plant), ("Trash", is_trash)],
        MIN_OBJECT_PIXELS,
        bottom_weight=True,
    )
    obj = choose_object(blobs)
    if obj is None:
        return None
    if strict and not reliable_search_object(obj):
        return None
    r, g, b = obj["rgb"]
    debug(
        f"Detected {obj['colour']}: pixels={obj['pixels']} "
        f"offset={obj['offset']:.2f} bottom={obj['bottom']:.2f} "
        f"rgb=({r:.1f},{g:.1f},{b:.1f})"
    )
    return obj


def detect_container(camera_handle, min_pixels=MIN_CONTAINER_PIXELS):
    blobs = scan_blobs(
        camera_handle,
        [("Red", is_red_container), ("Blue", is_blue_container)],
        min_pixels,
        bottom_weight=False,
    )
    if not blobs:
        return None
    blobs.sort(key=lambda item: item["pixels"], reverse=True)
    blob = blobs[0]
    return {"goal": blob["goal"], "pixels": blob["pixels"], "offset": blob["offset"]}


def front_container(min_pixels=MIN_CONTAINER_PIXELS):
    return detect_container(small_camera, min_pixels)


def top_container(min_pixels=MIN_CONTAINER_PIXELS):
    return detect_container(top_image_sensor._handle, min_pixels)


def top_colour():
    try:
        top_image_sensor._update_image()
        r, g, b = top_image_sensor.rgb()
        if is_yellow(r, g, b):
            return "Yellow"
        if is_red_container(r, g, b):
            return "Red"
        if is_blue_container(r, g, b):
            return "Blue"
    except Exception:
        pass
    return None


# -------------------------- Behaviours --------------------------

def back_away():
    timed_drive(BACK_SPEED, 0.0, 0.45)
    timed_drive(0.0, random.choice([-1.0, 1.0]) * 1.2, 0.55)


def recharge_behaviour():
    if get_battery() >= LOW_BATTERY:
        return None
    if top_colour() == "Yellow":
        stop()
        return "Charging"
    drive(0.65, 0.25)
    time.sleep(STEP_TIME)
    return "Looking for charger"


def avoid_wall_behaviour():
    distance = get_sonar()
    if distance == -1 or distance > WALL_DISTANCE:
        return None
    if detect_object(strict=False) is not None:
        return None
    back_away()
    return "Avoiding wall"


def avoid_container_behaviour(carrying=False, allowed_goal=None):
    front = front_container(90)
    if front is not None and front["goal"] in ("Red", "Blue"):
        if carrying and front["goal"] == allowed_goal:
            return None
        back_away()
        return f"Avoiding {front['goal']} container"

    colour = top_colour()
    if colour in ("Red", "Blue"):
        if carrying and colour == allowed_goal:
            return None
        back_away()
        return f"Leaving {colour} container area"

    return None


# -------------------------- Sorting FSM --------------------------

class Controller:
    def __init__(self):
        self.state = "SEARCH"
        self.object_colour = None
        self.target_goal = None
        self.lost = 0
        self.scan_dir = random.choice([-1.0, 1.0])

    def set_state(self, state, colour=None):
        if state != self.state:
            debug(f"State: {self.state} -> {state}")
        self.state = state
        if colour is not None:
            self.object_colour = colour
            self.target_goal = target_for(colour)
        if state == "SEARCH":
            self.object_colour = None
            self.target_goal = None
        self.lost = 0

    def search(self):
        obj = detect_object(strict=True)
        if obj is not None:
            self.set_state("APPROACH", obj["colour"])
            return self.approach(obj)
        drive(SEARCH_SPEED, 0.18 * self.scan_dir)
        time.sleep(STEP_TIME)
        if random.random() < 0.04:
            self.scan_dir *= -1.0
        return "Wandering"

    def approach(self, obj=None):
        if obj is None:
            obj = detect_object(strict=False)
        if obj is None:
            self.lost += 1
            drive(0.22, 0.0)
            time.sleep(STEP_TIME)
            if self.lost > 8:
                self.set_state("SEARCH")
                return "Lost object"
            return f"Closing on last seen {self.object_colour}"

        if obj["colour"] != self.object_colour:
            self.set_state("APPROACH", obj["colour"])
            self.target_goal = target_for(obj["colour"])

        self.lost = 0
        offset = obj["offset"]
        pixels = obj["pixels"]
        bottom = obj["bottom"]

        ready = (
            abs(offset) < OBJECT_READY_OFFSET
            and pixels >= OBJECT_READY_PIXELS
            and bottom >= OBJECT_READY_BOTTOM
        )
        if ready:
            timed_drive(0.28, clamp(offset * 0.15, -0.08, 0.08), 0.35)
            self.set_state("SCAN_GOAL", obj["colour"])
            stop()
            time.sleep(STEP_TIME)
            return f"{obj['colour']} held; looking for {self.target_goal}"

        if pixels > 55 or bottom > 0.60:
            turn = clamp(offset * 0.30, -0.16, 0.16)
            speed = 0.18 if abs(offset) > 0.35 else CLOSE_SPEED
            drive(speed, turn)
            time.sleep(STEP_TIME)
            return f"Close-aligning {self.object_colour}"

        turn = clamp(offset * 0.80, -0.45, 0.45)
        speed = 0.55 if abs(offset) > 0.55 else APPROACH_SPEED
        drive(speed, turn)
        time.sleep(STEP_TIME)
        return f"Approaching {self.object_colour}"

    def scan_goal(self):
        obj = detect_object(strict=False)
        if obj is not None and obj["colour"] != self.object_colour:
            self.set_state("APPROACH", obj["colour"])
            return f"Correcting label to {obj['colour']}"

        front = front_container(MIN_CONTAINER_PIXELS)
        if front is not None:
            if front["goal"] == self.target_goal:
                self.set_state("PUSH", self.object_colour)
                return f"{self.target_goal} found; pushing {self.object_colour}"
            back_away()
            return f"Wrong container: {front['goal']}"

        top = top_container(MIN_CONTAINER_PIXELS)
        if top is not None and top["goal"] == self.target_goal:
            turn = clamp(top["offset"] * 0.55, -0.28, 0.28)
            drive(0.12, turn)
            time.sleep(STEP_TIME)
            return f"Turning toward {self.target_goal}"

        drive(0.08, 0.22 * self.scan_dir)
        time.sleep(STEP_TIME)
        return f"Scanning for {self.target_goal}"

    def push(self):
        front = front_container(DELIVERY_CONTAINER_PIXELS)
        if front is not None and front["goal"] == self.target_goal:
            turn = clamp(front["offset"] * 0.35, -0.25, 0.25)
            timed_drive(0.65, turn, 0.6)
            back_away()
            self.set_state("SEARCH")
            return f"Delivered {self.object_colour} to {front['goal']}"
        if front is not None and front["goal"] != self.target_goal:
            back_away()
            self.set_state("SEARCH")
            return f"Stopped at wrong {front['goal']} container"

        target = front_container(MIN_CONTAINER_PIXELS)
        if target is None or target["goal"] != self.target_goal:
            self.set_state("SCAN_GOAL", self.object_colour)
            return f"Lost {self.target_goal}; scanning again"

        obj = detect_object(strict=False)
        object_turn = 0.0
        if obj is not None and obj["colour"] == self.object_colour:
            object_turn = clamp(obj["offset"] * 0.45, -0.25, 0.25)
        target_turn = clamp(target["offset"] * 0.65, -0.35, 0.35)
        drive(PUSH_SPEED, clamp(0.6 * object_turn + 0.4 * target_turn, -0.35, 0.35))
        time.sleep(STEP_TIME)
        return f"Pushing {self.object_colour} toward {self.target_goal}"

    def step(self):
        action = recharge_behaviour()
        if action is not None:
            return action

        if self.state == "SEARCH":
            action = avoid_container_behaviour()
            if action is not None:
                return action
            action = avoid_wall_behaviour()
            if action is not None:
                return action

        if self.state == "SEARCH":
            return self.search()
        if self.state == "APPROACH":
            return self.approach()
        if self.state == "SCAN_GOAL":
            return self.scan_goal()
        if self.state == "PUSH":
            action = avoid_container_behaviour(carrying=True, allowed_goal=self.target_goal)
            if action is not None:
                self.set_state("SEARCH")
                return action
            return self.push()

        self.set_state("SEARCH")
        return "Resetting"


sim.startSimulation()
time.sleep(2)

battery = 0.0
while battery <= 0.0:
    try:
        battery = get_battery()
    except Exception:
        battery = 0.0
    time.sleep(0.2)

controller = Controller()
print("Ready! Battery:", battery)
print("Controller:", CODE_VERSION)

while True:
    action = controller.step()
    debug(action)
    debug("-" * 40)
    time.sleep(0.05)
