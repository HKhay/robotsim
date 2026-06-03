import robots
from robots import *
import time
import random
from coppeliasim_zmqremoteapi_client import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

client = RemoteAPIClient()
sim = client.require("sim")

# HANDLES FOR ACTUATORS AND SENSORS
robot = Robot_OS(sim, DeviceNames.ROBOT_OS)

top_image_sensor = ImageSensor(sim, DeviceNames.TOP_IMAGE_SENSOR_OS)
small_image_sensor = ImageSensor(sim, DeviceNames.SMALL_IMAGE_SENSOR_OS)

left_motor = Motor(sim, DeviceNames.MOTOR_LEFT_OS, Direction.CLOCKWISE)
right_motor = Motor(sim, DeviceNames.MOTOR_RIGHT_OS, Direction.CLOCKWISE)

current_cube = None
container_direction = None

# HELPER FUNCTIO
def show_image(image):
    plt.imshow(image)
    plt.show()

def get_battery():
    signal = robot.get_string_signal("battery")
    if signal is None:
        return 0.0
    return float(signal.decode("utf-8") if isinstance(signal, bytes) else signal)

def get_sonar():
    try:
        sonar_handle = sim.getObject('/dr12/dr12_sonar')
        result = sim.readProximitySensor(sonar_handle)
        if result[0] > 0:
            return result[1]
        return -1
    except:
        return -1

def get_bumper():
    try:
        bumper_handle = sim.getObject('/dr12/dr12_bumperForceSensor_')
        result = sim.readForceSensor(bumper_handle)
        if result[0] > 0:
            force = sim.readForceSensor(bumper_handle)
            return (force[1][0], force[1][1], force[1][2])
        return (0.0, 0.0, 0.0)
    except:
        return (0.0, 0.0, 0.0)


def get_center_rgb(sensor):
    sensor._update_image()
    img = sensor.image
    h, w = img.shape[0], img.shape[1]
    # sample center region (middle 40% of image)
    cy, cx = h // 2, w // 2
    region = img[cy-h//5:cy+h//5, cx-w//5:cx+w//5]
    r = float(np.mean(region[:, :, 0])) / 255 * 100
    g = float(np.mean(region[:, :, 1])) / 255 * 100
    b = float(np.mean(region[:, :, 2])) / 255 * 100
    return r, g, b


def move_forward(left_speed, right_speed):
    left_motor.run(left_speed)
    right_motor.run(right_speed)

def turn(left_speed, right_speed):
    left_motor.run(left_speed)
    right_motor.run(-right_speed)

def stop():
    left_motor.run(0)
    right_motor.run(0)

def is_facing_container():
    small_image_sensor._update_image()
    r, g, b = small_image_sensor.rgb()
    is_red = r > 60 and g < 10 and b < 10
    is_blue = b > 15 and r < 5
    return is_red or is_blue


def wander():
    move_forward(2,2)
    time.sleep(0.5)
    turn(1, random.choice([-1,1]))
    time.sleep(0.3)

def detect_colour():
    r, g, b = get_center_rgb(small_image_sensor)
    is_plant = (g > r * 1.3) and (g > b * 1.3)
    is_trash = (r > 50) and (r > g * 1.5) and (b < 30)
    if is_plant:
        return "Plant"
    if is_trash:
        return "Trash"
    return None



# Robot should detect distance between itself and wall.
# if it is a wall it will turn in position.
#if there is wall return distance if there is none return None.

def avoid_wall():
    if is_facing_container():
        move_forward(-2, -2)
        time.sleep(1.0)
        turn(3, 3)
        time.sleep(0.8)
        return "Avoiding container"

    # Wall: r and g nearly equal, blue noticeably higher, all mid-range.
    small_image_sensor._update_image()
    r, g, b = small_image_sensor.rgb()
    is_wall = abs(r - g) < 3 and b > r + 6 and b > g + 6 and r < 55

    if is_wall:
        move_forward(-2, -2)
        time.sleep(0.8)
        turn(3, random.choice([-2, 2]))
        time.sleep(0.6)
        return "Avoiding wall"

    dist = get_sonar()
    colour = detect_colour()
    if dist != -1 and dist < 0.2 and colour is None:
        move_forward(-2, -2)
        time.sleep(0.5)
        turn(3, random.choice([-2, 2]))
        time.sleep(0.5)
        return "Avoiding wall"
    return None

def detect_goal():
    top_image_sensor._update_image()
    (r, g, b) = top_image_sensor.rgb()


    if r > 80 and g > 80 and b < 10:
        return "Yellow"
    elif b > 60 and r < 10:
        return "Blue"
    elif r > g * 1.5 and r > b * 1.5 and r > 50:
        return "Red"
    else:
        return None


## Robot needs to first get the current battery value
## if battery level is less than 30 and distance is not < 0.10 then detect charging station using detect_goal() == "Yellow"
# return dist and then robot should look for the charging station using detect_goal() == "Yellow"
## if recharge is True and detect goal ==  yellow then we need to stop()


def recharge():
    battery = get_battery()
    if battery < 0.4:
        goal = detect_goal()
        if goal == "Yellow":
            dist = get_sonar()
            if dist != -1 and dist < 0.15:
                stop()
                return "Charging"
            else:
                move_forward(1, 1)
                return "Moving to Charger"
        else:
            move_forward(1, 1)
            turn(1, 1)
            return "Looking for charging station"
    else:
        return None


def engage_cube():
    colour = detect_colour()
    dist = get_sonar()
    fx, fy, fz = get_bumper()

    if colour == "Plant" or colour == "Trash":
        if abs(fy) > 1.0:          # bumper confirms we're touching it
            stop()
            return colour
        else:
            move_forward(2, 2)      # camera sees cube -> drive in
            return None
    elif dist != -1 and dist < 0.4:
        # sonar sees something ahead but camera hasn't classified it yet
        move_forward(2, 2)          # close the gap so camera can see it
        return None
    else:
        return None


def push_to_container(colour):

    global container_direction

    # Step 1 - check if delivered
    if is_facing_container():
        container_direction = None
        return "Delivered"

    top_image_sensor._update_image()
    tr, tg, tb = top_image_sensor.rgb()
    goal = detect_goal()
    print(f"Pushing {colour}, TOP r={tr:.0f} g={tg:.0f} b={tb:.0f}, goal: {goal}, dir: {container_direction}")

    goal = detect_goal()
    print(f"Pushing {colour}, top camera sees: {goal}, direction: {container_direction}")
    # Step 2 - if we already know direction, keep pushing
    if container_direction is not None:
        move_forward(2, 2)
        return "Moving!"

    # Step 3 - look for the correct container

    if colour == "Trash" and goal == "Red":
        robot.compress()
        container_direction = "Red"
        move_forward(2, 2)
        return "Moving!"
    elif colour == "Plant" and goal == "Blue":
        container_direction = "Blue"
        move_forward(2, 2)
        return "Moving!"
    else:
        # Can't see container yet - spin to find it
        move_forward(-1, -1)
        time.sleep(0.3)
        turn(3, 3)
        time.sleep(0.3)
        return "Looking"

# Starts coppeliasim simulation if not done already
sim.startSimulation()
time.sleep(2)

# Wait until battery signal is ready
battery_ready = False
while not battery_ready:
    try:
        b = get_battery()
        if b > 0:
            battery_ready = True
    except:
        time.sleep(0.5)

print("Ready! Battery:", b)

# MAIN CONTROL LOOP

while True:
    if recharge() is not None:
        continue
    if avoid_wall() is not None:
        continue

    colour = detect_colour()
    dist = get_sonar()

    if colour == "Plant" or colour == "Trash":
        if dist != -1 and dist < 0.2:
            # Close enough - engage and push
            current_cube = colour
            push_to_container(current_cube)
            if is_facing_container():
                current_cube = None
        else:
            # See cube but not close - move toward it
            move_forward(2, 2)
        continue  # skip wander

    wander()
