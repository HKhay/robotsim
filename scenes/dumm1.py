from robots import *
import time
from coppeliasim_zmqremoteapi_client import *

client = RemoteAPIClient()
sim = client.require("sim")
left_motor = Motor(sim, DeviceNames.MOTOR_LEFT_OS, Direction.CLOCKWISE)
right_motor = Motor(sim, DeviceNames.MOTOR_RIGHT_OS, Direction.CLOCKWISE)

def get_bumper():
    try:
        h = sim.getObject('/dr12/dr12_bumperForceSensor_')
        res = sim.readForceSensor(h)
        if res[0] > 0:
            return (res[1][0], res[1][1], res[1][2])
        return (0.0, 0.0, 0.0)
    except:
        return (0.0, 0.0, 0.0)
def engage_cube():
    colour = detect_colour()
    fx, fy, fz = get_bumper()

    print(f"bumper y={fy:.2f} z={fz:.2f} | colour={colour}")

        if colour == "Plant" or colour == "Trash":
            # contact shows up as lateral force on y (and slight z); rest is ~0
            if abs(fy) > 0.2 or fz > 0.25:
                stop()
                return colour
            else:
                move_forward(2, 2)  # keep driving toward the cube
                return None
        else:
            return None

sim.startSimulation()
time.sleep(2)
print("Ready")

while True:
    left_motor.run(2)
    right_motor.run(2)   # drive straight into a cube/wall
    fx, fy, fz = get_bumper()
    print(f"x={fx:.2f} y={fy:.2f} z={fz:.2f}")
    time.sleep(0.1)