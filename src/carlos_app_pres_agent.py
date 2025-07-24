from vehicle import Vehicle
from lane import Lane
from environment import Environment
from simulation import Simulation
from sensor_array import SensorArray
import math
import layout_utils
import carlos_logging
import time
import matplotlib.pyplot as plt
import graphics

from presentation_agent import PresentationAgent


# Creating Log
def init_log(file_path: str = None):
    curr_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if file_path is None:
        carlos_logging.init_logger(f"./logs/{curr_time}_carlos_app.log")
    else:
        carlos_logging.init_logger(file_path)


init_log()
carlos_logging.log_message("Carlos App Initialized")

############# INITIALIZATION PARAMETERS ###############
LAYOUT_FILE_PATH = "./layouts/open_loop_0.txt"
MAX_STEPS = 200
MAX_EPISODES = 100  # 500
NUM_SENSORS = 9
SENSOR_LENGTH = 200.0
SENSOR_ANGLE_SPREAD = math.pi
TIME_STEP_SEC = 0.1  # seconds
INITIAL_SPEED_MPH = 25.0  # mph
INITIAL_LONGITUDE = 0.98  # 0 to 1
INITIAL_LATITUDE = 0.5  # 0 to 1
INITIAL_DIR_ANGLE_OFFSET = 0.0  # radians

x_lim = [0, 400]
y_lim = [0, 400]
#### Lane Initialization ####
lane_ctrl_points, lane_width, closed_loop = layout_utils.load_lane_from_file(
    LAYOUT_FILE_PATH
)
lane = Lane(
    control_points=lane_ctrl_points, lane_width=lane_width, closed_loop=closed_loop
)

#### Environment Initialization ####
env = Environment(lane)

#### Vehicle Initialization ####
vehicle = Vehicle()

#### Sensor Array Initialization ####
sensor_array = SensorArray(
    num_sensors=NUM_SENSORS,
    sensor_length=SENSOR_LENGTH,
    sensor_angle_spread=SENSOR_ANGLE_SPREAD,
)

agent = PresentationAgent(sensor_array)
#### Simulation Initialization ####
sim = Simulation(vehicle=vehicle, environment=env, agent=agent, dt=TIME_STEP_SEC)

carlos_logging.log_message("Simulation Initialized")


def elapsed_time(start_time: float) -> float:
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s"


#### Sim Execution ####
def execute_simulation(sim: Simulation, render: bool = False) -> list[float]:
    done = False
    steps = 0
    while not done and steps < MAX_STEPS:
        # Get action + step simulation
        reward = sim.sim_step()

        # Check status
        _, in_lane, in_motion = sim.get_sim_status()
        done = not in_lane or not in_motion

        if render:
            graphics.render_simulation(
                sim=sim,
            )
            graphics.show(
                title="CARLOS Execution Example",
                x_lim=x_lim,
                y_lim=y_lim,
            )

        total_reward += reward
        steps += 1

        speed += sim.vehicle.speed_mph

    avg_speed = speed / steps

    return steps, reward, avg_speed


def plot_rewards(reward_log, window=50):
    avg_rewards = []
    for i in range(len(reward_log)):
        start = max(0, i - window + 1)
        avg = sum(reward_log[start : i + 1]) / (i - start + 1)
        avg_rewards.append(avg)

    plt.figure(figsize=(10, 5))
    plt.plot(reward_log, label="Total Reward per Episode", alpha=0.3)
    plt.plot(avg_rewards, label=f"Moving Avg (window={window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


while True:
    reward_log = []
    step_count_log = []
    sim.sim_random_reset()
    start_time = time.time()
    total_reward, steps, avg_speed = execute_simulation(
        sim=sim, train=False, render=True
    )
    step_count_log.append(steps)
    reward_log.append(total_reward)
    carlos_logging.log_message(
        f"[{elapsed_time(start_time)}] | "
        f"Total Reward: {total_reward:.2f} | Steps: {steps} | "
        f"Distance: {round(sim.vehicle.distance_travelled_ft, 3)} ft | Avg. Speed: {round(avg_speed, 2)} mph"
    )
