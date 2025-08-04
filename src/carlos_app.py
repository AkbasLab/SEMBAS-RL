# app.py
from new_agent import NewAgent
from vehicle import Vehicle
from lane import Lane
from environment import Environment
from simulation import Simulation
from sensor_array import SensorArray
import math
import layout_utils
import carlos_logging
from summer_agent import SummerAgent
import time
import matplotlib.pyplot as plt
import graphics
import numpy as np


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
LAYOUT_FILE_PATH = "src/layouts/train_layout_5.txt"
MAX_STEPS = 1000
MAX_EPISODES = 150
NUM_SENSORS = 5
SENSOR_LENGTH = 200.0
SENSOR_ANGLE_SPREAD = math.pi
TIME_STEP_SEC = 0.01  # seconds
INITIAL_SPEED_MPH = 25.0  # mph
INITIAL_LONGITUDE = 0.0  # 0 to 1
INITIAL_LATITUDE = 0.5  # 0 to 1
INITIAL_DIR_ANGLE_OFFSET = 0.0  # radians


#### Lane Initialization ####
lane_ctrl_points, lane_width, closed_loop = layout_utils.load_lane_from_file(
    LAYOUT_FILE_PATH
)
print("WIDTH", lane_width)
lane = Lane(
    control_points=lane_ctrl_points, lane_width=lane_width, closed_loop=closed_loop
)
# lane = Lane(
#     control_points=[Point(50, 350), Point(350, 350)], lane_width=12.0, closed_loop=False
# )

#### Environment Initialization ####
env = Environment(lane)

#### Vehicle Initialization ####
vehicle = Vehicle(max_acceleration=50.0)

#### Sensor Array Initialization ####
sensor_array = SensorArray(
    num_sensors=NUM_SENSORS,
    sensor_length=SENSOR_LENGTH,
    sensor_angle_spread=SENSOR_ANGLE_SPREAD,
)

#### Agent Initialization ####
# action_dim: int=2, max_accel=5.0, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99
ACTION_DIM = 2
MAX_ACCEL = vehicle.max_acceleration_fps2
LR_ACTOR = 1e-3
LR_CRITIC = 1e-2
GAMMA = 0.99
obs_size = (
    NUM_SENSORS + 2 + 1
)  # Number of sensors + 2 for vehicle heading, speed, and wp_heading
# agent = SummerAgent(
#     sensor_array,
#     obs_dim=obs_size,
#     action_dim=ACTION_DIM,
#     max_accel=MAX_ACCEL,
#     lr_actor=LR_ACTOR,
#     lr_critic=LR_CRITIC,
#     gamma=GAMMA,
# )  # Placeholder for actual agent implementation

# agent = NewAgent(sensor_array)
agent = NewAgent(
    sensor_array,
    obs_dim=obs_size,
    action_dim=ACTION_DIM,
    max_accel=MAX_ACCEL,
    max_turn_rate=np.pi * 3,
    lr_actor=LR_ACTOR,
    lr_critic=LR_CRITIC,
    gamma=GAMMA,
    # lr_schedule=[(0.9, (1e-3, 1e-4))],
)  # Placeholder for actual agent implementation

#### Simulation Initialization ####
sim = Simulation(vehicle=vehicle, environment=env, agent=agent, dt=TIME_STEP_SEC)
sim.sim_reset(
    longitude=INITIAL_LONGITUDE,
    latitude=INITIAL_LATITUDE,
    dir_angle_offset=INITIAL_DIR_ANGLE_OFFSET,
    speed=INITIAL_SPEED_MPH,
)

carlos_logging.log_message("Simulation Initialized")


def elapsed_time(start_time: float) -> float:
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s"


#### Sim Execution ####
def execute_simulation(
    sim: Simulation,
    train: bool = True,
    render: bool = False,
    debug=False,
    num_episodes=MAX_EPISODES,
) -> list[float]:
    last_action_log = []
    loss_log = []
    reward_log = []
    step_count_log = []
    start_time = time.time()
    carlos_logging.log_message("Simulation Execution Started")
    sim.agent.training = train

    state_log = []

    for episode in range(num_episodes):
        has_valid = False
        while not has_valid:
            sim.sim_random_reset()
            sim.update_sim_status()
            has_valid = sim.get_sim_status()[1]

        total_reward = 0
        done = False
        steps = 0
        sim.agent.update_expl_noise(episode, num_episodes)
        sim.agent.update_lr(episode, num_episodes)

        losses = [0, 0]
        actor_loss, critic_loss = None, None

        # state = sim.get_state()
        last_action_log = []
        while not done and steps < MAX_STEPS:
            # Get action + step simulation
            state, action, reward, next_state = sim.sim_step(debug=debug)
            last_action_log.append(action)

            # print(action)
            # print(state)

            # Check status
            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion

            # Train agent
            if train:
                actor_loss, critic_loss = sim.agent.train_step(
                    state, action, reward, next_state, done
                )
                if actor_loss is not None:
                    losses[0] += actor_loss
                    losses[1] += critic_loss
            if render:
                graphics.render_simulation(sim=sim)
                graphics.show()
                input()

            if debug:
                print("Reward, act-l, crit-l:", reward, actor_loss, critic_loss)

            total_reward += reward
            steps += 1

        loss_log.append((losses[0] / len(losses), losses[1] / len(losses)))
        step_count_log.append(steps)
        reward_log.append(total_reward)
        carlos_logging.log_message(
            f"[{elapsed_time(start_time)}] | Episode {episode+1}/{num_episodes} | Total Reward: {total_reward:.2f} | Steps: {steps}"
        )

    return reward_log, step_count_log, state_log, loss_log, last_action_log


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


print("before")
print(sim.agent.actor.fc1.weight)

print("training")
import numpy as np

try:
    reward_log, step_count_log, state_log, loss_log, last_action_log = (
        execute_simulation(sim=sim, render=False, train=True, debug=False)
    )
    # reward_log, step_count_log, state_log, loss_log, last_action_log = (
    #     execute_simulation(sim=sim, render=True, train=True, debug=False)
    # )

    actor_loss, critic_loss = zip(*loss_log)

    fig, (axl, axm, axr) = plt.subplots(1, 3)
    fig.tight_layout()

    axl.set_title("loss")
    axl.set_xlabel("episode")
    axl.set_ylabel("loss")
    axl.plot(np.arange(len(actor_loss)), actor_loss)
    axl.plot(np.arange(len(critic_loss)), critic_loss)

    axm.set_title("reward")
    axm.set_xlabel("episode")
    axm.set_ylabel("reward")
    axm.plot(np.arange(len(reward_log)), reward_log)

    axr.set_title("step count")
    axr.set_xlabel("episode")
    axr.set_ylabel("step count")
    axr.plot(np.arange(len(step_count_log)), step_count_log)
    plt.show()

except KeyboardInterrupt:
    print("Canceling run")
# reward_log, step_count_log, state_log = execute_simulation(
#     sim=sim, render=False, train=True
# )

print("after")
print(agent.actor.fc1.weight)

# print("done")
sim.agent.debug = True
# graphics.render_simulation(sim=sim)
# graphics.show()
while True:
    try:
        reward_log, step_count_log, state_log, loss_log, last_action_log = (
            execute_simulation(
                sim=sim,
                train=False,
                render=True,
                debug=True,
                num_episodes=1,
            )
        )

    except KeyboardInterrupt:
        print("Canceling run")

    steering, acceleration = zip(*last_action_log)

    acceleration = np.array(acceleration)
    steering = np.array(steering) * 180 / np.pi

    print("Steering:", steering.min(), steering.max())
    print("Accel:", acceleration.min(), acceleration.max())

    fig, (axl, axr) = plt.subplots(1, 2)

    axl.set_title("Steer Action")
    axl.set_xlabel("step")
    axl.set_ylabel("steering (deg)")
    axl.plot(np.arange(len(steering)), steering)

    axr.set_title("Acceleration Action")
    axr.set_xlabel("step")
    axr.set_ylabel("acceleration")
    axr.plot(np.arange(len(acceleration)), acceleration)

    plt.show()

# while True:
#     sim.sim_step()
#     graphics.show()


# reward_log = execute_simulation(sim=sim, render=True)
# carlos_logging.log_message("Simulation Executed")
# plot_rewards(reward_log, window=50)
# carlos_logging.log_message("Carlos App Finished")
# input("Press Enter to begin test...")

# while True:
#     reward_log = execute_simulation(sim=sim, train=False, render=True)
