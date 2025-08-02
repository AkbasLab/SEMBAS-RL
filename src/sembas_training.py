from typing import Literal
from new_agent import NewAgent
from point import Point
from sembas_utils import map_norm, run_until_phase
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
import sembas_api as api
import torch
import logging

from numpy import ndarray


############# PARAMETERS ###############
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

# Agent Parameters
ACTION_DIM = 2
MAX_ACCEL = 50.0
LR_ACTOR = 1e-3
LR_CRITIC = 1e-2
# LR_ACTOR = 1e-4
# LR_CRITIC = 1e-3
GAMMA = 0.99
NUM_DIM = 4
SIM_LOW = np.array([0, 0.25, -np.pi / 5, 20.0])
SIM_HIGH = np.array([1, 0.75, np.pi / 5, 75.0])
# SIM_LOW = np.array([0, -np.pi / 5])
# SIM_HIGH = np.array([1, np.pi / 5])

EST_STEPS_PER_EP = 15  # early episodes are short

MSG_REACQ = "REACQ"
##########################################

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s:%(levelname)-8s:%(name)-15s: %(message)s"
)

logger = logging.getLogger("trainer")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("st.log")

file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s:%(levelname)-8s:%(name)-15s: %(message)s")
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)


# Creating Log
def init_log(file_path: str = None):
    curr_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if file_path is None:
        carlos_logging.init_logger(f"./logs/{curr_time}_carlos_app.log")
    else:
        carlos_logging.init_logger(file_path)


init_log()
carlos_logging.log_message("Carlos App Initialized")


def agent_factory():
    sensor_array = SensorArray(
        num_sensors=NUM_SENSORS,
        sensor_length=SENSOR_LENGTH,
        sensor_angle_spread=SENSOR_ANGLE_SPREAD,
    )

    obs_size = NUM_SENSORS + 2 + 1

    return NewAgent(
        sensor_array,
        obs_dim=obs_size,
        action_dim=ACTION_DIM,
        max_accel=MAX_ACCEL,
        max_turn_rate=np.pi * 3,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
    )


def setup_sim():
    lane_ctrl_points, lane_width, closed_loop = layout_utils.load_lane_from_file(
        LAYOUT_FILE_PATH
    )
    lane = Lane(
        control_points=lane_ctrl_points, lane_width=lane_width, closed_loop=closed_loop
    )

    #### Environment Initialization ####
    env = Environment(lane)

    #### Vehicle Initialization ####
    vehicle = Vehicle(max_acceleration=50.0)

    agent = agent_factory()

    #### Simulation Initialization ####
    sim = Simulation(vehicle=vehicle, environment=env, agent=agent, dt=TIME_STEP_SEC)
    x = torch.tensor(
        [
            INITIAL_LONGITUDE,
            INITIAL_LATITUDE,
            INITIAL_DIR_ANGLE_OFFSET,
            INITIAL_SPEED_MPH,
        ]
    )
    sim.sim_reset(*x)

    return sim


class EpisodeData:
    def __init__(
        self,
        parameters: ndarray,
        steps: list,
    ):
        self.parameters = parameters
        self.steps = steps

    @staticmethod
    def failed(parameters: ndarray) -> "EpisodeData":
        return EpisodeData(parameters, [])

    @property
    def num_steps(self):
        return len(self.steps)

    @property
    def total_reward(self):
        return sum(map(lambda s: s[2].item(), self.steps))


def run_episode(
    x: ndarray,
    sim: Simulation,
    train=True,
    step_limit=MAX_STEPS,
    display_mode: Literal["off", "play", "step"] = "off",
):
    "Runs the episode (if valid) and returns step history and class."
    xt = torch.tensor(x)
    sim.sim_reset(*xt)
    sim.update_sim_status()
    is_valid = sim.get_sim_status()[1]

    sim.agent.training = True

    if not is_valid:
        return EpisodeData.failed(x)

    steps = []
    done = False

    if display_mode != "off":
        graphics.render_simulation(sim)
        graphics.show()

    # run episode
    while not done and len(steps) < step_limit:
        state, action, reward, next_state = sim.sim_step()

        sim.update_sim_status()
        _, in_lane, in_motion = sim.get_sim_status()
        done = not in_lane or not in_motion

        steps.append((state, action, reward, next_state))

        if train:
            sim.agent.train_step(state, action, reward, next_state, done)

        if display_mode != "off":
            graphics.render_simulation(sim)
            graphics.show()

            if display_mode == "step":
                input("Press enter to continue")

    return EpisodeData(x, steps)


def train_standard(
    sim: Simulation, num_steps: int, use_noise=True
) -> list[EpisodeData]:
    episode_log = []

    total_steps = 0
    sim.agent.use_noise = use_noise

    while total_steps < num_steps:
        if use_noise:
            sim.agent.update_expl_noise(total_steps, num_steps)

        step_limit = num_steps - total_steps

        x = map_norm((SIM_LOW, SIM_HIGH), np.random.random(len(SIM_LOW)))
        ep = run_episode(x, sim, step_limit=step_limit)

        total_steps += ep.num_steps
        episode_log.append(ep)

    return episode_log


sim = setup_sim()
ep_log = train_standard(sim, 5000)
print([ep.num_steps for ep in ep_log])

run_episode(
    map_norm((SIM_LOW, SIM_HIGH), np.array([0.5] * 4)),
    sim,
    display_mode="play",
    train=False,
)
