import logging

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

from typing import Literal
from new_agent import NewAgent
from sembas_utils import map_norm, run_until_phase, reset
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


class EpisodeData:
    def __init__(
        self,
        parameters: ndarray,
        steps: list,
        distance_traveled: float,
        cls: bool,
        interestingness=None,
    ):
        self.parameters = parameters
        self.steps = steps
        self.cls = cls
        self.distance_traveled = distance_traveled
        self.interestingness = interestingness

    @staticmethod
    def failed(parameters: ndarray) -> "EpisodeData":
        return EpisodeData(parameters, [], 0.0, False)

    @property
    def is_failure(self):
        return len(self.steps) == 0

    @property
    def num_steps(self):
        return len(self.steps)

    @property
    def total_reward(self):
        return sum(map(lambda s: s[2].item(), self.steps))

    @staticmethod
    def get_step_history(ep_log: list["EpisodeData"]):
        return np.array([ep.num_steps for ep in ep_log])

    @staticmethod
    def get_reward_history(ep_log: list["EpisodeData"]):
        return np.array([ep.total_reward for ep in ep_log])

    @staticmethod
    def get_request_history(ep_log: list["EpisodeData"]):
        return np.array([ep.parameters for ep in ep_log])

    @staticmethod
    def get_distance_history(ep_log: list["EpisodeData"]):
        return np.array([ep.distance_traveled for ep in ep_log])


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
    sim.sim_reset(
        INITIAL_LONGITUDE,
        INITIAL_LATITUDE,
        INITIAL_DIR_ANGLE_OFFSET,
        INITIAL_SPEED_MPH,
    )

    return sim


def run_episode(
    x: ndarray,
    sim: Simulation,
    step_criteria: int = None,
    train=True,
    step_limit=MAX_STEPS,
    display_mode: Literal["off", "play", "step"] = "off",
    skip_failures=True,
    show_reward=False,
):
    "Runs the episode (if valid) and returns step history and class."
    # sim.sim_reset(*x)
    reset(sim, x)
    sim.update_sim_status()
    is_valid = sim.get_sim_status()[1]

    if skip_failures and not is_valid:
        return EpisodeData.failed(x)

    sim.agent.training = True
    steps = []
    done = False

    if display_mode != "off":
        graphics.render_simulation(sim)
        graphics.show()
    if show_reward:
        rewards = []
        fig, ax = plt.subplots()
        ax.set_title("reward")

    # run episode
    while not done and len(steps) < step_limit:
        state, action, reward, next_state = sim.sim_step()

        sim.update_sim_status()
        _, in_lane, in_motion = sim.get_sim_status()
        done = not in_lane or not in_motion

        steps.append((state, action, reward, next_state))
        if show_reward:
            ax.clear()
            rewards.append(reward.item())
            ax.plot(np.arange(len(rewards)), rewards, color="green")
            plt.pause(0.1)

        if train:
            sim.agent.train_step(state, action, reward, next_state, done)

        if display_mode != "off":
            graphics.render_simulation(sim)
            graphics.show()

            if display_mode == "step":
                input("Press enter to continue")

    return EpisodeData(
        x,
        steps,
        sim.vehicle.distance_travelled_ft,
        (len(steps) >= step_criteria) if step_criteria else None,
    )


def train_standard(
    sim: Simulation,
    num_steps: int,
    noise_step_start: int = None,
    noise_step_target: int = None,
    fixed_noise: float = None,
) -> list[EpisodeData]:
    assert (fixed_noise or noise_step_start) or not (noise_step_start and fixed_noise)
    dists = []
    episode_log = []

    total_steps = 0
    use_noise = noise_step_start is not None or fixed_noise is not None
    noise_step_target = noise_step_target or num_steps

    if fixed_noise:
        sim.agent.set_noise(fixed_noise)
    sim.agent.use_noise = use_noise

    while total_steps < num_steps:
        print(total_steps)
        if noise_step_start:
            sim.agent.update_expl_noise(
                noise_step_start + total_steps, noise_step_target
            )

        step_limit = num_steps - total_steps

        x = map_norm((SIM_LOW, SIM_HIGH), np.random.random(len(SIM_LOW)))
        ep = run_episode(x, sim, step_limit=step_limit)
        if ep.is_failure:
            continue

        total_steps += ep.num_steps
        episode_log.append(ep)

    return episode_log


def perf_test(sim: Simulation, num_episodes: int, max_steps=1000):
    ep_log = []
    sim.agent.use_noise = False
    for i in range(num_episodes):
        logger.info(f"Episode {i}")
        logger.info(f"Finding valid start...")
        has_valid = False
        while not has_valid:
            x = map_norm((SIM_LOW, SIM_HIGH), np.random.random(len(SIM_LOW)))
            # sim.sim_reset(*x)
            reset(sim, x)
            sim.update_sim_status()
            has_valid = sim.get_sim_status()[1]

        logger.info("Starting episode")
        ep_log.append(run_episode(x, sim, train=False, step_limit=max_steps))

    return ep_log


def train_sembas(
    session: api.SembasSession,
    sim: Simulation,
    step_criteria: int,
    step_count_target: int,
    batch_step_size: int,
    random_size: int = None,
    expl_batch_size_factor=1.5,
    max_global_search=1000,
    fixed_sembas_noise=0.3,
    fixed_random_noise=None,
):
    """
    Arguments:
    - session (SembasSession): The session with SEMBAS to handle parameter selection.
    - sim (Simulation): The simulation to run.
    - step_criteria (int): The number of steps defining a "success".
    - step_count_target (int): The number of training steps before termination.
    - batch_step_size (int): The number of SEMBAS boundary training steps for each batch of training.
        Keep this as some factor of @step_count_target.
    - random_size (int, optional): The number of random training steps between sembas training.
    - expl_batch_size_factor (float): If random_size is specified, this determines how many exploration
        steps to take (random_size * expl_batch_size_factor).
        This is needed due to train steps != exploration steps due to RNG. This provides a margin of
        additional training data to prevent not hitting the training target. If this is too small, it
        is possible/likely the total number of training steps will not equal @step_count_target.
    - max_global_search (int): The limit to the number of GS episodes taken before failing out.
    """
    # fig, (axl, axr) = plt.subplots(1, 2)
    # fig, axl = plt.subplots()
    # dists = []

    batch_upper_bound = int(batch_step_size * expl_batch_size_factor)
    training_data = []
    batch_episodes = []
    process = "NewSearch"
    total_train_steps = 0
    batch_steps = 0
    while total_train_steps < step_count_target or process == "Training":
        match process:
            case "NewSearch":
                logger.info("Starting new search")
                run_until_phase(
                    session,
                    sim,
                    step_criteria,
                    "BE",
                    max_steps=max_global_search,
                )
                if session.prev_known_phase == "BE":
                    process = "BE"
                else:
                    raise RuntimeError("Failed to find boundary during GS!")
            case "BE":
                if session.expect_phase() != api.SembasSession.PHASE_BOUNDARY_EXPL:
                    logger.info(
                        f"Phase change to {session.prev_known_phase}, assuming boundary complete. Starting new search."
                    )
                    process = "NewSearch"
                    continue
                elif batch_steps >= batch_upper_bound:
                    process = "Training"
                    continue

                x = session.receive_request()
                ep = run_episode(
                    x,
                    sim,
                    step_criteria,
                    train=False,
                    step_limit=batch_upper_bound - batch_steps,
                    # display_mode="play",
                    # show_reward=True,
                )
                batch_steps += ep.num_steps
                session.send_response(ep.cls)

                if not ep.is_failure:
                    batch_episodes.append(ep)

            case "Training":
                logger.info("Training")
                step_limit = min(batch_step_size, step_count_target - total_train_steps)

                if random_size:
                    logger.info("[Training] Random training")
                    train_standard(
                        sim,
                        random_size,
                        total_train_steps,
                        step_count_target,
                        fixed_noise=fixed_random_noise,
                    )

                logger.info("[Training] Sembas training")
                ep_log = rerun_and_train(
                    sim,
                    EpisodeData.get_request_history(batch_episodes),
                    step_criteria,
                    # cur_step=total_train_steps,
                    step_limit=step_limit,
                    target_steps=step_count_target,
                    fixed_noise=fixed_sembas_noise,
                )

                # test_eps = perf_test(sim, 50)
                # dists.append(np.array([ep.distance_traveled for ep in test_eps]).mean())
                # axl.clear()
                # axl.plot(np.arange(len(dists)), dists, color="blue")
                # plt.pause(0.5)

                total_train_steps += sum(EpisodeData.get_step_history(ep_log))
                training_data.extend(ep_log)

                process = "NewSearch"
                session.send_message(MSG_REACQ)
                batch_episodes = []
                batch_steps = 0

    return training_data


def warmup(
    sim: Simulation,
    target_step_count: int = None,
    target_distance: float = None,
    step_limit=None,
    window_size=10,
):
    assert target_step_count or target_distance
    sim.agent = agent_factory()
    i = 0
    num_steps = 0

    history = []
    target = target_step_count or target_distance

    rng = lambda: map_norm((SIM_LOW, SIM_HIGH), torch.rand(len(SIM_LOW)))

    num_passes = lambda: (np.array(history[-window_size:]) >= target).sum()

    while num_passes() < window_size // 2 and (
        num_steps is None or num_steps < step_limit
    ):
        sim.agent.update_expl_noise(i, step_limit or 1500)

        has_valid = False
        while not has_valid:
            x = rng()
            # sim.sim_reset(*x)
            reset(sim, x)
            sim.update_sim_status()
            has_valid = sim.get_sim_status()[1]

        ep = run_episode(x, sim, step_limit=step_limit - num_steps)
        num_steps += ep.num_steps
        history.append(ep.num_steps if target_step_count else ep.distance_traveled)

    return num_passes() >= window_size // 2


def rerun_and_train(
    sim: Simulation,
    requests: list[np.ndarray],
    step_criteria: int,
    cur_step: int = None,
    target_steps: int = None,
    step_limit: int = None,
    fixed_noise=None,
):
    """
    Arguments:
    - sim (Simulation): The simulation to run.
    - requests (list[ndarray]): The requests to re-run and train over.
    - cur_step (int): How many steps of training have occurred up until this point. If None, assumed to be 0.
    - target_steps (int): How many steps of training until complete. If None, assumed to be step_limit. If
        step_limit is None, will estimate based on the number of requests (50 steps per req).
        Used for determining how to scale exploration noise over the course of training.
    - step_limit (int): The maximum training steps before ending. If None, will run through all requests to
        completion.
    """

    ep_log = []
    total_steps_taken = 0

    target_steps = target_steps or step_limit or len(requests) * 50

    use_noise = cur_step is not None or fixed_noise is not None

    sim.agent.training = True
    if fixed_noise:
        sim.agent.use_noise = True
        sim.agent.exploration_noise = fixed_noise

    sim.agent.use_noise = use_noise

    for i, x in enumerate(requests):
        if total_steps_taken >= step_limit:
            break

        if use_noise and fixed_noise is None:
            sim.agent.update_expl_noise(cur_step + i, target_steps)

        # sim.sim_reset(*x)
        reset(sim, x)
        sim.update_sim_status()
        is_valid = sim.get_sim_status()[1]
        if not is_valid:
            continue

        ep = run_episode(
            x,
            sim,
            step_criteria,
            step_limit=step_limit - total_steps_taken,
        )
        ep_log.append(ep)
        total_steps_taken += ep.num_steps

    return ep_log


def timeit_test():
    from timeit import default_timer as timer

    times = []
    for i in range(10):
        t0 = timer()
        sim = setup_sim()
        ep_log = train_standard(sim, 2000)
        times.append(timer() - t0)

    print(sum(times) / len(times))


# timeit_test()

# 84.817

# sim = setup_sim()
# ep_log = train_standard(sim, 2000)
# print([ep.num_steps for ep in ep_log[-10:]])

# import pstats

# pstats.Stats("profile.out").sort_stats("cumtime").print_stats(20)

# run_episode(
#     map_norm((SIM_LOW, SIM_HIGH), np.array([0.5] * 4)),
#     sim,
#     display_mode="step",
#     train=False,
# )
