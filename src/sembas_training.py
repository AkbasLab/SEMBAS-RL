# app.py
from new_agent import NewAgent
from point import Point
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

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s:%(levelname)-8s:%(name)-15s: %(message)s"
)

logger = logging.getLogger("trainer")
logger.setLevel(logging.WARNING)

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

# Agent Parameters
ACTION_DIM = 2
MAX_ACCEL = 50.0
LR_ACTOR = 1e-3
LR_CRITIC = 1e-2
GAMMA = 0.99
NUM_DIM = 4
SIM_LOW = np.array([0, 0.25, -np.pi / 5, 20.0])
SIM_HIGH = np.array([1, 0.75, np.pi / 5, 75.0])
# SIM_LOW = np.array([0, -np.pi / 5])
# SIM_HIGH = np.array([1, np.pi / 5])

EST_STEPS_PER_EP = 15  # early episodes are short

MSG_REACQ = "REACQ"

#### Lane Initialization ####
lane_ctrl_points, lane_width, closed_loop = layout_utils.load_lane_from_file(
    LAYOUT_FILE_PATH
)
print("WIDTH", lane_width)
lane = Lane(
    control_points=lane_ctrl_points, lane_width=lane_width, closed_loop=closed_loop
)

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
# Number of sensors + 2 for vehicle heading, speed, and wp_heading
obs_size = NUM_SENSORS + 2 + 1

agent_factory = lambda: NewAgent(
    sensor_array,
    obs_dim=obs_size,
    action_dim=ACTION_DIM,
    max_accel=MAX_ACCEL,
    max_turn_rate=np.pi * 3,
    lr_actor=LR_ACTOR,
    lr_critic=LR_CRITIC,
    gamma=GAMMA,
)

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


def elapsed_time(start_time: float) -> float:
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s"


def map_norm(bounds, x) -> ndarray:
    x = np.array(x)
    return x * (bounds[1] - bounds[0]) + bounds[0]


def train_random(sim: Simulation, num_episodes: int = None, num_steps: int = None):
    """
    Trains the agent with an initial amount of random experience to
    establish an initial region of competence.
    """
    assert num_episodes or step_i, "Must specify either num eps or steps"

    agent: NewAgent = sim.agent

    reward_log = []
    step_log = []
    ep_i = 0
    step_i = 0
    total_steps = 0
    while (
        num_episodes is None
        or ep_i < num_episodes
        and (num_steps is None or total_steps < num_steps)
    ):
        # longitude: float, latitude: float, dir_angle_offset: float, speed: float
        # sim.sim_random_reset()
        agent.update_expl_noise(ep_i, num_episodes)

        # scale
        is_valid = False
        while not is_valid:
            x = np.random.random(4)
            x = map_norm((SIM_LOW, SIM_HIGH), x)
            sim.sim_reset(*x)
            sim.update_sim_status()
            is_valid = sim.get_sim_status()[1]

        done = False
        steps = 0
        running_reward = 0

        # run episode
        while (
            not done
            and steps < MAX_STEPS
            and (num_steps is None or total_steps < num_steps)
        ):
            state, action, reward, next_state = sim.sim_step()
            running_reward += reward

            sim.update_sim_status()
            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion

            sim.agent.train_step(state, action, reward, next_state, done)

            steps += 1
            total_steps += 1

        reward_log.append(running_reward.item())
        step_log.append(steps)
        if (ep_i + 1) % (num_episodes // 20) == 0:
            print(
                f"{ep_i+1} : {sum(step_log[-10:]) / 10}, {min(step_log[-10:])}, {max(step_log[-10:])}"
            )

        ep_i += 1

    return reward_log, step_log


def random_training_with_early_stopping(
    sim: Simulation, median_step_count_change: int, window_size: int = 10
):
    """
    Trains the agent with an initial amount of random experience to
    establish an initial region of competence.
    """
    num_steps = 0

    agent: NewAgent = sim.agent

    reward_log = []
    step_log = []
    i = 0
    while (
        len(step_log) < window_size
        and (
            (
                np.array(step_log[1 - window_size :])
                - np.array(step_log[-window_size:-1])
            )
            >= median_step_count_change
        ).sum()
        >= window_size // 2
    ):
        agent.update_expl_noise(i, 150)

        is_valid = False
        while not is_valid:
            x = np.random.random(4)
            x = map_norm((SIM_LOW, SIM_HIGH), x)
            sim.sim_reset(*x)
            sim.update_sim_status()
            is_valid = sim.get_sim_status()[1]

        done = False
        steps = 0
        running_reward = 0

        # run episode
        while not done and steps < MAX_STEPS:
            state, action, reward, next_state = sim.sim_step()
            running_reward += reward

            sim.update_sim_status()
            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion

            sim.agent.train_step(state, action, reward, next_state, done)

            num_steps += 1
            steps += 1

        reward_log.append(running_reward.item())
        step_log.append(steps)
        if (i + 1) % (num_episodes // 20) == 0:
            print(
                f"{i+1} : {sum(step_log[-10:]) / 10}, {min(step_log[-10:])}, {max(step_log[-10:])}"
            )
        i += 1

    return reward_log, step_log


def run_sembas_episode(session: api.SembasSession, sim: Simulation, crit_step_c: int):
    """
    Returns the training data developed on a given pass of the
    SEMBAS algorithm.
    """
    # longitude: float, latitude: float, dir_angle_offset: float, speed: float
    # x = receive_request(client)
    x = session.receive_request()
    # print("state", x.shape, x)
    # sim.sim_reset(*x)
    xt = torch.tensor(x)
    # sim.sim_reset(xt[0], 0.5, xt[1], 45)
    sim.sim_reset(*xt)
    # scale
    sim.update_sim_status()
    valid_init_state = sim.get_sim_status()[1]
    if not valid_init_state:
        session.send_response(False)
        return xt, False, []

    done = False
    steps = 0

    episode_data = []
    # run episode
    while not done and steps < MAX_STEPS:
        # Get action + step simulation
        state, action, reward, next_state = sim.sim_step()

        # Check status
        sim.update_sim_status()
        _, in_lane, in_motion = sim.get_sim_status()
        done = not in_lane or not in_motion

        # new_train_data.append((state, action, reward, next_state, done))
        episode_data.append((state, action, reward, next_state, done))
        steps += 1

    # Target performance is unsafe behavior (i.e. not in lane)
    # api.send_response(client, not in_lane)
    session.send_response(steps >= crit_step_c)

    return x, steps >= crit_step_c, episode_data


def warmup(
    sim: Simulation,
    target_step_c=50,
    max_episodes: int = None,
    apply_expl_noise=True,
    window_size=10,
):
    """
    Trains the agent with an initial amount of random experience to
    establish an initial region of competence.
    """
    # reset agent to fresh agent
    sim.agent = agent_factory()
    i = 0
    num_steps = 0
    if apply_expl_noise:
        agent.update_expl_noise(0, max_episodes or 100)
    else:
        agent.exploration_noise = 0.0

    step_history = []

    logger.debug(f"Target step count: {target_step_c}")

    rng = lambda: map_norm((SIM_LOW, SIM_HIGH), torch.rand(len(SIM_LOW)))
    # rng = lambda: torch.tensor(
    #     (np.random.rand(), 0.5, np.random.rand() * np.pi * 2 / 5 - np.pi / 5, 45.0)
    # )
    # reset = lambda: sim.sim_reset(
    #     np.random.rand(), 0.5, np.random.rand() * np.pi * 2 / 5 - np.pi / 5, 45.0
    # )

    invalid_states = []
    target_states = []
    nontarget_states = []

    while (
        np.array(step_history[-window_size:]) >= target_step_c
    ).sum() < window_size // 2:
        # longitude: float, latitude: float, dir_angle_offset: float, speed: float
        # sim.sim_random_reset()
        if i >= max_episodes:
            return False
        if apply_expl_noise:
            sim.agent.update_expl_noise(i, max_episodes or 100)

        # scale
        has_valid = False
        while not has_valid:
            x = rng()
            sim.sim_reset(*x)
            sim.update_sim_status()
            has_valid = sim.get_sim_status()[1]
            if not has_valid:
                invalid_states.append(x)

        done = False
        steps = 0

        # run episode
        while not done and steps < MAX_STEPS:
            state, action, reward, next_state = sim.sim_step()

            sim.update_sim_status()
            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion

            sim.agent.train_step(state, action, reward, next_state, done)

            num_steps += 1
            steps += 1

        step_history.append(steps)
        if steps >= target_step_c:
            target_states.append(x)
        else:
            nontarget_states.append(x)

        i += 1

    logger.debug(
        f"[Warmup] Observed: targets: {len(target_states)} non-targets: {len(nontarget_states)}"
    )

    return True


def rerun_and_train(
    sim: Simulation, requests: list[ndarray], ep: int, limit_steps: int = None
):
    """
    Trains the agent with an initial amount of random experience to
    establish an initial region of competence.
    """
    num_steps = 0

    agent: NewAgent = sim.agent

    reward_log = []
    step_log = []
    total_steps = 0

    for i, x in enumerate(requests):
        # longitude: float, latitude: float, dir_angle_offset: float, speed: float
        # sim.sim_random_reset()
        agent.update_expl_noise(ep + i, len(requests))

        # scale
        x = torch.tensor(x)
        # sim.sim_reset(x[0], 0.5, x[1], 45)
        sim.sim_reset(*x)

        sim.update_sim_status()
        is_valid = sim.get_sim_status()[1]
        if not is_valid:
            continue

        done = False
        steps = 0
        running_reward = 0

        # run episode
        while not done and steps < MAX_STEPS:
            if limit_steps is not None and total_steps >= limit_steps:
                break
            state, action, reward, next_state = sim.sim_step()
            running_reward += reward.item()

            sim.update_sim_status()
            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion

            sim.agent.train_step(state, action, reward, next_state, done)

            num_steps += 1
            steps += 1
            total_steps += 1

        reward_log.append(running_reward)
        step_log.append(steps)
        if limit_steps is not None and total_steps >= limit_steps:
            break

    print(
        f"{i} : {sum(step_log[-10:]) / 10}, {min(step_log[-10:])}, {max(step_log[-10:])}"
    )

    return reward_log, step_log


def run_until_phase(
    session: api.SembasSession, sim: Simulation, crit_step_c: int, target_phase: str
) -> dict[str, list[tuple[tuple, bool]]]:
    """
    Returns the training data developed on a given pass of the
    SEMBAS algorithm.
    """
    new_train_data = {}  # phase : list[(list[train_step], cls)]
    # session.force_continue()
    session.expect_phase()
    logger.debug(f"Running until {target_phase}, starting {session.prev_known_phase}")
    while session.prev_known_phase != target_phase:
        if session.prev_known_phase not in new_train_data:
            logger.debug(f"Starting phase {session.prev_known_phase}")
            new_train_data[session.prev_known_phase] = []

        # longitude: float, latitude: float, dir_angle_offset: float, speed: float
        # x = receive_request(client)
        x = session.receive_request()
        x = torch.tensor(x)
        # sim.sim_reset(x[0], 0.5, x[1], 45)
        sim.sim_reset(*x)
        # scale
        sim.update_sim_status()
        is_valid = sim.get_sim_status()[1]
        if not is_valid:
            # api.send_response(client, False)
            new_train_data[session.prev_known_phase].append(([], False))
            session.send_response(False)
            session.expect_phase()
            continue

        done = False
        steps = 0
        episode_data = []

        # run episode
        while not done and steps < MAX_STEPS:
            # Get action + step simulation
            state, action, reward, next_state = sim.sim_step()

            # Check status
            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion

            # new_train_data.append((state, action, reward, next_state, done))
            episode_data.append((state, action, reward, next_state, done))
            steps += 1

        # Target performance is unsafe behavior (i.e. not in lane)
        # api.send_response(client, not in_lane)
        new_train_data[session.prev_known_phase].append(
            (episode_data, steps >= crit_step_c)
        )
        session.send_response(steps >= crit_step_c)
        session.expect_phase()

    logger.debug(f"Phase {session.prev_known_phase} reached")
    return new_train_data


# def get_even_split()
def traditional_training(
    wup_path: str,
    group_size: int,
    num_groups: int,
    with_grouping=False,
):
    # re-use the warmed up model used by SEMBAS
    sim.agent.load(wup_path)
    reward_log = []
    step_log = []

    if with_grouping:
        for i in range(num_groups):
            r, s = train_random(sim, group_size)
            reward_log.extend(r)
            step_log.extend(s)
    else:
        reward_log, step_log = train_random(sim, group_size * num_groups)

    return reward_log, step_log


def traditional_training_by_steps(
    wup_path: str,
    num_steps: int,
):
    # re-use the warmed up model used by SEMBAS
    sim.agent.load(wup_path)
    reward_log = []
    step_log = []

    reward_log, step_log = train_random(sim, num_steps=num_steps)

    return reward_log, step_log


def sembas_reacquisition(
    session: api.SembasSession, sim: Simulation, crit_step_c: int
) -> dict[str, list[tuple[tuple, bool]]]:
    """
    Signals to SEMBAS that boundary reacquisition is necessary and samples until
    SEMBAS has reacquired the boundary and is exploring once more.
    """

    session.send_message(MSG_REACQ)
    new_train_data = run_until_phase(
        session, sim, crit_step_c, api.SembasSession.PHASE_BOUNDARY_EXPL
    )

    return new_train_data


def sembas_training(
    batch_size: int,
    crit_step_c: int,
    num_iterations: int = None,
    plot_samples=False,
    init_crit_step_c=None,
    include_warmup=True,
    save_warmup=True,
    wup_suffix: str = None,
    wup_subdir="misc",
    max_warmup_episodes=100,
    session=None,
    wup_override=None,
    target_training_steps: int = None,
):
    init_crit_step_c = init_crit_step_c or crit_step_c
    wup_suffix = f"-{wup_suffix}" if wup_suffix is not None else ""

    session = session or api.SembasSession(
        [SIM_LOW, SIM_HIGH], plot_samples=plot_samples
    )
    ep = 0

    reward_log = []
    step_log = []
    if include_warmup and wup_override is None:
        complete = False
        while not complete:
            logger.info("Warming up agent")
            complete = warmup(
                sim, target_step_c=init_crit_step_c, max_episodes=max_warmup_episodes
            )
    elif wup_override is not None:
        sim.agent.load(wup_override)

    if save_warmup:
        sim.agent.save(f".models/warmup/{wup_subdir}", f"warmup{wup_suffix}")

    training_batch = []
    requests = []

    process = "NewSearch"
    i = 0
    num_new_searches = 0
    t_c = 0
    nt_c = 0
    num_total_steps = 0
    try:
        while (num_iterations is None or i < num_iterations) and not (
            target_training_steps is not None
            and num_total_steps >= target_training_steps
        ):
            match process:
                case "NewSearch":
                    logger.info("Starting new search")
                    num_new_searches += 1
                    run_until_phase(
                        session, sim, crit_step_c, api.SembasSession.PHASE_BOUNDARY_EXPL
                    )
                    process = "BE"

                case "BE":
                    if session.expect_phase() == api.SembasSession.PHASE_BOUNDARY_EXPL:
                        x, cls, train_data = run_sembas_episode(
                            session, sim, crit_step_c
                        )

                        if cls:
                            t_c += 1
                        else:
                            nt_c += 1

                        requests.append(x)
                        training_batch.append(train_data)
                        if len(training_batch) >= batch_size:
                            process = "Training"
                    else:
                        logger.info(
                            f"Phase change to {session.prev_known_phase()}, assuming boundary complete. Starting new search."
                        )
                        process = "NewSearch"

                case "Training":
                    logger.info("Training")
                    ep_rewards, ep_steps = rerun_and_train(
                        sim,
                        requests,
                        ep,
                        limit_steps=(
                            None
                            if target_training_steps is None
                            else target_training_steps - num_total_steps
                        ),
                    )
                    reward_log.extend(ep_rewards)
                    step_log.extend(ep_steps)

                    # session._ax.clear()

                    training_batch = []
                    requests = []
                    ep += batch_size

                    sembas_reacquisition(session, sim, crit_step_c)

                    process = "BE"
                    i += 1
                    num_total_steps += len(ep_rewards)

    except KeyboardInterrupt:
        logger.warning("Ending training early")
    finally:
        sim.agent.save()

    logger.debug(f"Target and non-target counts: {t_c}, {nt_c}")
    logger.debug(f"Number of new searches: {num_new_searches}")

    return reward_log, step_log


def perf_test(sim: Simulation, num_episodes: int, max_steps=MAX_STEPS):
    reward_log = []
    step_log = []

    for i in range(num_episodes):
        logger.info(f"Episode {i}")
        logger.info(f"Finding valid start...")
        has_valid = False
        while not has_valid:
            sim.sim_random_reset()
            sim.update_sim_status()
            has_valid = sim.get_sim_status()[1]

        logger.info("Starting episode")
        running_reward = 0
        done = False
        steps = 0

        # run episode
        while not done and steps < max_steps:
            # Get action + step simulation
            state, action, reward, next_state = sim.sim_step()

            # Check status
            sim.update_sim_status()
            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion

            running_reward += reward.item()
            steps += 1

        reward_log.append(running_reward)
        step_log.append(steps)

    return reward_log, step_log


def watch(sim: Simulation, max_episodes: int = None):
    """
    Trains the agent with an initial amount of random experience to
    establish an initial region of competence.
    """
    i = 0
    sim.agent.training = False

    while max_episodes is None or i < max_episodes:
        # longitude: float, latitude: float, dir_angle_offset: float, speed: float
        has_valid = False
        print("Getting random state")
        while not has_valid:
            sim.sim_random_reset()
            sim.update_sim_status()
            has_valid = sim.get_sim_status()[1]
        print("running")
        done = False
        steps = 0

        graphics.render_simulation(sim=sim)
        graphics.show()
        # run episode
        while not done and steps < MAX_STEPS:
            state, action, reward, next_state = sim.sim_step()

            sim.update_sim_status()
            _, in_lane, in_motion = sim.get_sim_status()
            # done = not in_lane or not in_motion

            plt.pause(0.01)
            print(in_lane, in_motion)
            input()
            steps += 1
        i += 1


import traceback
import json


def main_trad():
    try:
        rewards, steps = traditional_training(
            ".models/warmup/test/agent_warmup-0.pt", 20, 5, with_grouping=False
        )

        print("Showing test results")
        rlog, slog = perf_test(sim, 50)

        with open("trad-meta.json", "w") as f:
            json.dump(
                {
                    "train-rewards": rewards,
                    "train-steps": steps,
                    "test-rewards": rlog,
                    "test-steps": slog,
                },
                f,
            )

        data = np.array(slog)
        print("Step count stats:", data.mean(), data.min(), data.max())

        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, len(slog)), slog)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Num Steps")

        plt.show()

    except KeyboardInterrupt:
        print("ending early.")
        sim.agent.save(".models/test", "broken")
    except Exception as e:
        print("Failure occurred!")
        print(traceback.print_exc())
        print(e)


def main_sembas():
    init_crit_step_c = 75

    # init_crit_step_c = 30
    # session = sembas_training(100, 500, init_crit_step_c)
    try:
        # wup_path = '.models/warmup/warmup.model/agent_latest.pt'
        wup_path = None
        rewards, steps = sembas_training(
            5, init_crit_step_c, 20, plot_samples=False, wup_override=wup_path
        )
        print(rewards)
        print(steps)

        print("Showing test results")
        rlog, slog = perf_test(sim, 10)

        with open("meta.json", "w") as f:
            json.dump(
                {
                    "train-rewards": rewards,
                    "train-steps": steps,
                    "test-rewards": rlog,
                    "test-steps": slog,
                },
                f,
            )

        data = np.array(slog)
        print("Step count stats:", data.mean(), data.min(), data.max())

        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, len(slog)), slog)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Num Steps")

        plt.show()

    except KeyboardInterrupt:
        print("ending early.")
    except Exception as e:
        print("Failure occurred!")
        print(traceback.print_exc())
        print(e)


if __name__ == "__main__":
    # step_count_test(sim, 5)
    # main_trad()
    # main_sembas()
    # sim.agent.load(".models/warmup/warmup.model/agent_latest.pt")
    # rlog, slog = perf_test(sim, 50)
    # data = np.array(slog)
    # print("Step count stats:", data.mean(), data.min(), data.max())

    # sim.agent.load("checkpoints/agent_latest.pt")
    # watch(sim)
    # 201 29 348

    input("Press enter to continue")
