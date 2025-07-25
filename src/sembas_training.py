# app.py
from new_agent import NewAgent
from point import Point
from vehicle import Vehicle
from lane import Lane
from environment import Environment
from simulation import Simulation
from sensor_array import SensorArray
import torch
import math
import layout_utils
import carlos_logging
from summer_agent import SummerAgent
import time
import matplotlib.pyplot as plt
import graphics
import numpy as np
import socket
import sembas_api as api


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
# longitude: float,
# latitude: float,
# dir_angle_offset: float,
# speed: float
T_SIM_LOW = torch.tensor([0, 0.25, -np.pi / 5, 20.0])
T_SIM_HIGH = torch.tensor([1, 0.75, np.pi / 5, 75.0])
SIM_LOW = torch.tensor([0, -np.pi / 5])
SIM_HIGH = torch.tensor([1, np.pi / 5])

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
# Number of sensors + 2 for vehicle heading, speed, and wp_heading
obs_size = NUM_SENSORS + 2 + 1

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


def elapsed_time(start_time: float) -> float:
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s"


def map_norm(bounds, x) -> torch.Tensor:
    x = torch.tensor(x)
    return x * (bounds[1] - bounds[0]) + bounds[0]

def train_random(sim: Simulation, num_episodes: int):
    """
    Trains the agent with an initial amount of random experience to
    establish an initial region of competence.
    """
    print("Rerunning and training over boundary")
    num_steps = 0

    agent: NewAgent = sim.agent

    reward_log = []
    step_log = []

    for i in range(num_episodes):
        # longitude: float, latitude: float, dir_angle_offset: float, speed: float
        # sim.sim_random_reset()
        agent.update_expl_noise(i, num_episodes)

        # scale
        is_valid = False
        while not is_valid:
            x = np.random.random(4)
            x = map_norm((T_SIM_LOW, T_SIM_HIGH), x)
            sim.sim_reset(x[0], 0.5, x[2], 45)
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
        if (i + 1) % (num_episodes // 10) == 0:
            print(f"{i} : {steps}")

    return reward_log, step_log

def run_random_group(sim: Simulation, crit_step_c: int, n_episodes: int = 10):
    """
    Returns the training data developed on a given pass of the
    SEMBAS algorithm.
    """
    episodic_train_data = []
    for i in range(n_episodes):
        # sim.sim_random_reset((SIM_LOW[3], SIM_HIGH[3]))
        x = np.random.random(4)
        x = map_norm((SIM_LOW, SIM_HIGH), x)
        sim.sim_reset(x[0], 0.5, x[2], 45)
        sim.update_sim_status()
        invalid_initial_state = not sim.get_sim_status()[1]
        if invalid_initial_state:
            episodic_train_data.append(([], False))
            continue

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
        episodic_train_data.append((episode_data, steps >= crit_step_c))

    return episodic_train_data


def run_episode(x: torch.Tensor, sim: Simulation, crit_step_c: int):
    """
    Returns the training data developed on a given pass of the
    SEMBAS algorithm.
    """
    # longitude: float, latitude: float, dir_angle_offset: float, speed: float
    # x = receive_request(client)
    # sim.sim_reset(*x)
    sim.sim_reset(x[0], 0.5, x[1], 45)
    # scale
    sim.update_sim_status()
    is_valid = sim.get_sim_status()[1]
    if not is_valid:
        return False, []

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

    return steps >= crit_step_c, episode_data


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
    sim.sim_reset(x[0], 0.5, x[1], 45)
    # scale
    sim.update_sim_status()
    valid_init_state = sim.get_sim_status()[1]
    if not valid_init_state:
        session.send_response(False)
        return x, False, []

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
    i = 0
    num_steps = 0
    if apply_expl_noise:
        agent.update_expl_noise(0, max_episodes or 100)
    else:
        agent.exploration_noise = 0.0

    step_history = []

    print(target_step_c)

    # rng = lambda: map_norm((SIM_LOW, SIM_HIGH), torch.rand(len(SIM_LOW)))
    rng = lambda: torch.tensor(
        (np.random.rand(), 0.5, np.random.rand() * np.pi * 2 / 5 - np.pi / 5, 45.0)
    )
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

    return (
        torch.vstack(target_states) if len(target_states) > 0 else None,
        torch.vstack(nontarget_states) if len(nontarget_states) > 0 else None,
        torch.vstack(invalid_states) if len(invalid_states) > 0 else None,
    )


def train_batch(agent: NewAgent, episodic_train_data):
    print("Training batch")
    reward_log = []
    step_log = []
    for ep_data in episodic_train_data:
        running_reward = 0
        for x in ep_data:
            running_reward += x[2]
            agent.train_step(*x)

        reward_log.append(running_reward)
        step_log.append(len(ep_data))

    return reward_log, step_log


def rerun_and_train(sim: Simulation, requests: list[torch.Tensor], ep: int):
    """
    Trains the agent with an initial amount of random experience to
    establish an initial region of competence.
    """
    print("Rerunning and training over boundary")
    num_steps = 0

    agent: NewAgent = sim.agent

    reward_log = []
    step_log = []

    for i, x in enumerate(requests):
        # longitude: float, latitude: float, dir_angle_offset: float, speed: float
        # sim.sim_random_reset()
        agent.update_expl_noise(ep + i, len(requests))

        # scale
        sim.sim_reset(x[0], 0.5, x[1], 45)
        # sim.sim_reset(*x)

        sim.update_sim_status()
        is_valid = sim.get_sim_status()[1]
        if not is_valid:
            continue

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
    print(f"Running until {target_phase}, starting {session.prev_known_phase}")
    session.expect_phase()
    while session.prev_known_phase != target_phase:
        if session.prev_known_phase not in new_train_data:
            print(f"Starting phase {session.prev_known_phase}")
            new_train_data[session.prev_known_phase] = []

        # longitude: float, latitude: float, dir_angle_offset: float, speed: float
        # x = receive_request(client)
        x = session.receive_request()
        sim.sim_reset(x[0], 0.5, x[1], 45)
        # sim.sim_reset(*x)
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

    return new_train_data


# def get_even_split()
def traditional_training(
    group_size: int, num_groups: int, with_grouping=True
):
    # re-use the warmed up model used by SEMBAS
    sim.agent.load(".models/warmup/warmup.model/agent_latest.pt")
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


def sembas_reacquisition(
    session: api.SembasSession, sim: Simulation, crit_step_c: int
) -> dict[str, list[tuple[tuple, bool]]]:
    """
    Signals to SEMBAS that boundary reacquisition is necessary and samples until
    SEMBAS has reacquired the boundary and is exploring once more.
    """

    print("TEST")
    session.send_message(MSG_REACQ)
    print("TEST")
    new_train_data = run_until_phase(
        session, sim, crit_step_c, api.SembasSession.PHASE_BOUNDARY_EXPL
    )

    return new_train_data


def sembas_training(
    batch_size: int, crit_step_c: int, num_iterations: int = None, plot_samples=False, init_crit_step_c=None
):
    init_crit_step_c = init_crit_step_c or crit_step_c
    print("Setting up connection...")
    # client = api.setup_socket(4)
    session = api.SembasSession([SIM_LOW, SIM_HIGH], plot_samples=plot_samples)
    total_episodes = batch_size * num_iterations
    ep = 0

    reward_log = []
    step_log = []
    # get enough data to fill memory and begin training ~ episodes
    print("Warmup...")
    warmup(sim, target_step_c=init_crit_step_c)
    sim.agent.save(".models/warmup/warmup.model")

    # plt.pause(0.01)

    # Get through the phases
    # while session.phase != api.SembasSession.PHASE_BOUNDARY_EXPL:

    training_batch = []
    requests = []

    process = "NewSearch"
    input("Press enter to continue")
    i = 0
    try:
        while num_iterations is None or i < num_iterations:
            match process:
                case "NewSearch":
                    print("New search...")
                    run_until_phase(
                        session, sim, crit_step_c, api.SembasSession.PHASE_BOUNDARY_EXPL
                    )
                    process = "BE"

                case "BE":
                    if session.expect_phase() == api.SembasSession.PHASE_BOUNDARY_EXPL:
                        x, cls, train_data = run_sembas_episode(
                            session, sim, crit_step_c
                        )
                        requests.append(x)
                        training_batch.append(train_data)
                        if len(training_batch) > batch_size:
                            process = "Training"
                    else:
                        print(
                            f"Phase change to {session.expect_phase()}, assuming boundary complete. Starting new search."
                        )
                        process = "NewSearch"

                case "Training":
                    print("Training")
                    ep_rewards, ep_steps = rerun_and_train(sim, requests, ep)
                    reward_log.extend(ep_rewards)
                    step_log.extend(ep_steps)

                    # session._ax.clear()

                    training_batch = []
                    requests = []
                    ep += batch_size

                    sembas_reacquisition(session, sim, crit_step_c)
                    process = "BE"
                    i += 1

    except KeyboardInterrupt:
        print("Ending training early")
    finally:
        sim.agent.save()

    return reward_log, step_log


def test(crit_step_c: int, group_size: int = 10, num_samples=None):
    print("Setting up connection...")
    # client = api.setup_socket(4)
    session = api.SembasSession([SIM_LOW, SIM_HIGH], plot_samples=True)
    reward_log = []
    step_log = []
    # get enough data to fill memory and begin training ~ episodes
    print("Warmup...")
    warmup(sim, target_step_c=crit_step_c)
    sim.agent.save(".models/warmup/warmup.model")

    # Get through the phases
    # while session.phase != api.SembasSession.PHASE_BOUNDARY_EXPL:
    print("Running until boundary exploration...")
    run_until_phase(session, sim, crit_step_c, api.SembasSession.PHASE_BOUNDARY_EXPL)

    # fig, ax = plt.subplots()
    # ax.set_title("Samples")
    # ax.set_xlim(SIM_LOW[0], SIM_HIGH[0])
    # ax.set_ylim(SIM_LOW[1], SIM_HIGH[1])
    # ax.set_xlabel("Longitude")
    # ax.set_ylabel("Angle")

    print("Beginning boundary training")
    i = 0
    try:
        while num_samples is None or i < num_samples:
            targets = []
            nontargets = []
            for i in range(group_size):
                x, cls, _ = run_sembas_episode(session, sim, crit_step_c)
                if cls:
                    targets.append(x)
                else:
                    nontargets.append(x)

            # if len(targets) > 0:
            #     ax.scatter(*np.array(targets).T, color="red")
            # if len(nontargets) > 0:
            #     ax.scatter(*np.array(nontargets).T, color="blue")
            # plt.pause(0.01)

    except KeyboardInterrupt:
        print("Backing out")

    return reward_log, step_log


def perf_test(sim: Simulation, num_episodes: int, max_steps=MAX_STEPS):
    reward_log = []
    step_log = []

    for i in range(num_episodes):
        print(f"Ep {i}")
        has_valid = False
        while not has_valid:
            sim.sim_random_reset()
            sim.update_sim_status()
            has_valid = sim.get_sim_status()[1]

        print("starting ep")
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

            reward += reward
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
        # has_valid = False
        # while not has_valid:
        #     sim.sim_random_reset()
        #     sim.update_sim_status()
        #     has_valid = sim.get_sim_status()[1]

        done = False
        steps = 0

        # run episode
        while not done and steps < MAX_STEPS:
            state, action, reward, next_state = sim.sim_step()

            sim.update_sim_status()
            _, in_lane, in_motion = sim.get_sim_status()
            done = not in_lane or not in_motion

            graphics.render_simulation(sim=sim)
            graphics.show()
            input()
            steps += 1
        i += 1


import traceback
import json

def main_trad():
    try:
        rewards, steps = traditional_training(15, 10, with_grouping=False)

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
                f
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


def main_sembas():
    init_crit_step_c = 75
    
    # init_crit_step_c = 30
    # session = sembas_training(100, 500, init_crit_step_c)
    try:
        rewards, steps = sembas_training(15, init_crit_step_c, 10, plot_samples=False)
        print(rewards)
        print(steps)

        print("Showing test results")
        rlog, slog = perf_test(sim, 50)

        with open("meta.json", "w") as f:
            json.dump(
                {
                    "train-rewards": rewards, 
                    "train-steps": steps,
                    "test-rewards": rlog,
                    "test-steps": slog,
                },
                f
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


def review():
    sim.agent.load("./checkpoints/agent_latest.pt")

    rlog, slog = perf_test(sim, 50)

    with open("meta.json", "w") as f:
        json.dump(
            {
                "test-rewards": rlog,
                "test-steps": slog,
            },
            f
        )

    data = np.array(slog)
    print("Step count stats:", data.mean(), data.min(), data.max())

if __name__ == "__main__":
    # main_trad()
    main_sembas()
    # sim.agent.load(".models/warmup/warmup.model/agent_latest.pt")
    # rlog, slog = perf_test(sim, 50)
    # data = np.array(slog)
    # print("Step count stats:", data.mean(), data.min(), data.max())

    input("Press enter to continue")
