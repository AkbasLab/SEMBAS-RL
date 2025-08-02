import numpy as np
from numpy import ndarray
import sembas_api as api
from simulation import Simulation
import logging
import torch


def map_norm(bounds, x) -> ndarray:
    x = np.array(x)
    return x * (bounds[1] - bounds[0]) + bounds[0]


def run_until_phase(
    session: api.SembasSession,
    sim: Simulation,
    crit_step_c: int,
    target_phase: str,
    logger: logging.Logger,
    max_steps=1000,
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
        while not done and steps < max_steps:
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
