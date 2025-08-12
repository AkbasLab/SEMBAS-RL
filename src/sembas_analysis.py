"""
wanting to look at the "interesting-ness" metric
how we will do this is by the following equation
|ds / dx|, where s and x are the score (steps taken) and the parameters respectively

"""

import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s:%(levelname)-8s:%(name)-15s: %(message)s"
)

logger = logging.getLogger("analysis")
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
import sembas_training as st


SIM_LOW = np.array([0, 0.45, -0.01, 20.0])
SIM_HIGH = np.array([0.75, 0.55, 0.01, 75.0])


def calculate_interestingness(b_ep: st.EpisodeData, p_ep: st.EpisodeData) -> float:
    x = p_ep.parameters - b_ep.parameters
    s = b_ep.num_steps - p_ep.num_steps

    return s / np.linalg.norm(x) ** 2


def find_boundaries(
    session: api.SembasSession,
    sim: Simulation,
    step_criteria: int,
    boundary_count: int,
    batch_size: int = None,
    max_global_search=1000,
) -> list[tuple[st.EpisodeData, st.EpisodeData]]:
    all_episodes = []
    b_pairs = []
    batch_episodes = []
    process = "NewSearch"
    ep_i = 0

    while len(b_pairs) < boundary_count:
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
                ep_i = 0
                prev_cls = None
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
                elif batch_size is not None and ep_i >= batch_size:
                    process = "NewSearch"
                    session.send_message("REACQ")
                    continue

                x = session.receive_request()
                ep = st.run_episode(
                    x,
                    sim,
                    step_criteria,
                    train=False,
                )
                ep_i += 1
                session.send_response(ep.cls)
                if (
                    prev_cls is not None
                    and prev_cls != ep.cls
                    and not (
                        all_episodes[-1] if not all_episodes[-1].cls else ep
                    ).is_failure
                ):
                    b_ep, p_ep = (
                        (ep, all_episodes[-1])
                        if ep.cls
                        else (
                            all_episodes[-1],
                            ep,
                        )
                    )
                    b_ep.interestingness = calculate_interestingness(b_ep, p_ep)
                    b_pairs.append((b_ep, p_ep))
                    prev_cls = None

                all_episodes.append(ep)
                prev_cls = ep.cls
                if not ep.is_failure:
                    batch_episodes.append(ep)

    return b_pairs


def test_interestingnss():
    # Let's compare the most-interesting to the least interesting
    sim = st.setup_sim()
    sim.agent.load(".checkpoints/agent_latest.pt")
    session = api.SembasSession([SIM_LOW, SIM_HIGH], plot_samples=False)
    step_criteria = 70
    pairs = find_boundaries(session, sim, step_criteria, 50, 20)
    # remove failure cases (domain OOB)
    pairs = [pair for pair in pairs if not pair[1].is_failure]

    pairs: list[st.EpisodeData] = sorted(
        pairs, key=lambda pair: pair[0].interestingness, reverse=True
    )

    top_five = pairs[:5]
    bottom_five = pairs[-5:]

    repeat = True
    while repeat:
        print("Top 5")
        try:
            for b_ep, p_ep in top_five:
                print(
                    f"Target Steps: {b_ep.num_steps}, Interestingness: {b_ep.interestingness}"
                )
                st.run_episode(
                    b_ep.parameters,
                    sim,
                    step_criteria,
                    train=False,
                    display_mode="step",
                    step_limit=60,
                )
                print(f"Non-Target Steps: {p_ep.num_steps}")
                st.run_episode(
                    p_ep.parameters,
                    sim,
                    step_criteria,
                    train=False,
                    display_mode="step",
                    step_limit=60,
                )
        except KeyboardInterrupt:
            print("Ending early")

        print("Bottom 5")
        try:
            for b_ep, p_ep in bottom_five:
                print(
                    f"Target Steps: {b_ep.num_steps}, Interestingness: {b_ep.interestingness}"
                )
                st.run_episode(
                    b_ep.parameters,
                    sim,
                    step_criteria,
                    train=False,
                    display_mode="step",
                    step_limit=60,
                )
                print(f"Non-Target Steps: {p_ep.num_steps}")
                st.run_episode(
                    p_ep.parameters,
                    sim,
                    step_criteria,
                    train=False,
                    display_mode="step",
                    step_limit=60,
                )
        except KeyboardInterrupt:
            print("Ending early")

        repeat = input("Repeat y/n") != "n"


def perf_endurance_test(num_episodes=50):
    sim = st.setup_sim()
    sim.agent.load(".checkpoints/agent_latest.pt")
    ep_log = []
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
        ep_log.append(st.run_episode(x, sim, train=False, step_limit=10000))

    return ep_log


def find_failure_region():
    fig, ax = plt.subplots()

    sim = st.setup_sim()
    x, y = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    x = x.flatten()
    y = y.flatten()

    classes = []

    for xi, yi in zip(x, y):
        params = np.random.random(4)
        params = st.map_norm((SIM_LOW, SIM_HIGH), params)
        params[0] = xi
        params[2] = yi

        ep = st.run_episode(
            params,
            sim,
        )

        classes.append(ep.is_failure)

    ax.scatter(x, y, color=["red" if cls else "blue" for cls in classes])
    plt.show()


# find_failure_region()

test_interestingnss()
