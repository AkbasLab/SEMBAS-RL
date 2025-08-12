from typing import Literal
import sembas_training as st
import sembas_api as api
import json
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("trainer")

WARMUP_STEP_TARGET = 75
STEP_CRITERIA = 20


def create_wup_list(wup_type: str, num_runs: int):
    return [f".models/warmup/{wup_type}/agent_warmup-{i}.pt" for i in range(num_runs)]


def create_result_dict(
    test_ep_log: list[st.EpisodeData], train_ep_log: list[st.EpisodeData]
):
    test_step_counts = st.EpisodeData.get_step_history(test_ep_log)
    test_dist = st.EpisodeData.get_distance_history(test_ep_log)
    test_rewards = st.EpisodeData.get_reward_history(test_ep_log)
    train_dist = st.EpisodeData.get_distance_history(train_ep_log)
    train_step_counts = st.EpisodeData.get_step_history(train_ep_log)
    train_rewards = st.EpisodeData.get_reward_history(train_ep_log)

    train_slog = [int(x) for x in train_step_counts]
    train_rlog = [float(x) for x in train_rewards]
    train_dlog = [float(x) for x in train_dist]
    test_slog = [int(x) for x in test_step_counts]
    test_rlog = [float(x) for x in test_rewards]
    test_dlog = [float(x) for x in test_dist]

    return {
        "test-step-mean": float(test_step_counts.mean()),
        "test-step-min": float(test_step_counts.min()),
        "test-step-max": float(test_step_counts.max()),
        "test-dist-mean": float(test_dist.mean()),
        "test-dist-min": float(test_dist.min()),
        "test-dist-max": float(test_dist.max()),
        "train-rewards": train_rlog,
        "train-steps": train_slog,
        "train-dist": train_dlog,
        "test-rewards": test_rlog,
        "test-steps": test_slog,
        "test_dist": test_dlog,
    }


def save_run(rdict: dict, title, i=None):
    suffix = f"-{i}" if i is not None else ""
    filepath = Path(f".results/{title}/run{suffix}.json")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(rdict, f)


def sembas_training(
    session: api.SembasSession,
    sim: st.Simulation,
    step_criteria: int,
    num_training_samples: int,
    batch_step_size: int,
    random_size: int = None,
    save_path=None,
    fixed_sembas_noise=0.3,
    fixed_random_noise=None,
):
    try:
        return st.train_sembas(
            session,
            sim,
            step_criteria,
            num_training_samples,
            batch_step_size,
            random_size,
            fixed_sembas_noise=fixed_sembas_noise,
            fixed_random_noise=fixed_random_noise,
        )
    except KeyboardInterrupt:
        print("Ending early")
    finally:
        if save_path:
            sim.agent.save(save_path)


def random_training(
    sim: st.Simulation,
    num_training_samples: int,
    fixed_noise: float = None,
    s_random_size: int = None,
    s_batch_size: int = None,
):
    """
    Arguments:
    sim (Simulation): The simulation to run.
    num_training_samples (int): The number of SEMBAS-equivalent steps to run.
    s_random_size (int): SEMBAS' equivalent number of random samples to run.
        This allows for an accurate number of training steps to occur in
        order to compare SEMBAS with random training. Requires batch size!
    s_batch_size (int): The batch size used by SEMBAS. Used in conjunction
        with @s_random_size to calculate the total samples executed by
        SEMBAS.
    """
    assert s_random_size is None or s_random_size and s_batch_size

    if s_random_size:
        # Between each batch, a fixed length random training occurs
        # This means, even if num isn't evenly divisible by batch size,
        # another batch of random training occurs. This is why we take
        # the ceiling of the num/batch.
        num_batches = np.ceil(num_training_samples / s_batch_size)
        random_samples = num_batches * s_random_size
        num_training_samples += random_samples

    try:
        return st.train_standard(
            sim,
            num_training_samples,
            noise_step_start=None if fixed_noise else 0,
            fixed_noise=fixed_noise,
        )
    except KeyboardInterrupt:
        print("Ending eearly")
    finally:
        sim.agent.save(".checkpoints/latest_agent-random.pt")


def perf_test(sim: st.Simulation, num_episodes: int, max_steps=1000):
    ep_log = []
    for i in range(num_episodes):
        logger.info(f"Episode {i}")
        logger.info(f"Finding valid start...")
        has_valid = False
        while not has_valid:
            x = st.map_norm(
                (st.SIM_LOW, st.SIM_HIGH), np.random.random(len(st.SIM_LOW))
            )
            # sim.sim_reset(*x)
            st.reset(sim, x)
            sim.update_sim_status()
            has_valid = sim.get_sim_status()[1]

        logger.info("Starting episode")
        ep_log.append(st.run_episode(x, sim, train=False, step_limit=max_steps))

    return ep_log


def sembas_warmup(
    title: str,
    sim: st.Simulation,
    index: int,
    target_distance=None,
    target_steps=None,
    max_steps=2500,
):
    print("Warmup")
    has_valid = False
    while not has_valid:
        has_valid = st.warmup(
            sim,
            target_distance=target_distance,
            target_step_count=target_steps,
            step_limit=max_steps,
        )
    sim.agent.save(f".models/warmup/{title}/agent_warmup-{index}.pt")


def create_warmups(title: str, num_models: int, target_distance=120, max_steps=5000):
    sim = st.setup_sim()
    for i in range(num_models):
        sembas_warmup(
            title, sim, i, target_distance=target_distance, max_steps=max_steps
        )


def test_sembas(
    name: str,
    num_runs=20,
    train_size: int = 5000,
    batch_size: int = 500,
    wup_paths: list[str] = None,
    random_size: int = None,
    fixed_sembas_noise=0.3,
    fixed_random_noise=None,
    session: api.SembasSession = None,
    step_criteria=STEP_CRITERIA,
    start_index=None,
):
    session = session or api.SembasSession(
        [st.SIM_LOW, st.SIM_HIGH], plot_samples=False
    )
    sim = st.setup_sim()
    num_runs = len(wup_paths) if wup_paths else num_runs

    for i in range(start_index or 0, num_runs):
        if wup_paths is not None:
            sim.agent.load(wup_paths[i])
        else:
            # sembas_warmup(sim, i)
            print("skipping warmup")
            sim.agent = st.agent_factory()

        train_eps = sembas_training(
            session,
            sim,
            step_criteria,
            train_size,
            batch_size,
            random_size=random_size,  # random_size=150
            fixed_sembas_noise=fixed_sembas_noise,
            fixed_random_noise=fixed_random_noise,
        )
        test_eps = perf_test(sim, num_episodes=50)

        result = create_result_dict(test_eps, train_eps)
        save_run(result, name, i)


def test_random(
    name: str,
    num_runs=20,
    train_size: int = 5000,
    wup_paths: list[str] = None,
    fixed_noise=None,
    s_random_size=None,
):
    sim = st.setup_sim()
    num_runs = len(wup_paths) if wup_paths else num_runs

    for i in range(num_runs):
        if wup_paths is not None:
            sim.agent.load(wup_paths[i])
        else:
            # sembas_warmup(sim, i)
            print("Skipping warmup")
            sim.agent = st.agent_factory()

        train_eps = random_training(
            sim,
            train_size,
            fixed_noise=fixed_noise,
            s_random_size=s_random_size,
        )
        test_eps = perf_test(sim, num_episodes=50)

        result = create_result_dict(test_eps, train_eps)
        save_run(result, name, i)


def review_last(title: str, num_runs: int):
    steps = 0
    dist = 0
    for i in range(num_runs):
        fn = f"run-{i}.json"
        with open(f".results/{title}/{fn}") as f:
            data = json.load(f)
            steps += data["test-step-mean"]
            dist += data["test-dist-mean"]

    print(f"{title}: mean steps: {steps / num_runs}, mean distance: {dist / num_runs}")


# def show_results(title: str, index: int):
#     fig, ax = plt

# sim = st.setup_sim()
# for i in range(20):
#     sim.agent.load(f".models/warmup/test/agent_warmup-{i}.pt")
#     eps = perf_test(sim, 20)

#     distances = np.array([ep.distance_traveled for ep in eps])

#     print("total distance:", sum(distances))
#     print("median distance:", np.median(distances))
#     print("stats distance:", distances.min(), distances.max(), distances.mean())

# create_warmups(20)
# review_last("sembas-test", 20)
# review_last("random-test", 20)


# from timeit import default_timer as timer

# t0 = timer()

# test_sembas(num_runs=10, wup_paths=create_wup_list(20))

# dists = []
# steps = []
# sim = st.setup_sim()
# for wup_path in create_wup_list(20):
#     sim.agent.load(wup_path)

#     eps = st.perf_test(sim, 50)
#     dists.append(st.EpisodeData.get_distance_history(eps).mean())
#     steps.append(st.EpisodeData.get_step_history(eps).mean())


# dists = np.array(dists)
# steps = np.array(steps)
# print("Distances")
# print(dists)
# print("Steps")
# print(steps)
# print(dists.mean())
# print(steps.mean())
###
def full_run(
    sembas_continue_from: tuple[str, float, int] = None,
    random_continue_from: str = None,
):
    from itertools import product

    total = 5000
    wup_types = ["experienced", "simple"]
    random_sizes = [None, 250, 500]
    # wup_types = ["simple"]
    # random_sizes = [None]
    criteria = [30, 50]
    ndim = st.SIM_LOW.shape[0]

    session = api.SembasSession([st.SIM_LOW, st.SIM_HIGH], plot_samples=False)
    for warmup_type, random_size in product(wup_types, random_sizes):
        if sembas_continue_from and (warmup_type, random_size) != sembas_continue_from:
            continue
        else:
            sembas_continue_from = None

        bs = 500

        s_total = total if random_size is None else total // (1 + random_size / bs)

        test_sembas(
            f"sembas-{warmup_type}-r{random_size}-{ndim}d",
            wup_paths=create_wup_list(warmup_type, 20),
            random_size=random_size,
            train_size=s_total,
            fixed_sembas_noise=0.3,
            session=session,
            step_criteria=STEP_CRITERIA,
        )

    for warmup_type in wup_types:
        if random_continue_from and warmup_type != random_continue_from:
            continue
        else:
            random_continue_from = None
        test_random(
            f"random-{warmup_type}-{ndim}d",
            wup_paths=create_wup_list(warmup_type, 20),
            train_size=total,
        )


def full_no_wup_run(
    sembas_continue_from=None,
):
    from itertools import product

    total = 10000
    random_sizes = [None, 250, 500]
    # wup_types = ["simple"]
    # random_sizes = [None]
    ndim = st.SIM_LOW.shape[0]

    session = api.SembasSession([st.SIM_LOW, st.SIM_HIGH], plot_samples=False)
    for random_size in random_sizes:
        if sembas_continue_from and (random_size) != sembas_continue_from:
            continue
        else:
            sembas_continue_from = None

        bs = 500

        s_total = total if random_size is None else total // (1 + random_size / bs)

        test_sembas(
            f"sembas-no_wup-r{random_size}-{ndim}d",
            random_size=random_size,
            train_size=s_total,
            fixed_sembas_noise=0.3,
            session=session,
            step_criteria=STEP_CRITERIA,
        )

    test_random(
        f"random-no_wup-{ndim}d",
        train_size=total,
    )


import os
import matplotlib.pyplot as plt


def plot_run(ax, title: str, index: int):

    with open(f".results/{title}/run-{index}.json") as f:
        data = json.load(f)

    train = data["train-dist"]
    test = data["test_dist"]

    ax.plot(np.arange(len(train)), train, color="cyan", label="train")
    ax.plot(np.arange(len(test)), test, color="green", label="test")
    ax.legend()


def plot_all_runs(
    ax,
    title: str,
    mode: Literal["all", "mean"] = "mean",
    skip_train=False,
    skip_test=False,
):
    assert not (skip_train and skip_test)

    train = []
    test = []

    for fn in os.listdir(f".results/{title}"):
        with open(f".results/{title}/{fn}") as f:
            data = json.load(f)

        train.append(np.array(data["train-dist"]))
        test.append(np.array(data["test_dist"]))

    if mode == "mean":
        # train = np.vstack(train)
        test = np.vstack(test)
        # train = train.mean(axis=0)
        test = test.mean(axis=0)

        # if not skip_train:
        #     ax.plot(np.arange(len(train)), train, color="cyan", label="train")
        if not skip_test:
            ax.plot(np.arange(len(test)), test, color="green", label="test")
    else:
        for i, run_train, run_test in zip(range(len(train)), train, test):
            if not skip_train:
                ax.plot(np.arange(len(run_train)), run_train, label=f"run_{i}-train")
            if not skip_test:
                ax.plot(np.arange(len(run_test)), run_test, label=f"run_{i}-test")

    ax.legend()


def plot_results(ax, title: str):
    ax.set_title(title)

    test = []
    train = []

    for fn in os.listdir(f".results/{title}"):
        with open(f".results/{title}/{fn}") as f:
            data = json.load(f)
            test.append(data["test-dist-mean"])
            train.append(np.array(data["train-dist"]).mean())

    ax.plot(np.arange(len(train)), train, color="cyan", label="train")
    ax.plot(np.arange(len(test)), test, color="green", label="test")
    ax.legend()


"""
"Interestingness" is the delta between target and non-target points.

d score / dx
We divide by the change in parameter (magnitude) since we can't really know
boundary pairs at this point. In the future we need to make that possible but...
"""

# full_no_wup_run()

# NOTE: another test to try is ground-up training. promising early results...
# review_last("sembas-experienced-rNone-4d", 20)
# review_last("random-experienced-4d", 20)

sim = st.setup_sim()
session = api.SembasSession([st.SIM_LOW, st.SIM_HIGH], plot_samples=False)
train_eps = sembas_training(
    session,
    sim,
    STEP_CRITERIA,
    10000,
    500,
    fixed_sembas_noise=0.3,
)

# sim.agent.save()
eps = perf_test(sim, 50)
print(st.EpisodeData.get_distance_history(eps).mean())
print(st.EpisodeData.get_step_history(eps).mean())

# fig, ax = plt.subplots()
# plot_all_runs(ax, "sembas-simple-s50-rNone-4d", mode="mean")
# plt.show()


# fig, (axl, axr) = plt.subplots(1, 2)
# axl.set_ylim(0, 800)
# axr.set_ylim(0, 800)
# plot_results(axl, "sembas-simple-s30-rNone-4d")
# plot_results(axr, "sembas-simple-s50-rNone-4d")
# plot_results(axl, "random-simple-4d")
# plot_results(axr, "random-experienced-4d")
# plt.show()
# for i in range(20):
#     fig, ax = plt.subplots()
#     # plot_run(ax, "random-simple-4d", i)
#     plot_run(ax, "sembas-simple-r250-4d", i)
#     plt.show()


# full_run()

# import os

# bl = [
#     # "sembas-experienced-rNone-4d",
#     # "sembas-simple-r500-4d",
#     "archive",
#     "sembas-experienced-2d",
# ]

# for fn in sorted(os.listdir(".results")):
#     if fn in bl:
#         continue

#     review_last(fn, 20)


# the score to beat: 230
# review_last(f"sembas-{warmup_type}-2d", 19)
# review_last(f"sembas-{warmup_type}-tmp-2d", 20)
# review_last(f"random-{warmup_type}-2d", 19)
# review_last(f"random-{warmup_type}-tmp-2d", 20)

# sim = st.setup_sim()
# center = sim.environment.lane.center_line

# lens = []
# for a, b in zip(center[:-1], center[1:]):
#     s = b - a
#     lens.append(np.linalg.norm(s))

# print(center[0], center[-1], center[-2])


# wup_types = ["experienced", "simple"]
# random_sizes = [None, 250, 500]
# # wup_types = ["simple"]
# # random_sizes = [None]
# criteria = [30, 50]
# ndim = st.SIM_LOW.shape[0]

# warmup_type = wup_types[1]
# random_size = random_sizes[0]
# crit = criteria[0]

# session = api.SembasSession([st.SIM_LOW, st.SIM_HIGH], plot_samples=False)

# total = 5000
# bs = 500

# s_total = total if random_size is None else total // (1 + random_size / bs)

# test_sembas(
#     f"sembas-{warmup_type}-s{crit}-r{random_size}-{ndim}d",
#     wup_paths=create_wup_list(warmup_type, 20),
#     random_size=random_size,
#     train_size=s_total,
#     fixed_sembas_noise=0.3,
#     session=session,
#     step_criteria=crit,
#     # start_index=12,
# )
