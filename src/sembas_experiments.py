import sembas_training as st
import sembas_api as api
import json
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("trainer")

WARMUP_STEP_TARGET = 75
STEP_CRITERIA = 20


def create_wup_list(num_runs: int):
    return [f".models/warmup/test/agent_warmup-{i}.pt" for i in range(num_runs)]


def create_result_dict(
    test_ep_log: list[st.EpisodeData], train_ep_log: list[st.EpisodeData]
):
    test_step_counts = st.EpisodeData.get_step_history(test_ep_log)
    test_rewards = st.EpisodeData.get_reward_history(test_ep_log)
    train_step_counts = st.EpisodeData.get_step_history(train_ep_log)
    train_rewards = st.EpisodeData.get_reward_history(train_ep_log)

    train_slog = [int(x) for x in train_step_counts]
    train_rlog = [float(x) for x in train_rewards]
    test_slog = [int(x) for x in test_step_counts]
    test_rlog = [float(x) for x in test_rewards]

    return {
        "test-mean": float(test_step_counts.mean()),
        "test-min": float(test_step_counts.min()),
        "test-max": float(test_step_counts.max()),
        "train-rewards": train_rlog,
        "train-steps": train_slog,
        "test-rewards": test_rlog,
        "test-steps": test_slog,
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
):
    try:
        return st.train_sembas(
            session,
            sim,
            step_criteria,
            num_training_samples,
            batch_step_size,
            random_size,
        )
    except KeyboardInterrupt:
        print("Ending early")
    finally:
        if save_path:
            sim.agent.save(save_path)


def random_training(
    sim: st.Simulation,
    num_training_samples: int,
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
    sim: st.Simulation,
    index: int,
    target_distance=None,
    target_steps=WARMUP_STEP_TARGET,
    max_steps=2500,
):
    print("Warmup")
    has_valid = False
    while not has_valid:
        has_valid = st.warmup(
            sim, target_distance=target_distance, step_limit=max_steps
        )
    sim.agent.save(f".models/warmup/test/agent_warmup-{index}.pt")


def create_warmups(num_models: int, target_distance=120, max_steps=5000):
    sim = st.setup_sim()
    for i in range(num_models):
        sembas_warmup(sim, i, target_distance=target_distance, max_steps=max_steps)


def test_sembas(num_runs=20, train_size: int = 5000, wup_paths: list[str] = None):
    session = api.SembasSession([st.SIM_LOW, st.SIM_HIGH], plot_samples=False)
    sim = st.setup_sim()

    for i in range(num_runs):
        if wup_paths is not None:
            sim.agent.load(wup_paths[i])
        else:
            sembas_warmup(sim, i)

        # sim.agent.set_lr(critic_lr=1e-3, actor_lr=1e-4)
        train_eps = sembas_training(
            session,
            sim,
            STEP_CRITERIA,
            train_size,
            500,  # random_size=150
        )
        # input("press enter")
        test_eps = perf_test(sim, num_episodes=50)

        result = create_result_dict(test_eps, train_eps)
        save_run(result, f"sembas-2", i)


def test_random(num_runs=20, train_size: int = 5000, wup_paths: list[str] = None):
    sim = st.setup_sim()

    for i in range(num_runs):
        if wup_paths is not None:
            sim.agent.load(wup_paths[i])
        else:
            sembas_warmup(sim, i)

        sim.agent.set_lr(critic_lr=1e-3, actor_lr=1e-4)
        train_eps = random_training(
            sim,
            train_size,
        )
        # input("Press enter")
        # test_eps = perf_test(sim, num_episodes=20)

        # result = create_result_dict(test_eps, train_eps)
        # save_run(result, f"random-test", i)


def review_last(title: str, num_runs: int):
    sr = 0
    for i in range(num_runs):
        fn = f"run-{i}.json"
        with open(f".results/{title}/{fn}") as f:
            sr += json.load(f)["test-mean"]

    print(f"{title}: {sr / num_runs}")


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
review_last("sembas-2", 10)
review_last("random-test", 10)

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

# test_random(wup_paths=create_wup_list(20))
# print(f"Took {timer() - t0}s")

# the score to beat: 230
