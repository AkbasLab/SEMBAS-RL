import sembas_training as st
import sembas_api as api
import json
import numpy as np
import logging


def create_wup_list(title: str, num_runs: int):
    return [f".models/warmup/{title}/agent_warmup-{i}.pt" for i in range(num_runs)]


def create_result_dict(train_slog, train_rlog, rlog, slog):
    data = np.array(slog)

    train_slog = [int(x) for x in train_slog]
    train_rlog = [float(x) for x in train_rlog]
    slog = [int(x) for x in slog]
    rlog = [float(x) for x in rlog]

    return {
        "test-mean": float(data.mean()),
        "test-min": float(data.min()),
        "test-max": float(data.max()),
        "train-rewards": train_rlog,
        "train-steps": train_slog,
        "test-rewards": rlog,
        "test-steps": slog,
    }


def save_run(rdict: dict, title, i=None):
    suffix = f"-{i}" if i is not None else ""
    with open(f".results/{title}/run{suffix}.json", "w") as f:
        json.dump(rdict, f)


def sembas(
    title: str,
    index: int,
    session,
    init_crit_step_c=75,
    crit_step_c=50,
    s_batch_size=15,
    # s_num_batches=10,
    s_num_batches=None,
    num_rand_tests=10,
    num_steps=1500,
):
    # hacking a bypass of warmup to re-use old models
    wup_path = create_wup_list(title, 20)[index]
    rewards, steps = st.sembas_training(
        s_batch_size,
        crit_step_c,
        s_num_batches,
        plot_samples=False,
        init_crit_step_c=init_crit_step_c,
        wup_suffix=index,
        save_warmup=False,
        wup_subdir=title,
        session=session,
        wup_override=wup_path,
        target_training_steps=num_steps,
    )
    # rewards, steps = st.sembas_training(s_batch_size, crit_step_c, s_num_batches, plot_samples=False, init_crit_step_c=init_crit_step_c, wup_suffix=index, save_warmup=save_warmup, wup_subdir=title, session=session)

    rlog, slog = st.perf_test(st.sim, num_rand_tests)

    return create_result_dict(steps, rewards, rlog, slog)


def random(wup_path, s_batch_size=15, s_num_batches=10, num_rand_tests=10):
    rewards, steps = st.traditional_training(
        wup_path,
        s_batch_size,
        s_num_batches,
    )

    rlog, slog = st.perf_test(st.sim, num_rand_tests)

    return create_result_dict(steps, rewards, rlog, slog)


def random_by_steps(wup_path, num_steps: int = 1500, num_rand_tests=10):
    rewards, steps = st.traditional_training(wup_path, num_steps=num_steps)

    rlog, slog = st.perf_test(st.sim, num_rand_tests)

    return create_result_dict(steps, rewards, rlog, slog)


def runner(foo, title: str, target_agg_key: str, num_runs=20, **kwargs):
    results = []
    for i in range(num_runs):
        rdict = foo(i, **kwargs)
        save_run(rdict, title, i)

        results.append(rdict[target_agg_key])

    return np.array(results)


def create_sembas_exp(title: str, session, **default_kwargs):
    def exp(i, **kwargs):
        return sembas(title, i, session, **default_kwargs, **kwargs)

    return exp


def create_trad_exp(wup_schedule: list[str], **default_kwargs):
    def exp(i, **kwargs):
        return random(wup_schedule[i], **default_kwargs, **kwargs)

    return exp


def main_run_both(num_runs=20, batch_size=15, num_batches=10):
    try:
        session = api.SembasSession([st.SIM_LOW, st.SIM_HIGH], plot_samples=False)
        exp = create_sembas_exp(
            "test", session, s_batch_size=batch_size, s_num_batches=num_batches
        )

        sembas_result = runner(exp, "sembas-test", "test-mean", num_runs=num_runs)
        wup_schedule = create_wup_list("test", num_runs)
        exp = create_trad_exp(
            wup_schedule, s_batch_size=batch_size, s_num_batches=num_batches
        )
        result = runner(exp, "trdtest", "test-mean", num_runs=num_runs)

        print("Sembas:", sembas_result.mean(), sembas_result.min(), sembas_result.max())
        print("Traditional:", result.mean(), result.min(), result.max())
    except KeyboardInterrupt:
        print("ending early")
        st.sim.agent.save(".test", "broken")


def main_by_steps(num_runs=20, num_steps=1500):
    try:
        session = api.SembasSession([st.SIM_LOW, st.SIM_HIGH], plot_samples=False)
        exp = create_sembas_exp("test", session, num_steps=num_steps)
        sembas_result = runner(exp, "sembas-test", "test-mean", num_runs=num_runs)

        wup_schedule = create_wup_list("test", num_runs)
        exp = create_trad_exp(wup_schedule, num_steps=num_steps)
        result = runner(exp, "trdtest", "test-mean", num_runs=num_runs)

        print("Sembas:", sembas_result.mean(), sembas_result.min(), sembas_result.max())
        print("Traditional:", result.mean(), result.min(), result.max())
    except KeyboardInterrupt:
        print("ending early")
        st.sim.agent.save(".test", "broken")


if __name__ == "__main__":
    main_run_both(20, 12, 5)
