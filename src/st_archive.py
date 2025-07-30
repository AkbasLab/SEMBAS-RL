




def step_count_test(sim: Simulation, median_step_count_change: int, window_size: int = 10):
    """
    Trains the agent with an initial amount of random experience to
    establish an initial region of competence.
    """
    num_steps = 0

    agent: NewAgent = sim.agent

    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Step count")

    g_steps = []
    g_steps_med = []
    g_dif = []
    g_dif_med = []

    reward_log = []
    step_log = []
    i = 0
    while True:
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
        
        window = np.array(step_log[-window_size:])

        if len(step_log) >= window_size:
            dif = (np.array(step_log[1-window_size:]) - np.array(step_log[-window_size:-1])) 
        else:
            dif = np.array([0])

        reward_log.append(running_reward.item())
        step_log.append(steps)

        g_steps.append(steps)
        g_steps_med.append(np.median(window))
        if len(step_log) > 1:
            g_dif.append(step_log[-1] - step_log[-2])
        else:
            g_dif.append(0)

        g_dif_med.append(np.median(dif))

        
        ax.clear()
        ax.scatter(np.arange(i+1), g_steps, color="red", label="steps", marker=".")
        ax.plot(np.arange(i+1), g_steps_med, color="orange", label="median steps")
        ax.scatter(np.arange(i+1), g_dif, color="blue", label="dif", marker=".")
        ax.plot(np.arange(i+1), g_dif_med, color="cyan", label="median dif")
        ax.legend()

        plt.pause(0.01)

        i += 1

    return reward_log, step_log