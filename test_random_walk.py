import time
import random
import torch
from PIL import Image

from vmas import make_env

from scenarios.ice_fishing import Scenario as IceFishingScenario

if __name__ == "__main__":
    device = "cpu"
    continuous_actions = True
    num_envs = 5
    scenario_name = "ice_fishing"

    env = make_env(
        scenario=IceFishingScenario(),
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        dict_spaces=True,
        wrapper=None,
        seed=None,
        # Environment specific variables
        n_agents=5,
    )

    n_steps = 100  # Number of steps before returning done

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    for s in range(n_steps):
        step += 1

        # VMAS actions can be either a list of tensors (one per agent)
        # or a dict of tensors (one entry per agent with its name as key)
        # Both action inputs can be used independently of what type of space its chosen
        dict_actions = random.choice([True, False])

        actions = {} if dict_actions else []
        for i, agent in enumerate(env.agents):

            # move with probability 0.2
            if random.random() < 0.1:
                simple_2d_action = [random.uniform(-1, 1), random.uniform(-1, 1)]
            else:
                simple_2d_action = [0, 0]

            action = torch.tensor(
                simple_2d_action,
                device=device,
            ).repeat(num_envs, 1)
            if dict_actions:
                actions.update({agent.name: action})
            else:
                actions.append(action)

        obs, rews, dones, info = env.step(actions)

        print(f"Step {step} rewards (env 0) {[int(r[0]) for a, r in rews.items()]}")

        frame_list.append(
            Image.fromarray(env.render(mode="rgb_array", agent_index_focus=None))
        )  # Can give the camera an agent index to focus on

    gif_name = scenario_name + ".gif"

    # Produce a gif
    frame_list[0].save(
        gif_name,
        save_all=True,
        append_images=frame_list[1:],
        duration=3,
        loop=0,
    )


    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )
