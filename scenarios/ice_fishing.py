#  Ice fishing scenario similar to the VMAS sampling scenario
# https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/sampling.py

from typing import Dict, Callable

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from vmas import render_interactively
from vmas.simulator.core import World, Line, Agent, Sphere, Entity
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, X, Y

from sensors.sensors import AgentDetector


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.get("n_agents", 3)
        self.shared_rew = kwargs.get("shared_rew", True)

        self.comms_range = kwargs.get("comms_range", 0.0)
        self.lidar_range = kwargs.get("lidar_range", 0.5)
        self.agent_radius = kwargs.get("agent_radius", 0.025)
        self.xdim = kwargs.get("xdim", 1)
        self.ydim = kwargs.get("ydim", 1)
        self.grid_spacing = kwargs.get("grid_spacing", 0.05)

        self.n_gaussians = kwargs.get("n_gaussians", 2)
        self.cov = 0.05

        assert (self.xdim / self.grid_spacing) % 1 == 0 and (
                self.ydim / self.grid_spacing
        ) % 1 == 0

        self.plot_grid = False
        self.n_x_cells = int((2 * self.xdim) / self.grid_spacing)
        self.n_y_cells = int((2 * self.ydim) / self.grid_spacing)
        self.max_pdf = torch.zeros((batch_dim,), device=device, dtype=torch.float32)
        self.alpha_plot: float = 0.5

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.xdim - self.agent_radius,
            y_semidim=self.ydim - self.agent_radius,
        )

        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent {i}",
                render_action=True,
                collide=True,
                shape=Sphere(radius=self.agent_radius),
                sensors=[AgentDetector(world), ],
            )

            world.add_agent(agent)

        self.sampled = torch.zeros(
            (batch_dim, self.n_x_cells, self.n_y_cells),
            device=device,
            dtype=torch.bool,
        )

        self.locs = [
            torch.zeros((batch_dim, world.dim_p), device=device, dtype=torch.float32)
            for _ in range(self.n_gaussians)
        ]
        self.cov_matrix = torch.tensor(
            [[self.cov, 0], [0, self.cov]], dtype=torch.float32, device=device
        ).expand(batch_dim, world.dim_p, world.dim_p)

        return world

    def reset_world_at(self, env_index: int = None):
        for i, loc in enumerate(self.locs):
            x = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.xdim, self.xdim)
            y = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.ydim, self.ydim)
            new_loc = torch.cat([x, y], dim=-1)
            if env_index is None:
                self.locs[i] = new_loc
            else:
                self.locs[i][env_index] = new_loc

        self.gaussians = [
            MultivariateNormal(
                loc=loc,
                covariance_matrix=self.cov_matrix,
            )
            for loc in self.locs
        ]

        if env_index is None:
            self.max_pdf[:] = 0
            self.sampled[:] = False
        else:
            self.max_pdf[env_index] = 0
            self.sampled[env_index] = False
        self.nomrlize_pdf(env_index=env_index)

        for agent in self.world.agents:
            agent.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.xdim, self.xdim),
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.ydim, self.ydim),
                    ],
                    dim=-1,
                ),
                batch_index=env_index,
            )
            agent.sample = self.sample(agent.state.pos, vel=agent.state.vel, update_sampled_flag=False)
            agent.sample_history = torch.zeros(
                (self.n_x_cells, self.n_y_cells)  # full grid
                if env_index is not None
                else (self.world.batch_dim, self.n_x_cells, self.n_y_cells),
                device=self.world.device,
                dtype=torch.float32,
            )
            agent.time_in_cell_history = torch.zeros(
                (self.n_x_cells, self.n_y_cells)  # full grid
                if env_index is not None
                else (self.world.batch_dim, self.n_x_cells, self.n_y_cells),
                device=self.world.device,
                dtype=torch.float32,
            )
            self.add_sample_to_history(agent)

    def sample(
            self,
            pos,
            vel=None,
            update_sampled_flag: bool = False,
            norm: bool = True,
    ):
        out_of_bounds = (
                (pos[:, X] < -self.xdim)
                + (pos[:, X] > self.xdim)
                + (pos[:, Y] < -self.ydim)
                + (pos[:, Y] > self.ydim)
        )

        pos, index = self.pos_to_index(pos)

        v = torch.stack([gaussian.log_prob(pos).exp() for gaussian in self.gaussians], dim=-1).sum(-1)

        if norm:
            v = v / self.max_pdf

        # not used
        sampled = self.sampled[torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]]
        v[sampled + out_of_bounds] = 0
        if update_sampled_flag:
            self.sampled[torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]] = True

        if vel is None:
            return v

        vel = self.check_velocity(vel)

        # make sure that the probability is between 0 and 1
        v = torch.clamp(v, 0, 1)
        # return zero or one with probability p multiplied by the time step
        return torch.bernoulli(v * self.world.dt * vel)

    def sample_single_env(
            self,
            pos,
            env_index,
            norm: bool = True,
    ):
        pos = pos.view(-1, self.world.dim_p)

        out_of_bounds = (
                (pos[:, X] < -self.xdim)
                + (pos[:, X] > self.xdim)
                + (pos[:, Y] < -self.ydim)
                + (pos[:, Y] > self.ydim)
        )
        pos, index = self.pos_to_index(pos)

        pos = pos.unsqueeze(1).expand(pos.shape[0], self.world.batch_dim, 2)

        v = torch.stack(
            [gaussian.log_prob(pos).exp() for gaussian in self.gaussians], dim=-1
        ).sum(-1)[:, env_index]
        if norm:
            v = v / self.max_pdf[env_index]

        sampled = self.sampled[env_index, index[:, 0], index[:, 1]]

        v[sampled + out_of_bounds] = 0

        return v

    def add_sample_to_history(self, agent: Agent):
        _, index = self.pos_to_index(agent.state.pos)
        inx = torch.arange(agent.sample_history.shape[0]).type_as(index)
        agent.time_in_cell_history[inx, index[:, X], index[:, Y]] += 1
        current_time_in_cell = agent.time_in_cell_history[inx, index[:, X], index[:, Y]]
        agent.sample_history[inx, index[:, X], index[:, Y]] += agent.sample / current_time_in_cell

    def check_velocity(self, vel, threshold: float = 1e-5):
        # get the magnitude of the velocity
        vel = torch.linalg.vector_norm(vel, dim=-1)

        # replace velocity < threshold with 1 and velocity > 1e-5 with 0 (agent can only sample when it is not moving)
        vel = (vel < threshold).float()
        return vel

    def pos_to_index(self, pos):
        pos[:, X].clamp_(-self.world.x_semidim, self.world.x_semidim)
        pos[:, Y].clamp_(-self.world.y_semidim, self.world.y_semidim)

        index = pos / self.grid_spacing
        index[:, X] += self.n_x_cells / 2
        index[:, Y] += self.n_y_cells / 2
        index = index.to(torch.long)
        return pos, index

    def nomrlize_pdf(self, env_index: int = None):
        xpoints = torch.arange(
            -self.xdim, self.xdim, self.grid_spacing, device=self.world.device
        )
        ypoints = torch.arange(
            -self.ydim, self.ydim, self.grid_spacing, device=self.world.device
        )
        if env_index is not None:
            ygrid, xgrid = torch.meshgrid(ypoints, xpoints, indexing="ij")
            pos = torch.stack((xgrid, ygrid), dim=-1).reshape(-1, 2)
            sample = self.sample_single_env(pos, env_index, norm=False)
            self.max_pdf[env_index] = sample.max()
        else:
            for x in xpoints:
                for y in ypoints:
                    pos = torch.tensor(
                        [x, y], device=self.world.device, dtype=torch.float32
                    ).repeat(self.world.batch_dim, 1)
                    sample = self.sample(pos, norm=False)
                    self.max_pdf = torch.maximum(self.max_pdf, sample)

    def reward(self, agent: Agent) -> Tensor:
        agent.sample = self.sample(agent.state.pos, vel=agent.state.vel, update_sampled_flag=False)
        self.add_sample_to_history(agent)
        return agent.sample

    def observation(self, agent: Agent) -> Tensor:
        observations = [
            agent.sample.unsqueeze(-1),
            agent.state.pos,
            agent.state.vel,
            agent.sensors[0].measure(),
            agent.sample_history.flatten(-2),
        ]

        return torch.cat(
            observations,
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {"agent_sample": agent.sample}

    def density_for_plot(self, env_index):
        def f(x):
            sample = self.sample_single_env(
                torch.tensor(x, dtype=torch.float32, device=self.world.device),
                env_index=env_index,
            )

            return sample

        return f

    def agent_sample_history_for_plot(self, env_index):
        def f(x):
            _, index = self.pos_to_index(torch.tensor(x, dtype=torch.float32, device=self.world.device))
            history = self.world.agents[0].sample_history[env_index]

            # if history is all zeros, add one value to the history
            if history.sum() == 0:
                history[0, 0] += 1e-5

            return history[index[:, X], index[:, Y]]

        return f

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        from vmas.simulator.rendering import render_function_util

        geoms = []

        # Function
        # geoms.extend(
        #     render_function_util(
        #         f=self.density_for_plot(env_index=env_index),
        #         plot_range=(self.xdim, self.ydim),
        #         cmap_alpha=self.alpha_plot,
        #     )
        # )

        # Agent 0 history
        geoms.extend(
            render_function_util(
                f=self.agent_sample_history_for_plot(env_index=env_index),
                plot_range=((-self.xdim, self.xdim), (-self.ydim, self.ydim)),
                cmap_range=(0, 1),
                cmap_alpha=self.alpha_plot,
            )
        )

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        # Perimeter
        for i in range(4):
            geom = Line(
                length=2
                       * ((self.ydim if i % 2 == 0 else self.xdim) - self.agent_radius)
                       + self.agent_radius * 2
            ).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                0.0
                if i % 2
                else (
                    self.world.x_semidim + self.agent_radius
                    if i == 0
                    else -self.world.x_semidim - self.agent_radius
                ),
                0.0
                if not i % 2
                else (
                    self.world.y_semidim + self.agent_radius
                    if i == 1
                    else -self.world.y_semidim - self.agent_radius
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__,
                         control_two_agents=True,
                         display_info=False)
