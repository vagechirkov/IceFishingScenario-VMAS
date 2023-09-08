import torch
import vmas
from vmas.simulator.sensors import Sensor


class AgentDetector(Sensor):

    def __init__(self, world: vmas.simulator.core.World):
        super().__init__(world)

    def measure(self):
        # iterate over all agents not including self
        m = torch.stack([
            torch.stack([agent.state.pos, agent.state.vel], dim=0)
            for agent in self._world.agents
            if agent is not self.agent
        ])

        # flatten and convert to tensor
        return m.flatten().unsqueeze(0)

    def render(self, env_index: int = 0):
        return []

    def to(self, device: torch.device):
        pass
