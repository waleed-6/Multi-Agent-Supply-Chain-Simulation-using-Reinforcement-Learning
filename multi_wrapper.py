
from pettingzoo import ParallelEnv
from gymnasium.spaces import Dict
from ray.rllib.env import MultiAgentEnv


class PettingZooToGymnasium(MultiAgentEnv):
    def __init__(self, parallel_env: ParallelEnv):
        super().__init__()
        self.env = parallel_env
        self.agents = self.env.possible_agents


        self.observation_space = Dict({
            agent: self.env.observation_space(agent)
            for agent in self.agents
        })

        self.action_space = Dict({
            agent: self.env.action_space(agent)
            for agent in self.agents
        })

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, actions):

        result = self.env.step(actions)
        print("Step result:", result)
        if result is None:
            raise RuntimeError("env.step() returned None, which is invalid!")
        obs, rewards, terminations, truncations, infos = result


        dones = {agent: terminations.get(agent, False) or truncations.get(agent, False) for agent in terminations}
        dones["__all__"] = all(dones.values())

        return obs, rewards, dones,False, infos

    def render(self):
        self.env.render()
