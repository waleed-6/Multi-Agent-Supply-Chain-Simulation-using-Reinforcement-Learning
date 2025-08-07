from gymnasium import spaces
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from enum import IntEnum


class ObservationIndex(IntEnum):
    INVENTORY_JEDDAH = 0
    INVENTORY_RIYADH = 1
    DEMAND_JEDDAH = 2
    DEMAND_RIYADH = 3
    LAST_DEMAND_JEDDAH = 4
    LAST_DEMAND_RIYADH = 5
    FULFILLED_JEDDAH = 6
    FULFILLED_RIYADH = 7
    SHORTAGE_JEDDAH = 8
    SHORTAGE_RIYADH = 9
    OVERSTOCK_JEDDAH = 10
    OVERSTOCK_RIYADH = 11
    TRANSPORT_CAPACITY = 12
    LAST_TRANSPORT_JEDDAH = 13
    LAST_TRANSPORT_RIYADH = 14
    LAST_DIST_JEDDAH = 15
    LAST_DIST_RIYADH = 16
    DAY_OF_WEEK = 17
    STEP_COUNT = 18
    AVG_DEMAND_JEDDAH = 19
    AVG_DEMAND_RIYADH = 20
    PENDING_JEDDAH = 21
    PENDING_RIYADH = 22


class SupplyChainEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        self.env_config = env_config or {}
        self.agents = ["inventory_agent", "transport_agent", "distribution_agent", "adaptation_agent"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {
            "inventory_agent": spaces.Discrete(3),
            "transport_agent": spaces.MultiDiscrete([201, 201]),
            "distribution_agent": spaces.MultiDiscrete([201, 201]),
            "adaptation_agent": spaces.Discrete(3),
        }

        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1000, shape=(23,), dtype=np.float32)
            for agent in self.agents
        }

        self._init_state()

    def _init_state(self):
        self.state = {
            "inventory": 600,
            "demand": {"jeddah": 100, "riyadh": 120},
            "day": 0,
            "in_transit": 0
        }
        self.inventory = {"jeddah": 300, "riyadh": 300}
        self.demand = {"jeddah": 100, "riyadh": 120}
        self.last_demand = {"jeddah": 100, "riyadh": 120}
        self.fulfilled = {"jeddah": 100, "riyadh": 120}
        self.shortage = {"jeddah": 0, "riyadh": 0}
        self.overstock = {"jeddah": 0, "riyadh": 0}
        self.last_transport = {"jeddah": 0, "riyadh": 0}
        self.last_distribution = {"jeddah": 0, "riyadh": 0}
        self.pending_delivery = {"jeddah": 0, "riyadh": 0}
        self.transport_capacity = 200
        self.step_count = 0
        self.day_of_week = 0
        self.history_demand_jeddah = [100]
        self.history_demand_riyadh = [120]

    def reset(self, *, seed=None, options=None):
        self._init_state()
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations, {}

    def _get_observation(self, agent):
        return np.array([
            self.inventory["jeddah"],
            self.inventory["riyadh"],
            self.demand["jeddah"],
            self.demand["riyadh"],
            self.last_demand["jeddah"],
            self.last_demand["riyadh"],
            self.fulfilled["jeddah"],
            self.fulfilled["riyadh"],
            self.shortage["jeddah"],
            self.shortage["riyadh"],
            self.overstock["jeddah"],
            self.overstock["riyadh"],
            self.transport_capacity,
            self.last_transport["jeddah"],
            self.last_transport["riyadh"],
            self.last_distribution["jeddah"],
            self.last_distribution["riyadh"],
            self.day_of_week,
            self.step_count,
            np.mean(self.history_demand_jeddah[-5:]),
            np.mean(self.history_demand_riyadh[-5:]),
            self.pending_delivery["jeddah"],
            self.pending_delivery["riyadh"],
        ], dtype=np.float32)

    def step(self, action_dict):
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        observations = {}

        for agent, action in action_dict.items():
            reward = 0
            if agent == "inventory_agent":
                if action == 1:
                    self.state["inventory"] += 100
                elif action == 2:
                    self.state["inventory"] += 200
                inventory = self.state["inventory"]
                if inventory < 300:
                    reward = -0.05 * (300 - inventory)
                elif inventory > 500:
                    reward = -0.05 * (inventory - 500)
                else:
                    reward = 5.0

            elif agent == "transport_agent":
                j, r = action
                total = j + r
                if self.state["inventory"] >= total:
                    self.state["inventory"] -= total
                    self.state["in_transit"] = total
                    reward = 10 * (min(j / max(1, self.demand["jeddah"]), 1.0) +
                                   min(r / max(1, self.demand["riyadh"]), 1.0))
                else:
                    reward = -10.0

            elif agent == "distribution_agent":
                j, r = action
                self.state["inventory"] -= j + r
                j_diff = abs(self.demand["jeddah"] - j) / max(1, self.demand["jeddah"])
                r_diff = abs(self.demand["riyadh"] - r) / max(1, self.demand["riyadh"])
                reward = max(0, 10 - 10 * (j_diff + r_diff))

            elif agent == "adaptation_agent":
                if action == 1:
                    self.state["inventory"] += 50
                    penalty = max(0, self.state["inventory"] - 1000) * 0.05
                    reward = 5.0 - penalty
                elif action == 2:
                    self.state["inventory"] = max(0, self.state["inventory"] - 50)
                    reward = 5.0 if self.state["inventory"] <= 500 else 0.0
                else:
                    reward = -1.0

            rewards[agent] = reward
            observations[agent] = self._get_observation(agent)
            terminations[agent] = self.state["day"] > 30
            truncations[agent] = False
            infos[agent] = {}

        self.state["day"] += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        self.step_count += 1
        self.last_demand = self.demand.copy()
        self.demand["jeddah"] = self._generate_city_demand()
        self.demand["riyadh"] = self._generate_city_demand()

        terminations["__all__"] = all(terminations.values())
        truncations["__all__"] = all(truncations.values())

        return observations, rewards, terminations, truncations, infos

    def _generate_city_demand(self):
        t = random.choices(["low", "medium", "high"], weights=[0.3, 0.5, 0.2])[0]
        return random.randint(20, 60) if t == "low" else random.randint(61, 100) if t == "medium" else random.randint(101, 150)

    def observation_space(self, agent):
        return self.observation_spaces[agent]


    def render(self):
        print(f"Day {self.state['day']} | Inventory: {self.state['inventory']} | Demand: {self.demand} | In Transit: {self.state['in_transit']}")

    def action_space(self, agent):
        return self.action_spaces[agent]
    def close(self):
        pass
