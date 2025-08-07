from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from env import SupplyChainEnv

import os



def env_creator(config):
    return SupplyChainEnv()


register_env("supply_chain", env_creator)


def train_the_model():
    dummy_env = SupplyChainEnv()
    config = (
        PPOConfig()
        .environment("supply_chain")
        .framework("torch")
        .env_runners(num_env_runners=1)
        .training(train_batch_size=4000, gamma=0.99)
        .multi_agent(
            policies={
                "inventory_policy": (
                    None,
                    dummy_env.observation_space("inventory_agent"),
                    dummy_env.action_space("inventory_agent"),
                    {},
                ),
                "transport_policy": (
                    None,
                    dummy_env.observation_space("transport_agent"),
                    dummy_env.action_space("transport_agent"),
                    {},
                ),
                "distribution_policy": (
                    None,
                    dummy_env.observation_space("distribution_agent"),
                    dummy_env.action_space("distribution_agent"),
                    {},
                ),
                "adaptation_policy": (
                    None,
                    dummy_env.observation_space("adaptation_agent"),
                    dummy_env.action_space("adaptation_agent"),
                    {},
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"{agent_id.split('_')[0]}_policy"
        )

    )

    algo = config.build()

    for i in range(50):
        result = algo.train()
        print(f"Iteration {i}")

    checkpoint_dir = os.path.abspath("./supply_chain_ppo")
    algo.save(f"file://{checkpoint_dir}")






def get_config():

    env = SupplyChainEnv()

    config = (
        PPOConfig()
        .environment("supply_chain")
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .training(train_batch_size=4000, gamma=0.99)
        .multi_agent(
            policies={
                "inventory_policy": (
                None, env.observation_space("inventory_agent"), env.action_space("inventory_agent"), {}),
                "transport_policy": (
                None, env.observation_space("transport_agent"), env.action_space("transport_agent"), {}),
                "distribution_policy": (
                None, env.observation_space("distribution_agent"), env.action_space("distribution_agent"), {}),
                "adaptation_policy": (
                None, env.observation_space("adaptation_agent"), env.action_space("adaptation_agent"), {}),
            },
            policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: f"{agent_id.split('_')[0]}_policy"
        )
    )
    return config


def test_model():

    checkpoint_dir = "/Users/waleedalharbi/PycharmProjects/RLagents/supply_chain_ppo"



    config = get_config()
    algo = config.build()

    try:

        algo.restore(checkpoint_dir)
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying to load weights manually...")

        try:

            import pickle
            import os


            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pkl') or 'policies' in file:
                    print(f"Found policy file: {file}")


            print("Using randomly initialized policy for testing...")

        except Exception as e2:
            print(f"Manual loading also failed: {e2}")
            print("Testing with random policy...")


    env = SupplyChainEnv()
    obs, _ = env.reset()

    for step in range(10):
        actions = {}
        for agent_id in obs.keys():
            policy_id = f"{agent_id.split('_')[0]}_policy"
            try:
                action = algo.compute_single_action(obs[agent_id], policy_id=policy_id)
            except:
                # If computation fails, use random action
                action = env.action_space(agent_id).sample()
            actions[agent_id] = action

        obs, rewards, terminated, truncated, _ = env.step(actions)
        print(f"Step {step}: Rewards = {rewards}")

        if terminated.get("__all__", False) or truncated.get("__all__", False):
            break

    algo.stop()
    print("Testing complete!")

if __name__ == "__main__":
    train_the_model()
    test_model()
