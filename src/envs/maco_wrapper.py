from pathlib import Path
import yaml

from .multiagentenv import MultiAgentEnv
from .maco_envs.aloha import AlohaEnv
from .maco_envs.disperse import DisperseEnv
from .maco_envs.gather import GatherEnv
from .maco_envs.hallway import HallwayEnv
from .maco_envs.pursuit import PursuitEnv
from .maco_envs.sensors import SensorEnv

MACO_CONFIG_DIR = Path(__file__).parent.parent / "config" / "envs" / "maco_configs"

MACO_ENV_MAP = {
    "gather": GatherEnv,
    "aloha": AlohaEnv,
    "disperse": DisperseEnv,
    "hallway": HallwayEnv,
    "pursuit": PursuitEnv,
    "sensor": SensorEnv,
}


def get_scenario_names():
    return [p.stem for p in MACO_CONFIG_DIR.glob("*.yaml")]


def load_scenario(map_name, **overrides):
    scenario_path = MACO_CONFIG_DIR / f"{map_name}.yaml"
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Could not find MACO scenario file at {scenario_path}")

    with open(scenario_path, "r") as f:
        scenario_cfg = yaml.load(f, Loader=yaml.FullLoader)

    env_args = scenario_cfg.get("env_args", {})
    env_args.update(overrides)
    env_args.pop("map_name", None)
    env_args.pop("reward_scalarisation", None)
    env_args.pop("common_reward", None)

    env_cls = MACO_ENV_MAP.get(map_name)
    if env_cls is None:
        raise ValueError(f"Unknown MACO env '{map_name}'. Available: {list(MACO_ENV_MAP)}")

    return env_cls(**env_args)


class MacoWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed=None, **kwargs):
        if seed is not None:
            kwargs.setdefault("seed", seed)
        self.env = load_scenario(map_name, **kwargs)
        self.n_agents = self.env.n_agents
        self.episode_limit = self.env.episode_limit

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        reward, terminated, info = self.env.step(actions)
        obss = self.get_obs()

        if (
            hasattr(self.env, "_episode_steps")
            and hasattr(self.env, "episode_limit")
            and self.env._episode_steps >= self.env.episode_limit
        ):
            info = dict(info)
            info["episode_limit"] = True

        truncated = False
        return obss, reward, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            try:
                self.env.seed(seed)
            except TypeError:
                pass

        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            obss = reset_out[0]
        else:
            obss = reset_out
        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        try:
            self.env.seed(seed)
        except TypeError:
            self.env.seed()

    def save_replay(self):
        """Save a replay."""
        self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()
