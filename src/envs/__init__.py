import os
import sys

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
from .smaclite_wrapper import SMACliteWrapper
from .maco_wrapper import MacoWrapper


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII") 


def _deduplicate_pysc2_maps(preferred_prefix=None):
    """
    Pysc2 raises DuplicateMapError when both SMAC and SMACv2 register maps.
    Override lib.get_maps so we keep a single entry per map name while
    optionally preferring classes from the requested package.
    """
    from pysc2.maps import lib as maps_lib

    def _get_maps_without_duplicates():
        maps = {}
        for mp in maps_lib.Map.all_subclasses():
            if not (mp.filename or mp.battle_net):
                continue

            map_name = mp.__name__
            existing = maps.get(map_name)
            if existing is None:
                maps[map_name] = mp
                continue

            if preferred_prefix:
                preferred_in_existing = existing.__module__.startswith(
                    preferred_prefix
                )
                preferred_in_new = mp.__module__.startswith(preferred_prefix)

                # Keep the preferred implementation if present
                if preferred_in_existing:
                    continue
                if preferred_in_new:
                    maps[map_name] = mp
                    continue

            # Otherwise keep the first one we saw
            continue

        return maps

    maps_lib.get_maps = _get_maps_without_duplicates


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


def __check_and_prepare_maco_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "MACO only supports common reward. Please set `common_reward=True`."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


def maco_fn(**kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_maco_kwargs(kwargs)
    return MacoWrapper(**kwargs)


REGISTRY = {}
REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn
REGISTRY["maco"] = maco_fn


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    _deduplicate_pysc2_maps(preferred_prefix="smac.env.")
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    _deduplicate_pysc2_maps(preferred_prefix="smacv2.")
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn
