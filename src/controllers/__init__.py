REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .dcg_controller import DeepCoordinationGraphMAC
from .dcg_noshare_controller import DCGnoshareMAC
from .cg_controller import SimpleCoordionationGraphMAC
from .low_rank_controller import LowRankMAC
from .low_rank_ns_controller import LowRankNSMAC
from .dicg_controller import DICGMAC
from .dicg_noshare_controller import DICGNonSharedMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC
REGISTRY["dcg_noshare_mac"] = DCGnoshareMAC
REGISTRY["dcg_ns_mac"] = DCGnoshareMAC
REGISTRY["cg_ns_mac"] = DCGnoshareMAC
REGISTRY["cg_mac"] = SimpleCoordionationGraphMAC
REGISTRY["low_rank_q"] = LowRankMAC
REGISTRY["low_rank_q_ns"] = LowRankNSMAC
REGISTRY["dicg_mac"] = DICGMAC
REGISTRY["dicg_noshare_mac"] = DICGNonSharedMAC
