from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .casec_rnn_agent import CasecRNNAgent, PairRNNAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["casec_rnn"] = CasecRNNAgent
REGISTRY["pair_rnn"] = PairRNNAgent
