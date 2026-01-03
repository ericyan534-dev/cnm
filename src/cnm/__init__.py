"""
CNM: Compositional Network Model for Chinese Pre-trained Language Models.

This package implements CNM-BERT, which treats Chinese characters as recursive
compositional functions of their sub-components (radicals and operators).
"""

__version__ = "0.1.0"

from cnm.data.tree import IDSTree, IndexedTree, BatchedTrees
from cnm.data.ids_parser import IDSParser
from cnm.data.vocab import CNMVocab
from cnm.model.tree_mlp import TreeMLPEncoder
from cnm.model.configuration_cnm import CNMConfig
from cnm.model.modeling_cnm import CNMBertModel, CNMBertForPreTraining

__all__ = [
    "__version__",
    "IDSTree",
    "IndexedTree",
    "BatchedTrees",
    "IDSParser",
    "CNMVocab",
    "TreeMLPEncoder",
    "CNMConfig",
    "CNMBertModel",
    "CNMBertForPreTraining",
]
