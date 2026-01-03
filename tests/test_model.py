"""Tests for CNM-BERT model."""

import pytest
import torch

from cnm.data.tree import IDSTree, BatchedTrees
from cnm.model.configuration_cnm import CNMConfig
from cnm.model.modeling_cnm import (
    CNMBertModel,
    CNMBertForPreTraining,
    CNMBertForSequenceClassification,
)


class TestCNMConfig:
    """Tests for CNMConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CNMConfig()
        assert config.hidden_size == 768
        assert config.struct_dim == 256
        assert config.max_tree_depth == 6

    def test_base_chinese_config(self):
        """Test base Chinese configuration."""
        config = CNMConfig.base_chinese()
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12

    def test_large_chinese_config(self):
        """Test large Chinese configuration."""
        config = CNMConfig.large_chinese()
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16


class TestCNMBertModel:
    """Tests for CNMBertModel."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return CNMConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            struct_dim=32,
            tree_hidden_dim=64,
            component_vocab_size=100,
            operator_vocab_size=16,
        )

    @pytest.fixture
    def model(self, small_config):
        """Create a test model."""
        return CNMBertModel(small_config)

    def test_model_init(self, model, small_config):
        """Test model initialization."""
        assert model.config.hidden_size == 64
        assert model.config.struct_dim == 32

    def test_forward_without_struct(self, model):
        """Test forward pass without structural embeddings."""
        batch_size, seq_length = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert outputs.last_hidden_state.shape == (batch_size, seq_length, 64)
        assert outputs.pooler_output.shape == (batch_size, 64)

    def test_forward_with_struct(self, model, small_config):
        """Test forward pass with structural embeddings."""
        batch_size, seq_length = 2, 8

        # Create input
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        # Create some trees
        comp_vocab = {str(i): i for i in range(100)}
        op_vocab = {"⿰": 2, "⿱": 3}

        trees = []
        for _ in range(4):  # 4 unique characters
            t = IDSTree.internal(
                "x", "⿰",
                (IDSTree.leaf("a"), IDSTree.leaf("b"))
            )
            trees.append(t.to_indexed(comp_vocab, op_vocab, 1, 1))

        unique_trees = BatchedTrees.from_trees(trees)

        # Mapping: positions 0-3 map to unique chars, rest are -1
        unique_to_position = torch.full((batch_size, seq_length), -1, dtype=torch.long)
        unique_to_position[0, 0] = 0
        unique_to_position[0, 1] = 1
        unique_to_position[1, 0] = 2
        unique_to_position[1, 1] = 3

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            unique_trees=unique_trees,
            unique_to_position=unique_to_position,
        )

        assert outputs.last_hidden_state.shape == (batch_size, seq_length, 64)

    def test_gradients_flow(self, model):
        """Test that gradients flow through the model."""
        input_ids = torch.randint(0, 1000, (1, 8))
        attention_mask = torch.ones(1, 8)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.last_hidden_state.sum()
        loss.backward()

        # Check that gradients exist in embeddings
        assert model.embeddings.word_embeddings.weight.grad is not None


class TestCNMBertForPreTraining:
    """Tests for CNMBertForPreTraining."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return CNMConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            struct_dim=32,
            tree_hidden_dim=64,
            component_vocab_size=100,
            operator_vocab_size=16,
            aux_loss_weight=0.1,
        )

    @pytest.fixture
    def model(self, small_config):
        """Create a test model."""
        return CNMBertForPreTraining(small_config)

    def test_forward_with_labels(self, model, small_config):
        """Test forward pass with MLM labels."""
        batch_size, seq_length = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        # MLM labels: -100 for non-masked, token_id for masked
        labels = torch.full((batch_size, seq_length), -100, dtype=torch.long)
        labels[0, 0] = 50  # Mask first token
        labels[1, 5] = 100  # Mask token at position 5

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert outputs.loss is not None
        assert outputs.mlm_loss is not None
        assert outputs.prediction_logits.shape == (batch_size, seq_length, 1000)

    def test_forward_with_aux_labels(self, model, small_config):
        """Test forward pass with auxiliary component labels."""
        batch_size, seq_length = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        # MLM labels
        labels = torch.full((batch_size, seq_length), -100, dtype=torch.long)
        labels[0, 0] = 50

        # Component labels (multi-hot)
        component_labels = torch.zeros(batch_size, seq_length, 100)
        component_labels[0, 0, 10] = 1.0
        component_labels[0, 0, 20] = 1.0

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            component_labels=component_labels,
        )

        assert outputs.loss is not None
        assert outputs.mlm_loss is not None
        assert outputs.aux_loss is not None
        assert outputs.component_logits.shape == (batch_size, seq_length, 100)


class TestCNMBertForSequenceClassification:
    """Tests for CNMBertForSequenceClassification."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        config = CNMConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            struct_dim=32,
            tree_hidden_dim=64,
            component_vocab_size=100,
            operator_vocab_size=16,
        )
        config.num_labels = 3
        return config

    @pytest.fixture
    def model(self, small_config):
        """Create a test model."""
        return CNMBertForSequenceClassification(small_config)

    def test_forward(self, model):
        """Test forward pass."""
        batch_size, seq_length = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert outputs.logits.shape == (batch_size, 3)

    def test_forward_with_labels(self, model):
        """Test forward pass with labels."""
        batch_size, seq_length = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randint(0, 3, (batch_size,))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, 3)
