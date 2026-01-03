"""Tests for Tree-MLP encoder and data structures."""

import pytest
import torch

from cnm.data.tree import IDSTree, IndexedTree, BatchedTrees, IDC_ARITY


class TestIDSTree:
    """Tests for IDSTree data structure."""

    def test_create_leaf(self):
        """Test creating a leaf node."""
        tree = IDSTree.leaf("日")
        assert tree.char == "日"
        assert tree.is_leaf
        assert tree.depth == 0
        assert tree.num_nodes() == 1
        assert tree.num_leaves() == 1

    def test_create_internal(self):
        """Test creating an internal node."""
        left = IDSTree.leaf("日")
        right = IDSTree.leaf("月")
        tree = IDSTree.internal("明", "⿰", (left, right))

        assert tree.char == "明"
        assert not tree.is_leaf
        assert tree.operator == "⿰"
        assert tree.depth == 1
        assert tree.num_nodes() == 3
        assert tree.num_leaves() == 2

    def test_nested_tree(self):
        """Test creating a nested tree."""
        # 森 = ⿱木⿰木木
        wood = IDSTree.leaf("木")
        bottom = IDSTree.internal("林", "⿰", (wood, wood))
        tree = IDSTree.internal("森", "⿱", (wood, bottom))

        assert tree.depth == 2
        assert tree.num_nodes() == 5
        assert tree.num_leaves() == 3

    def test_to_ids_string(self):
        """Test converting tree back to IDS string."""
        left = IDSTree.leaf("日")
        right = IDSTree.leaf("月")
        tree = IDSTree.internal("明", "⿰", (left, right))

        assert tree.to_ids_string() == "⿰日月"

    def test_leaves_iterator(self):
        """Test iterating over leaves."""
        left = IDSTree.leaf("日")
        right = IDSTree.leaf("月")
        tree = IDSTree.internal("明", "⿰", (left, right))

        leaves = list(tree.leaves())
        assert len(leaves) == 2
        assert leaves[0].char == "日"
        assert leaves[1].char == "月"

    def test_ternary_operator(self):
        """Test ternary operator (⿲)."""
        a = IDSTree.leaf("亻")
        b = IDSTree.leaf("口")
        c = IDSTree.leaf("刂")
        tree = IDSTree.internal("测", "⿲", (a, b, c))

        assert tree.arity == 3
        assert tree.num_leaves() == 3

    def test_invalid_arity(self):
        """Test that invalid arity raises error."""
        a = IDSTree.leaf("日")
        with pytest.raises(ValueError):
            # ⿰ expects 2 children, not 3
            IDSTree.internal("x", "⿰", (a, a, a))


class TestIndexedTree:
    """Tests for IndexedTree."""

    def test_to_indexed(self):
        """Test converting IDSTree to IndexedTree."""
        left = IDSTree.leaf("日")
        right = IDSTree.leaf("月")
        tree = IDSTree.internal("明", "⿰", (left, right))

        comp_vocab = {"日": 0, "月": 1}
        op_vocab = {"⿰": 0, "⿱": 1}

        indexed = tree.to_indexed(comp_vocab, op_vocab)

        assert indexed.operator_id == 0  # ⿰
        assert len(indexed.children) == 2
        assert indexed.children[0].component_id == 0  # 日
        assert indexed.children[1].component_id == 1  # 月


class TestBatchedTrees:
    """Tests for BatchedTrees."""

    def test_from_single_tree(self):
        """Test batching a single tree."""
        left = IDSTree.leaf("日")
        right = IDSTree.leaf("月")
        tree = IDSTree.internal("明", "⿰", (left, right))

        comp_vocab = {"日": 0, "月": 1}
        op_vocab = {"⿰": 0}

        indexed = tree.to_indexed(comp_vocab, op_vocab)
        batched = BatchedTrees.from_trees([indexed])

        assert batched.batch_size == 1
        assert batched.num_nodes == 3
        assert batched.max_depth == 1

    def test_from_multiple_trees(self):
        """Test batching multiple trees."""
        # Tree 1: 明 = ⿰日月
        t1_left = IDSTree.leaf("日")
        t1_right = IDSTree.leaf("月")
        t1 = IDSTree.internal("明", "⿰", (t1_left, t1_right))

        # Tree 2: 木 (leaf only)
        t2 = IDSTree.leaf("木")

        comp_vocab = {"日": 0, "月": 1, "木": 2}
        op_vocab = {"⿰": 0}

        indexed1 = t1.to_indexed(comp_vocab, op_vocab)
        indexed2 = t2.to_indexed(comp_vocab, op_vocab)
        batched = BatchedTrees.from_trees([indexed1, indexed2])

        assert batched.batch_size == 2
        assert batched.num_nodes == 4  # 3 from t1 + 1 from t2
        assert len(batched.root_indices) == 2

    def test_to_device(self):
        """Test moving to device."""
        tree = IDSTree.leaf("日")
        comp_vocab = {"日": 0}
        op_vocab = {}

        indexed = tree.to_indexed(comp_vocab, op_vocab)
        batched = BatchedTrees.from_trees([indexed])

        # Move to CPU (should work without GPU)
        batched_cpu = batched.to(torch.device("cpu"))
        assert batched_cpu.component_ids.device.type == "cpu"


class TestTreeMLPEncoder:
    """Tests for TreeMLPEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create a test encoder."""
        from cnm.model.tree_mlp import TreeMLPEncoder
        return TreeMLPEncoder(
            component_vocab_size=100,
            operator_vocab_size=16,
            embed_dim=64,
            hidden_dim=128,
            max_depth=6,
        )

    def test_forward_single_leaf(self, encoder):
        """Test forward pass with single leaf."""
        tree = IDSTree.leaf("日")
        comp_vocab = {"日": 10}
        op_vocab = {}

        indexed = tree.to_indexed(comp_vocab, op_vocab, unk_component_id=1, unk_operator_id=1)
        batched = BatchedTrees.from_trees([indexed])

        output = encoder(batched)
        assert output.shape == (1, 64)

    def test_forward_binary_tree(self, encoder):
        """Test forward pass with binary tree."""
        left = IDSTree.leaf("日")
        right = IDSTree.leaf("月")
        tree = IDSTree.internal("明", "⿰", (left, right))

        comp_vocab = {"日": 10, "月": 11}
        op_vocab = {"⿰": 2}

        indexed = tree.to_indexed(comp_vocab, op_vocab, unk_component_id=1, unk_operator_id=1)
        batched = BatchedTrees.from_trees([indexed])

        output = encoder(batched)
        assert output.shape == (1, 64)

    def test_forward_batch(self, encoder):
        """Test forward pass with batch of trees."""
        # Create multiple trees
        trees = []
        comp_vocab = {"日": 10, "月": 11, "木": 12, "林": 13}
        op_vocab = {"⿰": 2, "⿱": 3}

        # Tree 1: leaf
        t1 = IDSTree.leaf("木")
        trees.append(t1.to_indexed(comp_vocab, op_vocab, 1, 1))

        # Tree 2: binary
        t2 = IDSTree.internal("明", "⿰", (IDSTree.leaf("日"), IDSTree.leaf("月")))
        trees.append(t2.to_indexed(comp_vocab, op_vocab, 1, 1))

        # Tree 3: nested
        t3 = IDSTree.internal(
            "森", "⿱",
            (IDSTree.leaf("木"), IDSTree.internal("林", "⿰", (IDSTree.leaf("木"), IDSTree.leaf("木"))))
        )
        trees.append(t3.to_indexed(comp_vocab, op_vocab, 1, 1))

        batched = BatchedTrees.from_trees(trees)
        output = encoder(batched)

        assert output.shape == (3, 64)

    def test_gradients_flow(self, encoder):
        """Test that gradients flow through the encoder."""
        tree = IDSTree.internal("明", "⿰", (IDSTree.leaf("日"), IDSTree.leaf("月")))

        comp_vocab = {"日": 10, "月": 11}
        op_vocab = {"⿰": 2}

        indexed = tree.to_indexed(comp_vocab, op_vocab, 1, 1)
        batched = BatchedTrees.from_trees([indexed])

        output = encoder(batched)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert encoder.component_embed.weight.grad is not None
        assert encoder.operator_embed.weight.grad is not None
