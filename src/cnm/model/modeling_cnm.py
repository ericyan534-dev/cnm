"""
CNM-BERT model implementations.

This module provides CNM-BERT models that extend standard BERT with
structural embeddings from Chinese character decomposition (IDS).

Models:
    - CNMBertModel: Base model with fused embeddings
    - CNMBertForPreTraining: MLM + auxiliary component prediction
    - CNMBertForSequenceClassification: For CLUE benchmark tasks
    - CNMBertForTokenClassification: For NER and similar tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertOnlyMLMHead,
)

from cnm.data.tree import BatchedTrees
from cnm.model.configuration_cnm import CNMConfig
from cnm.model.embeddings import CNMEmbeddings
from cnm.model.tree_mlp import TreeMLPEncoder


@dataclass
class CNMPreTrainingOutput:
    """Output for CNM pretraining with MLM and auxiliary losses."""
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    component_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class CNMBertModel(BertPreTrainedModel):
    """
    CNM-BERT base model with fused structural embeddings.

    This model extends BERT by:
    1. Adding Tree-MLP encoder for structural embeddings
    2. Fusing BERT and structural embeddings in the embedding layer
    3. Keeping the transformer encoder unchanged

    The fusion preserves pretrained BERT weights through Identity+Zero
    initialization of the projection layer.
    """

    config_class = CNMConfig

    def __init__(self, config: CNMConfig):
        """
        Initialize CNMBertModel.

        Args:
            config: CNMConfig with model parameters
        """
        super().__init__(config)
        self.config = config

        # Create Tree-MLP encoder
        self.tree_encoder = TreeMLPEncoder(
            component_vocab_size=config.component_vocab_size,
            operator_vocab_size=config.operator_vocab_size,
            embed_dim=config.struct_dim,
            hidden_dim=config.tree_hidden_dim,
            max_depth=config.max_tree_depth,
            dropout=config.struct_dropout,
        )

        # Fused embeddings
        self.embeddings = CNMEmbeddings(config, tree_encoder=self.tree_encoder)

        # Standard BERT encoder (unchanged)
        self.encoder = BertEncoder(config)

        # Pooler
        self.pooler = BertPooler(config)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Get word embeddings."""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding):
        """Set word embeddings."""
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unique_trees: Optional[BatchedTrees] = None,
        unique_to_position: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Forward pass through CNM-BERT.

        Args:
            input_ids: [batch, seq] Token IDs
            attention_mask: [batch, seq] Attention mask
            token_type_ids: [batch, seq] Segment IDs
            position_ids: [batch, seq] Position IDs
            unique_trees: BatchedTrees for unique characters
            unique_to_position: [batch, seq] Mapping to unique char indices
            head_mask: Head mask for attention
            inputs_embeds: Pre-computed embeddings
            encoder_hidden_states: For cross-attention
            encoder_attention_mask: For cross-attention
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict or tuple

        Returns:
            Model outputs with last hidden state and pooled output
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Prepare attention mask for encoder
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # Prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Compute fused embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            unique_trees=unique_trees,
            unique_to_position=unique_to_position,
            inputs_embeds=inputs_embeds,
        )

        # Encode
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    @classmethod
    def from_pretrained_bert(
        cls,
        bert_name_or_path: str,
        cnm_config: Optional[CNMConfig] = None,
        **kwargs,
    ) -> "CNMBertModel":
        """
        Load pretrained BERT weights into CNM model.

        Args:
            bert_name_or_path: HuggingFace model name or path
            cnm_config: CNMConfig (created from BERT config if not provided)
            **kwargs: Additional arguments for from_pretrained

        Returns:
            CNMBertModel with pretrained BERT weights
        """
        # Load BERT model
        bert = BertModel.from_pretrained(bert_name_or_path, **kwargs)

        # Create CNM config from BERT config
        if cnm_config is None:
            cnm_config = CNMConfig.from_bert_config(bert.config)

        # Create CNM model
        model = cls(cnm_config)

        # Copy BERT weights
        model._load_bert_weights(bert)

        return model

    def _load_bert_weights(self, bert: BertModel):
        """Copy weights from a pretrained BERT model."""
        with torch.no_grad():
            # Copy embeddings
            self.embeddings.load_bert_embeddings(bert.embeddings)

            # Copy encoder
            self.encoder.load_state_dict(bert.encoder.state_dict())

            # Copy pooler
            self.pooler.load_state_dict(bert.pooler.state_dict())


class CNMBertForPreTraining(BertPreTrainedModel):
    """
    CNM-BERT for pretraining with MLM and auxiliary component prediction.

    This model has two training objectives:
    1. Masked Language Modeling (MLM): Standard BERT objective
    2. Component Prediction: Multi-label prediction of character components
       at masked positions (auxiliary loss)

    The auxiliary loss encourages the model to retain structural information.
    """

    config_class = CNMConfig

    def __init__(self, config: CNMConfig):
        """
        Initialize CNMBertForPreTraining.

        Args:
            config: CNMConfig with model parameters
        """
        super().__init__(config)

        self.bert = CNMBertModel(config)
        self.cls = BertOnlyMLMHead(config)

        # Auxiliary head for component prediction
        self.aux_head = nn.Linear(config.hidden_size, config.component_vocab_size)
        self.aux_loss_weight = config.aux_loss_weight

        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        """Get output embeddings for MLM."""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Set output embeddings for MLM."""
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unique_trees: Optional[BatchedTrees] = None,
        unique_to_position: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        component_labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CNMPreTrainingOutput]:
        """
        Forward pass for pretraining.

        Args:
            input_ids: [batch, seq] Token IDs
            attention_mask: [batch, seq] Attention mask
            token_type_ids: [batch, seq] Segment IDs
            position_ids: [batch, seq] Position IDs
            unique_trees: BatchedTrees for unique characters
            unique_to_position: [batch, seq] Mapping to unique char indices
            head_mask: Head mask for attention
            inputs_embeds: Pre-computed embeddings
            labels: [batch, seq] MLM labels (-100 for non-masked)
            component_labels: [batch, seq, component_vocab] Multi-hot labels
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict or tuple

        Returns:
            CNMPreTrainingOutput with losses and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            unique_trees=unique_trees,
            unique_to_position=unique_to_position,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        # MLM prediction
        prediction_scores = self.cls(sequence_output)

        # Auxiliary component prediction
        component_logits = self.aux_head(sequence_output)

        # Compute losses
        mlm_loss = None
        aux_loss = None
        total_loss = None

        if labels is not None:
            mlm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            total_loss = mlm_loss

        if component_labels is not None and labels is not None:
            # Only compute auxiliary loss on masked positions
            masked_positions = labels != -100

            if masked_positions.any():
                masked_logits = component_logits[masked_positions]
                masked_targets = component_labels[masked_positions]

                aux_loss = F.binary_cross_entropy_with_logits(
                    masked_logits, masked_targets.float()
                )

                if total_loss is not None:
                    total_loss = total_loss + self.aux_loss_weight * aux_loss
                else:
                    total_loss = self.aux_loss_weight * aux_loss

        if not return_dict:
            output = (prediction_scores, component_logits) + outputs[2:]
            return ((total_loss, mlm_loss, aux_loss) + output) if total_loss is not None else output

        return CNMPreTrainingOutput(
            loss=total_loss,
            mlm_loss=mlm_loss,
            aux_loss=aux_loss,
            prediction_logits=prediction_scores,
            component_logits=component_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CNMBertForSequenceClassification(BertPreTrainedModel):
    """
    CNM-BERT for sequence classification (CLUE benchmark tasks).
    """

    config_class = CNMConfig

    def __init__(self, config: CNMConfig):
        """Initialize with classification head."""
        super().__init__(config)
        self.num_labels = config.num_labels if hasattr(config, 'num_labels') else 2

        self.bert = CNMBertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unique_trees: Optional[BatchedTrees] = None,
        unique_to_position: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """Forward pass for sequence classification."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            unique_trees=unique_trees,
            unique_to_position=unique_to_position,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(), labels.squeeze())
            else:
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CNMBertForTokenClassification(BertPreTrainedModel):
    """
    CNM-BERT for token classification (NER, POS tagging, etc.).
    """

    config_class = CNMConfig

    def __init__(self, config: CNMConfig):
        """Initialize with token classification head."""
        super().__init__(config)
        self.num_labels = config.num_labels if hasattr(config, 'num_labels') else 2

        self.bert = CNMBertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unique_trees: Optional[BatchedTrees] = None,
        unique_to_position: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """Forward pass for token classification."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            unique_trees=unique_trees,
            unique_to_position=unique_to_position,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
