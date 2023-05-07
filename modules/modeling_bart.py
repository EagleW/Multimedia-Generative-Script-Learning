# based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py
import random
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers import BartTokenizer


import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MarginRankingLoss
import torch.nn.functional as F

from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.utils import logging

from transformers.models.bart.modeling_bart import (
    BartAttention,
    BartEncoderLayer,
    BartDecoderLayer, 
    BartEncoder, 
    BartDecoder,
    BartModel, 
    BartForConditionalGeneration,
    BartConfig, 
    BartLearnedPositionalEmbedding,
    shift_tokens_right, _expand_mask, ACT2FN
)
import math
import pickle
from .utils import AttnGate, AttnClsGate


import torch.nn.functional as F

logger = logging.get_logger(__name__)


class RetrieveBaseModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    retrieve_last_hidden_states: Optional[torch.FloatTensor] = None

class ContrastiveSeq2SeqModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    retrieve_last_hidden_states: Optional[torch.FloatTensor] = None

class ContrastiveSeq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    cl_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

class RetrieveBartEncoder(BartEncoder):

    def __init__(self, config: BartConfig, tokenizer: Optional[BartTokenizer]=None, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)

        self.gate_alpha = nn.Linear(config.d_model * 2, 1)
        self.gate_beta = nn.Linear(config.d_model * 2, 1)
        self.attention_cls_alpha = nn.MultiheadAttention(config.d_model, config.encoder_attention_heads, batch_first=True)
        self.attention_cls = nn.MultiheadAttention(config.d_model, config.encoder_attention_heads, batch_first=True)
        self.tokenizer = tokenizer

        self.mask_id = self.tokenizer.mask_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,

        max_hist: Optional[int]=1,
        hist_l: Optional[torch.LongTensor] = None,

        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return RetrieveBaseModelOutput(
            last_hidden_state=hidden_states, 
            hidden_states=encoder_states, 
            attentions=all_attentions
        )

class RetrieveBartDecoderLayer(BartDecoderLayer):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.retrieve_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.retrieve_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.ret_gate_w = nn.Linear(self.embed_dim * 2, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[Tuple[torch.Tensor]] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        
        src_hidden_states, knowl_hidden_states = encoder_hidden_states
        src_hidden_attn_mask, knowl_attn_mask = encoder_attention_mask

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value_src = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn_src cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value_src = past_key_value[-4:-2] if past_key_value is not None else None
            hidden_states, cross_attn_weights_src, cross_attn_present_key_value_src = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=src_hidden_states,
                attention_mask=src_hidden_attn_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value_src,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn_src to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value_src

            residual = hidden_states

            # cross_attn_ret cached key/values tuple is at positions 5,6 of present_key_value tuple
            cross_attn_past_key_value_ret = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value_ret = self.retrieve_attn(
                hidden_states=hidden_states,
                key_value_states=knowl_hidden_states,
                attention_mask=knowl_attn_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value_ret,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            # hidden_states = residual + hidden_states

            weight = self.ret_gate_w(torch.cat([hidden_states, residual], dim=-1)).sigmoid()
            hidden_states = (1 - weight) * residual + weight * hidden_states
            hidden_states = self.retrieve_attn_layer_norm(hidden_states)

            # add cross_attn_ret to positions 5,6 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value_ret

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights_src, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class RetrieveBartDecoder(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([RetrieveBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,

        retrieve_hidden_states: Optional[torch.FloatTensor] = None,
        retrieve_attention_mask: Optional[torch.LongTensor] = None,

        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            retrieve_attention_mask = _expand_mask(retrieve_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )
        
        # print(hidden_states.size())
        # print(attention_mask.size())

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    (encoder_hidden_states, retrieve_hidden_states),
                    (encoder_attention_mask, retrieve_attention_mask),
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=(encoder_hidden_states, retrieve_hidden_states),
                    encoder_attention_mask=(encoder_attention_mask, retrieve_attention_mask),
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
    
class ContrastiveHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        inner_dim: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, 1)

    def forward(self, hidden_states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.avg_pool(hidden_states, masks)
        hidden_states = torch.sigmoid(hidden_states)
        return hidden_states

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

class ContrastiveBartModel(BartModel):
    def __init__(self, config: BartConfig, tokenizer: BartTokenizer):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = RetrieveBartEncoder(config, tokenizer, self.shared)
        self.decoder = RetrieveBartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,

        max_hist: Optional[int]=1,
        hist_l: Optional[torch.LongTensor] = None,

        retrieve_ids: Optional[torch.LongTensor] = None,
        retrieve_attention_mask: Optional[torch.Tensor] = None,

        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:

            modified_attention = torch.clone(attention_mask[:,2:])

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,

                max_hist=max_hist,
                hist_l=hist_l,

                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            max_l = 30
            encoder_cls = encoder_outputs.last_hidden_state[:,1,:]
            seq_l = max_hist * 2 + 1

            encoder_outputs.last_hidden_state, batch_size, hidden_dim, mask_e = AttnClsGate(max_l, self.encoder.mask_id, input_ids.device, encoder_outputs.last_hidden_state, modified_attention, hist_l, self.encoder.embed_tokens, self.encoder.gate_alpha, encoder_cls, seq_l, self.encoder.attention_cls_alpha)

            modified_r_attention = torch.clone(retrieve_attention_mask[:,1:-1])

            retrieve_outputs = self.encoder(
                input_ids=retrieve_ids,
                attention_mask=retrieve_attention_mask,

                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            max_r = 28
            encoder_outputs.retrieve_last_hidden_states = AttnGate(encoder_cls, batch_size, hidden_dim, mask_e, max_r, input_ids.device, retrieve_outputs.last_hidden_state, modified_r_attention, self.encoder.gate_beta, self.encoder.attention_cls)


        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, RetrieveBaseModelOutput):
            encoder_outputs = RetrieveBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                retrieve_last_hidden_states=retrieve_outputs[0],
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,

            retrieve_hidden_states = encoder_outputs.retrieve_last_hidden_states,
            retrieve_attention_mask = retrieve_attention_mask,

            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return ContrastiveSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            retrieve_last_hidden_states=encoder_outputs.retrieve_last_hidden_states
        )


class ContrastiveBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, tokenizer: BartTokenizer):
        super().__init__(config)
        self.model = ContrastiveBartModel(config, tokenizer)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.contrastive_head = ContrastiveHead(config.d_model, config.dropout)
        self.d_model = config.d_model

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,

        max_hist: Optional[int]=1,
        hist_l: Optional[torch.LongTensor] = None,

        retrieve_ids: Optional[torch.LongTensor] = None,
        retrieve_attention_mask: Optional[torch.Tensor] = None,

        neg_ids: Optional[torch.LongTensor] = None,
        neg_num_total: Optional[int]=1,

        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,

            max_hist=max_hist,
            hist_l=hist_l,

            retrieve_ids = retrieve_ids,
            retrieve_attention_mask = retrieve_attention_mask,

            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        contrastive_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            contrastive_loss = self.contrastive(
                outputs=outputs,
                labels=labels,
                attention_mask=attention_mask,
                retrieve_attention_mask=retrieve_attention_mask,
                neg_ids=neg_ids,
                expand_size=neg_num_total,

                decoder_attention_mask=decoder_attention_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                loss_fct=loss_fct,
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,contrastive_loss,) + output) if masked_lm_loss is not None else output

        return ContrastiveSeq2SeqLMOutput(
            loss=masked_lm_loss,
            cl_loss=contrastive_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
    def contrastive(
        self,
        outputs,
        labels,
        attention_mask,
        retrieve_attention_mask,
        neg_ids,
        expand_size,

        decoder_attention_mask,
        decoder_head_mask,
        cross_attn_head_mask,
        past_key_values,
        decoder_inputs_embeds,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
        loss_fct
    ):
        pos_label_mask = labels != self.tokenizer.pad_token_id
        pos_emb = self.contrastive_head(outputs.last_hidden_state, pos_label_mask)

        decoder_input_ids = shift_tokens_right(
            neg_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )
        bs = labels.size(0)

        expanded_return_idx = (
            torch.arange(attention_mask.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(attention_mask.device)
        )
        encoder_outputs = RetrieveBaseModelOutput(
            last_hidden_state = outputs.encoder_last_hidden_state,
            hidden_states=outputs.encoder_hidden_states,
            attentions=outputs.encoder_attentions,
            retrieve_last_hidden_states=outputs.retrieve_last_hidden_states
        )

        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        )
        encoder_outputs["retrieve_last_hidden_states"] = encoder_outputs.retrieve_last_hidden_states.index_select(
            0, expanded_return_idx.to(encoder_outputs.retrieve_last_hidden_states.device)
        )

        attention_mask = attention_mask.index_select(0, expanded_return_idx).to(attention_mask.device)
        retrieve_attention_mask = retrieve_attention_mask.index_select(0, expanded_return_idx).to(retrieve_attention_mask.device)
        
        decoder = self.get_decoder()


        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,

            retrieve_hidden_states = encoder_outputs.retrieve_last_hidden_states,
            retrieve_attention_mask = retrieve_attention_mask,

            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        neg_label_mask = neg_ids != self.tokenizer.pad_token_id
        neg_emb = self.contrastive_head(decoder_outputs.last_hidden_state, neg_label_mask).view(bs, expand_size)
        # print(neg_emb.size())


        all_logit = torch.cat([pos_emb,neg_emb], dim=1)
        l = torch.zeros([bs], dtype=torch.long, device=neg_emb.device)
        cl_loss = loss_fct(all_logit, l)
        return cl_loss
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        retrieve_attention_mask=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "retrieve_attention_mask":retrieve_attention_mask
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        
        attention_mask = model_kwargs['attention_mask']
        hist_l = model_kwargs['hist_l']
        max_hist = model_kwargs['max_hist']
        
        modified_attention = torch.clone(attention_mask[:,2:])
        encoder_outputs = encoder(
                input_ids=inputs_tensor,
                attention_mask=attention_mask,
                hist_l=hist_l,
                return_dict=True,)

        retrieve_ids = encoder_kwargs["retrieve_ids"]
        retrieve_attention_mask = encoder_kwargs["retrieve_attention_mask"]


        max_l = 30
        encoder_cls = encoder_outputs.last_hidden_state[:,1,:]
        seq_l = max_hist * 2 + 1

        encoder_outputs.last_hidden_state, batch_size, hidden_dim, mask_e = AttnClsGate(max_l, encoder.mask_id, inputs_tensor.device, encoder_outputs.last_hidden_state, modified_attention, hist_l, encoder.embed_tokens, encoder.gate_alpha, encoder_cls, seq_l, encoder.attention_cls_alpha)


        modified_r_attention = torch.clone(retrieve_attention_mask[:,1:-1])

        retrieve_outputs = encoder(
                input_ids=retrieve_ids,
                attention_mask=retrieve_attention_mask,
                return_dict=True,
            )

        max_r = 28
        encoder_outputs.retrieve_last_hidden_states = AttnGate(encoder_cls, batch_size, hidden_dim, mask_e, max_r, inputs_tensor.device, retrieve_outputs.last_hidden_state, modified_r_attention, encoder.gate_beta, encoder.attention_cls)

        encoder_outputs.retrieve_last_hidden_states = retrieve_outputs.last_hidden_state

        model_kwargs["encoder_outputs"]: RetrieveBaseModelOutput = encoder_outputs

        return model_kwargs
    

    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        retrieve_attention_mask: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)
            model_kwargs["retrieve_attention_mask"] = retrieve_attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            encoder_outputs["retrieve_last_hidden_states"] = encoder_outputs.retrieve_last_hidden_states.index_select(
                0, expanded_return_idx.to(encoder_outputs.retrieve_last_hidden_states.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs


