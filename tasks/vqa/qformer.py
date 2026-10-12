"""Task-side learned queries using the ported BLIP-2 Q-Former layers."""
from types import SimpleNamespace

import torch
from torch import nn

from .config import QFORMER_DEFAULTS, validate_fusion_config
from .qformer_core import BertLayer


UPSTREAM = dict(
    model="BLIP-2", component="Q-Former attention/query FFN",
    repository="https://github.com/salesforce/LAVIS",
    revision="05cb670f47296522e24e7727e366b4a997d28582",
    paper="https://proceedings.mlr.press/v202/li23q.html",
    source_sha256="9d1375ffed630df9449e7e5e69072038bf20af5bd0e1cc22ae1f2337b586f95e",
    pretrained_qformer_weights=False,
    adaptation="YOLO spatial tokens; one pooled CLIP question token; task answer heads",
)


class QuestionQueryFormer(nn.Module):
    """Query/text self-attention, query-only visual cross-attention and FFNs.

    The pooled CLIP question is one text embedding. This adapter does not claim
    to recover word-level BERT features or pretrained BLIP-2 generation.
    """

    def __init__(self, config):
        super().__init__()
        if validate_fusion_config(config) != "blip2_qformer":
            raise ValueError("QuestionQueryFormer requires blip2_qformer configuration")
        options = {**QFORMER_DEFAULTS, **config}
        width = options["hidden_dim"]
        self.query_count = options["qformer_queries"]
        core_config = SimpleNamespace(
            hidden_size=width, encoder_width=width,
            num_attention_heads=options["qformer_heads"],
            intermediate_size=width * options["qformer_ffn_multiplier"],
            hidden_act="gelu", layer_norm_eps=1e-12,
            hidden_dropout_prob=options["qformer_dropout"],
            attention_probs_dropout_prob=options["qformer_dropout"],
            position_embedding_type="absolute", chunk_size_feed_forward=0,
            add_cross_attention=True,
            cross_attention_freq=options["qformer_cross_attention_freq"],
        )
        self.query_tokens = nn.Parameter(torch.empty(1, self.query_count, width))
        self.embedding_norm = nn.LayerNorm(width, eps=core_config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(core_config.hidden_dropout_prob)
        self.layers = nn.ModuleList(
            BertLayer(core_config, index) for index in range(options["qformer_layers"])
        )
        self.apply(self._init_weights)
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, visual_tokens, question):
        queries = self.query_tokens.expand(visual_tokens.shape[0], -1, -1)
        sequence = torch.cat((queries, question.unsqueeze(1)), dim=1)
        sequence = self.embedding_dropout(self.embedding_norm(sequence))
        for layer in self.layers:
            sequence = layer(
                sequence, encoder_hidden_states=visual_tokens,
                query_length=self.query_count,
            )[0]
        return sequence[:, :self.query_count], sequence[:, self.query_count]
