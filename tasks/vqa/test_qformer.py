"""Compare the port with pinned official code and verify real task integration."""
import ast
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .config import (ARCHITECTURE_IDS, DEFAULTS, QFORMER_ARCHITECTURE_ID, QFORMER_DEFAULTS,
                     architecture_id, config_from_file, read_json, sha256, write_json)
from .inference import load_bundle
from .model import VQAHead
from .qformer import QuestionQueryFormer, UPSTREAM
from .qformer_core import BertLayer
from .test_contract import labels
from .train import inference_config, save_bundle


def reference_classes():
    source = Path(__file__).parent / "third_party/lavis/Qformer.upstream.txt"
    if sha256(source) != UPSTREAM["source_sha256"]:
        raise AssertionError("Pinned upstream Q-Former source changed")
    names = {"BertSelfAttention", "BertSelfOutput", "BertAttention",
             "BertIntermediate", "BertOutput", "BertLayer"}
    nodes = [n for n in ast.parse(source.read_text(encoding="utf-8")).body
             if isinstance(n, ast.ClassDef) and n.name in names]
    assert len(nodes) == 6
    # Execute the independent official classes, without importing its full stack.
    def unchunked(fn, size, dim, *args):
        assert size == 0
        return fn(*args)
    namespace = dict(torch=torch, nn=nn, math=math, ACT2FN={"gelu": F.gelu},
                     apply_chunking_to_forward=unchunked)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class QueryFormerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.config = {**DEFAULTS, **QFORMER_DEFAULTS,
                       "fusion_backend": "blip2_qformer", "hidden_dim": 32,
                       "grids": [2, 1, 1], "dropout": 0., "qformer_dropout": 0.,
                       "qformer_queries": 3}
        self.labels = labels()

    def test_six_class_source_port(self):
        upstream = reference_classes()
        source = Path(__file__).parent / "third_party/lavis/Qformer.upstream.txt"
        port = Path(__file__).parent / "qformer_core.py"
        expected = {n.name: n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name in upstream}
        actual = {n.name: n for n in ast.parse(port.read_text()).body if isinstance(n, ast.ClassDef)}
        expected["BertAttention"].body = [n for n in expected["BertAttention"].body
            if not isinstance(n, ast.FunctionDef) or n.name != "prune_heads"]
        for name in actual:
            self.assertEqual(ast.dump(actual[name]), ast.dump(expected[name]), name)

    def test_official_layer_outputs_and_gradients(self):
        reference = reference_classes()
        config = SimpleNamespace(hidden_size=16, encoder_width=16, num_attention_heads=4,
            intermediate_size=64, hidden_act="gelu", layer_norm_eps=1e-12,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
            position_embedding_type="absolute", chunk_size_feed_forward=0,
            add_cross_attention=True, cross_attention_freq=2)
        for index in (0, 1):
            actual = BertLayer(config, index).eval()
            expected = reference["BertLayer"](config, index).eval()
            expected.load_state_dict(actual.state_dict(), strict=True)
            text = torch.randn(2, 4, 16, requires_grad=True)
            visual = torch.randn(2, 5, 16, requires_grad=True)
            ref_text = text.detach().clone().requires_grad_(True)
            ref_visual = visual.detach().clone().requires_grad_(True)
            mask = torch.zeros(2, 1, 1, 5); mask[:, :, :, -1] = -10000.
            output = actual(text, encoder_hidden_states=visual,
                encoder_attention_mask=mask, query_length=3, output_attentions=True)[0]
            target = expected(ref_text, encoder_hidden_states=ref_visual,
                encoder_attention_mask=mask, query_length=3, output_attentions=True)[0]
            torch.testing.assert_close(output, target, rtol=0, atol=0)
            output.square().sum().backward(); target.square().sum().backward()
            torch.testing.assert_close(text.grad, ref_text.grad, rtol=0, atol=0)
            if index == 0:
                torch.testing.assert_close(visual.grad, ref_visual.grad, rtol=0, atol=0)
            for (name, parameter), (_, reference_parameter) in zip(actual.named_parameters(), expected.named_parameters()):
                if parameter.grad is not None:
                    torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)

    def test_query_and_language_gradients_and_input_sensitivity(self):
        head = VQAHead(self.config, self.labels)
        visual = [torch.randn(2, c, g, g) for c, g in zip(self.config["channels"], self.config["grids"])]
        question = torch.randn(2, 768)
        types, answers = head(visual, question)
        (types.square().mean() + sum(value.square().mean() for value in answers.values())).backward()
        for name, parameter in head.query_former.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
        self.assertGreater(float(head.query_former.query_tokens.grad.norm()), 0)
        self.assertTrue(all(float(p.weight.grad.norm()) > 0 for p in head.projections))
        head.eval()
        result = head(visual, question)[1]["color"]
        different_question = head(visual, question.roll(1, 0))[1]["color"]
        different_image = head([v.roll(1, 0) for v in visual], question)[1]["color"]
        self.assertGreater(float((result - different_question).abs().max()), 0)
        self.assertGreater(float((result - different_image).abs().max()), 0)

    def test_queries_only_cross_attention(self):
        former = QuestionQueryFormer(self.config).eval()
        seen = []
        hook = former.layers[0].crossattention.self.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].shape[1]))
        try:
            queries, question = former(torch.randn(2, 6, 32), torch.randn(2, 32))
        finally:
            hook.remove()
        self.assertEqual(seen, [3])
        self.assertEqual(tuple(queries.shape), (2, 3, 32))
        self.assertEqual(tuple(question.shape), (2, 32))
        self.assertFalse(former.layers[1].has_cross_attention)

    def test_v2_save_reload_and_architecture_mismatch(self):
        head = VQAHead(self.config, self.labels).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = []
            asset_paths = {}
            for key in ("visual_backbone", "text_encoder"):
                path = root / (key + ".bin"); path.write_bytes(key.encode())
                assets.append(dict(asset_id=key, sha256=sha256(path)))
                asset_paths[key] = path
            save_bundle(root, head, self.config, self.labels, 1, 0., assets, {"python": "test"})
            manifest, config, _, loaded, _ = load_bundle(root, asset_paths)
            self.assertEqual(manifest["architecture_id"], QFORMER_ARCHITECTURE_ID)
            self.assertEqual(config["qformer_queries"], 3)
            self.assertFalse(manifest["fusion_source"]["pretrained_qformer_weights"])
            for name, tensor in head.state_dict().items():
                torch.testing.assert_close(tensor, loaded.state_dict()[name], rtol=0, atol=0)
            manifest["architecture_id"] = "hfsa_frozen_spatial_vqa_v1"
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "architecture/version"):
                load_bundle(root, asset_paths)

    def test_defaults_always_build_queryformer_and_version_two(self):
        head = VQAHead(self.config, self.labels)
        self.assertTrue(hasattr(head, "query_former"))
        self.assertEqual(inference_config(DEFAULTS)["fusion_backend"], "blip2_qformer")
        self.assertEqual(ARCHITECTURE_IDS, (QFORMER_ARCHITECTURE_ID,))
        self.assertEqual(architecture_id(DEFAULTS), QFORMER_ARCHITECTURE_ID)
        for key, value in QFORMER_DEFAULTS.items():
            self.assertEqual(DEFAULTS[key], value)
            self.assertEqual(inference_config(DEFAULTS)[key], value)
        head.load_state_dict(head.state_dict(), strict=True)

    def test_missing_queryformer_state_cannot_be_accepted_as_new_model(self):
        head = VQAHead(self.config, self.labels)
        legacy_state = {key: value for key, value in head.state_dict().items()
                        if not key.startswith("query_former.")}
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            head.load_state_dict(legacy_state, strict=True)

    def test_invalid_configs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for key, value in (("fusion_backend", "wrong"), ("fusion_backend", "spatial_attention"), ("qformer_queries", 0),
                               ("qformer_heads", 5), ("qformer_dropout", 1.0)):
                write_json(path, {**self.config, key: value})
                with self.assertRaises(ValueError):
                    config_from_file(path)


if __name__ == "__main__":
    unittest.main()
