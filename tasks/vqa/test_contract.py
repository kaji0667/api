"""Focused checks for semantic budgets, gradient flow and answer contracts."""
import unittest
import tempfile,json
from pathlib import Path
import torch
from .config import DEFAULTS,SUPPORTED,TYPES
from .data import normalize_answer,assert_split_isolation,question_type,comparison_intent
from .inference import decode_prediction,validate_modes
from .model import FrozenText,VQAHead
from .cache import FeatureCache
from .config import sha256

def labels():
    return dict(types=list(TYPES),supported=list(SUPPORTED),answers={t:["no","yes"] if t=="existence" else ["gray","green"] for t in SUPPORTED},aliases={t:{} for t in SUPPORTED})

class VQAContractTests(unittest.TestCase):
    def setUp(self):
        self.labels=labels();self.t=torch.full((len(TYPES),),-9.);self.t[1]=9.
        self.a={t:torch.tensor([0.,3.]) for t in SUPPORTED}
    def test_output_formats(self):
        r=decode_prediction(self.t,self.a,self.labels,"What color is the roof?",choices={"roof-green":"green","roof-gray":"grey"},image_size=(512,256))
        self.assertEqual(r["answer"],"roof-green");self.assertEqual(r["evidence"]["target_boxes"],[])
        r=decode_prediction(self.t,self.a,self.labels,"What color is the roof?",allowed_answers=["Green","Grey"])
        self.assertEqual(r["answer"],"Green")
    def test_unmatched_is_explicit(self):
        r=decode_prediction(self.t,self.a,self.labels,"What color?",choices={"A":"purple","B":"yellow"})
        self.assertEqual(r["reason"],"choice_unmatched");self.assertIsNone(r["answer"])
    def test_partial_choices_use_learned_scores_and_original_keys(self):
        for options in ({"unknown":"purple","roof-gray":"grey","roof-green":"green"},
                        {"roof-green":"green","roof-gray":"grey","unknown":"purple"}):
            with self.subTest(options=options):
                r=decode_prediction(self.t,self.a,self.labels,"What color?",choices=options)
                self.assertEqual(r["status"],"ok");self.assertEqual(r["answer"],"roof-green")
                self.assertEqual(r["answer_text"],"green")
                self.assertEqual(r["choice_coverage"],dict(provided=3,matched=2,unmatched_keys=["unknown"]))
                self.assertAlmostEqual(r["confidence"],float(self.a["color"].softmax(-1)[1]))
    def test_partial_choices_score_only_allowed_known_candidates(self):
        r=decode_prediction(self.t,self.a,self.labels,"What color?",choices={"original-gray":"grey","unknown":"purple"})
        self.assertEqual(r["answer"],"original-gray")
        self.assertAlmostEqual(r["confidence"],float(self.a["color"].softmax(-1)[0]))
    def test_partial_choices_preserve_alias_ambiguity_rejection(self):
        with self.assertRaises(ValueError):
            decode_prediction(self.t,self.a,self.labels,"What color?",choices={"A":"grey","B":"gray","unknown":"purple"})
    def test_enum_still_requires_full_vocabulary_coverage(self):
        r=decode_prediction(self.t,self.a,self.labels,"What color?",allowed_answers=["Green","Purple"])
        self.assertEqual(r["reason"],"choice_unmatched");self.assertIsNone(r["answer"])
        r=decode_prediction(self.t,self.a,self.labels,"What color?",choices={"A":"gray","B":"green"})
        self.assertNotIn("choice_coverage",r)
    def test_invalid_modes(self):
        for ch,en in [({},None),({"A":""},None),(None,[]),({"A":"x"},["x"])]:
            with self.assertRaises(ValueError):validate_modes(ch,en)
    def test_unsupported_guard(self):
        for q in ["How many planes?","Where is the red roof?","Which object is larger?"]:
            r=decode_prediction(self.t,self.a,self.labels,q);self.assertEqual(r["status"],"unsupported");self.assertIsNone(r["answer"])
    def test_multicolor_preserved(self):
        self.assertEqual(normalize_answer(" Red and Blue. "),"red and blue")
        self.assertNotEqual(normalize_answer("red and blue"),"red")
    def test_size_qualifiers_preserve_attribute_intent(self):
        for q in ["What color is the smaller airplane?","What shape is the larger roof?"]:
            self.assertFalse(comparison_intent(q))
            self.assertEqual(decode_prediction(self.t,self.a,self.labels,q)["status"],"ok")
        self.assertEqual(question_type("object color","What color is the smaller airplane?"),"color")
        self.assertEqual(question_type("object position","Where is the larger boat?"),"position")
        for q in ["Which object is larger?","Is the red boat smaller than the white boat?","Are the two courts the same size?"]:
            self.assertTrue(comparison_intent(q))
    def test_parent_leakage(self):
        with self.assertRaises(ValueError):assert_split_isolation([{"image_id":"x_1","parent_image_id":"x"}],[{"image_id":"x_2","parent_image_id":"x"}])
    def test_cache_binding_allows_only_audited_type_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original=root/'original.jsonl';final=root/'final.jsonl'
            row=dict(sample_id='val:x:0',image_id='x',image_path='x.jpg',question='What color is the smaller airplane?',
                     original_type='object color',normalized_answer='gray',question_type='comparison')
            original.write_text(json.dumps(row)+'\n',encoding='utf-8')
            obj=FeatureCache.__new__(FeatureCache);obj.metadata={'data_sha256':{str(original):sha256(original)}}
            row['question_type']='color';final.write_text(json.dumps(row)+'\n',encoding='utf-8')
            self.assertEqual(obj.verify_datasets([final])['verified_datasets'][0]['type_only_corrections'],1)
            row['normalized_answer']='green';final.write_text(json.dumps(row)+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):obj.verify_datasets([final])
    def test_chunk_retains_all_tokens(self):
        import open_clip
        obj=FrozenText.__new__(FrozenText);obj.tokenizer=open_clip.get_tokenizer("ViT-L-14")
        text="The small red car on the far left beside the road. "*20
        xs,ns=obj.chunks(text)
        self.assertGreater(len(xs),1)
        self.assertEqual([int(i) for x,n in zip(xs,ns) for i in x[1:1+n]],obj.tokenizer.encode(text))
        self.assertTrue(all(int(x[1+n])==obj.tokenizer.eot_token_id for x,n in zip(xs,ns)))
    def test_question_and_spatial_gradients(self):
        torch.manual_seed(42);c={**DEFAULTS,"grids":[2,1,1],"hidden_dim":32,"dropout":0.}
        h=VQAHead(c,self.labels)
        v=[torch.randn(4,ch,g,g) for ch,g in zip(c["channels"],c["grids"])];q=torch.randn(4,768)
        tl,al=h(v,q);loss=tl.square().mean()+al["color"].square().mean();loss.backward()
        self.assertGreater(float(h.question[0].weight.grad.norm()),0)
        self.assertTrue(all(float(p.weight.grad.norm())>0 for p in h.projections))
        self.assertGreater(float(h.query_former.query_tokens.grad.norm()),0)
        h.eval();r1=h(v,q)[1]["color"];r2=h(v,q.roll(1,0))[1]["color"]
        self.assertGreater(float((r1-r2).abs().max()),0)

if __name__ == "__main__":unittest.main()
