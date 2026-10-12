"""Focused checks for semantic budgets, gradient flow and answer contracts."""
import unittest
import tempfile,json
from pathlib import Path
from unittest.mock import patch
import torch
from .config import DEFAULTS,SUPPORTED,TYPES
from .data import normalize_answer,assert_split_isolation,question_type,comparison_intent,read_rows,build_labels
from .inference import decode_prediction,validate_modes,load_bundle
from .model import FrozenText,VQAHead
from .cache import FeatureCache
from .config import sha256,ARCHITECTURE_ID,FORMAT_VERSION

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
        for q in ["这张图里有多少架飞机？","Compare the first and second image.","What has changed between the two images?"]:
            r=decode_prediction(self.t,self.a,self.labels,q)
            self.assertEqual(r["status"],"unsupported")
            self.assertIsNone(r["answer"])
    def test_multicolor_preserved(self):
        self.assertEqual(normalize_answer(" Red and Blue. "),"red and blue")
        self.assertNotEqual(normalize_answer("red and blue"),"red")
    def test_delivery_normalization_covers_numbers_shapes_and_preserved_suffixes(self):
        examples={"Zero":"0","One":"1","Twenty":"20","The same size.":"same size",
                  "Bottom-right":"bottom right","Semi-circle":"semicircle",
                  "semi circle":"semicircle","Ships":"ship","Various":"various",
                  "Tennis":"tennis","Status":"status","Grass":"grass",
                  "The grey ships.":"gray ship"}
        for original,expected in examples.items():
            with self.subTest(answer=original):
                self.assertEqual(normalize_answer(original),expected)
    def test_read_rows_recomputes_stale_answers_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"data.jsonl"
            rows=[dict(sample_id="test:1",image_id="one",image_path="one.jpg",
                       question="How many ships?",question_type="quantity",
                       answer="The Two Ships.",normalized_answer="obsolete"),
                  dict(sample_id="test:2",image_id="two",image_path="two.jpg",
                       question="Where is the ship?",question_type="position",
                       normalized_answer="Bottom-right")]
            path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
            original=path.read_bytes();loaded=read_rows(path)
            self.assertEqual([row["normalized_answer"] for row in loaded],["2 ship","bottom right"])
            self.assertEqual(loaded[0]["answer"],"The Two Ships.")
            self.assertEqual(path.read_bytes(),original)
    def test_training_labels_keep_singleton_answers_for_all_twelve_types(self):
        self.assertEqual(DEFAULTS["min_answer_count"],1)
        rows=[dict(question_type=t,answer="Ships",normalized_answer=normalize_answer("Ships"))
              for t in SUPPORTED]
        result=build_labels(rows,min_count=DEFAULTS["min_answer_count"])
        self.assertEqual(result["supported"],list(SUPPORTED))
        self.assertTrue(all(values==["ship"] for values in result["answers"].values()))
    def test_all_twelve_types_preserve_three_answer_formats(self):
        examples={"category":("What object is shown?",["ship","airplane"]),
                  "color":("What color is the roof?",["gray","green"]),
                  "shape":("What shape is the roof?",["square","circle"]),
                  "existence":("Is there an airplane?",["no","yes"]),
                  "scene":("What scene is shown?",["airport","harbor"]),
                  "quantity":("How many ships are there?",["1","2"]),
                  "position":("Where is the ship?",["left","right"]),
                  "size":("What size is the ship?",["small","large"]),
                  "direction":("Which direction does the airplane face?",["left","right"]),
                  "comparison":("Which object is larger?",["airplane","ship"]),
                  "relation":("Is the road inside or outside the park?",["inside","outside"]),
                  "other":("What else is shown?",["first","second"])}
        self.assertEqual(tuple(examples),SUPPORTED)
        vocabulary=dict(types=list(TYPES),supported=list(SUPPORTED),
                        answers={t:values for t,(_,values) in examples.items()},
                        aliases={t:{} for t in SUPPORTED})
        answer_logits={t:torch.tensor([0.,3.]) for t in SUPPORTED}
        for index,(kind,(question,values)) in enumerate(examples.items()):
            type_logits=torch.full((len(TYPES),),-9.);type_logits[index]=9.
            display=["One","Two"] if kind=="quantity" else values
            for mode,options,expected in (
                    ("short_text",{},values[1]),
                    ("single_choice",{"choices":{"original-low":display[0],"original-high":display[1]}},"original-high"),
                    ("enum",{"allowed_answers":display},display[1])):
                with self.subTest(question_type=kind,mode=mode):
                    result=decode_prediction(type_logits,answer_logits,vocabulary,question,**options)
                    self.assertEqual(result["status"],"ok")
                    self.assertEqual(result["question_type"],kind)
                    self.assertEqual(result["answer"],expected)
                    self.assertEqual(result["answer_text"],values[1])
    def test_all_twelve_answer_heads_are_built_and_receive_gradients(self):
        expected=("category","color","shape","existence","scene","quantity",
                  "position","size","direction","comparison","relation","other")
        self.assertEqual(SUPPORTED,expected);self.assertEqual(TYPES,expected)
        config={**DEFAULTS,"grids":[2,1,1],"hidden_dim":32,"dropout":0.}
        head=VQAHead(config,self.labels)
        visual=[torch.randn(2,ch,g,g) for ch,g in zip(config["channels"],config["grids"])]
        type_logits,answers=head(visual,torch.randn(2,768))
        self.assertEqual(tuple(type_logits.shape),(2,12));self.assertEqual(tuple(answers),expected)
        (type_logits.square().mean()+sum(value.square().mean() for value in answers.values())).backward()
        for kind in expected:
            with self.subTest(question_type=kind):
                self.assertEqual(tuple(answers[kind].shape),(2,2))
                self.assertGreater(float(head.answers[kind].weight.grad.norm()),0)
    def test_bundle_rejects_five_or_mismatched_heads_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"inference.json").write_text(json.dumps(DEFAULTS),encoding="utf-8")
            cases=[(list(SUPPORTED[:5]),self.labels),
                   (list(SUPPORTED),{**self.labels,"supported":list(SUPPORTED[:5])}),
                   (list(SUPPORTED),{**self.labels,"types":list(reversed(TYPES))}),
                   (list(SUPPORTED),{**self.labels,"answers":{t:self.labels["answers"][t] for t in SUPPORTED[:5]}})]
            for supported,vocabulary in cases:
                with self.subTest(supported=supported,label_keys=list(vocabulary["answers"])):
                    (root/"manifest.json").write_text(json.dumps(dict(format_version=FORMAT_VERSION,
                        architecture_id=ARCHITECTURE_ID,supported_question_types=supported,
                        file_sha256={},external_assets=[],configs=dict(inference="inference.json",labels="labels.json"))),encoding="utf-8")
                    (root/"labels.json").write_text(json.dumps(vocabulary),encoding="utf-8")
                    with patch("tasks.vqa.inference.torch.load") as load,patch("tasks.vqa.inference.VQAHead") as head:
                        with self.assertRaisesRegex(ValueError,"supported question types"):
                            load_bundle(root,{})
                        load.assert_not_called();head.assert_not_called()
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
