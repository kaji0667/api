import unittest
from tasks.detection.text_consensus import consensus_selection


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.records=[{"target_score":.2,"contrast_score":-.1},
                      {"target_score":.3,"contrast_score":.05}]

    def test_agreement_requires_both_visual_identity_measures(self):
        self.assertEqual(consensus_selection(0,1,1,self.records),(1,"text_consensus_selected"))
        for key in ("target_score","contrast_score"):
            data=[dict(r) for r in self.records]
            data[1][key]=data[0][key]
            self.assertEqual(consensus_selection(0,1,1,data)[0],0)

    def test_disagreement_and_missing_winners_keep_original(self):
        for first,second in ((0,1),(1,0),(None,1),(1,None),(True,1),(2,2)):
            self.assertEqual(consensus_selection(0,first,second,self.records)[0],0)

    def test_missing_or_nonfinite_identity_cannot_override(self):
        for invalid in (None,float("nan"),float("inf"),float("-inf")):
            for key in ("target_score","contrast_score"):
                data=[dict(r) for r in self.records]
                data[1][key]=invalid
                self.assertEqual(consensus_selection(0,1,1,data)[0],0)

    def test_empty_pool_stays_empty(self):
        self.assertEqual(consensus_selection(None,None,None,[])[0],None)


if __name__=="__main__":
    unittest.main()
