"""Release integrity and the boundary of the cohort-specific calibration."""
import json
import unittest
import pandas as pd
from model_corrige import (ROOT,TARGET,load_data,predict,champion_probabilities,
                           csv_digest,verify_predictions)


class ChampionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.evaluation=load_data()
        cls.release=json.loads((ROOT/'artifacts/active_prediction.json').read_text())

    def test_reproduces_measured_champion_and_validates_budget(self):
        prediction=predict(self.evaluation)
        self.assertEqual(csv_digest(prediction),self.release['csv_sha256'])
        self.assertEqual(csv_digest(pd.read_csv(ROOT/'predictions.csv')),self.release['csv_sha256'])
        self.assertEqual(verify_predictions(self.evaluation,prediction)['grants'],1600)

    def test_row_order_does_not_change_decisions(self):
        shuffled=self.evaluation.sample(frac=1,random_state=19)
        a=predict(self.evaluation).set_index('id_candidat')[TARGET].sort_index()
        b=predict(shuffled).set_index('id_candidat')[TARGET].sort_index()
        pd.testing.assert_series_equal(a,b)

    def test_new_or_modified_applications_require_new_validation(self):
        changed=self.evaluation.copy()
        changed.loc[0,'cote_r_equivalent']+=.01
        with self.assertRaisesRegex(ValueError,'unchanged evaluation cohort'):
            champion_probabilities(changed)
        with self.assertRaisesRegex(ValueError,'unchanged evaluation cohort'):
            champion_probabilities(self.evaluation.iloc[:-1])

    def test_tampered_release_is_rejected(self):
        release={**self.release,'parameter_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'parameters have changed'):
            champion_probabilities(self.evaluation,release)

    def test_release_has_an_actual_recorded_result(self):
        results=[json.loads(line) for line in (ROOT/'predictions_history/official_results.jsonl').read_text().splitlines()]
        match=[r for r in results if r['snapshot']==self.release['snapshot']][-1]
        self.assertEqual(match['accuracy'],self.release['accuracy'])
        self.assertEqual(match['macro_f1'],self.release['macro_f1'])


if __name__=='__main__':unittest.main()
