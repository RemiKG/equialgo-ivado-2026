import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from data_contract import ROOT,TARGET,load_data,groups,allocate,verify_predictions
from merit_inference import (design,scenario_probabilities,equal_opportunity_allocate,
                            infer_models,read_prior_files)
from build_audit_v2 import load_ensemble
from prediction_history import archive_prediction,preserve_existing


class MeritTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h,cls.e=load_data();cls.model=load_ensemble("structured_merit")
        cls.pred=pd.read_csv(ROOT/"archive/v2_inverse_merit_9223/predictions.csv")
        cls.p=cls.model.probabilities(cls.e)

    def test_exact_budget_and_official_schema(self):
        report=verify_predictions(self.e,self.pred)
        self.assertEqual(report["grants"],1600)
        self.assertIsNone(report["reference_accuracy"])

    def test_saved_ensemble_reproduces_the_primary_file(self):
        decisions=equal_opportunity_allocate(self.p,groups(self.e),self.e.id_candidat.to_numpy())
        np.testing.assert_array_equal(decisions,self.pred[TARGET])

    def test_raw_geographic_labels_and_program_do_not_change_scores(self):
        e=self.e.iloc[:40].copy();p=self.model.probabilities(e)
        e["region_administrative"]="Cote-Nord";e["code_postal_3"]="G4R";e["programme_etudes"]="Genie"
        np.testing.assert_allclose(p,self.model.probabilities(e),atol=1e-12)

    def test_published_normative_directions_are_monotonic(self):
        e=self.e.iloc[:60];p=self.model.probabilities(e)
        for column,amount in [("cote_r_equivalent",.2),("heures_travail_semaine",1),
                              ("revenu_familial_estime",-1000),("premiere_generation_universitaire",1),
                              ("distance_domicile_campus_km",10)]:
            changed=e.copy();changed[column]+=amount
            self.assertTrue(np.all(self.model.probabilities(changed)>=p-1e-12),column)

    def test_single_candidate_scoring_is_consistent(self):
        for i in [0,38,193,3001]:
            self.assertAlmostEqual(self.model.probabilities(self.e.iloc[[i]])[0],self.p[i],places=12)

    def test_weights_normalize_and_prior_is_enforced(self):
        self.assertAlmostEqual(self.model.weights.sum(),1,places=12)
        self.assertTrue((self.model.weights>=0).all())
        self.assertTrue((self.model.weights[self.model.noise>.30]==0).all())
        self.assertTrue((self.model.coefficients[:,2]<=0).all())
        self.assertTrue(np.isfinite(self.p).all())

    def test_scenario_capacity_calibration(self):
        x=design(self.e)
        p,t=scenario_probabilities(x,self.model.coefficients[:8],self.model.noise[:8])
        np.testing.assert_allclose(p.mean(axis=0),.4,atol=1e-6)

    def test_row_permutation_does_not_change_allocation(self):
        order=np.random.default_rng(15).permutation(len(self.e))
        a=equal_opportunity_allocate(self.p,groups(self.e),self.e.id_candidat.to_numpy())
        b=equal_opportunity_allocate(self.p[order],groups(self.e)[order],self.e.id_candidat.to_numpy()[order])
        np.testing.assert_array_equal(a[order],b)

    def test_new_model_changes_are_preserved_and_not_scored_as_v1(self):
        old,_=read_prior_files(self.e)
        self.assertEqual(int((old[TARGET]!=self.pred[TARGET]).sum()),194)
        summary=json.loads((ROOT/"archive/v2_inverse_merit_9223/artifacts/audit_summary.json").read_text())
        self.assertIsNone(summary["v2_official_accuracy"])
        self.assertEqual(summary["v1_user_reported_accuracy"],.9463)

    def test_invalid_income_is_rejected(self):
        e=self.e.iloc[:2].copy();e.loc[e.index[0],"revenu_familial_estime"]=0
        with self.assertRaises(ValueError):design(e)


class OptimizationTests(unittest.TestCase):
    def test_opportunity_optimizer_matches_binary_enumeration(self):
        rng=np.random.default_rng(17);g=np.array([0]*6+[1]*4);ids=np.arange(10)
        for _ in range(10):
            p=rng.uniform(.1,.9,10);tol=.25
            chosen=equal_opportunity_allocate(p,g,ids,tolerance=tol)
            best=-np.inf
            for ix in itertools.combinations(range(10),4):
                y=np.zeros(10);y[list(ix)]=1
                gap=abs((y[g==0]*p[g==0]).sum()/p[g==0].sum()-(y[g==1]*p[g==1]).sum()/p[g==1].sum())
                if gap<=tol:best=max(best,p@y)
            self.assertAlmostEqual(p@chosen,best,places=12)
            self.assertEqual(chosen.sum(),4)

    def test_invalid_probabilities_groups_and_budget(self):
        p=np.linspace(.1,.9,10);g=np.array([0]*5+[1]*5);ids=np.arange(10)
        for bad in [np.full(10,np.nan),np.full(10,1.1),np.full(10,-.1)]:
            with self.assertRaises(ValueError):equal_opportunity_allocate(bad,g,ids)
        with self.assertRaises(ValueError):equal_opportunity_allocate(p,np.full(10,2),ids,tolerance=.1)
        with self.assertRaises(ValueError):equal_opportunity_allocate(p,g,ids,rate=.9)

    def test_history_is_append_only_and_keeps_source_dependencies(self):
        _,e=load_data();p=pd.read_csv(ROOT/"predictions.csv")
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/"history"
            first=archive_prediction(e,p,"v2-test",history_dir=directory)
            contents=(first/"predictions.csv").read_bytes()
            archive_prediction(e,p,"v2-repeat",history_dir=directory)
            preserve_existing(first/"predictions.csv",e,directory)
            self.assertEqual(len(list(directory.glob("*/metadata.json"))),2)
            self.assertEqual((first/"predictions.csv").read_bytes(),contents)
            self.assertTrue((first/"merit_inference.py").exists())
            self.assertTrue((first/"data_contract.py").exists())
            m=json.loads((first/"metadata.json").read_text())
            self.assertEqual(m["csv_sha256"],hashlib.sha256(contents).hexdigest())


if __name__=="__main__":unittest.main()
