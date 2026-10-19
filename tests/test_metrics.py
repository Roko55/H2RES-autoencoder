"""Check that averaging or reference construction cannot create an easy score."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from analysis.compare_profiles import summarize_paths, baseline_profile


class MetricDefinitions(unittest.TestCase):
    def reference(self):
        return {"scales":np.ones(3),"night":np.zeros(8760,dtype=bool),"low":-.5,"high":.5}

    def test_ensemble_smoothing_does_not_replace_individual_errors(self):
        truth=np.zeros((5,3,8760))
        summary,individual,central,*_=summarize_paths("example",{101:np.ones_like(truth),102:-np.ones_like(truth)},truth,self.reference())
        self.assertTrue(all(row["calendar_MAE"]==1. for row in individual))
        self.assertTrue(all(row["MAE"]==0. for row in central))

    def test_pooled_RMSE_precedes_square_root(self):
        truth=np.zeros((5,3,8760))
        summary,*_=summarize_paths("example",{101:np.ones_like(truth),102:3*np.ones_like(truth)},truth,self.reference())
        for channel in summary["channels"].values():
            self.assertAlmostEqual(channel["pooled_calendar_RMSE"],np.sqrt(5.))

    def test_nonfinite_paths_and_incomplete_years_are_rejected(self):
        truth=np.zeros((5,3,8760));invalid=truth.copy();invalid[0,0,0]=np.nan
        for value in (invalid,truth[:4]):
            with self.assertRaises(ValueError):summarize_paths("example",{101:value},truth,self.reference())

    def test_offshore_definition_changes_capacity_weighted_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            pd.DataFrame({"System":["HR"]*3,"technology":["PHOT","WTON","WTON"],"unit_name":["Solar","Wind","Offshore"],"cap_mw":[2.,1.,3.]}).to_csv(root/"genco.csv",index=False)
            pd.DataFrame({"period":np.arange(8760),"Solar":np.full(8760,.1),"Wind":np.full(8760,.2),"Offshore":np.full(8760,.8)}).to_csv(root/"ncre.csv",index=False)
            train=np.full((2,3,8760),12.)
            both,_=baseline_profile(root/"genco.csv",root/"ncre.csv",train,onshore_only=False)
            land,_=baseline_profile(root/"genco.csv",root/"ncre.csv",train,onshore_only=True)
            np.testing.assert_allclose(both[1],.65);np.testing.assert_allclose(land[1],.2)
            np.testing.assert_allclose(both[2],12.)
