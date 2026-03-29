"""CSP evaluation utilities.

Adapted from FlowMM:
- flowmm/src/flowmm/old_eval/reconstruction_metrics.py
"""

from __future__ import annotations

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm


class RecEval:
    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def process_one(self, pred, gt, is_valid):
        if not is_valid:
            return None
        try:
            rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
            if rms_dist is None:
                return None
            return rms_dist[0]
        except Exception:
            return None

    def get_match_rate_and_rms(self):
        validity = [c1.valid and c2.valid for c1, c2 in zip(self.preds, self.gts)]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(self.process_one(self.preds[i], self.gts[i], validity[i]))

        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class RecEvalBatch(RecEval):
    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        if len(pred_crys) == 0:
            raise ValueError("pred_crys must contain at least one candidate list.")
        if any(len(batch) != len(gt_crys) for batch in pred_crys):
            raise ValueError("Each candidate list must match gt_crys length.")
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)
        self.all_rms_dis = None

    def _compute_all_rms(self) -> None:
        if self.all_rms_dis is not None:
            return
        self.all_rms_dis = np.full((self.batch_size, len(self.gts)), np.nan, dtype=np.float64)
        for i in tqdm(range(len(self.gts))):
            gt = self.gts[i]
            for j in range(self.batch_size):
                pred = self.preds[j][i]
                rmsd = self.process_one(pred, gt, pred.valid and gt.valid)
                if rmsd is not None:
                    self.all_rms_dis[j][i] = float(rmsd)

    def get_match_rate_and_rms_for_k(self, k: int):
        if k <= 0 or k > self.batch_size:
            raise ValueError(f"k must be in [1, {self.batch_size}], got {k}.")
        self._compute_all_rms()
        subset = self.all_rms_dis[:k, :]
        finite_mask = np.isfinite(subset)
        has_match = finite_mask.any(axis=0)
        safe_subset = np.where(finite_mask, subset, np.inf)
        best_rms = safe_subset.min(axis=0)
        best_rms[~has_match] = np.nan
        match_rate = float(has_match.mean()) if has_match.size > 0 else 0.0
        mean_rms_dist = (
            float(best_rms[has_match].mean()) if np.any(has_match) else float("nan")
        )
        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

    def get_match_rate_and_rms(self):
        return self.get_match_rate_and_rms_for_k(self.batch_size)
