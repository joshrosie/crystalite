from collections import Counter
import pickle
from types import SimpleNamespace

import pytest
import torch
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Lattice, Structure

from src.eval.stability import _compute_thermo_metrics
import src.utils.stability_logger as stability_logger
from src.utils.stability_logger import StabilityLogger, _ThermoConfig


class NeighborOverflowError(Exception):
    pass


OVERFLOW_MESSAGE = (
    "The number of neighbors is larger than the maximum allowed: 58121 > 1584."
)
TAG = "sample_ema"


def _structures(count, bad=()):
    return [
        Structure(
            Lattice.cubic(4.0),
            ["Si", "Si"],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
            properties={"case_id": index, "bad": index in bad},
        )
        for index in range(count)
    ]


class FakeBatchRelaxer:
    def __init__(
        self, error_type=NeighborOverflowError, message=OVERFLOW_MESSAGE, *, indexed=False
    ):
        self.error_type = error_type
        self.message = message
        self.indexed = indexed
        self.calls = []
        self.successful = []
        self.failed_singletons = []

    @staticmethod
    def _result(structure):
        if structure.properties.get("malformed_result"):
            return {"final_structure": structure}
        energy = (-1.0 + structure.properties.get("e_above_hull", 0.0)) * len(structure)
        return {
            "final_structure": structure,
            "trajectory": SimpleNamespace(energies=[energy]),
        }

    def relax_many(self, structures, *, steps, verbose):
        assert steps == 2
        assert verbose is False
        ids = tuple(structure.properties["case_id"] for structure in structures)
        self.calls.append(ids)
        bad = [index for index, struct in enumerate(structures) if struct.properties["bad"]]
        if bad:
            if len(structures) == 1:
                self.failed_singletons.extend(ids)
            message = self.message
            if self.indexed:
                message += f" systems {bad}"
            raise self.error_type(message)
        self.successful.extend(ids)
        return [self._result(structure) for structure in structures]

    def relax(self, structure, *, steps, verbose):
        return self.relax_many([structure], steps=steps, verbose=verbose)[0]


@pytest.fixture
def make_logger(monkeypatch, tmp_path):
    # Exercise the real phase-diagram loader and hull calculation, with no ML model.
    ppd_path = tmp_path / "hull.pkl"
    ppd_path.write_bytes(pickle.dumps(PhaseDiagram([PDEntry("Si", -1.0)])))

    def make(relaxer, batch_size):
        monkeypatch.setattr(
            stability_logger,
            "make_nequip_batch_relaxer",
            lambda **kwargs: (None, relaxer, "cpu", kwargs["compile_path"]),
        )
        return StabilityLogger(
            gamma_cfg=None,
            thermo_cfg=_ThermoConfig(
                batch_size=batch_size,
                relax_steps=2,
                ppd_path=str(ppd_path),
                device="cpu",
                mlip="nequip",
                nequip_compile_path=str(tmp_path / "fake.nequip.pth"),
                nequip_relax_mode="batch",
            ),
        )

    return make


def _update(logger, structures):
    records = []

    def log_fn(payload, *, step, enabled):
        assert step == 300
        assert enabled is True
        records.append(payload)

    logger.update(structures, tag=TAG, step=300, log_fn=log_fn, enabled=True)
    assert len(records) == 1
    assert not logger.thermo_buffers[TAG]
    return records[0]


@pytest.mark.parametrize(
    "count,bad", [(1, (0,)), (6, (1,)), (6, (1, 4)), (6, tuple(range(6)))]
)
def test_neighbor_overflow_isolates_only_bad_structures(make_logger, count, bad):
    relaxer = FakeBatchRelaxer()
    logger = make_logger(relaxer, count)
    metrics = _update(logger, _structures(count, bad))

    good = set(range(count)) - set(bad)
    assert Counter(relaxer.successful) == Counter({index: 1 for index in good})
    assert Counter(relaxer.failed_singletons) == Counter({index: 1 for index in bad})
    assert metrics[f"{TAG}/thermo_checked"] == count
    assert metrics[f"{TAG}/thermo_success"] == len(good)
    assert metrics[f"{TAG}/thermo_failed"] == len(bad)
    assert metrics[f"{TAG}/thermo_divergence"] == 0
    assert metrics[f"{TAG}/thermo_stable_count"] == len(good)
    assert metrics[f"{TAG}/thermo_metastable_count"] == len(good)
    assert {key: value for key, value in metrics.items() if "/thermo_fail/" in key} == {
        f"{TAG}/thermo_fail/exception:NeighborOverflowError": float(len(bad))
    }
    assert len(relaxer.calls) <= 2 * count - 1


@pytest.mark.parametrize(
    "error_type,message,reason,divergence",
    [
        (NeighborOverflowError, OVERFLOW_MESSAGE, "exception:NeighborOverflowError", 0),
        (RuntimeError, "force divergence detected", "relax_divergence", 1),
        (RuntimeError, "non-finite forces detected", "relax_non_finite", 1),
        (
            torch.linalg.LinAlgError,
            "linalg.inv: the diagonal element 1 is zero, the inversion could not be completed",
            f"exception:{torch.linalg.LinAlgError.__name__}",
            0,
        ),
    ],
)
def test_failure_reasons_survive_thermo_aggregation(
    make_logger, error_type, message, reason, divergence
):
    relaxer = FakeBatchRelaxer(error_type, message)
    logger = make_logger(relaxer, 3)
    metrics = _compute_thermo_metrics(
        logger, _structures(3, bad=(1,)), tag=TAG, step=300, enabled=True
    )

    assert metrics[f"{TAG}/thermo_checked_total"] == 3
    assert metrics[f"{TAG}/thermo_success_total"] == 2
    assert metrics[f"{TAG}/thermo_failed_total"] == 1
    assert metrics[f"{TAG}/thermo_divergence_total"] == divergence
    assert metrics[f"{TAG}/thermo_fail/{reason}"] == 1
    assert Counter(relaxer.successful) == Counter({0: 1, 2: 1})


@pytest.mark.parametrize("error_type", [NeighborOverflowError, RuntimeError])
def test_offender_indices_preserve_survivors_without_bisection(make_logger, error_type):
    message = OVERFLOW_MESSAGE if error_type is NeighborOverflowError else "force divergence"
    relaxer = FakeBatchRelaxer(error_type, message, indexed=True)
    logger = make_logger(relaxer, 6)
    metrics = _update(logger, _structures(6, bad=(1, 4)))

    assert relaxer.calls == [(0, 1, 2, 3, 4, 5), (0, 2, 3, 5)]
    assert Counter(relaxer.successful) == Counter({0: 1, 2: 1, 3: 1, 5: 1})
    assert not relaxer.failed_singletons
    assert metrics[f"{TAG}/thermo_success"] == 4
    assert metrics[f"{TAG}/thermo_failed"] == 2
    reason = "exception:NeighborOverflowError" if error_type is NeighborOverflowError else "relax_divergence"
    assert metrics[f"{TAG}/thermo_fail/{reason}"] == 2


def test_no_failures_keep_one_batch_call_and_original_metrics(make_logger):
    structures = _structures(4)
    for structure, energy in zip(structures, [0.0, 0.05, 0.2, -0.1]):
        structure.properties["e_above_hull"] = energy
    relaxer = FakeBatchRelaxer()
    metrics = _update(make_logger(relaxer, 4), structures)

    expected = {
        f"{TAG}/thermo_checked": 4.0,
        f"{TAG}/thermo_success": 4.0,
        f"{TAG}/thermo_failed": 0.0,
        f"{TAG}/thermo_divergence": 0.0,
        f"{TAG}/thermo_stable_count": 2.0,
        f"{TAG}/thermo_metastable_count": 3.0,
        f"{TAG}/thermo_ehull_is_mp2020_like": 0.0,
        f"{TAG}/thermo_ehull_is_uncorrected": 1.0,
        f"{TAG}/thermo_stable_rate": 0.5,
        f"{TAG}/thermo_metastable_rate": 0.75,
        f"{TAG}/thermo_e_above_hull_mean": 0.0375,
    }
    assert metrics == pytest.approx(expected)
    assert relaxer.calls == [(0, 1, 2, 3)]
    assert Counter(relaxer.successful) == Counter({index: 1 for index in range(4)})

    sequential_logger = make_logger(FakeBatchRelaxer(), 4)
    sequential_logger._thermo_nequip_relax_mode = "sequential"
    assert metrics == _update(sequential_logger, structures)


def test_result_failure_does_not_retry_or_double_count_completed_relaxations(make_logger):
    structures = _structures(3)
    structures[1].properties["malformed_result"] = True
    relaxer = FakeBatchRelaxer()
    metrics = _update(make_logger(relaxer, 3), structures)

    assert relaxer.calls == [(0, 1, 2)]
    assert relaxer.successful == [0, 1, 2]
    assert metrics[f"{TAG}/thermo_checked"] == 3
    assert metrics[f"{TAG}/thermo_success"] == 2
    assert metrics[f"{TAG}/thermo_failed"] == 1
    assert metrics[f"{TAG}/thermo_fail/exception:KeyError"] == 1


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(make_logger, error_type):
    relaxer = FakeBatchRelaxer(error_type, "stop")
    logger = make_logger(relaxer, 3)
    with pytest.raises(error_type, match="stop"):
        _update(logger, _structures(3, bad=(1,)))
    assert relaxer.calls == [(0, 1, 2)]
    assert not relaxer.successful
