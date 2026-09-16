from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from iwpu.models import IWPUModel
from iwpu.models import build_model
from iwpu.trainers import (
    TrainSettings,
    _draw_four,
    _run_ratio_pretraining,
    evaluate_test_accuracy,
    train_one,
)


@dataclass
class TinyBundle:
    src_pos: torch.Tensor
    src_unl: torch.Tensor
    tgt_pos: torch.Tensor
    tgt_unl: torch.Tensor
    val_pos: torch.Tensor
    val_unl: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    input_dim: int
    metadata: dict[str, Any]


@pytest.fixture
def tiny_bundle() -> TinyBundle:
    generator = torch.Generator().manual_seed(31)

    def positive(count: int) -> torch.Tensor:
        return torch.randn(count, 4, generator=generator) * 0.2 + 0.8

    def negative(count: int) -> torch.Tensor:
        return torch.randn(count, 4, generator=generator) * 0.2 - 0.8

    def unlabeled(count: int) -> torch.Tensor:
        positive_count = count // 2
        return torch.cat((positive(positive_count), negative(count - positive_count)))

    test_positive = positive(6)
    test_negative = negative(6)
    return TinyBundle(
        src_pos=positive(8),
        src_unl=unlabeled(12),
        tgt_pos=positive(3),
        tgt_unl=unlabeled(5),
        val_pos=positive(4),
        val_unl=unlabeled(6),
        test_x=torch.cat((test_positive, test_negative)),
        test_y=torch.cat((torch.ones(6), -torch.ones(6))),
        input_dim=4,
        metadata={"dataset": "diabetes", "pi_src": 0.5, "pi_tgt": 0.5},
    )


def settings(**updates: Any) -> TrainSettings:
    values: dict[str, Any] = {
        "steps_per_epoch": 1,
        "max_epochs": 1,
        "pretrain_epochs": 1,
        "patience": 1,
        "source_batch_size": 4,
        "target_batch_size": 4,
        "lr_classifier": 1.0e-3,
        "lr_importance": 1.0e-3,
        "device": "cpu",
        "seed": 5,
    }
    values.update(updates)
    return TrainSettings(**values)


@pytest.mark.parametrize(
    "method",
    [
        "iwpu",
        "ours",
        "tepu",
        "trpu",
        "ftpu",
        "mtspu",
        "mtpu",
        "dapu",
        "udapu",
        "giw",
        "two_step",
        "2step",
    ],
)
def test_all_supported_methods_return_complete_result(
    tiny_bundle: TinyBundle, method: str
) -> None:
    result = train_one(tiny_bundle, method, settings())
    assert isinstance(result["model"], IWPUModel)
    assert torch.isfinite(torch.tensor(result["best_val_risk"]))
    assert result["epochs"] >= 1
    assert result["runtime"] >= 0
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["history"]


def test_mtpu_registers_two_independent_classifier_heads(
    tiny_bundle: TinyBundle,
) -> None:
    model = train_one(tiny_bundle, "mtpu", settings())["model"]
    assert hasattr(model, "u_source")
    assert model.u_source is not model.u
    assert set(model.u_source.state_dict()) == set(model.u.state_dict())


def test_early_stopping_uses_target_validation_pu_risk(
    tiny_bundle: TinyBundle,
) -> None:
    result = train_one(
        tiny_bundle,
        "tepu",
        settings(max_epochs=4, pretrain_epochs=0, patience=1, min_delta=1.0e6),
    )
    # The first checkpoint is accepted; the next cannot clear the deliberately
    # huge min_delta and exhausts patience.
    assert result["epochs"] == 2
    classifier_records = [
        record for record in result["history"] if record["phase"] == "classifier"
    ]
    assert len(classifier_records) == 2
    assert all(record["val_risk"] is not None for record in classifier_records)


def test_iwpu_ablation_flags_are_executable(tiny_bundle: TinyBundle) -> None:
    result = train_one(
        tiny_bundle,
        "iwpu",
        settings(correct_iw=False, correct_cl=False),
    )
    assert result["method"] == "iwpu"
    assert "importance_loss" in result["history"][-1]


def test_giw_reports_approximation_and_both_pseudo_pretraining_phases(
    tiny_bundle: TinyBundle,
) -> None:
    result = train_one(tiny_bundle, "giw", settings())
    assert result["method"] == "giw"
    assert result["approximation_due_to_missing_official_code"] is True
    assert result["metadata"]["approximation_due_to_missing_official_code"] is True
    assert result["metadata"]["giw_observed_positives_forced_positive"] is False
    phases = {record["phase"] for record in result["history"]}
    assert "giw_trpu_pretrain" in phases
    assert "giw_tepu_pretrain" in phases
    assert "classifier" in phases
    assert "importance_loss" in result["history"][-1]


def test_replacement_sampling_supports_batch_larger_than_target_sets(
    tiny_bundle: TinyBundle,
) -> None:
    # tgt_pos contains only three rows, but the target-domain total batch of
    # 16 is split into independent replacement P/U component batches of 8.
    result = train_one(
        tiny_bundle,
        "iwpu",
        settings(source_batch_size=16, target_batch_size=16),
    )
    assert result["epochs"] == 1


def test_reported_batch_sizes_are_per_domain_totals(
    tiny_bundle: TinyBundle,
) -> None:
    configured = settings(source_batch_size=10, target_batch_size=6)
    generator = torch.Generator().manual_seed(13)
    batches = _draw_four(tiny_bundle, configured, generator, torch.device("cpu"))
    assert [tensor.shape[0] for tensor in batches[:4]] == [5, 5, 3, 3]


def test_two_step_ratio_phase_trains_h_and_v_but_not_classifier(
    tiny_bundle: TinyBundle,
) -> None:
    configured = settings()
    torch.manual_seed(configured.seed)
    model = build_model("diabetes", tiny_bundle.input_dim, configured.alpha)
    before_h = [parameter.detach().clone() for parameter in model.h.parameters()]
    before_u = [parameter.detach().clone() for parameter in model.u.parameters()]
    before_v = [parameter.detach().clone() for parameter in model.v.parameters()]

    _run_ratio_pretraining(
        model,
        tiny_bundle,
        0.5,
        0.5,
        configured,
        torch.Generator().manual_seed(configured.seed),
        torch.device("cpu"),
        [],
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_h, model.h.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_v, model.v.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_u, model.u.parameters())
    )


def test_training_can_defer_test_evaluation_until_after_selection(
    tiny_bundle: TinyBundle,
) -> None:
    result = train_one(
        tiny_bundle,
        "tepu",
        settings(),
        evaluate_test=False,
    )
    assert "test_accuracy" not in result
    accuracy = evaluate_test_accuracy(result["model"], tiny_bundle, "cpu")
    assert 0.0 <= accuracy <= 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps_per_epoch", 0),
        ("source_batch_size", 0),
        ("target_batch_size", 1),
        ("alpha", 0.0),
        ("beta", 1.1),
        ("lr_classifier", 0.0),
        ("deterministic_algorithms", "yes"),
    ],
)
def test_train_settings_reject_invalid_values(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        settings(**{field: value})


def test_unknown_method_is_rejected(tiny_bundle: TinyBundle) -> None:
    with pytest.raises(ValueError, match="unsupported method"):
        train_one(tiny_bundle, "not-a-method", settings())
