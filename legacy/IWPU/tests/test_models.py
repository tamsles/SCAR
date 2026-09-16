from __future__ import annotations

import pytest
import torch

from iwpu.models import IWPUModel, build_model


@pytest.mark.parametrize(
    ("dataset", "input_dim", "input_shape", "feature_dim"),
    [
        ("mnist", 784, (4, 1, 28, 28), 128),
        ("fashion_mnist", 784, (4, 784), 128),
        ("diabetes", 142, (4, 142), 32),
        ("foodstamp", 239, (4, 239), 32),
    ],
)
def test_mlp_models_have_expected_shapes(
    dataset: str,
    input_dim: int,
    input_shape: tuple[int, ...],
    feature_dim: int,
) -> None:
    model = build_model(dataset, input_dim, alpha=0.5)
    assert isinstance(model, IWPUModel)
    x = torch.randn(input_shape)
    features = model.encode(x)
    assert features.shape == (4, feature_dim)
    assert model(x).shape == (4,)
    assert model.importance(x, +1).shape == (4,)


def test_cifar10_model_accepts_images_and_flattened_inputs() -> None:
    model = build_model("cifar10", 3072, alpha=0.5)
    image = torch.randn(2, 3, 32, 32)
    assert model.encode(image).shape == (2, 84)
    assert model(image).shape == (2,)
    assert model(image.flatten(start_dim=1)).shape == (2,)


def test_ratio_output_is_bounded_by_reciprocal_alpha() -> None:
    alpha = 0.1
    model = build_model("mnist", 784, alpha)
    x = torch.randn(8, 784)
    for label in (-1, +1):
        weights = model.importance(x, label)
        assert torch.all(weights > 0)
        assert torch.all(weights < 1.0 / alpha)


def test_one_hot_and_signed_labels_produce_identical_weights() -> None:
    model = build_model("diabetes", 142, alpha=0.5)
    x = torch.randn(3, 142)
    signed = torch.tensor([-1, +1, -1])
    one_hot = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    torch.testing.assert_close(
        model.importance(x, signed), model.importance(x, one_hot)
    )


def test_detach_features_updates_v_but_not_h() -> None:
    model = build_model("diabetes", 142, alpha=0.5)
    x = torch.randn(5, 142)
    model.importance(x, +1, detach_features=True).sum().backward()
    assert all(parameter.grad is None for parameter in model.h.parameters())
    assert any(parameter.grad is not None for parameter in model.v.parameters())


def test_detach_output_removes_weight_path_from_classifier_gradient() -> None:
    torch.manual_seed(7)
    model = build_model("diabetes", 142, alpha=0.5)
    x = torch.randn(5, 142)

    model(x).sum().backward()
    baseline = [parameter.grad.clone() for parameter in model.h.parameters()]
    model.zero_grad(set_to_none=True)

    objective = model(x).sum() + model.importance(x, +1, detach_output=True).sum()
    objective.backward()
    for expected, parameter in zip(baseline, model.h.parameters()):
        torch.testing.assert_close(parameter.grad, expected)
    assert all(parameter.grad is None for parameter in model.v.parameters())


def test_build_model_rejects_unsupported_or_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_model("unknown", 10, 0.5)
    with pytest.raises(ValueError, match="alpha"):
        build_model("mnist", 784, 0.0)
    with pytest.raises(ValueError, match="3072"):
        build_model("cifar10", 784, 0.5)
