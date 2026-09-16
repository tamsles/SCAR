import torch
import torch.nn as nn
import torch.nn.functional as F


def logistic_loss(pred):
    negative_logistic = nn.LogSigmoid()
    return -negative_logistic(pred)


def complementary_nll(outputs, labels):
    """Pointwise NLL for the event that the sample is not its given class."""
    labels = labels.long().view(-1)
    probabilities = F.softmax(outputs, dim=1)
    complementary_probabilities = probabilities.gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)
    epsilon = torch.finfo(outputs.dtype).eps
    complementary_probabilities = complementary_probabilities.clamp(
        min=0.0, max=1.0 - epsilon
    )
    return -torch.log1p(-complementary_probabilities)


def supervised_loss_vector(outputs, labels):
    return F.cross_entropy(outputs, labels.long(), reduction="none")


def SCARCE_loss(outputs, labels, device=None, sample_weight=None):
    """Compute the SCARCE risk, optionally with DIW sample weights."""
    del device  # outputs determines the device; retained for API compatibility.
    labels = labels.long().view(-1)
    if outputs.shape[0] != labels.shape[0]:
        raise ValueError("outputs and labels must have the same batch size")

    if sample_weight is None:
        sample_weight = outputs.new_ones(outputs.shape[0])
    else:
        sample_weight = sample_weight.to(
            device=outputs.device, dtype=outputs.dtype
        ).view(-1)
        if sample_weight.shape[0] != outputs.shape[0]:
            raise ValueError("sample_weight must contain one value per sample")
        if not torch.isfinite(sample_weight).all().item():
            raise ValueError("sample_weight contains NaN or infinite values")
        if (sample_weight < 0).any().item():
            raise ValueError("sample_weight must be non-negative")

    num_classes = outputs.shape[1]
    complementary_mask = F.one_hot(
        labels, num_classes=num_classes
    ).to(dtype=outputs.dtype)
    weighted_complementary_mask = (
        complementary_mask * sample_weight.unsqueeze(1)
    )

    positive_loss = logistic_loss(outputs)
    negative_loss = logistic_loss(-outputs)
    epsilon = torch.finfo(outputs.dtype).eps

    complementary_weight = weighted_complementary_mask.sum(dim=0).clamp_min(
        epsilon
    )
    negative_risk = (
        negative_loss * weighted_complementary_mask
    ).sum(dim=0) / complementary_weight
    positive_complementary_risk = (
        positive_loss * weighted_complementary_mask
    ).sum(dim=0) / complementary_weight

    total_weight = sample_weight.sum().clamp_min(epsilon)
    positive_unlabeled_risk = (
        positive_loss * sample_weight.unsqueeze(1)
    ).sum(dim=0) / total_weight

    class_prior = outputs.new_full((num_classes,), 1.0 / num_classes)
    complementary_selection_probability = 1.0 - class_prior
    loss_negative = (
        complementary_selection_probability * negative_risk
    ).sum()
    uncorrected_positive = (
        positive_unlabeled_risk
        - complementary_selection_probability * positive_complementary_risk
    )
    loss_positive = torch.abs(uncorrected_positive).sum()
    return loss_negative + loss_positive


def accuracy_check(loader, model, device):
    was_training = model.training
    model.eval()
    total = 0
    num_samples = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[:2]
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    if was_training:
        model.train()
    return 100.0 * total / num_samples


def chosen_loss_c(
        f,
        K,
        labels,
        ccp,
        meta_method,
        device,
        sample_weight=None):
    del K, ccp
    if meta_method not in ("SCARCE", "DIW"):
        raise ValueError("Unknown complementary-label method: {}".format(meta_method))
    final_loss = SCARCE_loss(
        outputs=f,
        labels=labels,
        device=device,
        sample_weight=sample_weight,
    )
    return final_loss, None
