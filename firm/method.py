"""Backpropagation-free decision adaptation used by FIRM.

The network remains frozen.  For sample t, FIRM first rebalances the source
posterior prefix 1..t and then queries audio, video, and fusion prototypes that
contain samples 1..t-1 only.  The prediction is emitted before the current
sample updates the prototype sufficient statistics.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def balanced_responsibilities(
    probabilities: torch.Tensor,
    iterations: int = 20,
    target_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project a posterior matrix onto row and class-marginal constraints."""
    responsibilities = probabilities.double().clamp_min(1e-12)
    samples, classes = responsibilities.shape
    if target_prior is None:
        target_prior = torch.full(
            (classes,),
            1.0 / classes,
            device=responsibilities.device,
            dtype=responsibilities.dtype,
        )
    else:
        target_prior = target_prior.to(
            responsibilities.device, dtype=responsibilities.dtype
        ).clamp_min(1e-12)
        target_prior = target_prior / target_prior.sum()

    target_mass = samples * target_prior
    for _ in range(iterations):
        responsibilities /= responsibilities.sum(dim=1, keepdim=True)
        responsibilities *= target_mass / responsibilities.sum(
            dim=0, keepdim=True
        ).clamp_min(1e-12)
    responsibilities /= responsibilities.sum(dim=1, keepdim=True)
    return responsibilities.float()


class FIRMState:
    """Causal decision state for a frozen audio-visual classifier."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        device: torch.device | str,
        classifier_weight: torch.Tensor,
        prototype_temperature: float = 0.07,
        evidence_weight: float = 0.25,
        projection_iterations: int = 20,
        drift_gate: bool = True,
    ) -> None:
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.temperature = float(prototype_temperature)
        self.evidence_weight = float(evidence_weight)
        self.projection_iterations = int(projection_iterations)
        self.drift_gate = bool(drift_gate)

        self.source_anchor = F.normalize(
            classifier_weight.detach().to(self.device).float(), dim=1
        )
        self.source_probabilities: list[torch.Tensor] = []
        self.prototype_sums = {
            view: torch.zeros(
                self.num_classes, self.feature_dim, device=self.device
            )
            for view in ("fusion", "audio", "video")
        }
        self.prototype_mass = torch.zeros(
            self.num_classes, device=self.device
        )

    def _rebalanced_current(
        self, source_probability: torch.Tensor
    ) -> torch.Tensor:
        self.source_probabilities.append(source_probability)
        prefix = torch.cat(self.source_probabilities, dim=0)
        prior = torch.full(
            (self.num_classes,),
            1.0 / self.num_classes,
            device=self.device,
        )
        projected = balanced_responsibilities(
            prefix,
            iterations=self.projection_iterations,
            target_prior=prior,
        )[-1:]
        if not self.drift_gate:
            return projected

        hard_counts = torch.bincount(
            prefix.argmax(dim=1), minlength=self.num_classes
        ).float()
        expected = len(prefix) * prior.clamp_min(1e-6)
        chi_square = float(
            (((hard_counts - expected).square()) / expected).sum().item()
        )
        degrees = max(self.num_classes - 1, 1)
        chi_z = (chi_square - degrees) / math.sqrt(2.0 * degrees)
        confidence = 2.0 * (
            0.5 * (1.0 + math.erf(chi_z / math.sqrt(2.0)))
        ) - 1.0
        confidence = max(0.0, min(1.0, confidence))
        return (
            (1.0 - confidence) * source_probability
            + confidence * projected
        )

    def _prototypes(self, view: str) -> torch.Tensor:
        target = F.normalize(self.prototype_sums[view], dim=1)
        supported = self.prototype_mass[:, None] > 1e-8
        return torch.where(supported, target, self.source_anchor)

    def _prototype_log_probability(
        self, current: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        evidence = torch.zeros(
            1, self.num_classes, device=self.device
        )
        for view in ("fusion", "audio", "video"):
            query = F.normalize(current[view].float(), dim=1)
            evidence += F.log_softmax(
                query @ self._prototypes(view).transpose(0, 1)
                / self.temperature,
                dim=1,
            )
        return evidence / 3.0

    @torch.no_grad()
    def process_sample(
        self,
        fusion_feature: torch.Tensor,
        audio_feature: torch.Tensor,
        video_feature: torch.Tensor,
        source_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Emit predictions, then update state without accessing a label."""
        current = {
            "fusion": fusion_feature.detach().to(self.device).float(),
            "audio": audio_feature.detach().to(self.device).float(),
            "video": video_feature.detach().to(self.device).float(),
        }
        source = torch.softmax(
            source_logits.detach().to(self.device).float(), dim=1
        )
        if len(source) != 1:
            raise ValueError("process_sample requires exactly one sample")

        rebalanced = self._rebalanced_current(source)
        prototype_log = self._prototype_log_probability(current)
        firm = torch.softmax(
            rebalanced.clamp_min(1e-12).log()
            + self.evidence_weight * prototype_log,
            dim=1,
        )

        # Query-before-write: the prediction above cannot use sample t's own
        # feature when constructing its prototype evidence.
        responsibility = rebalanced.squeeze(0)
        self.prototype_mass += responsibility
        for view in self.prototype_sums:
            feature = F.normalize(current[view], dim=1).squeeze(0)
            self.prototype_sums[view] += (
                responsibility[:, None] * feature[None, :]
            )

        return {
            "source": source,
            "rebalance": rebalanced,
            "firm": firm,
        }

