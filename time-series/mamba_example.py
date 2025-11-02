"""Example implementation of a Mamba-style state space block and loss functions.

This script translates the mathematical definitions shown in the reference image into
PyTorch code. It implements the recurrent update

    h_t^(i) = lambda^(i) * h_{t-1}^(i) + u_t^(i)
    m_t^(i) = h_t^(i) + tau_i

followed by the readout

    y_t = sum_k w_k * ReLU(m_t^(k)),

with the decay coefficients computed from SSM parameters, convolution-based input
accumulation, a forget gate, and the composite loss

    J = J_ce + lambda * J_con + mu * J_phy.

Running this module will build a tiny synthetic example showing how each component
contributes to the overall loss and how gradients can be propagated end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossTerms:
    """Container for the different loss components used in the example."""

    cross_entropy: torch.Tensor
    contrastive: torch.Tensor
    physics: torch.Tensor

    def total(self, lambda_con: float, mu_phy: float) -> torch.Tensor:
        """Combine the individual losses into the final objective value."""

        return self.cross_entropy + lambda_con * self.contrastive + mu_phy * self.physics


class BilinearSimilarity(nn.Module):
    """Implements the C(u_i, u_j) similarity used in the contrastive loss."""

    def __init__(self, feature_dim: int, regularization_strength: float = 1e-4) -> None:
        super().__init__()
        self.c_sum = nn.Parameter(torch.randn(feature_dim))
        self.c_diff = nn.Parameter(torch.randn(feature_dim))
        self.regularization_strength = regularization_strength

    def forward(self, u_i: torch.Tensor, u_j: torch.Tensor) -> torch.Tensor:
        """Compute the learnable similarity score C(u_i, u_j).

        The score is the sum of a symmetric component (controlled by ``c_sum``)
        and an asymmetric component (controlled by ``c_diff``).
        """

        symmetric = self.c_sum * (u_i + u_j)
        asymmetric = self.c_diff * torch.abs(u_i - u_j)
        return torch.sum(symmetric + asymmetric, dim=-1)

    def weight_decay(self) -> torch.Tensor:
        """Return the ||C^(i)||^2 regularization term from the contrastive loss."""

        return torch.sum(self.c_sum ** 2 + self.c_diff ** 2)


class MambaStateSpaceBlock(nn.Module):
    """Minimal Mamba-style block based on the equations from the reference image."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        # Input projection and convolutional accumulator for U(t).
        self.input_proj = nn.Linear(input_dim, state_dim)
        self.kernel = nn.Parameter(torch.randn(state_dim, 1, kernel_size))
        self.kernel_initial = nn.Parameter(torch.randn(state_dim))

        # Parameters for computing lambda from Gamma and Delta.
        self.gamma = nn.Parameter(torch.zeros(state_dim))
        self.t_a = nn.Parameter(torch.zeros(state_dim))
        self.t_b = nn.Parameter(torch.zeros(state_dim))

        # Bias tau and readout weights w_k.
        self.bias_tau = nn.Parameter(torch.zeros(state_dim))
        self.output_weights = nn.Parameter(torch.randn(state_dim, output_dim))

        # Forget gate projection.
        self.forget_proj = nn.Linear(input_dim, output_dim)

    def _compute_lambda(self) -> torch.Tensor:
        """Compute the decay coefficients lambda^(i)."""

        delta = torch.exp(self.t_a) + torch.exp(self.t_b) - 1.0
        decay = torch.exp(-delta * torch.exp(self.gamma))
        return decay

    def _accumulate_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional kernel to build the U(t) sequence."""

        projected = self.input_proj(x)  # (B, T, S)
        base = projected.transpose(1, 2)  # (B, S, T)
        conv = F.conv1d(base, self.kernel, padding=self.kernel_size - 1, groups=self.state_dim)
        conv = conv[:, :, : projected.size(1)].transpose(1, 2)
        u0 = projected[:, 0:1, :] * self.kernel_initial.unsqueeze(0).unsqueeze(0)
        return conv + u0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate a sequence through the state-space recurrence.

        Returns
        -------
        outputs: torch.Tensor
            The sequence of z_t values after applying the forget gate.
        states: torch.Tensor
            The hidden state trajectory h_t for analysis or regularization.
        """

        lambdas = self._compute_lambda().unsqueeze(0)
        inputs = self._accumulate_inputs(x)
        batch_size, seq_len, _ = inputs.shape

        h_t = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)
        states = []
        outputs = []
        for t in range(seq_len):
            h_t = lambdas * h_t + inputs[:, t, :]
            m_t = h_t + self.bias_tau
            y_t = F.relu(m_t) @ self.output_weights
            forget_gate = torch.sigmoid(self.forget_proj(x[:, t, :]))
            z_t = y_t + forget_gate
            states.append(h_t.unsqueeze(1))
            outputs.append(z_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), torch.cat(states, dim=1)


def contrastive_loss(
    similarity: BilinearSimilarity,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Compute J_con using a hinge formulation with learnable similarity."""

    positive_scores = similarity(anchors, positives)
    negative_scores = similarity(anchors, negatives)
    hinge = F.relu(margin - positive_scores + negative_scores)
    return hinge.mean() + similarity.regularization_strength * similarity.weight_decay()


def physics_informed_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smoothing_weight: float = 1e-2,
) -> torch.Tensor:
    """Compute J_phy = J_reg + J_data for the state/output trajectories."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")

    data_term = F.mse_loss(predictions, targets)
    reg_term = smoothing_weight * torch.mean((predictions[:, 1:, :] - predictions[:, :-1, :]) ** 2)
    return data_term + reg_term


def build_synthetic_batch(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create random inputs and labels for the demonstration."""

    inputs = torch.randn(batch_size, seq_len, input_dim)
    labels = torch.randint(0, num_classes, size=(batch_size,))
    return inputs, labels


def run_demo(
    batch_size: int = 8,
    seq_len: int = 6,
    input_dim: int = 12,
    state_dim: int = 10,
    num_classes: int = 4,
    margin: float = 0.3,
    lambda_con: float = 0.2,
    mu_phy: float = 0.1,
) -> LossTerms:
    """Run a full forward/backward pass demonstrating the combined loss."""

    torch.manual_seed(0)

    block = MambaStateSpaceBlock(
        input_dim=input_dim,
        state_dim=state_dim,
        output_dim=num_classes,
        kernel_size=4,
    )
    similarity = BilinearSimilarity(feature_dim=num_classes)

    inputs, labels = build_synthetic_batch(batch_size, seq_len, input_dim, num_classes)
    outputs, states = block(inputs)

    logits = outputs[:, -1, :]
    ce_loss = F.cross_entropy(logits, labels)

    anchors = outputs[:, -1, :]
    positives = outputs[:, -2, :]
    negatives = torch.roll(outputs[:, -1, :], shifts=1, dims=0)
    con_loss = contrastive_loss(similarity, anchors, positives, negatives, margin)

    target_trajectory = torch.zeros_like(outputs)
    phy_loss = physics_informed_loss(outputs, target_trajectory)

    total_loss = LossTerms(ce_loss, con_loss, phy_loss)
    total_loss.total(lambda_con, mu_phy).backward()

    return total_loss


if __name__ == "__main__":
    terms = run_demo()
    print("Cross-entropy loss:", terms.cross_entropy.item())
    print("Contrastive loss:", terms.contrastive.item())
    print("Physics-informed loss:", terms.physics.item())
