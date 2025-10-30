"""Benchmark smartphone image denoising algorithms on the SIDD dataset.

This module implements a compact yet complete evaluation pipeline for comparing
traditional population-based optimizers (BFO, PSO, GA) against hybrid
BFO-enhanced deep models (BFO-LSTM and BFO-Transformer) on the SIDD dataset.
It expects the dataset to be available locally and referenced through the
``--data-root`` command-line argument.

The implementation focuses on reproducibility and clarity:
* each algorithm exposes a unified ``DenoisingAlgorithm`` interface
* optimisation hyper-parameters are intentionally lightweight to keep the demo
  tractable on CPUs
* PSNR, SSIM and FSIM scores are reported alongside FLOPs and runtime

The code avoids any proprietary dependencies so that it can run in a restricted
environment once the dataset is present.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import random
import time
from glob import glob
from typing import Callable, Iterable, List, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - defensive
    raise SystemExit(
        "This script requires PyTorch. Please install torch before running the benchmark."
    ) from exc


Tensor = torch.Tensor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_image(path: str) -> Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


@dataclasses.dataclass
class SiddPair:
    noisy_path: str
    clean_path: str


class SIDDPairDataset:
    """Lightweight loader for the paired SIDD smartphone dataset."""

    def __init__(self, root: str) -> None:
        pattern = os.path.join(root, "**", "*_NOISY_SRGB.png")
        noisy_files = sorted(glob(pattern, recursive=True))
        if not noisy_files:
            raise FileNotFoundError(
                "Could not locate any SIDD files. Ensure the dataset is unpacked "
                "under the provided root path."
            )
        self.pairs: List[SiddPair] = []
        for noisy in noisy_files:
            clean = noisy.replace("_NOISY_SRGB.png", "_GT_SRGB.png")
            if os.path.exists(clean):
                self.pairs.append(SiddPair(noisy, clean))
        if not self.pairs:
            raise FileNotFoundError(
                "Found noisy files but no matching ground truth images."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def load_pair(self, index: int) -> Tuple[Tensor, Tensor]:
        pair = self.pairs[index]
        noisy = _load_image(pair.noisy_path)
        clean = _load_image(pair.clean_path)
        return noisy, clean

    def iter_pairs(self, limit: int | None = None) -> Iterable[Tuple[Tensor, Tensor]]:
        total = len(self.pairs) if limit is None else min(limit, len(self.pairs))
        for idx in range(total):
            yield self.load_pair(idx)

    def sample_training_patches(
        self,
        total_patches: int,
        patch_size: int,
        seed: int = 123,
    ) -> Tuple[Tensor, Tensor]:
        """Randomly crop patches from the dataset for algorithm calibration."""

        _set_seed(seed)
        patches_noisy: List[Tensor] = []
        patches_clean: List[Tensor] = []
        per_image = max(1, total_patches // len(self.pairs))
        for noisy, clean in self.iter_pairs():
            noisy_p, clean_p = _extract_random_patches(noisy, clean, patch_size, per_image)
            patches_noisy.append(noisy_p)
            patches_clean.append(clean_p)
            if sum(p.shape[0] for p in patches_noisy) >= total_patches:
                break
        noisy_tensor = torch.cat(patches_noisy, dim=0)[:total_patches]
        clean_tensor = torch.cat(patches_clean, dim=0)[:total_patches]
        return noisy_tensor, clean_tensor


# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------


def _extract_random_patches(
    noisy: Tensor,
    clean: Tensor,
    patch_size: int,
    count: int,
) -> Tuple[Tensor, Tensor]:
    _, height, width = noisy.shape
    patches_noisy = []
    patches_clean = []
    for _ in range(count):
        top = random.randint(0, height - patch_size)
        left = random.randint(0, width - patch_size)
        patches_noisy.append(noisy[:, top : top + patch_size, left : left + patch_size])
        patches_clean.append(clean[:, top : top + patch_size, left : left + patch_size])
    return torch.stack(patches_noisy, dim=0), torch.stack(patches_clean, dim=0)


def _extract_non_overlapping_patches(image: Tensor, patch_size: int) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
    """Return non-overlapping patches and metadata for reconstruction."""

    _, height, width = image.shape
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect")
    patches = F.unfold(padded, kernel_size=patch_size, stride=patch_size)
    patches = patches.transpose(1, 2)
    return patches, (height, width), (pad_h, pad_w)


def _reconstruct_from_patches(
    patches: Tensor,
    patch_size: int,
    original_hw: Tuple[int, int],
    padding_hw: Tuple[int, int],
) -> Tensor:
    batch = patches.transpose(1, 2)
    padded_h = original_hw[0] + padding_hw[0]
    padded_w = original_hw[1] + padding_hw[1]
    output = F.fold(batch, output_size=(padded_h, padded_w), kernel_size=patch_size, stride=patch_size)
    counts = F.fold(
        torch.ones_like(batch),
        output_size=(padded_h, padded_w),
        kernel_size=patch_size,
        stride=patch_size,
    )
    reconstructed = output / counts
    if padding_hw != (0, 0):
        reconstructed = reconstructed[:, :, : original_hw[0], : original_hw[1]]
    return reconstructed.squeeze(0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_psnr(denoised: Tensor, reference: Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((denoised - reference) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse.item()))


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> Tensor:
    radius = window_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window


def compute_ssim(denoised: Tensor, reference: Tensor, data_range: float = 1.0) -> float:
    K1, K2 = 0.01, 0.03
    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    window = _gaussian_window(11, 1.5, denoised.device).unsqueeze(0).unsqueeze(0)
    mu_x = F.conv2d(denoised.unsqueeze(0), window, padding=5, groups=denoised.shape[0])
    mu_y = F.conv2d(reference.unsqueeze(0), window, padding=5, groups=reference.shape[0])
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d((denoised.unsqueeze(0) ** 2), window, padding=5, groups=denoised.shape[0]) - mu_x2
    sigma_y2 = F.conv2d((reference.unsqueeze(0) ** 2), window, padding=5, groups=reference.shape[0]) - mu_y2
    sigma_xy = F.conv2d((denoised * reference).unsqueeze(0), window, padding=5, groups=denoised.shape[0]) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def _sobel_gradients(image: Tensor) -> Tuple[Tensor, Tensor]:
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=image.device) / 8.0
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.expand(image.shape[0], 1, -1, -1)
    kernel_y = kernel_y.expand(image.shape[0], 1, -1, -1)
    grad_x = F.conv2d(image.unsqueeze(0), kernel_x, padding=1, groups=image.shape[0])
    grad_y = F.conv2d(image.unsqueeze(0), kernel_y, padding=1, groups=image.shape[0])
    return grad_x.squeeze(0), grad_y.squeeze(0)


def compute_fsim(denoised: Tensor, reference: Tensor) -> float:
    """Approximate feature similarity index using gradients and phase congruency."""

    eps = 1e-8
    grad_dx, grad_dy = _sobel_gradients(denoised)
    grad_rx, grad_ry = _sobel_gradients(reference)

    gradient_mag_d = torch.sqrt(grad_dx ** 2 + grad_dy ** 2 + eps)
    gradient_mag_r = torch.sqrt(grad_rx ** 2 + grad_ry ** 2 + eps)
    gradient_similarity = (2 * gradient_mag_d * gradient_mag_r + eps) / (
        gradient_mag_d ** 2 + gradient_mag_r ** 2 + eps
    )

    phase_d = torch.atan2(grad_dy, grad_dx + eps)
    phase_r = torch.atan2(grad_ry, grad_rx + eps)
    phase_similarity = (1 + torch.cos(phase_d - phase_r)) / 2

    pc_weight = torch.max(_phase_congruency(denoised), _phase_congruency(reference))
    fsim_map = gradient_similarity * phase_similarity
    numerator = torch.sum(fsim_map * pc_weight)
    denominator = torch.sum(pc_weight) + eps
    return float((numerator / denominator).item())


def _phase_congruency(image: Tensor, scales: int = 4) -> Tensor:
    """Simplified log-Gabor based phase congruency."""

    _, height, width = image.shape
    device = image.device
    y = torch.fft.fftfreq(height, device=device).reshape(-1, 1)
    x = torch.fft.fftfreq(width, device=device).reshape(1, -1)
    radius = torch.sqrt(x ** 2 + y ** 2)
    radius[0, 0] = 1.0

    orientations = 4
    result = torch.zeros(1, height, width, device=device)
    for o in range(orientations):
        angle = o * math.pi / orientations
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        filter_response = torch.zeros_like(radius)
        for scale in range(scales):
            wavelength = 3 * (2 ** scale)
            sigma_f = math.log(2) / (math.pi * (0.55))
            fo = 1.0 / wavelength
            log_gabor = torch.exp(-(torch.log(radius / fo) ** 2) / (2 * sigma_f ** 2))
            log_gabor[radius <= 0] = 0
            spread = torch.exp(-((x * cos_angle + y * sin_angle) ** 2) * (orientation_bandwidth := (math.pi / orientations)) ** -2)
            filter_response += log_gabor * spread
        fft_image = torch.fft.fft2(image.mean(dim=0, keepdim=True))
        response = torch.fft.ifft2(fft_image * filter_response).real
        result = torch.maximum(result, response.unsqueeze(0).abs())
    result = result / (result.max() + 1e-8)
    return result


# ---------------------------------------------------------------------------
# Meta-heuristic optimisers and base denoiser model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ParameterSpace:
    lower: Tensor
    upper: Tensor

    def clamp(self, values: Tensor) -> Tensor:
        return torch.max(torch.min(values, self.upper), self.lower)

    def sample(self, population: int) -> Tensor:
        return self.lower + torch.rand(population, len(self.lower)) * (self.upper - self.lower)


class AdaptiveGaussianDenoiser:
    """Simple denoiser parameterised by spatial blur and blend factor."""

    def __init__(self) -> None:
        self.sigma: float = 1.0
        self.mix: float = 0.5

    def configure(self, sigma: float, mix: float) -> None:
        self.sigma = float(sigma)
        self.mix = float(mix)

    def _kernel(self, sigma: float, device: torch.device) -> Tensor:
        radius = max(1, int(math.ceil(3 * sigma)))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel2d = kernel[:, None] @ kernel[None, :]
        return kernel2d

    def apply(self, image: Tensor, mix_override: float | None = None) -> Tensor:
        kernel = self._kernel(self.sigma, image.device)
        padding = kernel.shape[0] // 2
        weight = kernel.expand(image.shape[0], 1, -1, -1)
        blurred = F.conv2d(image.unsqueeze(0), weight, padding=padding, groups=image.shape[0]).squeeze(0)
        mix = self.mix if mix_override is None else float(mix_override)
        return mix * image + (1.0 - mix) * blurred

    def estimate_flops(
        self,
        channels: int,
        height: int,
        width: int,
        extra_evals: int = 0,
    ) -> float:
        kernel_size = self._kernel(self.sigma, torch.device("cpu")).shape[0]
        conv_ops = channels * height * width * (kernel_size ** 2)
        blend_ops = channels * height * width * 3
        calibration_ops = extra_evals * (kernel_size ** 2)
        return float(conv_ops + blend_ops + calibration_ops)


class MetaHeuristicOptimizer:
    def __init__(self, space: ParameterSpace, objective: Callable[[Tensor], float]) -> None:
        self.space = space
        self.objective = objective
        self.evaluations = 0

    def evaluate(self, values: Tensor) -> float:
        values = self.space.clamp(values)
        loss = self.objective(values)
        self.evaluations += 1
        return loss

    def optimise(self) -> Tuple[Tensor, float]:
        raise NotImplementedError


class BacterialForagingOptimizer(MetaHeuristicOptimizer):
    def __init__(
        self,
        space: ParameterSpace,
        objective: Callable[[Tensor], float],
        population: int = 6,
        chemotactic_steps: int = 12,
        swim_length: int = 4,
        reproduction_steps: int = 4,
        elimination_prob: float = 0.25,
    ) -> None:
        super().__init__(space, objective)
        self.population = population
        self.chemotactic_steps = chemotactic_steps
        self.swim_length = swim_length
        self.reproduction_steps = reproduction_steps
        self.elimination_prob = elimination_prob

    def optimise(self) -> Tuple[Tensor, float]:
        bacteria = self.space.sample(self.population)
        health = torch.zeros(self.population)
        best_value = float("inf")
        best_solution = bacteria[0]
        for _ in range(self.reproduction_steps):
            for _ in range(self.chemotactic_steps):
                costs = []
                for idx, cell in enumerate(bacteria):
                    cost = self.evaluate(cell)
                    costs.append(cost)
                    health[idx] += cost
                    if cost < best_value:
                        best_value = cost
                        best_solution = cell.clone()
                    direction = torch.randn_like(cell)
                    direction = direction / (torch.norm(direction) + 1e-12)
                    cell_new = cell + 0.1 * direction
                    cost_new = self.evaluate(cell_new)
                    swim_steps = 0
                    while cost_new < cost and swim_steps < self.swim_length:
                        cell = cell_new
                        cost = cost_new
                        if cost < best_value:
                            best_value = cost
                            best_solution = cell.clone()
                        cell_new = cell + 0.1 * direction
                        cost_new = self.evaluate(cell_new)
                        swim_steps += 1
                    bacteria[idx] = cell
            # reproduction
            _, indices = torch.sort(health)
            survivors = bacteria[indices[: self.population // 2]]
            bacteria = torch.cat([survivors, survivors.clone()], dim=0)
            health = torch.zeros(self.population)
            # elimination-dispersal
            mask = torch.rand(self.population) < self.elimination_prob
            if mask.any():
                bacteria[mask] = self.space.sample(mask.sum().item())
        return best_solution, best_value


class ParticleSwarmOptimizer(MetaHeuristicOptimizer):
    def __init__(
        self,
        space: ParameterSpace,
        objective: Callable[[Tensor], float],
        swarm_size: int = 12,
        inertia: float = 0.6,
        cognitive: float = 1.5,
        social: float = 1.5,
        iterations: int = 40,
    ) -> None:
        super().__init__(space, objective)
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.iterations = iterations

    def optimise(self) -> Tuple[Tensor, float]:
        positions = self.space.sample(self.swarm_size)
        velocities = torch.zeros_like(positions)
        personal_best = positions.clone()
        personal_scores = torch.tensor([self.evaluate(p) for p in positions])
        best_idx = torch.argmin(personal_scores)
        global_best = personal_best[best_idx].clone()
        global_score = float(personal_scores[best_idx].item())
        for _ in range(self.iterations):
            for i in range(self.swarm_size):
                r1 = torch.rand_like(positions[i])
                r2 = torch.rand_like(positions[i])
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * r1 * (personal_best[i] - positions[i])
                    + self.social * r2 * (global_best - positions[i])
                )
                positions[i] = self.space.clamp(positions[i] + velocities[i])
                score = self.evaluate(positions[i])
                if score < personal_scores[i]:
                    personal_scores[i] = score
                    personal_best[i] = positions[i].clone()
                    if score < global_score:
                        global_score = float(score)
                        global_best = positions[i].clone()
        return global_best, global_score


class GeneticOptimizer(MetaHeuristicOptimizer):
    def __init__(
        self,
        space: ParameterSpace,
        objective: Callable[[Tensor], float],
        population: int = 20,
        generations: int = 30,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
    ) -> None:
        super().__init__(space, objective)
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimise(self) -> Tuple[Tensor, float]:
        population = self.space.sample(self.population)
        scores = torch.tensor([self.evaluate(individual) for individual in population])
        for _ in range(self.generations):
            parents_idx = torch.multinomial(1 / (scores + 1e-8), self.population, replacement=True)
            next_population = []
            for i in range(0, self.population, 2):
                parent1 = population[parents_idx[i]].clone()
                parent2 = population[parents_idx[(i + 1) % self.population]].clone()
                if random.random() < self.crossover_rate:
                    alpha = torch.rand_like(parent1)
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                else:
                    child1, child2 = parent1, parent2
                for child in (child1, child2):
                    if random.random() < self.mutation_rate:
                        child += 0.05 * torch.randn_like(child)
                    next_population.append(self.space.clamp(child))
            population = torch.stack(next_population[: self.population])
            scores = torch.tensor([self.evaluate(individual) for individual in population])
        best_idx = torch.argmin(scores)
        return population[best_idx], float(scores[best_idx].item())


# ---------------------------------------------------------------------------
# Algorithm implementations
# ---------------------------------------------------------------------------


class DenoisingAlgorithm:
    name: str = ""

    def fit(self, noisy_patches: Tensor, clean_patches: Tensor) -> None:
        raise NotImplementedError

    def denoise(self, noisy_image: Tensor) -> Tensor:
        raise NotImplementedError

    def estimate_flops(self, image_shape: Tuple[int, int, int]) -> float:
        raise NotImplementedError


class BFOAlgorithm(DenoisingAlgorithm):
    name = "BFO"

    def __init__(self) -> None:
        self.denoiser = AdaptiveGaussianDenoiser()
        lower = torch.tensor([0.3, 0.0])
        upper = torch.tensor([3.0, 1.0])
        self.space = ParameterSpace(lower, upper)
        self.extra_evals = 0

    def fit(self, noisy_patches: Tensor, clean_patches: Tensor) -> None:
        objective = self._objective(noisy_patches, clean_patches)
        optimizer = BacterialForagingOptimizer(self.space, objective)
        solution, _ = optimizer.optimise()
        self.denoiser.configure(solution[0].item(), solution[1].item())
        self.extra_evals = optimizer.evaluations

    def _objective(self, noisy: Tensor, clean: Tensor) -> Callable[[Tensor], float]:
        def fn(params: Tensor) -> float:
            sigma, mix = params
            self.denoiser.configure(float(sigma), float(mix))
            denoised = torch.stack([self.denoiser.apply(n) for n in noisy])
            loss = F.mse_loss(denoised, clean)
            return float(loss.item())

        return fn

    def denoise(self, noisy_image: Tensor) -> Tensor:
        return self.denoiser.apply(noisy_image)

    def estimate_flops(self, image_shape: Tuple[int, int, int]) -> float:
        c, h, w = image_shape
        return self.denoiser.estimate_flops(c, h, w, self.extra_evals)


class PSOAlgorithm(BFOAlgorithm):
    name = "PSO"

    def fit(self, noisy_patches: Tensor, clean_patches: Tensor) -> None:
        objective = self._objective(noisy_patches, clean_patches)
        optimizer = ParticleSwarmOptimizer(self.space, objective)
        solution, _ = optimizer.optimise()
        self.denoiser.configure(solution[0].item(), solution[1].item())
        self.extra_evals = optimizer.evaluations


class GAAlgorithm(BFOAlgorithm):
    name = "GA"

    def fit(self, noisy_patches: Tensor, clean_patches: Tensor) -> None:
        objective = self._objective(noisy_patches, clean_patches)
        optimizer = GeneticOptimizer(self.space, objective)
        solution, _ = optimizer.optimise()
        self.denoiser.configure(solution[0].item(), solution[1].item())
        self.extra_evals = optimizer.evaluations


class BFOLSTMAlgorithm(DenoisingAlgorithm):
    name = "BFO-LSTM"

    def __init__(self, patch_size: int, hidden_size: int = 64, epochs: int = 3) -> None:
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.model: PatchLSTM | None = None
        self.mix = 0.6
        self.sigma = 1.2
        self.extra_evals = 0

    def fit(self, noisy_patches: Tensor, clean_patches: Tensor) -> None:
        device = noisy_patches.device
        self.model = PatchLSTM(self.patch_size, self.hidden_size).to(device)
        train_patch_model(self.model, noisy_patches, clean_patches, epochs=self.epochs)

        def objective(params: Tensor) -> float:
            sigma, mix = params
            denoiser = AdaptiveGaussianDenoiser()
            denoiser.configure(float(sigma), float(mix))
            outputs = self.model.forward_patches(noisy_patches)
            blurred = torch.stack([denoiser.apply(n, mix_override=0.0) for n in noisy_patches])
            combined = mix * outputs + (1 - mix) * blurred
            loss = F.mse_loss(combined, clean_patches)
            return float(loss.item())

        lower = torch.tensor([0.3, 0.3])
        upper = torch.tensor([2.5, 0.9])
        optimizer = BacterialForagingOptimizer(ParameterSpace(lower, upper), objective)
        solution, _ = optimizer.optimise()
        self.sigma = solution[0].item()
        self.mix = solution[1].item()
        self.extra_evals = optimizer.evaluations

    def denoise(self, noisy_image: Tensor) -> Tensor:
        assert self.model is not None
        patches, original_hw, padding_hw = _extract_non_overlapping_patches(noisy_image, self.patch_size)
        patch_batch = patches.reshape(-1, noisy_image.shape[0], self.patch_size, self.patch_size)
        outputs = self.model.forward_patches(patch_batch)
        denoiser = AdaptiveGaussianDenoiser()
        denoiser.configure(self.sigma, self.mix)
        blurred = torch.stack([denoiser.apply(p, mix_override=0.0) for p in patch_batch])
        combined = self.mix * outputs + (1 - self.mix) * blurred
        combined = combined.reshape(patches.shape[0], -1)
        reconstructed = _reconstruct_from_patches(
            combined.unsqueeze(0),
            self.patch_size,
            original_hw,
            padding_hw,
        )
        return reconstructed

    def estimate_flops(self, image_shape: Tuple[int, int, int]) -> float:
        c, h, w = image_shape
        seq_len = self.patch_size
        input_dim = self.patch_size * c
        lstm_ops = 4 * self.hidden_size * (input_dim + self.hidden_size) * seq_len
        output_ops = self.hidden_size * (self.patch_size * self.patch_size * c)
        num_patches = (math.ceil(h / self.patch_size) * math.ceil(w / self.patch_size))
        total_ops = (lstm_ops + output_ops) * num_patches
        denoiser = AdaptiveGaussianDenoiser()
        denoiser.configure(self.sigma, self.mix)
        blur_ops = denoiser.estimate_flops(c, self.patch_size, self.patch_size, self.extra_evals)
        return float(total_ops + blur_ops * num_patches)


class BFOTransformerAlgorithm(DenoisingAlgorithm):
    name = "BFO-Transformer"

    def __init__(self, patch_size: int, embed_dim: int = 96, depth: int = 2, epochs: int = 3) -> None:
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.epochs = epochs
        self.model: PatchTransformer | None = None
        self.mix = 0.5
        self.sigma = 1.0
        self.extra_evals = 0

    def fit(self, noisy_patches: Tensor, clean_patches: Tensor) -> None:
        device = noisy_patches.device
        self.model = PatchTransformer(self.patch_size, self.embed_dim, self.depth).to(device)
        train_patch_model(self.model, noisy_patches, clean_patches, epochs=self.epochs)

        def objective(params: Tensor) -> float:
            sigma, mix = params
            denoiser = AdaptiveGaussianDenoiser()
            denoiser.configure(float(sigma), float(mix))
            outputs = self.model.forward_patches(noisy_patches)
            blurred = torch.stack([denoiser.apply(n, mix_override=0.0) for n in noisy_patches])
            combined = mix * outputs + (1 - mix) * blurred
            loss = F.mse_loss(combined, clean_patches)
            return float(loss.item())

        lower = torch.tensor([0.3, 0.3])
        upper = torch.tensor([2.5, 0.9])
        optimizer = BacterialForagingOptimizer(ParameterSpace(lower, upper), objective)
        solution, _ = optimizer.optimise()
        self.sigma = solution[0].item()
        self.mix = solution[1].item()
        self.extra_evals = optimizer.evaluations

    def denoise(self, noisy_image: Tensor) -> Tensor:
        assert self.model is not None
        patches, original_hw, padding_hw = _extract_non_overlapping_patches(noisy_image, self.patch_size)
        patch_batch = patches.reshape(-1, noisy_image.shape[0], self.patch_size, self.patch_size)
        outputs = self.model.forward_patches(patch_batch)
        denoiser = AdaptiveGaussianDenoiser()
        denoiser.configure(self.sigma, self.mix)
        blurred = torch.stack([denoiser.apply(p, mix_override=0.0) for p in patch_batch])
        combined = self.mix * outputs + (1 - self.mix) * blurred
        combined = combined.reshape(patches.shape[0], -1)
        reconstructed = _reconstruct_from_patches(
            combined.unsqueeze(0),
            self.patch_size,
            original_hw,
            padding_hw,
        )
        return reconstructed

    def estimate_flops(self, image_shape: Tuple[int, int, int]) -> float:
        c, h, w = image_shape
        patch_dim = self.patch_size * self.patch_size * c
        num_patches = math.ceil(h / self.patch_size) * math.ceil(w / self.patch_size)
        # Transformer encoder rough estimate: 4 * dim^2 per MLP block + attention cost
        attention_ops = (self.embed_dim ** 2) * num_patches * self.depth
        mlp_ops = 4 * (self.embed_dim ** 2) * num_patches * self.depth
        proj_ops = patch_dim * self.embed_dim * num_patches
        denoiser = AdaptiveGaussianDenoiser()
        denoiser.configure(self.sigma, self.mix)
        blur_ops = denoiser.estimate_flops(c, self.patch_size, self.patch_size, self.extra_evals)
        return float(attention_ops + mlp_ops + proj_ops + blur_ops * num_patches)


# ---------------------------------------------------------------------------
# Patch models
# ---------------------------------------------------------------------------


class PatchLSTM(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.input_dim = patch_size * 3
        self.lstm = nn.LSTM(self.input_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, patch_size * patch_size * 3)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        rows = x.view(batch, 3, self.patch_size, self.patch_size).permute(0, 2, 1, 3)
        rows = rows.reshape(batch, self.patch_size, self.input_dim)
        out, _ = self.lstm(rows)
        features = out[:, -1, :]
        flat = self.output(features)
        return flat

    def forward_patches(self, patches: Tensor) -> Tensor:
        flat = self.forward(patches)
        return flat.view(-1, 3, self.patch_size, self.patch_size)


class PatchTransformer(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int, depth: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.input_dim = patch_size * patch_size * 3
        self.project = nn.Linear(self.input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output = nn.Linear(embed_dim, self.input_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        flat = x.view(batch, -1)
        tokens = self.project(flat).unsqueeze(1)
        encoded = self.encoder(tokens).squeeze(1)
        flat_out = self.output(encoded)
        return flat_out

    def forward_patches(self, patches: Tensor) -> Tensor:
        flat = self.forward(patches)
        return flat.view(-1, 3, self.patch_size, self.patch_size)


def train_patch_model(model: nn.Module, noisy: Tensor, clean: Tensor, epochs: int = 3) -> None:
    device = noisy.device
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        permutation = torch.randperm(noisy.shape[0])
        for idx in permutation.split(16):
            batch_noisy = noisy[idx].to(device)
            batch_clean = clean[idx].to(device)
            optimiser.zero_grad()
            output = model.forward_patches(batch_noisy)
            loss = F.mse_loss(output, batch_clean)
            loss.backward()
            optimiser.step()


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


def evaluate_algorithm(
    algorithm: DenoisingAlgorithm,
    dataset: SIDDPairDataset,
    patch_size: int,
    patches_per_image: int,
    max_images: int | None = None,
    device: torch.device | None = None,
) -> None:
    device = device or torch.device("cpu")
    noisy_patches, clean_patches = dataset.sample_training_patches(
        total_patches=patches_per_image * (max_images or len(dataset)),
        patch_size=patch_size,
    )
    noisy_patches = noisy_patches.to(device)
    clean_patches = clean_patches.to(device)

    start_fit = time.perf_counter()
    algorithm.fit(noisy_patches, clean_patches)
    fit_time = time.perf_counter() - start_fit

    psnr_scores: List[float] = []
    ssim_scores: List[float] = []
    fsim_scores: List[float] = []
    runtimes: List[float] = []
    flops: List[float] = []

    for noisy, clean in dataset.iter_pairs(limit=max_images):
        noisy = noisy.to(device)
        clean = clean.to(device)
        start = time.perf_counter()
        denoised = algorithm.denoise(noisy)
        runtimes.append(time.perf_counter() - start)
        psnr_scores.append(compute_psnr(denoised, clean))
        ssim_scores.append(compute_ssim(denoised, clean))
        fsim_scores.append(compute_fsim(denoised, clean))
        flops.append(algorithm.estimate_flops(tuple(noisy.shape)))

    print(f"Algorithm: {algorithm.name}")
    print(f"Training time: {fit_time:.2f}s")
    print(f"Average runtime per image: {np.mean(runtimes):.2f}s")
    print(f"PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f}")
    print(f"SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"FSIM: {np.mean(fsim_scores):.4f} ± {np.std(fsim_scores):.4f}")
    print(f"Estimated FLOPs: {np.mean(flops) / 1e9:.3f} GFLOPs")


ALGORITHMS = {
    "bfo": lambda patch_size: BFOAlgorithm(),
    "pso": lambda patch_size: PSOAlgorithm(),
    "ga": lambda patch_size: GAAlgorithm(),
    "bfo_lstm": lambda patch_size: BFOLSTMAlgorithm(patch_size),
    "bfo_transformer": lambda patch_size: BFOTransformerAlgorithm(patch_size),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIDD denoising benchmark")
    parser.add_argument("--data-root", required=True, help="Path to the unpacked SIDD dataset")
    parser.add_argument("--algorithm", choices=ALGORITHMS.keys(), help="Algorithm to evaluate")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--patches-per-image", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None, help="Limit the number of image pairs")
    parser.add_argument("--device", default="cpu", help="PyTorch device identifier")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    dataset = SIDDPairDataset(args.data_root)
    algorithm_builder = ALGORITHMS[args.algorithm]
    algorithm = algorithm_builder(args.patch_size)
    evaluate_algorithm(
        algorithm,
        dataset,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        max_images=args.max_images,
        device=torch.device(args.device),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
