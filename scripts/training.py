"""Training utilities for the genotype-alone effect model.

Reliability weighting (from n_envs_tested) is applied identically across
GBLUP and both MLP variants via sample weights in the loss / kernel-ridge
fit, so architecture comparisons aren't confounded by one model handling
the weighting differently than another.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.kernel_ridge import KernelRidge

from .genotype_models import GenotypeMLP, apply_group_lasso_proximal


def reliability_weights(n_envs_tested: np.ndarray, cap_percentile: float = 95.0) -> np.ndarray:
    """Inverse-variance-style sample weights from reliability counts.

    Weight = min(n_envs_tested, cap), capped at the given percentile so a
    single extreme outlier (e.g. a check hybrid tested in 259 environments)
    doesn't dominate the loss -- capped rather than uncapped n_envs_tested.
    """
    cap = np.percentile(n_envs_tested, cap_percentile)
    return np.minimum(n_envs_tested, cap).astype(np.float32)


def make_loader(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    """Wraps (X, y, weights) as a DataLoader of float32 tensors."""
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
        torch.as_tensor(weights, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean squared error, normalized by sum(weight) rather than
    n -- a plain `.mean()` on `weight * error**2` scales the loss (and
    therefore the effective gradient step) by the absolute magnitude of the
    weights, which for reliability weights up to ~cap (e.g. 24+) makes
    training badly conditioned and effectively changes the learning rate
    depending on the weight scale. Normalizing by sum(weight) keeps the loss
    on the same numeric scale as unweighted MSE regardless of weight
    magnitude, so LR/epochs/patience behave consistently.
    """
    return (weight * (pred - target) ** 2).sum() / weight.sum()


def fit_mlp(model: GenotypeMLP, train_loader: DataLoader, val_loader: DataLoader,
            l1_layer_indices: list[int], l1_lambda: float, lr: float, num_epochs: int,
            patience: int, device: torch.device,
            checkpoint_path: Path | None = None, log_every: int = 10,
            verbose: bool = True) -> tuple[GenotypeMLP, dict[str, list[float]]]:
    """Trains an MLP variant with weighted MSE (SGD + momentum) plus a
    group-lasso sparsity penalty applied via an explicit proximal step
    (once per epoch), and early stopping.

    Uses SGD+momentum rather than Adam specifically because of the sparsity
    mechanism: proximal-gradient theory (ISTA/FISTA) for group lasso assumes
    plain gradient descent, and Adam's per-parameter adaptive scaling turned
    out to actively fight the proximal step in testing -- with Adam, the
    lambda-vs-sparsity relationship was non-monotonic and unstable (e.g.
    lambda=0.015 zeroed every column, lambda=0.02-0.04 zeroed none, despite
    higher loss) rather than the smooth, monotonic sparsity increase SGD
    produces. Gradient clipping (norm 1.0) keeps SGD stable across both the
    1- and 2-hidden-layer architectures (MLP-2 diverged to NaN without it
    at learning rates that worked fine for MLP-1).

    The group-lasso term is NOT added to the backprop loss itself --
    `apply_group_lasso_proximal` runs once per epoch (not per batch) after
    all of that epoch's optimizer steps; see its docstring for why per-batch
    was also wrong (total shrinkage would depend on batch count). train_loss
    /val_loss below are therefore pure weighted MSE, not loss+penalty.

    Prints progress every `log_every` epochs and on early stop, matching
    nb06_mlp_variant_comparison.ipynb's training-loop convention -- an
    intentional exception to "no print inside functions", since training
    here can behave very differently across the lambda grid and silent
    multi-minute runs are hard to debug otherwise.
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    history: dict[str, list[float]] = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch, w_batch in train_loader:
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = weighted_mse(pred, y_batch, w_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Proximal group-lasso step once per epoch (not per batch) -- see
        # apply_group_lasso_proximal's docstring for why per-batch was wrong.
        apply_group_lasso_proximal(model, l1_layer_indices, lam=l1_lambda)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch, w_batch in val_loader:
                X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)
                pred = model(X_batch)
                val_losses.append(weighted_mse(pred, y_batch, w_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        if verbose and epoch % log_every == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    return model, history


def fit_mlp_adam(model: GenotypeMLP, train_loader: DataLoader, val_loader: DataLoader,
                  penalty_fn, lr: float, num_epochs: int, patience: int, device: torch.device,
                  checkpoint_path: Path | None = None, log_every: int = 10,
                  verbose: bool = True) -> tuple[GenotypeMLP, dict[str, list[float]]]:
    """Trains an MLP with Adam and an optional smooth penalty added directly
    to the loss -- for variants that don't need the proximal-step machinery
    `fit_mlp` uses for group lasso (L2, or no regularization at all). Unlike
    group lasso, L2 is a smooth penalty Adam optimizes correctly on its own,
    and "no regularization" obviously needs no special handling either, so
    plain Adam (no gradient clipping, no proximal step) is the appropriate
    and simpler choice here.

    `penalty_fn`: callable(model) -> scalar tensor, or None for no penalty.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch, w_batch in train_loader:
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = weighted_mse(pred, y_batch, w_batch)
            if penalty_fn is not None:
                loss = loss + penalty_fn(model)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch, w_batch in val_loader:
                X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)
                pred = model(X_batch)
                val_losses.append(weighted_mse(pred, y_batch, w_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        if verbose and epoch % log_every == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    return model, history


class GBLUPModel:
    """Wraps KernelRidge with explicit mean-centering.

    The VanRaden kinship kernel is inherently centered (row sums are
    approximately zero), so fitting KernelRidge directly on it recovers only
    deviations around zero -- it has no mechanism to learn the population
    mean yield. Standard GBLUP practice is y = mu + genetic_deviation, with
    mu handled separately; this wrapper subtracts the (sample-weighted)
    training mean before fitting and adds it back at predict time.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.kernel_ridge = KernelRidge(alpha=alpha, kernel='precomputed')
        self.y_mean_: float | None = None

    def fit(self, K_train: np.ndarray, y_train: np.ndarray,
            sample_weight: np.ndarray | None = None) -> 'GBLUPModel':
        self.y_mean_ = float(np.average(y_train, weights=sample_weight))
        self.kernel_ridge.fit(K_train, y_train - self.y_mean_, sample_weight=sample_weight)
        return self

    def predict(self, K: np.ndarray) -> np.ndarray:
        if self.y_mean_ is None:
            raise RuntimeError("GBLUPModel must be fit before predict.")
        return self.kernel_ridge.predict(K) + self.y_mean_


def fit_gblup(kinship_train: np.ndarray, y_train: np.ndarray, weights_train: np.ndarray,
              kinship_val: np.ndarray, y_val: np.ndarray,
              alpha_grid: list[float]) -> tuple[GBLUPModel, float, dict[str, float]]:
    """Fits mean-centered kernel ridge regression (GBLUP-equivalent) with a
    precomputed kinship kernel, selecting the ridge penalty alpha by
    validation Pearson r.

    kinship_train: train-vs-train kinship submatrix (n_train x n_train).
    kinship_val: val-vs-train kinship submatrix (n_val x n_train) -- NOT
    val-vs-val, since predict needs the kernel between new points and the
    training points the model was fit on.
    """
    from scipy.stats import pearsonr

    best_alpha, best_model, best_r = None, None, -np.inf
    for alpha in alpha_grid:
        model = GBLUPModel(alpha=alpha)
        model.fit(kinship_train, y_train, sample_weight=weights_train)
        pred_val = model.predict(kinship_val)
        r = pearsonr(y_val, pred_val)[0] if len(y_val) > 1 else np.nan
        if not np.isnan(r) and r > best_r:
            best_alpha, best_model, best_r = alpha, model, r

    if best_model is None:
        # Degenerate case (e.g. all folds produced NaN r) -- fall back to the
        # smallest alpha rather than silently returning nothing.
        best_alpha = alpha_grid[0]
        best_model = GBLUPModel(alpha=best_alpha)
        best_model.fit(kinship_train, y_train, sample_weight=weights_train)

    return best_model, best_alpha, {'val_pearson_r': best_r}
