import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def to_dl_inputs(x, model_type):
    """Convert (C,T,N) -> proper tensor for each model"""
    C, T, N = x.shape
    if model_type.lower() in ["eegnet", "shallowconvnet", "deepconvnet"]:
        X = np.transpose(x, (2, 0, 1))[:, None, :, :]  # (N,1,C,T)
    elif model_type.lower() == "mlp":
        X = x.reshape(C * T, N).T  # (N, C*T)
    else:
        raise ValueError("Unsupported model type for DL input")
    return X


def make_loaders_3way(x_train, y_train, x_test, y_test, model_type, batch_size=64):
    """
    Create 80/10/10 train/val/test loaders.
    Validation and test sets come from the original test data (split 50/50).
    """
    # ---- Convert numpy arrays to model-compatible tensors ----
    Xtr = to_dl_inputs(x_train, model_type)
    Xte = to_dl_inputs(x_test, model_type)

    Xtr_t, Xte_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(
        Xte, dtype=torch.float32
    )
    ytr_t, yte_t = torch.tensor(y_train, dtype=torch.long), torch.tensor(
        y_test, dtype=torch.long
    )

    # ---- Split test set into validation and test halves ----
    N_te = len(yte_t)
    n_val = N_te // 2
    n_test = N_te - n_val

    val_set, test_set = random_split(TensorDataset(Xte_t, yte_t), [n_val, n_test])
    train_set = TensorDataset(Xtr_t, ytr_t)

    # ---- DataLoaders ----
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
