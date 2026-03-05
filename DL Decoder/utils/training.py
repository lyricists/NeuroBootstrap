import time, torch, numpy as np
from sklearn.metrics import balanced_accuracy_score


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_score, model):
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_and_eval(
    model,
    train_loader,
    test_loader,
    val_loader,  # always provided now
    epochs=50,
    lr=1e-3,
    weight_decay=0.0,
    patience=10,
    device=None,
    verbose=False,
):
    """Train model with early stopping on validation set, evaluate on test set."""
    device = torch.device("mps")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=patience)

    t0 = time.time()

    # ---------- TRAINING LOOP ----------
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        # ---------- VALIDATION ----------
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(torch.argmax(logits, 1).cpu().numpy())
                true.append(yb.numpy())

        val_bacc = balanced_accuracy_score(np.concatenate(true), np.concatenate(preds))
        if stopper.step(val_bacc, model):
            if verbose:
                print(f"⏹️ Early stopped @ epoch {ep}")
            break

        if verbose and (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs} | val_bacc={val_bacc:.3f}")

    # ---------- RESTORE BEST MODEL ----------
    model.load_state_dict(stopper.best_state)
    train_time = time.time() - t0

    # ---------- FINAL TEST EVALUATION ----------
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(torch.argmax(logits, 1).cpu().numpy())
            true.append(yb.numpy())

    y_pred, y_true = np.concatenate(preds), np.concatenate(true)
    bacc = balanced_accuracy_score(y_true, y_pred)

    return bacc, train_time, stopper.best_score
