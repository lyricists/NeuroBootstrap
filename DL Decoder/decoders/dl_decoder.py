import pickle, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.factory import get_dl_model
from utils.data import make_loaders_3way
from utils.training import train_and_eval


class DLDecoder:
    def __init__(
        self,
        fPath,
        fileName_train,
        fileName_test,
        model_type="eegnet",
        tWin=None,
        Trial_num=None,
        k_fold=5,
        n_classes=2,
        hyper=None,
        saveName=None,
    ):
        self.fPath, self.fileName_train, self.fileName_test, self.model_type = (
            fPath,
            fileName_train,
            fileName_test,
            model_type,
        )
        self.tWin, self.Trial_num, self.kfold, self.n_classes = (
            tWin,
            Trial_num,
            k_fold,
            n_classes,
        )
        self.saveName = saveName
        self.hyper = hyper or {}
        self._load()

    def _load(self):
        with open(self.fPath + "train/" + self.fileName_train, "rb") as f:
            D_train = pickle.load(f)

        with open(self.fPath + "test/" + self.fileName_test, "rb") as f:
            D_test = pickle.load(f)

        self.augDataset_train = D_train["augDataset"]
        self.augDataset_test = D_test["augDataset"]
        self.classIdx = {
            "Train": D_train["classIdx"]["Train"],
            "Test": D_test["classIdx"]["Test"],
        }
        self.subIdx = D_train["subIdx"]

    def _equal_select(self, X, y, total, split=200):
        half = total // 2
        idxA, idxB = np.arange(0, split), np.arange(split, X.shape[-1])
        sel = np.concatenate(
            [np.random.choice(idxA, half, False), np.random.choice(idxB, half, False)]
        )
        np.random.shuffle(sel)
        return X[..., sel], y[sel]

    def _decode(self):
        decode_scores, times = [], []
        print(f"Running deep model: {self.model_type.upper()} (80/10/10 split)")

        for n in tqdm(range(len(self.subIdx))):
            sub_scores, sub_times = [], []

            for k in range(self.kfold):
                Xtr = self.augDataset_train[n][k]["TrainData"][:, self.tWin, :]
                Xte = self.augDataset_test[n][k]["TestData"][:, self.tWin, :]
                ytr, yte = self.classIdx["Train"], self.classIdx["Test"]

                if self.Trial_num:
                    Xtr, ytr = self._equal_select(Xtr, ytr, self.Trial_num, split=200)
                    Xte, yte = self._equal_select(
                        Xte, yte, max(10, self.Trial_num // 5), split=50
                    )

                C, T, _ = Xtr.shape

                # build model
                if self.model_type.lower() == "mlp":
                    model = get_dl_model(
                        self.model_type, C, T, self.n_classes, input_dim=C * T
                    )
                else:
                    model = get_dl_model(self.model_type, C, T, self.n_classes)

                # -----------------------------------------------------
                # Prepare 80/10/10 splits: validation/test from 20% test set
                # -----------------------------------------------------
                tr_loader, val_loader, te_loader = make_loaders_3way(
                    Xtr,
                    ytr,
                    Xte,
                    yte,
                    self.model_type,
                    batch_size=self.hyper.get("batch_size", 64),
                )

                # -----------------------------------------------------
                # Train and evaluate
                # -----------------------------------------------------
                bacc, ttime, best_val = train_and_eval(
                    model,
                    tr_loader,
                    te_loader,
                    val_loader=val_loader,
                    epochs=self.hyper.get("epochs", 50),
                    lr=self.hyper.get("lr", 1e-3),
                    weight_decay=self.hyper.get("weight_decay", 0.0),
                    patience=self.hyper.get("patience", 10),
                    verbose=False,
                )

                sub_scores.append(bacc)
                sub_times.append(ttime)

            decode_scores.append(np.mean(sub_scores))
            times.append(np.sum(sub_times))

        mean, se = np.mean(decode_scores), np.std(decode_scores) / np.sqrt(
            len(decode_scores)
        )
        avg_time = np.mean(times)

        self.Results = {"Decode": np.array(decode_scores), "Time": np.array(times)}
        print(
            f"{self.model_type.upper()} mean={mean:.3f}, SE={se:.3f}, avg_time={avg_time:.1f}s"
        )

        if self.saveName:
            with open(self.fPath + "Results/" + self.saveName, "wb") as f:
                pickle.dump(
                    {
                        "decodeScore": self.Results["Decode"],
                        "time": self.Results["Time"],
                        "subIdx": self.subIdx,
                    },
                    f,
                )

        return mean, se, avg_time
