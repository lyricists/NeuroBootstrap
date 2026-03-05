# ============================================================
# spatialPCA: Random trial splitting + PCA projection
# Author: Woojae Jeong
# ============================================================

import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import mat73
from tqdm import tqdm


class spatialPCA:
    def __init__(
        self,
        fPath: str = None,
        bPath: str = None,
        fileName: str = None,
        IdxName: str = None,
        logName: str = None,
        sentiName: str = None,
        chName: str = None,
        k_fold: int = 5,
        numPC: int = None,
        saveName: str = None,
        state: int = 42,
    ):
        self.fPath = fPath
        self.bPath = bPath
        self.logName = logName
        self.IdxName = IdxName
        self.fileName = fileName
        self.sentiName = sentiName
        self.chName = chName
        self.kfold = k_fold
        self.state = state
        self.numPC = numPC
        self.saveName = saveName

        # --- Pipeline ---
        self.load_EEG()
        self.train_test_split()
        self.compute_PCA()
        self.projectPCA()
        self.saveData()

    # ------------------------------------------------------------
    # Helper: Safe numpy conversion
    # ------------------------------------------------------------
    def _as_numpy(self, arr):
        return np.array(arr, dtype=float)

    # ------------------------------------------------------------
    # Load EEG, behavioral, and weighting data
    # ------------------------------------------------------------
    def load_EEG(self):
        print("Loading dataset")

        # EEG data
        with open(self.fPath + self.fileName, "rb") as file:
            self.Dataset = pickle.load(file)

        # Subject index
        self.subIdx = mat73.loadmat(self.bPath + self.IdxName)["subject_index"]

        # Channel index
        self.goodCh = mat73.loadmat(self.bPath + self.chName)["Channel"].astype(int) - 1

        # Sentence (trial) indices
        with open(self.bPath + self.logName, "rb") as file:
            self.senId = pickle.load(file)["Sentiment"]

        # Channel filtering
        self.Dataset = self.Dataset[self.goodCh, :, :, :]

    # ------------------------------------------------------------
    # Stratified train/test split + random subsets
    # ------------------------------------------------------------
    def train_test_split(self):
        print("Performing train-test split")
        skf = StratifiedKFold(
            n_splits=self.kfold, shuffle=True, random_state=self.state
        )

        n_subj = len(self.senId)
        self.split_data = {
            "Bio": [None] * n_subj,
            "Int": [None] * n_subj,
            "BI": [None] * n_subj,
            "BI_h": [None] * n_subj,
            "BI_l": [None] * n_subj,
        }

        for n in range(n_subj):
            pos_B = np.asarray(self.senId[n]["Biography"]["positive"], dtype=int)
            neg_B = np.asarray(self.senId[n]["Biography"]["negative"], dtype=int)
            pos_I = np.asarray(self.senId[n]["Intention"]["positive"], dtype=int)
            neg_I = np.asarray(self.senId[n]["Intention"]["negative"], dtype=int)

            X_B = np.concatenate([pos_B, neg_B])
            y_B = np.concatenate([np.ones(len(pos_B)), np.zeros(len(neg_B))])
            X_I = np.concatenate([pos_I, neg_I])
            y_I = np.concatenate([np.ones(len(pos_I)), np.zeros(len(neg_I))])

            bio_folds, int_folds, bi_folds, bih_folds, bil_folds = [], [], [], [], []

            for (train_B, test_B), (train_I, test_I) in zip(
                skf.split(X_B, y_B), skf.split(X_I, y_I)
            ):
                # ------------------------
                # 1) Biography split
                # ------------------------
                fold_B = {
                    "positive": X_B[train_B][y_B[train_B] == 1],
                    "negative": X_B[train_B][y_B[train_B] == 0],
                    "test_positive": X_B[test_B][y_B[test_B] == 1],
                    "test_negative": X_B[test_B][y_B[test_B] == 0],
                }

                # ------------------------
                # 2) Intention split
                # ------------------------
                fold_I = {
                    "positive": X_I[train_I][y_I[train_I] == 1],
                    "negative": X_I[train_I][y_I[train_I] == 0],
                    "test_positive": X_I[test_I][y_I[test_I] == 1],
                    "test_negative": X_I[test_I][y_I[test_I] == 0],
                }

                # ------------------------
                # 3) Combined BI split
                # ------------------------
                fold_BI = {
                    "positive": np.concatenate(
                        [fold_B["positive"], fold_I["positive"]]
                    ).astype(int),
                    "negative": np.concatenate(
                        [fold_B["negative"], fold_I["negative"]]
                    ).astype(int),
                    "test_positive": np.concatenate(
                        [fold_B["test_positive"], fold_I["test_positive"]]
                    ).astype(int),
                    "test_negative": np.concatenate(
                        [fold_B["test_negative"], fold_I["test_negative"]]
                    ).astype(int),
                }

                # ------------------------
                # 4) Random subsets per cohort
                # ------------------------
                np.random.seed(self.state)

                # --- BIO ---
                pos_B = fold_B["positive"]
                neg_B = fold_B["negative"]

                idx_pos_B = np.random.permutation(pos_B.shape[0])
                idx_neg_B = np.random.permutation(neg_B.shape[0])

                split_k_pos_B = min(20, pos_B.shape[0] // 2)
                split_k_neg_B = min(20, neg_B.shape[0] // 2)

                bio_high_pos = pos_B[idx_pos_B[:split_k_pos_B]]
                bio_low_pos = pos_B[idx_pos_B[split_k_pos_B : split_k_pos_B * 2]]
                bio_high_neg = neg_B[idx_neg_B[:split_k_neg_B]]
                bio_low_neg = neg_B[idx_neg_B[split_k_neg_B : split_k_neg_B * 2]]

                # --- INT ---
                pos_I = fold_I["positive"]
                neg_I = fold_I["negative"]

                idx_pos_I = np.random.permutation(pos_I.shape[0])
                idx_neg_I = np.random.permutation(neg_I.shape[0])

                split_k_pos_I = min(20, pos_I.shape[0] // 2)
                split_k_neg_I = min(20, neg_I.shape[0] // 2)

                int_high_pos = pos_I[idx_pos_I[:split_k_pos_I]]
                int_low_pos = pos_I[idx_pos_I[split_k_pos_I : split_k_pos_I * 2]]
                int_high_neg = neg_I[idx_neg_I[:split_k_neg_I]]
                int_low_neg = neg_I[idx_neg_I[split_k_neg_I : split_k_neg_I * 2]]

                # ------------------------
                # 5) Combined BI_h / BI_l with class separation
                # ------------------------
                fold_BIh = {
                    "positive": np.concatenate([bio_high_pos, int_high_pos]).astype(
                        int
                    ),
                    "negative": np.concatenate([bio_high_neg, int_high_neg]).astype(
                        int
                    ),
                    "test_positive": np.concatenate(
                        [fold_B["test_positive"], fold_I["test_positive"]]
                    ).astype(int),
                    "test_negative": np.concatenate(
                        [fold_B["test_negative"], fold_I["test_negative"]]
                    ).astype(int),
                }

                fold_BIl = {
                    "positive": np.concatenate([bio_low_pos, int_low_pos]).astype(int),
                    "negative": np.concatenate([bio_low_neg, int_low_neg]).astype(int),
                    "test_positive": np.concatenate(
                        [fold_B["test_positive"], fold_I["test_positive"]]
                    ).astype(int),
                    "test_negative": np.concatenate(
                        [fold_B["test_negative"], fold_I["test_negative"]]
                    ).astype(int),
                }

                # ------------------------
                # 6) Save per-fold
                # ------------------------
                bio_folds.append(fold_B)
                int_folds.append(fold_I)
                bi_folds.append(fold_BI)
                bih_folds.append(fold_BIh)
                bil_folds.append(fold_BIl)

            # ------------------------
            # Save per subject
            # ------------------------
            self.split_data["Bio"][n] = bio_folds
            self.split_data["Int"][n] = int_folds
            self.split_data["BI"][n] = bi_folds
            self.split_data["BI_h"][n] = bih_folds
            self.split_data["BI_l"][n] = bil_folds

    # ------------------------------------------------------------
    # Compute PCA
    # ------------------------------------------------------------
    def compute_PCA(self):
        self.pcaModel = []
        print("Computing PCA")

        for k in tqdm(range(self.kfold)):
            cPos, cNeg, dPos, dNeg, sPos, sNeg = [], [], [], [], [], []

            for n in range(len(self.subIdx)):
                pos_idx = self.split_data["BI"][n][k]["positive"]
                neg_idx = self.split_data["BI"][n][k]["negative"]

                if self.subIdx[n] == 1:
                    cPos.append(np.mean(self.Dataset[:, :, pos_idx, n], axis=2))
                    cNeg.append(np.mean(self.Dataset[:, :, neg_idx, n], axis=2))
                elif self.subIdx[n] == 2:
                    dPos.append(np.mean(self.Dataset[:, :, pos_idx, n], axis=2))
                    dNeg.append(np.mean(self.Dataset[:, :, neg_idx, n], axis=2))
                else:
                    sPos.append(np.mean(self.Dataset[:, :, pos_idx, n], axis=2))
                    sNeg.append(np.mean(self.Dataset[:, :, neg_idx, n], axis=2))

            pcaInput = np.concatenate(
                (
                    (np.mean(cPos, axis=0) + np.mean(cNeg, axis=0)) / 2,
                    (np.mean(dPos, axis=0) + np.mean(dNeg, axis=0)) / 2,
                    (np.mean(sPos, axis=0) + np.mean(sNeg, axis=0)) / 2,
                ),
                axis=1,
            )

            pca = PCA(n_components=self.numPC)
            pca.fit(pcaInput.T)
            self.pcaModel.append({"pcaCoeff": pca.components_, "pcaModel": pca})

    # ------------------------------------------------------------
    # Project PCA
    # ------------------------------------------------------------
    def projectPCA(self):
        print("PCA projection")
        n_components = self.numPC
        Dataset = self.Dataset
        n_channels = Dataset.shape[0]
        data_shape = list(Dataset.shape)
        channel_axis = data_shape.index(n_channels)

        self.pcaDataset = []
        for k in tqdm(range(self.kfold)):
            X_transform_pca = Dataset.reshape(n_channels, -1).T
            X_transform_pca = self.pcaModel[k]["pcaModel"].transform(X_transform_pca)
            X_shape = list(data_shape)
            X_shape[channel_axis] = n_components
            X_transform_pca = X_transform_pca.T.reshape(X_shape)
            self.pcaDataset.append({"pcaData": X_transform_pca})

    # ------------------------------------------------------------
    # Save Results
    # ------------------------------------------------------------
    def saveData(self):
        print("Saving data")
        Results = {
            "split_data": self.split_data,
            "pcaModel": self.pcaModel,
            "pcaData": self.pcaDataset,
            "subIdx": self.subIdx,
        }

        save_path = (
            "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/Modeling/Data/"
            + self.saveName
        )
        with open(save_path, "wb") as file:
            pickle.dump(Results, file)
        print(f"Saved results to: {save_path}")


# ============================================================
# Run Module
# ============================================================
if __name__ == "__main__":
    runPCA = spatialPCA(
        fPath="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/",
        bPath="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/",
        fileName="Data_sen_lepoch.pkl",
        IdxName="subject_index.mat",
        logName="senIdx_TOI.pkl",
        chName="GoodChannel.mat",
        numPC=3,
        saveName="train_test_PCA_BI_w.pkl",
    )
