# Import library

import glob, json, pickle
import numpy as np
from tqdm import tqdm


class EEG_augmentation:
    def __init__(
        self,
        fPath: str = None,
        bPath: str = None,
        fileName: str = None,
        k_fold: int = 5,
        numPC: int = None,
        Trial_num: int = None,
        TOI_name: str = None,
        senId: str = None,
        avg_num: int = None,
        saveName: str = None,
        state: int = 42,
    ):

        self.fPath = fPath
        self.bPath = bPath
        self.fileName = fileName
        self.kfold = k_fold
        self.state = state
        self.Trial_num = Trial_num
        self.TOI_name = TOI_name
        self.senId = senId
        self.numPC = numPC
        self.avg_num = avg_num
        self.saveName = saveName

        self.load_EEG()
        self.augData()
        self.genClass()
        self.saveData()

    # Data load
    def load_EEG(self):
        print("Loading dataset")
        with open(self.fPath + self.fileName, "rb") as file:
            Dataset = pickle.load(file)

        self.subIdx = Dataset["subIdx"]
        self.split_data = Dataset["split_data"][self.TOI_name]
        self.pcaDataset = Dataset["pcaData"]

        # Sentence weight
        with open(self.bPath + self.senId, "rb") as file:
            weightFile = pickle.load(file)

        self.senIdx = weightFile["Index"]
        self.senWeight = weightFile["Weight_LOVO"]

    # ----------------------------
    # Weighted bootstrap sampling
    # ----------------------------
    def weighted_bootstrap_indices(
        self, x, w, n_boot=1, avg_num=None, random_state=None
    ):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x)
        p = np.asarray(w, dtype=float)

        # Random weight
        p = np.random.permutation(p)
        p = p / p.sum()
        size = len(x) if avg_num is None else avg_num

        samples = [rng.choice(x, size=size, replace=True, p=p) for _ in range(n_boot)]
        return samples

    def augData(self):
        print(f"Data augmentation {self.TOI_name}")
        self.augDataset = []

        for n in tqdm(range(len(self.subIdx))):
            foldData = []

            # Assign weight for each sentence
            wIndex = {}
            for group, w in zip(self.senIdx[n], self.senWeight[n, :]):
                for x in group:
                    wIndex[x] = w

            for k in range(self.kfold):
                # --- TRAIN ---

                # Weighted
                tmpPos, tmpNeg = [], []
                for label_type in ["positive", "negative"]:
                    data_indices = self.split_data[n][k][label_type]
                    weights = np.array([wIndex.get(x, 0.0) for x in data_indices])
                    rIdx = self.weighted_bootstrap_indices(
                        data_indices,
                        weights,
                        n_boot=int(self.Trial_num * 0.8),
                        avg_num=self.avg_num,
                    )

                    arr = [
                        np.mean(self.pcaDataset[k]["pcaData"][:, :, bs, n], axis=2)
                        for bs in rIdx
                    ]
                    (tmpPos if label_type == "positive" else tmpNeg).extend(arr)

                TrainData = np.concatenate(
                    (
                        np.array(tmpPos).transpose(1, 2, 0),
                        np.array(tmpNeg).transpose(1, 2, 0),
                    ),
                    axis=2,
                )

                foldData.append({"TrainData": TrainData})

            self.augDataset.append(foldData)

    # Generating class for augmented trials
    def genClass(self):
        print("Generate class")
        train_class = np.concatenate(
            (
                np.zeros(int(self.Trial_num * 0.8), dtype=int),
                np.ones(int(self.Trial_num * 0.8), dtype=int),
            )
        )

        self.classIdx = {"Train": train_class}

    # Saving data
    def saveData(self):
        print(f"Saving data as {self.saveName}")
        Results = {
            "classIdx": self.classIdx,
            "augDataset": self.augDataset,
            "subIdx": self.subIdx,
        }

        with open(self.fPath + "train/" + self.saveName, "wb") as file:
            pickle.dump(Results, file)


if __name__ == "__main__":
    # Configuration
    fPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/Modeling/Data/"
    bPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/"
    fileName = "train_test_PCA_BI_w.pkl"
    senId = "weight_TOI.pkl"
    Trial_num = 250
    TOI_list = ["BI"]

    for TOI in TOI_list:
        saveName = f"AugEEG_train_rw{TOI}_16.pkl"
        print(f"\n=== Running EEG augmentation for TOI = {TOI} ===\n")
        EEG_augmentation(
            fPath=fPath,
            bPath=bPath,
            fileName=fileName,
            Trial_num=Trial_num,
            avg_num=16,
            TOI_name=TOI,
            senId=senId,
            saveName=saveName,
        )
