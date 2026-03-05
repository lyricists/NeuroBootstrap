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

    # Data augmentation (Bootstrapping)
    def augData(self):
        self.augDataset = []
        print(f"Data augmentation with avg_num = {self.avg_num}")

        for n in tqdm(range(len(self.subIdx))):
            foldData = []

            for k in range(self.kfold):
                tmpPos, tmpNeg = [], []

                # ----- Training data -----
                for _ in range(int(self.Trial_num * 0.8)):
                    rIdx = np.random.choice(
                        len(self.split_data[n][k]["positive"]), self.avg_num
                    )
                    tmpPos.append(
                        np.mean(
                            self.pcaDataset[k]["pcaData"][
                                :,
                                :,
                                self.split_data[n][k]["positive"][rIdx],
                                n,
                            ],
                            axis=2,
                        )
                    )

                    rIdx = np.random.choice(
                        len(self.split_data[n][k]["negative"]), self.avg_num
                    )
                    tmpNeg.append(
                        np.mean(
                            self.pcaDataset[k]["pcaData"][
                                :,
                                :,
                                self.split_data[n][k]["negative"][rIdx],
                                n,
                            ],
                            axis=2,
                        )
                    )

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
    Trial_num = 250
    # TOI_list = ["Bio", "Int", "BI", "BI_h", "BI_l"]
    TOI_list = ["BI"]

    for TOI in TOI_list:
        saveName = f"AugEEG_train_{TOI}_16.pkl"
        print(f"\n=== Running EEG augmentation for avg_num = {TOI} ===\n")
        EEG_augmentation(
            fPath=fPath,
            bPath=bPath,
            fileName=fileName,
            Trial_num=Trial_num,
            avg_num=16,
            TOI_name=TOI,
            saveName=saveName,
        )
