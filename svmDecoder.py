# Import library

import glob, json, pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class SVM_decoder:
    def __init__(
        self,
        fPath: str = None,
        fileName_train: str = None,
        fileName_test: str = None,
        k_fold: int = 5,
        Trial_num: int = None,
        saveName: str = None,
        state: int = 42,
    ):

        self.fPath = fPath
        self.fileName_train = fileName_train
        self.fileName_test = fileName_test
        self.kfold = k_fold
        self.state = state
        self.Trial_num = Trial_num
        self.saveName = saveName

        self.Decoder()

    # Data load
    def load_EEG(self):
        print("Loading dataset")
        # Load EEG dataset
        with open(self.fPath + "train/" + self.fileName_train, "rb") as file:
            Dataset_train = pickle.load(file)

        with open(self.fPath + "test/" + self.fileName_test, "rb") as file:
            Dataset_test = pickle.load(file)

        self.augDataset_train = Dataset_train["augDataset"]
        self.augDataset_test = Dataset_test["augDataset"]
        self.classIdx = {
            "Train": Dataset_train["classIdx"]["Train"],
            "Test": Dataset_test["classIdx"]["Test"],
        }
        self.subIdx = Dataset_train["subIdx"]

    # SVM decoder
    def classifier(self, trainData, testData):

        if self.Trial_num < 250:
            trainNum = int(self.Trial_num * 0.8)
            testNum = int(self.Trial_num * 0.2)

            # Define index offsets for the two conditions
            train_offsets = [0, 200]
            test_offsets = [0, 50]

            # Concatenate data and labels for training and testing
            x_train = np.concatenate(
                [
                    trainData[:, np.arange(offset, offset + trainNum)]
                    for offset in train_offsets
                ],
                axis=1,
            )
            x_test = np.concatenate(
                [
                    testData[:, np.arange(offset, offset + testNum)]
                    for offset in test_offsets
                ],
                axis=1,
            )

            y_train = np.concatenate(
                [
                    self.classIdx["Train"][np.arange(offset, offset + trainNum)]
                    for offset in train_offsets
                ]
            )
            y_test = np.concatenate(
                [
                    self.classIdx["Test"][np.arange(offset, offset + testNum)]
                    for offset in test_offsets
                ]
            )
        else:
            x_train, x_test = trainData, testData
            y_train, y_test = self.classIdx["Train"], self.classIdx["Test"]

        rIdx_train, rIdx_test = np.arange(x_train.shape[1]), np.arange(x_test.shape[1])
        np.random.shuffle(rIdx_train), np.random.shuffle(rIdx_test)

        x_train, x_test = x_train[:, rIdx_train], x_test[:, rIdx_test]
        y_train, y_test = y_train[rIdx_train], y_test[rIdx_test]

        scalar = StandardScaler()

        x_train = scalar.fit_transform(x_train.transpose(1, 0))
        x_test = scalar.transform(x_test.transpose(1, 0))

        # Train
        clf = SVC(
            kernel="linear",
            C=1,
            gamma="auto",
            class_weight="balanced",
            tol=1e-3,
            random_state=42,
        )
        clf.fit(x_train, y_train)

        # Test
        pred = clf.predict(x_test)
        score = balanced_accuracy_score(y_test, pred)

        return score

    # Saving data
    def saveData(self):
        Results = {
            "decodeScore": self.Results["Decode"],
            "subIdx": self.subIdx,
        }

        with open(
            self.fPath + "Results/" + self.saveName,
            "wb",
        ) as file:
            pickle.dump(Results, file)

    def Decoder(self):

        # Load EEG data
        self.load_EEG()

        # Classifying
        decode = []

        print("Performing svm decoding")
        for n in tqdm(range(len(self.subIdx))):

            sub_decode = []

            for k in range(self.kfold):

                score = []

                for t in range(425):

                    tmpScore = self.classifier(
                        self.augDataset_train[n][k]["TrainData"][:, t, :],
                        self.augDataset_test[n][k]["TestData"][:, t, :],
                    )

                    score.append(tmpScore)

                sub_decode.append(score)

            decode.append(np.mean(np.array(sub_decode), axis=0))

        self.Results = {"Decode": np.array(decode)}

        print("Save data")
        self.saveData()


if __name__ == "__main__":
    TOI = ["wBI_h"]

    for toi in TOI:

        print(f"\nRunning EEG decoding with TOI = {toi}")
        svm = SVM_decoder(
            fPath="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/Modeling/Data/",  # EEG data directory
            fileName_train=f"AugEEG_train_{toi}.pkl",  # EEG data
            fileName_test=f"AugEEG_test_{toi}.pkl",
            Trial_num=125,  # Number of augmented trials
            saveName=f"svmDecode_{toi}.pkl",
        )
