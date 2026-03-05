import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from models.classifiers import get_classifier


# Trial decoder
class tDecoder:
    def __init__(
        self,
        fPath,
        fileName_train,
        fileName_test,
        model_type="svm",
        k_fold=5,
        Trial_num=None,
        tWin=None,
        state=42,
    ):

        self.fPath = fPath
        self.fileName_train = fileName_train
        self.fileName_test = fileName_test
        self.model_type = model_type
        self.kfold = k_fold
        self.state = state
        self.Trial_num = Trial_num
        self.tWin = tWin

        self.Decoder()

    def load_EEG(self):
        print("Loading dataset")
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

    def extract_features(self, x):
        feats = []
        for ch in range(x.shape[0]):
            feats.append(np.mean(x[ch]))
        return feats

    def balanced_random_select(self, x_data, y_data, ratio, random_state=42):

        if self.Trial_num < 250:
            n_select = int(self.Trial_num * ratio)

        np.random.seed(random_state)

        split = int(x_data.shape[1] / 2)

        idx_A = np.arange(split)
        idx_B = np.arange(split, x_data.shape[1])

        sel_A = np.random.choice(idx_A, size=n_select, replace=False)
        sel_B = np.random.choice(idx_B, size=n_select, replace=False)

        sel_idx = np.concatenate([sel_A, sel_B])

        x_subset, y_subset = x_data[:, sel_idx], y_data[sel_idx]

        return x_subset, y_subset

    def classifier(self, trainData, testData):
        if self.Trial_num < 250:

            x_train, y_train = self.balanced_random_select(
                trainData, self.classIdx["Train"], ratio=0.8
            )
            x_test, y_test = self.balanced_random_select(
                testData, self.classIdx["Test"], ratio=0.2
            )

        else:
            x_train, x_test = trainData, testData
            y_train, y_test = self.classIdx["Train"], self.classIdx["Test"]

        rIdx_train, rIdx_test = np.arange(x_train.shape[1]), np.arange(x_test.shape[1])
        np.random.shuffle(rIdx_train)
        np.random.shuffle(rIdx_test)

        x_train, x_test = x_train[:, rIdx_train], x_test[:, rIdx_test]
        y_train, y_test = y_train[rIdx_train], y_test[rIdx_test]

        scalar = StandardScaler()
        x_train = scalar.fit_transform(x_train.T)
        x_test = scalar.transform(x_test.T)

        clf = get_classifier(self.model_type, self.state)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        return balanced_accuracy_score(y_test, pred)

    def Decoder(self):
        self.load_EEG()
        decode = []

        print(f"Performing {self.model_type} decoding...")
        for n in tqdm(range(len(self.subIdx))):
            sub_decode = []
            for k in range(self.kfold):
                trainInput = np.array(
                    [
                        self.extract_features(x)
                        for x in np.transpose(
                            self.augDataset_train[n][k]["TrainData"][:, self.tWin, :],
                            (2, 0, 1),
                        )
                    ]
                )
                testInput = np.array(
                    [
                        self.extract_features(x)
                        for x in np.transpose(
                            self.augDataset_test[n][k]["TestData"][:, self.tWin, :],
                            (2, 0, 1),
                        )
                    ]
                )

                score = self.classifier(trainInput.T, testInput.T)
                sub_decode.append(score)

            decode.append(np.mean(sub_decode))

        self.Results = {"Decode": np.array(decode)}
        print(
            f"{self.model_type.upper()} mean={np.mean(decode):.3f}, SE={np.std(decode)/np.sqrt(len(decode)):.3f}"
        )
