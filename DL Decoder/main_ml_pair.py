# main_ml.py

from decoders.EEGDecoder import tDecoder
import numpy as np
import pickle

DATA_PATH = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/Modeling/Data/"
TWIN = np.arange(150, 200)

models = ["svm"]

# TOI = [
#     "Bio",
#     "Int",
#     "BI_h",
#     "wBI_h",
#     "rwBI_h",
#     "BI",
#     "wBI",
#     "rwBI",
#     "BI_12",
#     "wBI_12",
#     "rwBI_12",
# ]

TOI_train = ["Int", "wBI_12"]
TOI_test = ["Int", "wBI_12"]
# --- store all results in one nested dict ---
results = {}

for train in TOI_train:
    results[train] = {}

    for test in TOI_test:
        results[train][test] = {}

        FILE_NAME_TRAIN = f"AugEEG_train_{train}.pkl"
        FILE_NAME_TEST = f"AugEEG_test_{test}.pkl"

        for model in models:
            print(f"\n=== Running {model.upper()} (Train: {train}, Test: {test}) ===")

            decoder = tDecoder(
                fPath=DATA_PATH,
                fileName_train=FILE_NAME_TRAIN,
                fileName_test=FILE_NAME_TEST,
                Trial_num=125,
                tWin=TWIN,
                model_type=model,
            )

            # save model-specific decode results
            results[train][test][model] = decoder.Results["Decode"]

        print(f"\nAblation results (Train: {train}, Test: {test}):")
        for model, v in results[train][test].items():
            print(f"{model.upper():<10}: {np.mean(v):.3f}")

# --- save everything once at the end ---
SAVE_PATH = DATA_PATH + "Results/" + "ablation_summary_ML_w.pkl"
with open(SAVE_PATH, "wb") as f:
    pickle.dump(results, f)

print(f"\nAll ablation results saved to {SAVE_PATH}")
