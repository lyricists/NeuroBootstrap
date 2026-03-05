# main_ml.py

from decoders.EEGDecoder import tDecoder
import numpy as np
import pickle

DATA_PATH = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/Modeling/Data/"
TWIN = np.arange(150, 200)

models = ["svm", "rf", "knn"]

TOI = ["wBI_16"]
# --- store all results in one nested dict ---
results = {}

for toi in TOI:
    results[toi] = {}

    FILE_NAME_TRAIN = f"AugEEG_train_{toi}.pkl"
    FILE_NAME_TEST = f"AugEEG_test_{toi}.pkl"

    for model in models:
        print(f"\n=== Running {model.upper()} (TOI: {toi}) ===")

        decoder = tDecoder(
            fPath=DATA_PATH,
            fileName_train=FILE_NAME_TRAIN,
            fileName_test=FILE_NAME_TEST,
            Trial_num=125,
            tWin=TWIN,
            model_type=model,
        )

        # save model-specific decode results
        results[toi][model] = decoder.Results["Decode"]

    print(f"\nAblation results (Toi: {toi}):")
    for model, v in results[toi].items():
        print(f"{model.upper():<10}: {np.mean(v):.3f}")

# # --- save everything once at the end ---
SAVE_PATH = DATA_PATH + "Results/" + "ablation_summary_ML_w.pkl"
with open(SAVE_PATH, "wb") as f:
    pickle.dump(results, f)

print(f"\nAll ablation results saved to {SAVE_PATH}")
