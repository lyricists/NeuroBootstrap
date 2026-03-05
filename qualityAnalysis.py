# ============================================================
# ERP Difference & SNR Quality Test across TOIs
# Author: Woojae Jeong
# ============================================================

import numpy as np
import pickle
from tqdm import tqdm

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
fPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/Modeling/Data/"
bPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/"
fileName = "train_test_PCA_BI_w.pkl"
TOI_list = ["Bio", "Int", "BI_h", "BI"]

# Time windows
baseWin = np.arange(0, 50)  # −200 to 0 ms
sigWin = np.arange(150, 200)  # 300–600 ms

n_sub = 137
n_fold = 5

results = {"ERP_diff": {}, "SNR": {}}

# ------------------------------------------------------------
# Loop over TOIs
# ------------------------------------------------------------
for TOI in TOI_list:
    print(f"\nProcessing TOI: {TOI}")

    with open(fPath + fileName, "rb") as file:
        Dataset = pickle.load(file)

    split_data = Dataset["split_data"][TOI]
    pcaDataset = Dataset["pcaData"]

    erp_all = []
    snr_all = []

    # --------------------------------------------------------
    # Subject Loop
    # --------------------------------------------------------
    for sub in tqdm(range(n_sub), desc=f"Subject Loop ({TOI})"):
        erp_sub = []
        snr_sub = []

        for k in range(n_fold):
            EEG = pcaDataset[k]["pcaData"][:, :, :, sub]

            # ===== ERP Difference =====
            ERP_type1 = EEG[:, :, split_data[sub][k]["positive"]][:, sigWin, :]
            ERP_type2 = EEG[:, :, split_data[sub][k]["negative"]][:, sigWin, :]
            diff_val = np.mean(np.mean(np.mean(ERP_type1 - ERP_type2, axis=2), axis=1))
            erp_sub.append(diff_val)

            # ===== SNR =====
            # ERP estimate (average across trials)
            EEG_all = np.concatenate(
                (
                    EEG[:, :, split_data[sub][k]["positive"]],
                    EEG[:, :, split_data[sub][k]["negative"]],
                ),
                axis=2,
            )
            ERP_est = np.mean(EEG_all, axis=2)  # [channels x time]

            # Mean-square of ERP signal and baseline
            ms_signal = np.mean(ERP_est[:, sigWin] ** 2)
            ms_baseline = np.mean(ERP_est[:, baseWin] ** 2)

            # SNR
            snr_val = 20 * np.log10(np.sqrt(ms_signal) / (np.sqrt(ms_baseline) + 1e-12))
            snr_sub.append(snr_val)

        erp_all.append(np.mean(erp_sub))
        snr_all.append(np.mean(snr_sub))

    # --------------------------------------------------------
    # Store subject-wise results
    # --------------------------------------------------------
    erp_all = np.array(erp_all)
    snr_all = np.array(snr_all)

    results["ERP_diff"][TOI] = {
        "mean": float(np.mean(erp_all)),
        "sem": float(np.std(erp_all) / np.sqrt(n_sub)),
        "values": erp_all,
    }

    results["SNR"][TOI] = {
        "mean": float(np.mean(snr_all)),
        "sem": float(np.std(snr_all) / np.sqrt(n_sub)),
        "values": snr_all,
    }

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
save_path = fPath + "Results/QualityTest_results.pkl"
with open(save_path, "wb") as file:
    pickle.dump(results, file)

print("\nSaved all results to:", save_path)
