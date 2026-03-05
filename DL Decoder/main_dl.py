import argparse, numpy as np, csv, os, datetime
from decoders.dl_decoder import DLDecoder
import torch, random


# ------------------------------------------------------------
#  Set deterministic behavior for reproducibility
# ------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed(42)


# ------------------------------------------------------------
#  Default hyperparameters per model
# ------------------------------------------------------------
def get_hparams(model):
    presets = {
        "eegnet": dict(epochs=100, lr=1e-3, batch_size=64, patience=10),
        "shallowconvnet": dict(epochs=80, lr=1e-3, batch_size=64, patience=10),
        "deepconvnet": dict(epochs=80, lr=1e-3, batch_size=64, patience=10),
        "mlp": dict(epochs=60, lr=1e-3, batch_size=64, patience=8),
    }
    return presets.get(model.lower(), dict(epochs=50, lr=1e-3, batch_size=64))


# ------------------------------------------------------------
#  CSV logging helper
# ------------------------------------------------------------
def append_csv(log_path, row):
    header = [
        "model",
        "TOI",
        "mean_bacc",
        "se",
        "avg_train_time_s",
        "n_subjects",
        "datetime",
    ]
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# ------------------------------------------------------------
#  Main runner
# ------------------------------------------------------------
def main(args):
    print(
        f"\n=== Running {args.model.upper()} ablation "
        f"(TOI: {args.TOI}, 80/10/10 split) ==="
    )

    hyper = get_hparams(args.model)

    # Construct output filename
    save_name = f"{args.model}_{args.TOI}_ablation.pkl"

    # Initialize decoder (80/10/10 split built inside)
    decoder = DLDecoder(
        fPath=args.path,
        fileName_train=args.file_train,
        fileName_test=args.file_test,
        model_type=args.model,
        k_fold=args.kfold,
        Trial_num=args.trialnum,
        tWin=np.arange(args.tstart, args.tend),
        n_classes=2,
        hyper=hyper,
        saveName=save_name,
    )

    mean, se, avg_time = decoder._decode()  # returns tuple

    # Log results to CSV
    log_file = os.path.join(args.path, "Results/ablation_summary.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    append_csv(
        log_file,
        [
            args.model,
            args.TOI,
            f"{mean:.3f}",
            f"{se:.3f}",
            f"{avg_time:.1f}",
            len(decoder.subIdx),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ],
    )

    print(f"Logged results to {log_file}")


# ------------------------------------------------------------
#  CLI Arguments
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Deep-learning EEG ablation study (80/10/10)"
    )
    p.add_argument(
        "--model",
        type=str,
        default="eegnet",
        choices=["eegnet", "shallowconvnet", "deepconvnet", "mlp"],
    )
    p.add_argument(
        "--path",
        type=str,
        default="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/Modeling/Data/",
    )
    p.add_argument("--file_train", type=str, default="AugEEG_12_train.pkl")
    p.add_argument("--file_test", type=str, default="AugEEG_12_test.pkl")
    p.add_argument("--tstart", type=int, default=150)
    p.add_argument("--tend", type=int, default=200)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--trialnum", type=int, default=125)
    p.add_argument("--TOI", type=str, required=True)
    args = p.parse_args()

    main(args)
