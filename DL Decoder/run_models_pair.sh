#!/bin/bash
# ============================================================
# Sequentially run all deep learning EEG ablation models
# Author: Woojae Jeong
# ============================================================

PYTHON_EXEC="python3"
MAIN_SCRIPT="main_dl_pair.py"

# --- Default data configuration ---
DATA_PATH="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/Modeling/Data/"

# --- Common decoding parameters ---
KFOLD=5
TRIALNUM=125
TSTART=150
TEND=200

# --- Define per-model hyperparameters (portable across shells) ---
get_epochs() {
  case "$1" in
    eegnet) echo 100 ;;
    shallowconvnet) echo 80 ;;
    deepconvnet) echo 80 ;;
    mlp) echo 60 ;;
    *) echo 50 ;;  # default
  esac
}

get_lr() {
  case "$1" in
    eegnet|shallowconvnet|deepconvnet|mlp) echo 0.001 ;;
    *) echo 0.001 ;;
  esac
}

get_batch() {
  case "$1" in
    eegnet|shallowconvnet|deepconvnet|mlp) echo 64 ;;
    *) echo 64 ;;
  esac
}

# --- List of models to run sequentially ---
# MODELS=("eegnet" "deepconvnet" "mlp")
MODELS=("deepconvnet")
TOI_Train=("Bio" "Int" "BI_16" "wBI_16")
TOI_Test=("Bio" "Int" "BI_16" "wBI_16")

echo "==========================================================="
echo " Running deep learning EEG ablation sequentially"
echo "==========================================================="

# --- Run each model in sequence ---

# --- Loop over each mode and avg_num ---
for Train in "${TOI_Train[@]}"; do
  for Test in "${TOI_Test[@]}"; do

    DATA_FILE_TRAIN="AugEEG_train_${Train}.pkl"
    DATA_FILE_TEST="AugEEG_test_${Test}.pkl"

    echo ""
    echo ">>> Train: ${Train}, Test: ${Test}"
    echo "==========================================================="

    # --- Run each model in sequence ---
    for MODEL in "${MODELS[@]}"; do
        EPOCHS=$(get_epochs ${MODEL})
        LR=$(get_lr ${MODEL})
        BATCH=$(get_batch ${MODEL})

        echo ""
        echo "▶ Starting model: ${MODEL}"
        echo "   epochs=${EPOCHS}, lr=${LR}, batch=${BATCH}"
        echo "-----------------------------------------------------------"

        ${PYTHON_EXEC} ${MAIN_SCRIPT} \
            --model ${MODEL} \
            --path ${DATA_PATH} \
            --file_train ${DATA_FILE_TRAIN} \
            --file_test ${DATA_FILE_TEST} \
            --kfold ${KFOLD} \
            --trialnum ${TRIALNUM} \
            --tstart ${TSTART} \
            --tend ${TEND} \
            --trainTOI ${Train} \
            --testTOI ${Test}

        echo "✅ Finished ${MODEL}"
        echo "-----------------------------------------------------------"
    done
  done
done

echo ""
echo "==========================================================="
echo " ✅ All models finished!"
echo "==========================================================="