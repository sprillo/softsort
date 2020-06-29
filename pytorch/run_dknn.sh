ROOT_DIR=run_dknn_results
mkdir -p ${ROOT_DIR}

NUM_EPOCHS=200
for DATASET in mnist cifar10 fashion-mnist
do
    for METHOD in stochastic deterministic
    do
        GRID_SEARCH_RESULTS_DIR="${DATASET}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for TAU in 1 4 16 64 128 512
        do
            for k in 1 3 5 9
            do  
                for LR in 3 4 5
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/${DATASET}_${METHOD}_TAU_${TAU}_k_${k}_lr_${LR}.txt"
                    python3 run_dknn.py --k=${k} --tau=${TAU} --nloglr=${LR} --method=${METHOD} --dataset=${DATASET} --num_epochs=${NUM_EPOCHS} --simple 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done
