ROOT_DIR=run_sort_results
mkdir -p ${ROOT_DIR}

NUM_EPOCHS=100
L=4
M=20
for N in 3 5 7
do
    for METHOD in deterministic_neuralsort deterministic_softsort stochastic_neuralsort stochastic_softsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.005
        do
            for TAU in 1024
            do
                for REPETITION in 0 1 2 3 4 5 6 7 8 9
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_sort.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=20
for N in 9 15
do
    for METHOD in deterministic_neuralsort deterministic_softsort stochastic_neuralsort stochastic_softsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.005
        do
            for TAU in 128
            do
                for REPETITION in 0 1 2 3 4 5 6 7 8 9
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_sort.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done