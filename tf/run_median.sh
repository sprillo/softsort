ROOT_DIR=run_median_results
mkdir -p ${ROOT_DIR}

NUM_EPOCHS=100
L=4
M=5
for N in 5
do
    for METHOD in deterministic_neuralsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 1024
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 5
do
    for METHOD in stochastic_neuralsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 2048
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 5
do
    for METHOD in deterministic_softsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 2048
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 5
do
    for METHOD in stochastic_softsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 4096
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 9
do
    for METHOD in deterministic_neuralsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 512
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 9
do
    for METHOD in stochastic_neuralsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 512
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 9
do
    for METHOD in deterministic_softsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 2048
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 9
do
    for METHOD in stochastic_softsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 2048
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 15
do
    for METHOD in deterministic_neuralsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 1024
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 15
do
    for METHOD in stochastic_neuralsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 4096
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 15
do
    for METHOD in deterministic_softsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 256
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

NUM_EPOCHS=100
L=4
M=5
for N in 15
do
    for METHOD in stochastic_softsort
    do
        GRID_SEARCH_RESULTS_DIR="N_${N}_${METHOD}"
        mkdir ${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}
        for LR in 0.001
        do
            for TAU in 2048
            do
                for REPETITION in 0
                do
                    OUTPUT_FILE="${ROOT_DIR}/${GRID_SEARCH_RESULTS_DIR}/N_${N}_${METHOD}_TAU_${TAU}_LR_${LR}_E_${NUM_EPOCHS}_REP_${REPETITION}.txt"
                    python3 run_median.py --num_epochs ${NUM_EPOCHS} --l=${L} --lr=${LR} --M=${M} --n=${N} --tau=${TAU} --method=${METHOD} --repetition=${REPETITION} 2>&1 | tee ${OUTPUT_FILE}
                done
            done
        done
    done
done

