for FRAMEWORK in tf pytorch
do
    ROOT_DIR=benchmark_results_${FRAMEWORK}
    mkdir -p ${ROOT_DIR}
    for ((N=100;N<=4000;N+=100))
    do
        for METHOD in neuralsort softsort
        do
            RESULTS_DIR="N_${N}_${METHOD}"
            mkdir ${ROOT_DIR}/${RESULTS_DIR}
            for DEVICE in cpu cuda
            do
                OUTPUT_FILE="${ROOT_DIR}/${RESULTS_DIR}/N_${N}_${METHOD}_DEVICE_${DEVICE}.txt"
                python3 ${FRAMEWORK}/synthetic_experiment_speed_comparison.py --batch_size 20 --n ${N} --epochs 100 --device ${DEVICE} --method ${METHOD} --burnin 1 2>&1 | tee ${OUTPUT_FILE}
            done
        done
    done
done
