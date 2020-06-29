ROOT_DIR=benchmark_results_learning_curve
BATCH_SIZE=20
N=4000
EPOCHS=100
DEVICE=cpu

mkdir -p ${ROOT_DIR}

FRAMEWORK=tf
python3 ${FRAMEWORK}/synthetic_experiment_learning_curves.py \
    --batch_size ${BATCH_SIZE} \
    --n ${N} \
    --epochs ${EPOCHS} \
    --device ${DEVICE} \
    --method softsort \
    --tau 0.03 \
    --pow 2.0 \
    2>&1 | tee ${ROOT_DIR}/benchmark_results_learning_curve_softsort_p2_${FRAMEWORK}_${N}_${BATCH_SIZE}.txt


python3 ${FRAMEWORK}/synthetic_experiment_learning_curves.py \
    --batch_size ${BATCH_SIZE} \
    --n ${N} \
    --epochs ${EPOCHS} \
    --device ${DEVICE} \
    --method softsort \
    --tau 0.1 \
    --pow 1.0 \
    2>&1 | tee ${ROOT_DIR}/benchmark_results_learning_curve_softsort_p1_${FRAMEWORK}_${N}_${BATCH_SIZE}.txt


python3 ${FRAMEWORK}/synthetic_experiment_learning_curves.py \
    --batch_size ${BATCH_SIZE} \
    --n ${N} \
    --epochs ${EPOCHS} \
    --device ${DEVICE} \
    --method neuralsort \
    --tau 100.0 \
    2>&1 | tee ${ROOT_DIR}/benchmark_results_learning_curve_neuralsort_${FRAMEWORK}_${N}_${BATCH_SIZE}.txt
