#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <benchmark_name>"
    exit 1
fi

# Root Directories #mustafaaljadery #gradientai #winglian
GPUS="1" # GPU size for tensor_parallel (set to 1 as specified).
ROOT_DIR="root" # the path that stores generated task samples and model predictions.
MODEL_DIR="mustafaaljadery" # the path that contains individual model folders from Huggingface.
ENGINE_DIR="" # the path that contains individual engine folders from TensorRT-LLM.

# Model and Tokenizer
source config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

# Benchmark and Tasks
source config_tasks.sh
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start server (you may want to run in other container.)
if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python pred/serve_vllm.py \
        --model=microsoft/Phi-3-mini-128k-instruct \
        --tensor-parallel-size=1 \
        --dtype=bfloat16 \
        --disable-custom-all-reduce \
        --trust-remote-code \
        --batch-size=1 \
        --precision=bf16 \
        --offload \
        > server_vllm.log 2>&1 &
    SERVER_PID=$!
elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path=${MODEL_PATH} \
        > server_trtllm.log 2>&1 &
    SERVER_PID=$!
fi

# Wait for the server to start
sleep 10

# # Check if the server is running
# if ps -p $SERVER_PID > /dev/null
# then
#    echo "Server is running"
# else
#    echo "Failed to start server"
#    exit 1
# fi

# Start client (prepare data / call model API / obtain final metrics)
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}

        python pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            ${STOP_WORDS}
    done

    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done
