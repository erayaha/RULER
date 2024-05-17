# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(
    #1048576
    #262144
    65536
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3
    
    case $MODEL_NAME in #winglian/llama-3-1m-context-gradient-lora #hus960/llama-3-8b-1m-PoSE-Q4_K_M-GGUF
        #MoMonir/dolphin-2.9-llama3-8b-1m-GGUF
        llama3_1m_lora)
            MODEL_PATH="${MODEL_DIR}/dolphin-2.9-llama3-8b-1m-GGUF" 
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="hf"
            ;;
        chatglm6_128k)
            MODEL_PATH="${MODEL_DIR}/chatglm3-6b-128k" 
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="vllm"
            ;;
        gemma2_10M)
            MODEL_PATH="${MODEL_DIR}/gemma-2B-10M" #/TinyLlama-1.1B-Chat-v1.0"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="hf"
            ;;
        phi3_128k)
            MODEL_PATH="${MODEL_DIR}/Phi-3-mini-128k-instruct" #/TinyLlama-1.1B-Chat-v1.0"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="hf"
            ;;
        llama3_1M)
            MODEL_PATH="${MODEL_DIR}/Llama-3-8B-Instruct-Gradient-1048k" #/TinyLlama-1.1B-Chat-v1.0"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="hf"
            ;;
        llama3)
            MODEL_PATH="${MODEL_DIR}/Meta-Llama-3-8B" #/TinyLlama-1.1B-Chat-v1.0"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="hf"
            ;;
        gpt-3.5-turbo)
            MODEL_PATH="gpt-3.5-turbo-0125"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="openai"
            TOKENIZER_PATH="cl100k_base"
            TOKENIZER_TYPE="openai"
            OPENAI_API_KEY=""
            AZURE_ID=""
            AZURE_SECRET=""
            AZURE_ENDPOINT=""
            ;;
        gpt-4-turbo)
            MODEL_PATH="gpt-4"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="openai"
            TOKENIZER_PATH="cl100k_base"
            TOKENIZER_TYPE="openai"
            OPENAI_API_KEY=""
            AZURE_ID=""
            AZURE_SECRET=""
            AZURE_ENDPOINT=""
            ;;
        gemini_1.0_pro)
            MODEL_PATH="gemini-1.0-pro-latest"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="gemini"
            TOKENIZER_PATH=$MODEL_PATH
            TOKENIZER_TYPE="gemini"
            GEMINI_API_KEY=""
            ;;
        gemini_1.5_pro)
            MODEL_PATH="gemini-1.5-pro-latest"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="gemini"
            TOKENIZER_PATH=$MODEL_PATH
            TOKENIZER_TYPE="gemini"
            GEMINI_API_KEY=""
            ;;
    esac


    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi


    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}