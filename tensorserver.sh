#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LOG_DIR="${DIR}/logs/server/server.log"
MODEL=${1:-"resnet"}

tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=fashion_model \
  --model_base_path="$DIR/models/$MODEL" > $LOG_DIR 2>&1