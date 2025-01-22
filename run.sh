#bin/bash

CUDA_VISIBLE_DEVICES=0 python -m src.stages.format_datasets --config params.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.stages.translate_datasets --config params.yaml --batch-size 5

