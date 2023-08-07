#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate diffusers-learning

folder_path="/Users/reasonknow/Downloads/models/sd"

for file_path in $folder_path/*; do
    if [ -f "$file_path" ]; then
        string=$(basename "$file_path")
        model_name="${string%%.*}"
        file_extension=".${string#*.}"

        echo "Model Name: $model_name"

        python /Volumes/workspace/python/diffusers-learning/pipeline/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "/Users/reasonknow/Downloads/models/sd/$string" \
            --scheduler_type dpm \
            --device cuda \
            --from_safetensors \
            --dump_path "/Users/reasonknow/Downloads/models/d/$model_name"
    fi
done