CUDA_VISIBLE_DEVICES=4,5,6,7  python launch.py \
    --config configs/multi-prompt_benchmark/mse_mv_triplane_transformer_volsdf_10k.yaml \
    --train \
    system.prompt_processor.prompt_library="dreamfusion_415_prompt_library"
