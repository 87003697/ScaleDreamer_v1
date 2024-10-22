CUDA_VISIBLE_DEVICES=4,5,6,7  python launch.py \
    --config configs/group_7/3DTopia__base2_step_4__asd_mv+rd_volsdf+mc-cpu_05_bg_v1__hexa_16_vanilla_16_none_lora_prompt_r2.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 