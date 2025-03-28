CUDA_VISIBLE_DEVICES=4,5,6,7  python launch.py \
    --config configs/group_8-1/3DTopia__base2_step_4__asd_mv+rd+sd_volsdf+diffmc-0_hexa_16_vanilla_16_wobias_lora_prompt_v4.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 
