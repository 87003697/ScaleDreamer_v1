CUDA_VISIBLE_DEVICES=4,5,6  python launch.py \
    --config configs/group_6/3DTopia__turbo*2_step_4__asd_mv+rd_volsdf+diffmc_grad_01__hexa_16_vanilla_16_none_lora_prompt_v3-2.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 
