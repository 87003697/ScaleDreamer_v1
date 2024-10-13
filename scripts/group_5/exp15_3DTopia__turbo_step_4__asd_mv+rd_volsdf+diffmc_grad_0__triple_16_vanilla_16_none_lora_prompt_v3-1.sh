# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  python launch.py \
#     --config configs/group_5/3DTopia__turbo_step_4__asd_mv+rd_volsdf+diffmc_grad_0__triple_16_vanilla_16_none_lora_prompt_v3-1.yaml \
#     --train \
#     data.prompt_library="3DTopia_361k_prompt_library" \
#     data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
#     data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

CUDA_VISIBLE_DEVICES=0  python launch.py \
    --config configs/group_5/3DTopia__turbo_step_4__asd_mv+rd_volsdf+diffmc_grad_0__triple_16_vanilla_16_none_lora_prompt_v3-1.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 
