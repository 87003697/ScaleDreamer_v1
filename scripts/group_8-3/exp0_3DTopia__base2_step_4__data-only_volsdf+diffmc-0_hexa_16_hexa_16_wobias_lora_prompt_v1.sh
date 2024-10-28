CUDA_VISIBLE_DEVICES=0  python launch.py \
    --config configs/group_8-3/3DTopia__base2_step_4__data-only_volsdf+diffmc-0_hexa_16_vanilla_16_wobias_lora_prompt_v1.yaml \
    --train \
    data.obj_library="Gobjaverse_53197_obj_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_Gobjaverse" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_Gobjaverse" 
