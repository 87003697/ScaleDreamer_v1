# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
#     --config configs/group_14/DE+MJ__grid-diffmc-001_dmd_v0_iters_3w_ac_1__eik_0_eps_1e-3_spars_10-50.yaml \
#     --train \
#     data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
#     data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" \
#     data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" 

# CUDA_VISIBLE_DEVICES=1 python launch.py \
#     --config configs/group_14/DE+MJ__grid-diffmc-001_dmd_v0_iters_3w_ac_1__eik_0_eps_1e-3_spars_10-50.yaml \
#     --train \
#     data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
#     data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" \
#     data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" 


CUDA_VISIBLE_DEVICES=0 python launch.py \
    --config configs/group_14/DE+MJ__grid-diffmc-001_dmd_v0_iters_3w_ac_1__eik_0_eps_1e-3_spars_10-50.yaml \
    --train \
    data.prompt_library="dreamfusion_415_prompt_library" \
    trainer.val_check_interval=10 \
    data.batch_size=8
      
