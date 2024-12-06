NUM_TRAINERS=8

torchrun \
    --nnodes=1 \
    --standalone \
    --nproc-per-node=$NUM_TRAINERS \
    launch.py \
        --config configs/group_13/DE+MJ__diffmc-001_dmd_v3_iters_3w_ac_1__eik_0_eps_1e-3_spars_10-50.yaml \
        --train \
        trainer.num_nodes=$2 \
        data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
        data.condition_processor.cache_dir="/media/test/.threestudio_cache/text_embeddings_DALLE_Midjourney" \
        data.guidance_processor.cache_dir="/media/test/.threestudio_cache/text_embeddings_DALLE_Midjourney" 
