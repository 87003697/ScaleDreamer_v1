
NUM_TRAINERS=8
HOST_NODE_ADDR=10.21.21.181


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 torchrun \
    --nnodes=$2 \
    --nproc-per-node=$NUM_TRAINERS \
    --max-restarts=1 \
    --node_rank=$3 \
    --master_port=$1 \
    --master_addr=$HOST_NODE_ADDR \
    launch.py \
        --config configs/group_13/DE+MJ__diffmc-001_dmd_v3_iters_3w_ac_1__eik_0_eps_1e-3_spars_10-50.yaml \
        --train \
        data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
        data.condition_processor.cache_dir="/media/test/.threestudio_cache/text_embeddings_DALLE_Midjourney" \
        data.guidance_processor.cache_dir="/media/test/.threestudio_cache/text_embeddings_DALLE_Midjourney" 
