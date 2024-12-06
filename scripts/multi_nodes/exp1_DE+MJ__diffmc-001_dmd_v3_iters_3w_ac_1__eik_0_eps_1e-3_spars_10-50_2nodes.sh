
NUM_NODES=2
NUM_TRAINERS=8
HOST_NODE_ADDR=10.21.21.181

if [ -z "$1" ]; then
  echo "error: missing JOB_ID"
  exit 1
fi
JOB_ID=$1

torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    python launch.py \
        --config configs/group_13/DE+MJ__diffmc-001_dmd_v3_iters_3w_ac_1__eik_0_eps_1e-3_spars_10-50.yaml \
        --train \
        data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
        data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" \
        data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" 
