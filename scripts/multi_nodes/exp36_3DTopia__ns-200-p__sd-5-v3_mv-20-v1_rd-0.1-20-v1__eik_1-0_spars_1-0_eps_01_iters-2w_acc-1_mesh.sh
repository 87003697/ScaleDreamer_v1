
HOST_NODE_ADDR=10.21.21.181

port=$1
num_nodes=$2
node_rank=$3
nproc_per_node=8

# Calculate the total number of GPUs (for reference)
total_gpus=$((num_nodes * nproc_per_node))

# Export the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((total_gpus - 1)))

# # count the number of GPUs given the nnodes and nproc-per-node
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
torchrun \
    --nnodes=$num_nodes \
    --nproc-per-node=$nproc_per_node \
    --max-restarts=1 \
    --node_rank=$node_rank \
    --master_port=$port \
    --master_addr=$HOST_NODE_ADDR \
    launch.py \
        --config configs/to_release_3/3DTopia__ns-200-p__sd-5-v3_mv-20-v1_rd-0.1-20-v1__eik_1-0_spars_1-0_eps_01_iters-2w_acc-1_mesh.yaml \
        --train \
        trainer.num_nodes=$2 \
        data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
        data.condition_processor.cache_dir="/dfs/ai-storage/Scaledreamer/.threestudio_cache/text_embeddings_DALLE_Midjourney" \
        data.guidance_processor.cache_dir="/dfs/ai-storage/Scaledreamer/.threestudio_cache/text_embeddings_DALLE_Midjourney" 
