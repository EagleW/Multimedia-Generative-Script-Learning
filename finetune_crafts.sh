CUDA_VISIBLE_DEVICES=0 python \
    wikihow_contrastive.py \
    --batch_size 16 \
    --lr 2e-5 \
    --clip_grad_norm 1.0 \
    --valid_batch_size 60 \
    --dataset_dir crafts_data\
    --output wikihow_contrastive_crafts_stp \
    --epoch 30 \
    --patient 10 \
    --num_beams 5 \
    --neg_num_total 10 \
    --neg_num 5 \
    --wandb