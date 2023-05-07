CUDA_VISIBLE_DEVICES=0 python \
    wikihow_contrastive.py \
    --batch_size 16 \
    --lr 1e-5 \
    --clip_grad_norm 1.0 \
    --valid_batch_size 60 \
    --dataset_dir gardening_data\
    --output wikihow_contrastive_garden_stp \
    --epoch 30 \
    --patient 10 \
    --num_beams 5 \
    --neg_num_total 5 \
    --neg_num 4 \
    --load wikihow_contrastive_garden_stp/BEST \
    --test_only