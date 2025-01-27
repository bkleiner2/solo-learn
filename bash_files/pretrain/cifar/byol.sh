python3 ../../../main_pretrain.py \
    --dataset $1 \
    --backbone resnet18 \
    --data_dir ./datasets \
    --train_dir cifar100/train \
    --max_epochs 200 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 1e-3 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.1 \
    --weight_decay 15e-6 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --horizontal_flip_prob 0.5 \
    --color_jitter_prob 0.8 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name byol-$1-l2 \
    --project ssl-project \
    --entity cu-ssl-project \
    --save_checkpoint \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.99 \
    --momentum_classifier \
    --loss_function_to_use l2_dist \
    --wandb
#    --checkpoint_dir "trained_models/byol/32xuerht" \
#    --resume_from_checkpoint "trained_models/byol/32xuerht/byol-cifar100-32xuerht-ep=172.ckpt" \
