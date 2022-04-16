python3 ../../../main_pretrain.py \
    --dataset $1 \
    --backbone resnet18 \
    --data_dir ./datasets \
    --max_epochs 200 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer adamw \
    --scheduler warmup_cosine \
    --lr 0.001 \
    --classifier_lr 0.1 \
    --weight_decay 0.01 \
    --batch_size 256 \
    --num_workers 4 \
    --crop_size 32 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --zero_init_residual \
    --name simsiam-$1-momentum99-995 \
    --project ssl-project \
    --entity cu-ssl-project \
    --save_checkpoint \
    --method simsiammom \
    --proj_hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --proj_output_dim 2048 \
    --wandb \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.995 \
    --momentum_classifier
#    --checkpoint_dir "trained_models/simsiammom/simsiam-cifar100-momentum_asdfasd" \
#    --resume_from_checkpoint "trained_models/simsiammom/simsiam-cifar100-momentum_asdfasd/simsiam-cifar100-momentum99-995-simsiam-cifar100-momentum_asdfasd-ep=23.ckpt"
#    --wandb_version simsiam-$1-momentum_asdfasd \
