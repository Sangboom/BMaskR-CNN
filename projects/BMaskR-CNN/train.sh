# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0

# python train_net.py --config-file configs/bmask_rcnn_R_50_FPN_1x_multiscale.yaml --num-gpus 8 --eval-only --resume
# python train_net.py --config-file configs/bmask_rcnn_R_50_FPN_cityscapes.yaml --num-gpus 4
python train_net.py --config-file configs/bmask_rcnn_R_50_FPN_1x_armbench_ObjectOnly.yaml --num-gpus 1