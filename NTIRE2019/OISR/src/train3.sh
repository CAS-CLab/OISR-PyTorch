python main.py --model EDSR --scale 1 --patch_size 68 --save OISR --data_train DIV2K --data_test B100 --n_feats 512 --dir_data ../../data2 --ext bin --gpu-id 0,1,2,3,4,5,6,7 --chop --epochs 100 --lr_decay 250 --n_GPUs 8 --loss 1*SL1 --lr 1.5e-5 --pre_train ./model_best.pt --data_range 1-52/85-105