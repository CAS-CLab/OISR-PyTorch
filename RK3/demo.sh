# Standard benchmarks (Multi-GPU)
# OISR LF-s model (x2)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 2 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --pre_train ./LF-s/x2.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results
# OISR LF-s model (x3)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 3 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --pre_train ./LF-s/x3.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results
# OISR LF-s model (x4)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --pre_train ./LF-s/x4.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results

# OISR LF-m model (x2)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 2 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --n_feats 122 --pre_train ./LF-m/x2.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results
# OISR LF-m model (x3)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 3 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --n_feats 122 --pre_train ./LF-m/x3.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results
# OISR LF-m model (x4)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --n_feats 122 --pre_train ./LF-m/x4.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results

# OISR RK2-s model (x2)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 2 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --pre_train ./RK2-s/x2.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results
# OISR RK2-s model (x3)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 3 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --pre_train ./RK2-s/x3.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results
# OISR RK2-s model (x4)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --pre_train ./RK2-s/x4.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results

# OISR RK2-m model (x2)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 2 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --n_feats 122 --pre_train ./RK2-m/x2.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results
# OISR RK2-m model (x3)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 3 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --n_feats 122 --pre_train ./RK2-m/x2.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results
# OISR RK2-m model (x4)
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 8 --n_feats 122 --pre_train ./RK2-m/x2.pt --gpu-id 4,5,6,7 --chop --n_GPUs 4 --save_results

# OISR RK3 model (x2)
python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 2 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 22 --n_feats 256 --gpu-id 4,5,6,7 --n_GPUs 4 --chop --pre_train ./RK3/x2.pt --save_results
# OISR RK3 model (x3)
python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 3 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 22 --n_feats 256 --gpu-id 4,5,6,7 --n_GPUs 4 --chop --pre_train ./RK3/x3.pt
# OISR RK3 model (x4)
python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 22 --n_feats 256 --gpu-id 4,5,6,7 --n_GPUs 4 --chop --pre_train ./RK3/x4.pt

# Standard benchmarks (Single-GPU)
# E.g., RK3 x2
# python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 2 --test_only --ext bin --dir_data /your/downloaded/data/path --n_resblocks 22 --n_feats 256 --gpu-id 4 --pre_train ./RK3/x2.pt

# How to train / fine-tune on your own datasets?
# Please refer to our NTIRE2019 example (https://github.com/HolmesShuan/OISR-PyTorch/tree/master/NTIRE2019).


# EDSR baseline model (x2) + JPEG augmentation
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results

# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

