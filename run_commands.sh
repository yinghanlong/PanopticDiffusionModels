ssh long273@cbric-gpu21.ecn.purdue.edu
cd Documents
source myenv/bin/activate
cd diffusers/U-ViT
source ~/.bashrc

find . -depth -name "*_text.txt" -exec sh -c 'f="{}"; mv -- "$f" "${f%_text.txt}.txt"' \;

python -m clip_score /home/nano01/a/long273/results/mscoco_uvit_mid/coco2017-3-mid-patch2/samples /home/nano01/a/long273/coco256_features/val2017 --device cuda:3
python -m clip_score /home/nano01/a/long273/results/mscoco_uvit_small/coco2017-3-mask4-tanh-noise2/samples /home/nano01/a/long273/coco256_features/val2017 --device cuda:3
CUDA_VISIBLE_DEVICES=3
# CIFAR10 (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/cifar10_uvit_small.py

# CelebA 64x64 (U-ViT-S/4)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/celeba64_uvit_small.py 

# ImageNet 64x64 (U-ViT-M/4)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/imagenet64_uvit_mid.py

# ImageNet 64x64 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train.py --config=configs/imagenet64_uvit_large.py

# ImageNet 256x256 (U-ViT-L/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/imagenet256_uvit_large.py

# ImageNet 256x256 (U-ViT-H/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm_discrete.py --config=configs/imagenet256_uvit_huge.py

# ImageNet 512x512 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/imagenet512_uvit_large.py

# ImageNet 512x512 (U-ViT-H/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm_discrete.py --config=configs/imagenet512_uvit_huge.py

# MS-COCO (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_mid.py
accelerate launch  --num_processes 1 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py

# MS-COCO (U-ViT-S/2, Deep)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py --config.nnet.depth=16