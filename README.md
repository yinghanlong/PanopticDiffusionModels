## Panoptic Diffusion Models: co-generation of images and segmentation maps <sub><small>(CVPR 2025 Workshop on Generative Models for Computer Vision)</small></sub>

💡 This is the official implementation of Panoptic Diffusion Models.
[Short version for CVPR Workshop](https://generative-vision.github.io/workshop-CVPR-25/papers/56.pdf)
[ArXiv](https://arxiv.org/abs/2412.02929)

* More detailed instructions will be added

This implementation is based on
* [UViT](https://github.com/baofff/U-ViT/tree/main) (Provide the baseline model)
* [Extended Analytic-DPM](https://github.com/baofff/Extended-Analytic-DPM) (provide the FID reference statistics on CIFAR10 and CelebA 64x64)
* [guided-diffusion](https://github.com/openai/guided-diffusion) (provide the FID reference statistics on ImageNet)
* [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (provide the official implementation of FID to PyTorch)
* [dpm-solver](https://github.com/LuChengTHU/dpm-solver) (provide the sampler)
