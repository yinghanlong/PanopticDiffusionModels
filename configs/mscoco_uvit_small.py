import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    #Set latent size
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=2000000,#1000000
        batch_size= 64, #128,#256
        log_interval=20,
        eval_interval=5000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,#lr=0.0002,
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit_t2i',
        img_size=32,
        in_chans=4,
        patch_size=2,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        clip_dim=768,
        num_clip_token=77,
        enable_panoptic=True, use_ground_truth=False, separate=True, num_panoptic_class=8, patch_factor=2 #1,2,4
    )

    config.dataset = d(
        name='mscoco256_features',
        path='/home/nano01/a/long273/coco256_features', #resolution=256 / 512
        cfg= True,
        p_uncond= 0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size= 32,
        cfg=True,
        scale=1.,
        path='/home/nano01/a/long273/results/sample'
    )
    #config.use_twophases=True
    config.use_unet=False
    config.mask_channel=1 #1 by default
    #config.pretrained = '/home/nano01/a/long273/results/mscoco_uvit_small/coco2017-1-baseline/ckpts/1000000.ckpt/nnet.pth'
    config.pretrained = '/home/min/a/long273/Documents/diffusers/U-ViT/pretrained_model/mscoco_uvit_small.pth'
    return config
