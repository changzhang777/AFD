<div align="center">

<h1>AFD: Mitigating Feature Gap for Adversarial  Robustness by Feature Disentanglement</h1>

<div>
    <a href='https://scholar.google.com/citations?user=wxC_XDMAAAAJ' target='_blank'>Nuoyan Zhou</a><sup>1</sup>&emsp;
    <a target='_blank'>Dawei Zhou</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=5fHHi24AAAAJ' target='_blank'>Decheng Liu</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=SRBn7oUAAAAJ' target='_blank'>Nannan Wang</a><sup>1</sup>&emsp;
          <a href='https://scholar.google.com/citations?user=VZVTOOIAAAAJ' target='_blank'>Xinbo Gao</a><sup>2</sup>
</div>
<div>
    <sup>1</sup>State Key Laboratory of Integrated Services Networks, Xidian University, Xi'an, China&emsp;<br>
    <sup>2</sup>Chongqing Key Laboratory of Image Cognition, Chongqing University of Posts and Telecommunications, Chongqing, China;<br>
</div>
<div>
</div>
<div>
    <strong>AAAI 2025</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://arxiv.org/abs/2410.18666" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv%20paper-2410.18666-b31b1b.svg">
        </a>
        <a href="https://huggingface.co/shallowdream204/DreamClear/tree/main" target='_blank'>
        <img src="https://img.shields.io/badge/🤗%20Weights-DreamClear-yellow">
        </a>
        <img src="https://visitor-badge.laobi.icu/badge?page_id=shallowdream204/DreamClear">
    </h4>
</div>

⭐ If AFD is helpful to your projects, please help star this repo. Thanks! 🤗


</div>

<be>


## 🔥 News
- **2024.12.10**: This repo is created.

## 📸 Pipeline
<img src="assets/overview.png" height="400px"/>


## 🔧 Dependencies and Installation

1. Clone this repo and navigate to DreamClear folder

   ```bash
   git clone https://github.com/changzhang777/AFD.git
   cd AFD
   ```

2. Create Conda Environment and Install Package

   ```bash
   conda create -n AFD python=3.9 -y
   conda activate AFD
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
   pip install 
   ```
3. Download Pre-trained Models (All models except for llava can be downloaded at [Huggingface](https://huggingface.co/shallowdream204/DreamClear/tree/main) for convenience.)
      #### Base Model:
      * `PixArt-α-1024`: [PixArt-XL-2-1024-MS.pth](https://huggingface.co/PixArt-alpha/PixArt-alpha/blob/main/PixArt-XL-2-1024-MS.pth)
      * `VAE`: [sd-vae-ft-ema](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema)
      * `T5 Text Encoder`: [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl)
      * `LLaVA`: [llava-v1.6-vicuna-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b)
      * `SwinIR`: [general_swinir_v1.ckpt](https://huggingface.co/lxq007/DiffBIR/blob/main/general_swinir_v1.ckpt)
      #### Ours provided Model:
      * `DreamClear`: [DreamClear-1024.pth](https://huggingface.co/shallowdream204/DreamClear/blob/main/DreamClear-1024.pth)
      * `RMT for Segmentation`: [rmt_uper_s_2x.pth](https://huggingface.co/shallowdream204/DreamClear/blob/main/rmt_uper_s_2x.pth)
      * `RMT for Detection`: [rmt_maskrcnn_s_1x.pth](https://huggingface.co/shallowdream204/DreamClear/blob/main/rmt_maskrcnn_s_1x.pth)
      
## 🎰 Train
#### I - Prepare training data




#### II - Training for DreamClear
Run the following command to train ADF with default settings:
```shell
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... \
    train_dreamclear.py configs/DreamClear/DreamClear_Train.py \
    --load_from /path/to/PixArt-XL-2-1024-MS.pth \
    --vae_pretrained /path/to/sd-vae-ft-ema \
    --swinir_pretrained /path/to/general_swinir_v1.ckpt \
    --val_image /path/to/RealLQ250/lq/val_image.png \
    --val_npz /path/to/RealLQ250/npz/val_image.npz \
    --work_dir experiments/train_dreamclear
```
Please modify the path of training datasets in `ADF/tl+sl.py`. You can also modify the training hyper-parameters (e.g., `lr`, `train_batch_size`, `gradient_accumulation_steps`) in this file, according to your own GPU machines.
## ⚡ Inference
We provide the AutoAttack
#### Testing ADF for AutoAttack\PGD-20\CW


Run the following command:
```shell
python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 \
    test.py configs/DreamClear/DreamClear_Test.py \
    --dreamclear_ckpt /path/to/DreamClear-1024.pth \
    --swinir_ckpt /path/to/general_swinir_v1.ckpt \
    --vae_ckpt /path/to/sd-vae-ft-ema \
    --t5_ckpt /path/to/t5-v1_1-xxl \
    --llava_ckpt /path/to/llava-v1.6-vicuna-13b \
    --lre --cfg_scale 4.5 --color_align wavelet \
    --image_path /path/to/input/images \
    --save_dir validation \
    --mixed_precision fp16 \
    --upscale 4
```
## 🪪 License

The provided code and pre-trained weights are licensed under the [Apache 2.0 license](LICENSE).

## 🤗 Acknowledgement

This code is based on [ANCRA](https://github.com/changzhang777/ANCRA). 

## 📧 Contact
If you have any questions, please feel free to reach me out at nuoyanzhou@stu.xidian.edu.cn. 

## 📖 Citation
If you find our work useful for your research, please consider citing our paper:
```
@article{zhou2024mitigating,
  title={Mitigating Feature Gap for Adversarial Robustness by Feature Disentanglement},
  author={Zhou, Nuoyan and Zhou, Dawei and Liu, Decheng and Gao, Xinbo and Wang, Nannan},
  journal={arXiv preprint arXiv:2401.14707},
  year={2024}
}
```
