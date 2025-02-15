# Diffusion Self-Distillation for Zero-Shot Customized Image Generation

**arXiv 2024**

This repository represents the official implementation of the paper titled "Diffusion Self-Distillation for Zero-Shot Customized Image Generation".

*This repository is still under construction, many updates will be applied in the near future.*

[![Website](docs/badge-website.svg)](https://primecai.github.io/dsd/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2411.18616)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-Demo-yellow)](https://huggingface.co/papers/2411.18616)

[Shengqu Cai](https://primecai.github.io/),
[Eric Ryan Chan](https://ericryanchan.github.io/about.html),
[Yunzhi Zhang](https://cs.stanford.edu/~yzzhang/),
[Leonidas Guibas](https://www.cs.stanford.edu/people/leonidas-guibas),
[Jiajun Wu](https://jiajunwu.com/),
[Gordon Wetzstein](https://stanford.edu/~gordonwz/ )

Text-to-image diffusion models produce impressive results but are frustrating tools for artists who desire fine-grained control. For example, a common use case is to create images of a specific instance in novel contexts, i.e., "identity-preserving generation". This setting, along with many other tasks (e.g., relighting), is a natural fit for image+text-conditional generative models. However, there is insufficient high-quality paired data to train such a model directly. We propose Diffusion Self-Distillation, a method for using a pre-trained text-to-image model to generate its own dataset for text-conditioned image-to-image tasks. We first leverage a text-to-image diffusion model's in-context generation ability to create grids of images and curate a large paired dataset with the help of a Visual-Language Model. We then fine-tune the text-to-image model into a text+image-to-image model using the curated paired dataset. We demonstrate that Diffusion Self-Distillation outperforms existing zero-shot methods and is competitive with per-instance tuning techniques on a wide range of identity-preservation generation tasks, without requiring test-time optimization.

![teaser](docs/teaser.jpg)


## 🛠️ Setup

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/primecai/diffusion-self-distillation.git
cd diffusion-self-distillation
```

### 💻 Dependencies
For the environment, run:

```
pip install -r requirements.txt
```
You may need to setup Google Gemini API key to use the prompt enhancement feature, which is optional but highly recommended.


### 📦 Pretrained Models
Download our pretrained models from [Google Drive](https://drive.google.com/file/d/1z6cR3PbqnrlVjXJJlk6AYxdl8z18hvtL/view?usp=sharing) and unzip. You should have the following files:
- `transformers`
    - `config.json`
    - `diffusion_pytorch_model.safetensors`
- `pytorch_lora_weights.safetensors`
<!-- 
### 🔧 Configuration
Depends on where you store the data and checkpoints, you may need to change a few things in the configuration ```yaml``` file. We marked out the important lines you may want to take a look at in our exempler configurations files. -->

## 🏃 Inference
To generate subject-preserving images, simply run:

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path /PATH/TO/transformer \                         # Path to the 'transformer' folder
    --lora_path /PATH/TO/pytorch_lora_weights.safetensors \     # Path to the 'pytorch_lora_weights.safetensors' file
    --image_path /PATH/TO/conditioning_image.png \              # Path to the conditioning image
    --text "this character sitting on a chair" \                # Text prompt
    --output_path output.png \                                  # Path to save the output image
    --guidance 3.5 \                                            # Guidance scale
    --i_guidance 1.0 \                                          # True image guidance scale, set to >1.0 if you want to enhance the image conditioning
    --t_guidance 1.0 \                                          # True text guidance scale, set to >1.0 if you want to enhance the text conditioning
    # --disable_gemini_prompt \                                 # Disable Gemini prompt enhancement, not recommended unless you have a very detailed prompt
```




## Training
TBD

## 🎓 Citation

Please cite our paper:

```bibtex
@inproceedings{cai2024dsd,
    author={Cai, Shengqu and Chan, Eric Ryan and Zhang, Yunzhi and Guibas, Leonidas and Wu, Jiajun and Wetzstein, Gordon.},
    title={Diffusion Self-Distillation for Zero-Shot Customized Image Generation},
    booktitle={arXiv},
    year={2024}
}       
```