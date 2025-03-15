import gradio as gr
import torch
from PIL import Image

from diffusers.utils import load_image
from pipeline import FluxConditionalPipeline
from transformer import FluxTransformer2DConditionalModel
from recaption import enhance_prompt
import os

pipe = None

CHECKPOINT = "primecai/dsd_model"

device = "cpu"
dtype = torch.float32
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
#dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

transformer = FluxTransformer2DConditionalModel.from_pretrained(
    CHECKPOINT,
    subfolder="transformer",
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
#    use_auth_token=os.getenv("HF_TOKEN"),
)
pipe = FluxConditionalPipeline.from_pretrained(
    "cocktailpeanut/xulf-d",
#    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=dtype,
#    use_auth_token=os.getenv("HF_TOKEN"),
)
pipe.load_lora_weights(
    CHECKPOINT,
    weight_name="pytorch_lora_weights.safetensors",
#    use_auth_token=os.getenv("HF_TOKEN"),
)
pipe.enable_model_cpu_offload()
#pipe.to(device, dtype=dtype)

def generate_image(
    image: Image.Image,
    text: str,
    gemini_prompt: bool = True,
    guidance: float = 3.5,
    i_guidance: float = 1.0,
    t_guidance: float = 1.0,
    num_images: int = 1,
):
    w, h, min_size = image.size[0], image.size[1], min(image.size)
    image = image.crop(
        ((w - min_size) // 2, (h - min_size) // 2, (w + min_size) // 2, (h + min_size) // 2)
    ).resize((512, 512))

    control_image = load_image(image)
    text_list = []
    for _ in range(num_images):
        if gemini_prompt:
            text = enhance_prompt(image, text.strip())
        text_list.append(text.strip())
    result_image = pipe(
        prompt=text_list,
        negative_prompt="",
        num_inference_steps=28,
        height=512,
        width=1024,
        guidance_scale=guidance,
        image=control_image,
        guidance_scale_real_i=i_guidance,
        guidance_scale_real_t=t_guidance,
        gemini_prompt=gemini_prompt,
    ).images[0]

    return result_image


def get_samples():
    sample_list = [
        {
            "image": "assets/hf-logo.png",
            "text": "This item, holding a sign that reads 'DSD!', is placed on a shiny glass table.",
        },
        {
            "image": "assets/seededit_example.png",
            "text": "an adorable small creature with big round orange eyes, fluffy brown fur, wearing a blue scarf with a golden charm, sitting atop a towering stack of colorful books in the middle of a vibrant futuristic city street with towering buildings and glowing neon signs, soft daylight illuminating the scene, detailed and whimsical 3D style.",
        },
        {
            "image": "assets/wanrong_character.png",
            "text": "A chibi-style girl with pink hair, green eyes, wearing a black and gold ornate dress, dancing gracefully in a flower garden, anime art style with clean and detailed lines.",
        },
        {
            "image": "assets/ben_character_squared.png",
            "text": "A confident green-eye young woman with platinum blonde hair in a high ponytail, wearing an oversized orange jacket and black pants, is striking a dynamic pose, anime-style with sharp details and vibrant colors.",
        },
        {
            "image": "assets/action_hero_figure.jpeg",
            "text": "A cartoonish muscular action hero figure with long blue hair and red headband sits on a crowded sidewalk on a Christmas evening, covered in snow and wearing a Christmas hat, holding a sign that reads 'DSD!', dramatic cinematic lighting, close-up view, 3D-rendered in a stylized, vibrant art style.",
        },
        {
            "image": "assets/anime_soldier.jpeg",
            "text": "An adorable cartoon goat soldier sits under a beach umbrella with 'DSD!' written on it, bright teal background with soft lighting, 3D-rendered in a playful and vibrant art style.",
        },
        {
            "image": "assets/goat_logo.jpeg",
            "text": "A shirt with this logo on it.",
        },
        {
            "image": "assets/cartoon_cat.png",
            "text": "A cheerful cartoon orange cat sits under a beach umbrella with 'DSD!' written on it under a sunny sky, simplistic and humorous comic art style.",
        },
    ]
    return [
        [
            Image.open(sample["image"]),
            sample["text"],
        ]
        for sample in sample_list
    ]


demo = gr.Blocks()

with demo:
    iface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Image(type="pil", width=300),
            gr.Textbox(lines=2, label="text", info="Could be something as simple as 'this character playing soccer'."),
            gr.Checkbox(label="Gemini prompt", value=True, info="Use Gemini to enhance the prompt. This is recommended for most cases, unless you have a specific prompt similar to the examples in mind."),
            gr.Slider(minimum=1.0, maximum=6.0, step=0.5, value=3.5, label="guidance scale", info="Tip: start with 3.5, then gradually increase if the consistency is consistently off"),
            gr.Slider(minimum=1.0, maximum=2.0, step=0.05, value=1.5, label="real guidance scale for image", info="Tip: increase if the image is not consistent"),
            gr.Slider(minimum=1.0, maximum=2.0, step=0.05, value=1.0, label="real guidance scale for prompt", info="Tip: increase if the prompt is not consistent"),
            # gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Number of images", info="Select how many images to generate"),
        ],
        outputs=gr.Image(type="pil"),
        # outputs=gr.Gallery(label="Generated Images", height=544),
        # examples=get_samples(),
        live=False,
    )
    gr.Examples(
        examples=get_samples(),
        inputs=iface.input_components,
        outputs=iface.output_components,
        run_on_click=False  # Prevents auto-submission
    )

    gr.HTML(
        """
        <div style="text-align: center;">
            * We borrowed some prompts from the awesome <a href="https://arxiv.org/abs/2411.15098" target="_blank">OminiControl</a>.
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(debug=False, share=True, ssr_mode=False)
