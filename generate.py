import argparse
import torch
from PIL import Image
from diffusers.utils import load_image

from pipeline import FluxConditionalPipeline
from transformer import FluxTransformer2DConditionalModel
from recaption import enhance_prompt

pipe = None

def init_pipeline(model_path, lora_path):
    """Initialize the global pipeline (pipe)."""
    global pipe
    transformer = FluxTransformer2DConditionalModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )
    pipe = FluxConditionalPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(
        lora_path
    )
    # pipe.enable_model_cpu_offload()
    pipe.to("cuda")

def process_image_and_text(image, text, gemini_prompt, guidance, i_guidance, t_guidance):
    """Process the given image and text using the global pipeline."""
    # center-crop image
    w, h = image.size
    min_size = min(w, h)
    image = image.crop(((w - min_size) // 2, 
                        (h - min_size) // 2, 
                        (w + min_size) // 2, 
                        (h + min_size) // 2))
    image = image.resize((512, 512))

    control_image = load_image(image)

    if gemini_prompt:
        text = enhance_prompt(image, text.strip().replace("\n", "").replace("\r", ""))
    
    result = pipe(
        prompt=text.strip().replace("\n", "").replace("\r", ""),
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

    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Run Diffusion Self-Distillation.")

    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/transformer", 
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--lora_path", 
        type=str, 
        default="/home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/pytorch_lora_weights.safetensors", 
        help="Path to the lora checkpoint."
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True,
        help="Path to the input image."
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=True, 
        help="The text prompt."
    )
    parser.add_argument(
        "--disable_gemini_prompt", 
        action="store_true",
        help="Flag to disable gemini prompt. If not set, gemini_prompt is True."
    )
    parser.add_argument(
        "--guidance", 
        type=float, 
        default=3.5, 
        help="Guidance scale for the pipeline."
    )
    parser.add_argument(
        "--i_guidance", 
        type=float, 
        default=1.0, 
        help="Image guidance scale."
    )
    parser.add_argument(
        "--t_guidance", 
        type=float, 
        default=1.0, 
        help="Text guidance scale."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output.png", 
        help="Path to save the output image."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize pipeline
    init_pipeline(args.model_path, args.lora_path)
    
    # Open the image
    image = Image.open(args.image_path).convert("RGB")
    
    # Process image and text
    result_image = process_image_and_text(
        image, 
        args.text, 
        not args.disable_gemini_prompt, 
        args.guidance, 
        args.i_guidance, 
        args.t_guidance
    )
    
    # Save the output
    result_image.save(args.output_path)
    print(f"Output saved to {args.output_path}")

if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=7 python generate.py --model_path /home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/transformer --lora_path /home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/pytorch_lora_weights.safetensors --image_path /home/shengqu/repos/dreambench_plus/conditioning_images/seededit_example.png --text "this character sitting on a chair" --output_path output.png
