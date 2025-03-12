import argparse
import torch
from PIL import Image
from diffusers.utils import load_image

from transformers import AutoModelForCausalLM

from pipeline import FluxConditionalPipeline
from transformer import FluxTransformer2DConditionalModel


def process_image_and_text(
    pipe, image, text, gemini_prompt, guidance, i_guidance, t_guidance, steps
):
    """Process the given image and text using the global pipeline."""
    # center-crop image
    w, h = image.size
    min_size = min(w, h)
    image = image.crop(
        (
            (w - min_size) // 2,
            (h - min_size) // 2,
            (w + min_size) // 2,
            (h + min_size) // 2,
        )
    )
    image = image.resize((512, 512))

    control_image = load_image(image)

    if gemini_prompt:
        from recaption import enhance_prompt
        text = enhance_prompt(image, text.strip().replace("\n", "").replace("\r", ""))

    result = pipe(
        prompt=text.strip().replace("\n", "").replace("\r", ""),
        negative_prompt="",
        num_inference_steps=steps,
        height=512,
        width=1024,
        guidance_scale=guidance,
        image=control_image,
        guidance_scale_real_i=i_guidance,
        guidance_scale_real_t=t_guidance,
    ).images[0]

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run Diffusion Self-Distillation.")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/transformer",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/pytorch_lora_weights.safetensors",
        help="Path to the lora checkpoint.",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument("--text", type=str, required=True, help="The text prompt.")
    parser.add_argument(
        "--disable_gemini_prompt",
        action="store_true",
        help="Flag to disable gemini prompt. If not set, gemini_prompt is True.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3.5, help="Guidance scale for the pipeline."
    )
    parser.add_argument(
        "--i_guidance", type=float, default=1.0, help="Image guidance scale."
    )
    parser.add_argument(
        "--t_guidance", type=float, default=1.0, help="Text guidance scale."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Path to save the output image.",
    )
    parser.add_argument(
        "--sequential_offload",
        action="store_true",
        help="Sequentially offload to CPU",
    )
    parser.add_argument(
        "--model_offload", action="store_true", help="Offload full models"
    )
    parser.add_argument("--steps", type=int, default=28, help="Steps to generate")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model_path}")
    print(f"LoRA: {args.lora_path}")

    # Initialize pipeline
    transformer = FluxTransformer2DConditionalModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
    )
    pipe = FluxConditionalPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    assert isinstance(pipe, FluxConditionalPipeline)
    pipe.load_lora_weights(args.lora_path)
    if args.model_offload:
        pipe.enable_model_cpu_offload()
    if args.sequential_offload:
        pipe.enable_sequential_cpu_offload()

    # Open the image
    image = Image.open(args.image_path).convert("RGB")

    print(f"Process image: {args.image_path}")

    # Process image and text
    result_image = process_image_and_text(
        pipe,
        image,
        args.text,
        not args.disable_gemini_prompt,
        args.guidance,
        args.i_guidance,
        args.t_guidance,
        args.steps
    )

    # Save the output
    result_image.save(args.output_path)
    print(f"Output saved to {args.output_path}")


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=7 python generate.py --model_path /home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/transformer --lora_path /home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/pytorch_lora_weights.safetensors --image_path /home/shengqu/repos/dreambench_plus/conditioning_images/seededit_example.png --text "this character sitting on a chair" --output_path output.png
