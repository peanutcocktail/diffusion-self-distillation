from safetensors.torch import load_file, save_file

# 1) Load the state dict from the .safetensors file
state_dict = load_file("/home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/transformer/diffusion_pytorch_model.safetensors")

# 2) Filter/rename the keys. For example, remove "base_layer." from each key.
filtered_state_dict = {
    key.replace("base_layer.", ""): value
    for key, value in state_dict.items()
}

# 3) (Optional) Save the filtered state dict back to a new .safetensors file
save_file(filtered_state_dict, "/home/shengqu/repos/SimpleTuner/output/1x2_v1/checkpoint-172000/transformer/diffusion_pytorch_model.safetensors")

print("Filtered state_dict saved to filtered_diffusion_pytorch_model.safetensors")
