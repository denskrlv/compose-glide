import torch

from compose_glide import ComposeGlide
from PIL import Image


compositional_prompts = [
    "No Smiling AND NOT Glasses AND NOT Female",
    "Smiling AND NOT (No Glasses) AND NOT Female",
    "NOT (No Smiling) AND No Glasses AND NOT Male",
    "NOT (No Smiling) AND NOT (No Glasses) AND Male",
    "Smiling AND NOT (No Glasses) AND NOT Male"
]

NUM_VARIANTS = 20


def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a PIL Image."""
    # Scale from [-1, 1] to [0, 255]
    scaled = ((tensor + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
    
    # Rearrange dimensions from CxHxW to HxWxC
    if scaled.dim() == 3:  # Single image
        img = scaled.permute(1, 2, 0).numpy()
    else:  # Batch of images
        img = scaled[0].permute(1, 2, 0).numpy()  # Take the first image
        
    return Image.fromarray(img)


if __name__ == "__main__":

    compose_glide = ComposeGlide(model_name='glide_faces', verbose=True)
    print(compose_glide)

    for i, prompt in enumerate(compositional_prompts):
        for j in range(NUM_VARIANTS):
            result, _ = compose_glide.generate(
                prompt, 
                num_images=1, 
                upsample=True, 
                upsample_temp=0.995,
                save_intermediate_steps=10,
                return_attention_maps=True
            )

            image = tensor_to_image(result)
            image_path = f"../outputs/prompt_{i}_variant_{j}.png"
            image.save(image_path)
            print(f"Saved: {image_path}!")
