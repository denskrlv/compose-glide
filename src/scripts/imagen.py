import torch

from compose_glide import ComposeGlide
from PIL import Image


compositional_prompts = [
    "a man AND NOT Smiling AND NOT Glasses",
    "NOT a woman AND Smiling AND NOT Glasses",
    "a man AND Smiling AND Glasses",
    "NOT a man AND NOT Smiling AND NOT Glasses",
    "a woman AND Smiling AND NOT Glasses",
    "a woman AND Smiling AND Glasses"
]

standard_prompts = [
    "a man",
    "a man smiling",
    "a man smiling with glasses",
    "a woman",
    "a woman smiling",
    "a woman smiling with glasses",
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

            print(f"Generating image for prompt {i} variant {j}")

            result, _ = compose_glide.generate(
                prompt, 
                num_images=1, 
                upsample=True, 
                upsample_temp=0.995,
                save_intermediate_steps=10,
                return_attention_maps=True
            )

            image = tensor_to_image(result)
            image_path = f"/Users/deniskrylov/Developer/University/compose-glide/outputs/compositional/prompt_{i}_variant_{j}.png"
            image.save(image_path)
            print(f"Saved: {image_path}!")
    
    for i, prompt in enumerate(standard_prompts):
        for j in range(NUM_VARIANTS):

            print(f"Generating image for prompt {i} variant {j}")

            result, _ = compose_glide.generate(
                prompt, 
                num_images=1, 
                upsample=True, 
                upsample_temp=0.995,
                save_intermediate_steps=10,
                return_attention_maps=True
            )

            image = tensor_to_image(result)
            image_path = f"/Users/deniskrylov/Developer/University/compose-glide/outputs/standard/prompt_{i}_variant_{j}.png"
            image.save(image_path)
            print(f"Saved: {image_path}!")
