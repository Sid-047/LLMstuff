import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif

pipe = ShapEPipeline.from_pretrained("openai/shap-e").to("cuda" if torch.cuda.is_available() else "cpu")
guidance_scale = 15.0
prompt = "<prompt>"
images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    size=256,
).images

gif_path = export_to_gif(images, "outGif.gif")
