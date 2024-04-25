import time
import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif


prompt = input("Enter the Prompt Yo: ")
t1 = time.time()
pipe = ShapEPipeline.from_pretrained("openai/shap-e").to("cuda" if torch.cuda.is_available() else "cpu")
guidance_scale = 15.0
images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64
).images[0]


gif_path = export_to_gif(images, "outGif.gif")
t2 = time.time()
print("ExecTime: ", (t2-t1))