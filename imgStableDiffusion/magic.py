from diffusers import DiffusionPipeline
import torch
import tqdm
import time

prompt = '<input prompt>'
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = pipe.to("cuda")

t1 = time.time()
for i in tqdm.trange(10, colour="red"):
    img = pipe(prompt).images[0]
    img.save("GenImages\{}.png".format(time.time()))
t2 = time.time()

print("\nCompleteExecTime: ", (t2-t1))