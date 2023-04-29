from keras_cv.models import StableDiffusionV2
from PIL import Image

file_name = "prompt.png"
model = StableDiffusionV2(img_height=512, img_width=512, jit_compile=True)
img = model.text_to_image(
    prompt="Cool programmer dog",
    batch_size=1,  # How many images to generate at once
    num_steps=30,  # Number of iterations (controls image quality)
    seed=-1,  # Set this to always get the same image from the same prompt
)

Image.fromarray(img[0]).save(file_name)
print(f"saved at {file_name}")
