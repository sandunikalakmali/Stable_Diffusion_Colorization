from PIL import Image

img = Image.open("./dataset/preprocessed_imagenet_filtered_flat/train/n01440764_n01440764_78.png")
print("Mode:", img.mode)

# Number of channels
channels = len(img.getbands())
print("Number of channels:", channels)
