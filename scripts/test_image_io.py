from controlled_editor.image.io import load_image

image = load_image("data/samples/test.jpeg")

print("shape:", image.shape)
print("dtype:", image.dtype)
print("min:", image.min())
print("max:", image.max())
