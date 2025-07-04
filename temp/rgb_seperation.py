from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load your RGB image
# Replace 'your_image.jpg' with the actual path to your image file
try:
    img = Image.open("E:/Pain/bp4d+_multi_modal_pretrain/bp4d+_multi_modal_pretrain/BP4D+/Texture_crop_crop_images_DB/Texture_crop/F001/T1/1203.jpg")
    if img.mode != 'RGB':
        img = img.convert('RGB')
except FileNotFoundError as e:
    print(e)
    # Create a dummy RGB image for demonstration if file not found
    print("Creating a dummy RGB image for demonstration.")
    # Create a blank RGB image (100x100)
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_array[:, :, 0] = 255  # Red channel
    img_array[20:80, 20:80, 1] = 255 # Green channel in a square
    img_array[40:60, 40:60, 2] = 255 # Blue channel in a smaller square
    img = Image.fromarray(img_array, 'RGB')


# Split the image into individual bands
r_channel, g_channel, b_channel = img.split()

# Convert back to NumPy arrays for easier plotting if needed
r_array = np.array(r_channel)
g_array = np.array(g_channel)
b_array = np.array(b_channel)

# Display the images
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title('Original RGB Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(r_array, cmap='gray')
plt.title('Red Channel (Grayscale)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(g_array, cmap='gray')
plt.title('Green Channel (Grayscale)')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(b_array, cmap='gray')
plt.title('Blue Channel (Grayscale)')
plt.axis('off')

plt.show()

# Ensure 'temp' directory exists
os.makedirs('temp', exist_ok=True)

# Save colorized channel images (R as red, G as green, B as blue)
r_rgb = np.zeros((r_array.shape[0], r_array.shape[1], 3), dtype=np.uint8)
r_rgb[..., 0] = r_array
r_img = Image.fromarray(r_rgb, 'RGB')
r_img.save('temp/red_channel_rgb.jpg')

g_rgb = np.zeros((g_array.shape[0], g_array.shape[1], 3), dtype=np.uint8)
g_rgb[..., 1] = g_array
g_img = Image.fromarray(g_rgb, 'RGB')
g_img.save('temp/green_channel_rgb.jpg')

b_rgb = np.zeros((b_array.shape[0], b_array.shape[1], 3), dtype=np.uint8)
b_rgb[..., 2] = b_array
b_img = Image.fromarray(b_rgb, 'RGB')
b_img.save('temp/blue_channel_rgb.jpg')

# Visualize the colorized channels
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(r_rgb)
plt.title('Red Channel (Color)')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(g_rgb)
plt.title('Green Channel (Color)')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(b_rgb)
plt.title('Blue Channel (Color)')
plt.axis('off')
plt.show()