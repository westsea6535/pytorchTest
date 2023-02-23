import numpy as np
import matplotlib.pyplot as plt

def visualize_stl10_data(bin_file_path):
    # Read binary data into a numpy array
    with open(bin_file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)

    # Reshape the data into an (N, H, W, C) tensor, where N is the number of images
    num_images = int(data.shape[0] / (96 * 96 * 3))
    data = data.reshape(num_images, 96, 96, 3)

    # Plot the first 10 images using Matplotlib
    for i in range(10):
        image = data[i]
        plt.imshow(image)
        plt.axis('off')
        plt.show()

# Call the function to visualize the STL-10 data
bin_file_path = './data/stl10_binary/test_X.bin'
visualize_stl10_data(bin_file_path)
