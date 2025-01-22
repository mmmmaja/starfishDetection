import math

import matplotlib.pyplot as plt


def visualize_dataset(dataset, num_images=4):
    # Create a grid of images
    rows = int(math.sqrt(num_images))
    cols = int(math.ceil(num_images / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    for i in range(num_images):
        try:
            dataset.plot_sample(i, axs=axes[i // cols, i % cols])
        except IndexError:
            break

    plt.show()
