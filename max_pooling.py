# Given image dimensions
width = 640
height = 960

# Determining the maximum number of pooling layers
# Pooling layers usually reduce the dimensions by a factor of 2
# We'll calculate how many times we can divide the dimensions by 2 until we reach a minimum size (usually at least 1)

def max_pooling_layers(dimension):
    count = 0
    while dimension >= 2:
        dimension /= 2
        count += 1
    return count

max_pooling_layers_width = max_pooling_layers(width)
max_pooling_layers_height = max_pooling_layers(height)

max_pooling_layers_width, max_pooling_layers_height

print("Max pooling layers for width: ", max_pooling_layers_width)