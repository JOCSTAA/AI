when you have added training, predicting and all that :

add this test :

# def convert_output_to_images(neural_output, image1_shape, image2_shape):
#     batch_size, sequence_length, output_dimension = neural_output.shape
#     image1_size = np.prod(image1_shape)

#     RGB_images = neural_output[:, :, :image1_size].reshape((batch_size, sequence_length) + image1_shape)
#     BW_images = neural_output[:, :, image1_size:].reshape((batch_size, sequence_length) + image2_shape)

#     return RGB_images, BW_images

# def concatenate_flattened_arrays(arr1, arr2):
#     # Get the shapes of the input arrays
#     b, s, l, w, o = arr1.shape

#     # Flatten both arrays along the l, w, o axes
#     arr1_flattened = arr1.reshape((b, s, -1))  # -1 automatically calculates the new size
#     arr2_flattened = arr2.reshape((b, s, -1))

#     # Concatenate the flattened arrays along the last axis (k)
#     concatenated_array = np.concatenate((arr1_flattened, arr2_flattened), axis=-1)

#     return concatenated_array

# image1_shape = (2,2,3)
# image2_shape = (2,2,1)
# sample_nn_output = np.random.rand(1,1,16)
# print(sample_nn_output)

# RGB_images, BW_images = convert_output_to_images(sample_nn_output, image1_shape, image2_shape)
# print(RGB_images.shape, BW_images.shape)

# return_array = concatenate_flattened_arrays(RGB_images, BW_images)

# print(return_array)
# print(np.array_equal(sample_nn_output, return_array))