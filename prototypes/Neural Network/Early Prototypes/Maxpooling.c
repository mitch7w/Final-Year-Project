// Takes in matrix from conv. operation and outputs maxpooled matrix

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

float *maxpooling(float ***previous_layer_output, int previous_layer_output_x, int previous_layer_output_y, int previous_layer_output_num_filters, int pool_size, int stride_size)
{
    float maxpooled_values[(((previous_layer_output_x - pool_size) / stride_size) + 1) * (((previous_layer_output_y - pool_size) / stride_size) + 1) * (previous_layer_output_num_filters)] ;
    int maxpool_counter = 0;
    // inputX and inputY are counters over input layer
    int inputX = 0;
    int inputY = 0;
    // iterate over the entire input matrix
    while (inputX + pool_size < previous_layer_output_x + 1)
    {
        while (inputY + pool_size < previous_layer_output_y + 1)
        {
            for (int filter_num = 0; filter_num < previous_layer_output_num_filters; filter_num++)
            {
                // filterX and filterY are the x and y positions going across the length of the imaginary pooling matrix
                // iterate over the whole imaginary pooling matrix
                // find max value in sub matrix
                float max_value = 10000.0;
                for (int filter_x = 0; filter_x < pool_size; filter_x++)
                {
                    for (int filter_y = 0; filter_y < pool_size; filter_y++)
                    {
                        if (previous_layer_output[inputX + filter_x][inputY + filter_y][filter_num] < max_value)
                        {
                            max_value = previous_layer_output[inputX + filter_x][inputY + filter_y][filter_num];
                        }
                    }
                }
            }
            inputY += stride_size;
        }
        inputX += stride_size;
        inputY = 0;
    }
    return maxpooled_values;
}

// reduce size images for training and test time
// integrate C code with Python and time
// world coordinates tomorrow