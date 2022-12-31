import numpy as np

sample_list = [[23, 22, 24, 23],[1,2,3,4,5]]
new_array = np.array(sample_list)
# Displaying the array
file = open("sample.txt", "w+")
# Saving the array in a text file
content = str(new_array)
file.write(content)
# file.close()
