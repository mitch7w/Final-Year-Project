import numpy as np

prediction_result_array = np.load("Realtime Predictions/realtime_predictions_dsu_down.npy")
# print(prediction_result_array)
print(np.bincount(prediction_result_array))
accuracy = np.bincount(prediction_result_array)[2]/1000
print(accuracy)


# 0 = open. 1 = fist.

# np.load("realtime_predictions_dfu_down.npy")

#  Down side up classifier (z axis rotation)
#     if(down_side_up_index_min == 0): # down
#         glRotatef(-1,0,0,1) # rotate z by -1 degrees
#         z_rotation_degrees -= 1
#         rotation_history.append(["z",-1])
#         # print("Down")
#     if(down_side_up_index_min == 1): # side
#         # Leave rotation as is
#         # print("Side")
#         pass
#     if(down_side_up_index_min == 2): # up
#         glRotatef(1,0,0,1) # rotate z by 1 degrees
#         z_rotation_degrees += 1
#         rotation_history.append(["z",1])
#         # print("Up")

#     # Left towards right classifier (y axis rotation)
#     if(ltr_directions_index_min == 0): # left
#         glRotatef(-1,0,1,0) # rotate y by -1 degrees
#         y_rotation_degrees -= 1
#         rotation_history.append(["y",-1])
#         # print("Left")
#     if(ltr_directions_index_min == 1): # towards
#         # leave rotation as is
#         # print("Towards")
#         pass
#     if(ltr_directions_index_min == 2): # right
#         glRotatef(1,0,1,0) # rotate y by 1 degrees
#         y_rotation_degrees += 1
#         rotation_history.append(["y",1])
#         # print("Right")
            
#     # Down forwards up classifier (x axis rotation)
#     if(down_forwards_up_index_min == 0): # down
#         glRotatef(1,1,0,0) # rotate x by -1 degrees
#         x_rotation_degrees -= 1
#         rotation_history.append(["x",-1])
#         # print("Down")
#     if(down_forwards_up_index_min == 1): # forwards
#         # leave rotation as is
#         # print("Forwards")
#         pass
#     if(down_forwards_up_index_min == 2): # up
#         glRotatef(-1,1,0,0) # rotate x by 1 degrees
#         x_rotation_degrees += 1
#         rotation_history.append(["x",1])
#         # print("Up")
