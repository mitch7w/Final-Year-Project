import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageEnhance, Image
from math import sqrt
import cv2
sns.set_theme()
import freenect
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)
import numba as nb
import time


while True:
    counter_new = 0
    normal_list = []
    start_time = time.time()
    # read in from Kinect
    videoInput = freenect.sync_get_video()[0]
    depthInput = freenect.sync_get_depth()[0]
    np.save("surface_normals_depthInput.npy", depthInput)
    depthInput = np.double(depthInput)
    start_time = time.time()
    # find normal of each pixel to the plane it rests on
    @nb.jit(nopython=True)
    def return_normals(depthIn):
        normals = np.zeros((480,640,3))
        for x in range(1,depthInput.shape[0]-1,1):
            for y in range(1,depthInput.shape[1]-1,1):
                if(depthInput[x][y] == 2047):
                    normals[x][y] = 0
                else:
                    dzdx = np.double(depthInput[x+1][y] - depthInput[x-1][y]) / 2
                    dzdy = (depthInput[x][y+1] - depthInput[x][y-1]) / 2
                    vector_d = [-dzdx, -dzdy, 1.0]
                    vector_mag = np.sqrt(vector_d[0]**2 + vector_d[1]**2 + vector_d[2]**2)
                    new_normal = np.array(vector_d)/vector_mag
                    normals[x][y] = new_normal
        return normals
    normals = return_normals(depthInput)
    # print("Normal calcs: ", 1/(time.time()-start_time), " fps")
            # print(new_normal)
    # print("normals[420][450]: ", normals[420][450])
    # find most common normal
    
    # # print two side by side normals and check discrepancies between them
    # print("normals[100][100]: ", normals[10][10])
    # print("normals[100][101]: ", normals[300][300])
    # # normals[300:330][400:430] = 0 # see where coords are

    # gather all normals part of the chosen plane
    # chosen_normal = [np.mean(normals[400:420,350:370][:,:,0]),np.mean(normals[400:420,350:370][:,:,1]),np.mean(normals[400:420,350:370][:,:,2])]
    
    # print(chosen_normal)
    def find_plane(normals):
        chosen_normal = [0.9,-0.2,0.6]
        normals_detected = np.zeros((normals.shape)) # coloured output plane
        for x in range(1,depthInput.shape[0]-1,1):
            for y in range(1,depthInput.shape[1]-1,1):
                if(np.allclose(normals[x][y], chosen_normal, 0.2,0.2) == True):
                    normals_detected[x][y] = 255
        return normals_detected
    
    normals_detected = find_plane(normals)
    
    # find plane regions above threshold
    plane_r_regions = []
    plane_c_regions = []
    plane_region_sums = []
    for col in range(0,480,40):
        for row in range(0,640,40):
            # print("Region colrow ", col, " ,", row, ": ", np.sum(normals_detected[col:col+40,row:row+40]))
            if(np.sum(normals_detected[col:col+40,row:row+40]) > 50000): # if above threshold add to regions
                plane_region_sums.append(np.sum(normals_detected[col:col+40,row:row+40]))
                plane_r_regions.append(row)
                plane_c_regions.append(col)
                np.argmax(plane_region_sums)

    # get all pixels of surface by same colour (RANSAC-inspired algorithm)
    starting_surface_point_row_coord = plane_r_regions[np.argmax(plane_region_sums)]
    starting_surface_point_col_coord = plane_c_regions[np.argmax(plane_region_sums)]
    surface_pixel_rows = [starting_surface_point_row_coord]
    surface_pixel_cols = [starting_surface_point_col_coord]
    first_coord = str(starting_surface_point_col_coord) + "," + str(starting_surface_point_row_coord)
    surface_pixel_coords=[first_coord]
    pixel_counter = 0
    while(pixel_counter<len(surface_pixel_rows)):
        current_row_coord = surface_pixel_rows[pixel_counter]
        current_col_coord = surface_pixel_cols[pixel_counter]
        new_coord_1 = str(current_col_coord+1) + "," + str(current_row_coord)
        if(current_col_coord+1 < 480):
            # print("current_col_coord: ", current_col_coord)
            if(abs(np.mean(videoInput[starting_surface_point_col_coord][starting_surface_point_row_coord]) - np.mean(videoInput[current_col_coord+1][current_row_coord])) < 20 and new_coord_1 not in surface_pixel_coords):
                surface_pixel_rows.append(current_row_coord)
                surface_pixel_cols.append(current_col_coord+1)
                surface_pixel_coords.append(new_coord_1)
        new_coord_2 = str(current_col_coord-1) + "," + str(current_row_coord)
        if(current_col_coord-1 > -1 ):
            if(abs(np.mean(videoInput[starting_surface_point_col_coord][starting_surface_point_row_coord]) - np.mean(videoInput[current_col_coord-1][current_row_coord])) < 20 and new_coord_2 not in surface_pixel_coords):
                surface_pixel_rows.append(current_row_coord)
                surface_pixel_cols.append(current_col_coord-1)
                surface_pixel_coords.append(new_coord_2)
        new_coord_3 = str(current_col_coord) + "," + str(current_row_coord+1)
        if(current_row_coord+1 < 640):
            if(abs(np.mean(videoInput[starting_surface_point_col_coord][starting_surface_point_row_coord]) - np.mean(videoInput[current_col_coord][current_row_coord+1])) < 20 and new_coord_3 not in surface_pixel_coords):
                surface_pixel_rows.append(current_row_coord+1)
                surface_pixel_cols.append(current_col_coord)
                surface_pixel_coords.append(new_coord_3)
        new_coord_4 = str(current_col_coord) + "," + str(current_row_coord-1)
        if(current_row_coord-1 > -1):
            if(abs(np.mean(videoInput[starting_surface_point_col_coord][starting_surface_point_row_coord]) - np.mean(videoInput[current_col_coord][current_row_coord-1])) < 20 and new_coord_4 not in surface_pixel_coords):
                surface_pixel_rows.append(current_row_coord-1)
                surface_pixel_cols.append(current_col_coord)
                surface_pixel_coords.append(new_coord_4)
        pixel_counter += 1
        # print("pixel_counter: ", pixel_counter)
    
    coloured_surface = np.zeros(videoInput.shape)
    for each_pixel in range(len(surface_pixel_rows)):
        coloured_surface[surface_pixel_cols[each_pixel]][surface_pixel_rows[each_pixel]] = 255


    cv2.imshow("Image", np.array(videoInput))
    cv2.imshow("Depth", np.array(depthInput))
    cv2.imshow("Normals", normals)
    cv2.imshow("Detected normals", normals_detected)
    # cv2.imshow("Detected surface", cv2.rectangle(normals_detected, (plane_left,plane_top), (plane_right,plane_bottom), (0,0,255), 2))
    cv2.imshow("Detected surface", cv2.rectangle(normals_detected, (plane_r_regions[np.argmax(plane_region_sums)],plane_c_regions[np.argmax(plane_region_sums)]), (plane_r_regions[np.argmax(plane_region_sums)]+40,plane_c_regions[np.argmax(plane_region_sums)]+40), (0,0,255), 2))
    cv2.imshow("Coloured surface: ", coloured_surface)
    # print("Sum of region: ", np.argmax(plane_region_sums))
    cv2.waitKey(1)
    print("Total runtime: ", time.time()-start_time)

    # Total runtime is ~100s