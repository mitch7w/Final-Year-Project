import cv2
import numpy as np
import freenect
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)

top_left_coordinates = [50,50]
width_cube = 100
cube_depth = 0 # z axis = 0


def moveCube(x_coord, y_coord):
    global top_left_coordinates
    top_left_coordinates[0] = x_coord
    top_left_coordinates[1] = y_coord
    # z coord requires shifting of x and y in order to stay in that place but appear to "shrink" or grow
    

def handTracking(frame):
    # hand tracker
        image = cv2.resize(frame, (320,240), interpolation = cv2.INTER_AREA)
        image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
        # print(image_ycrcb[0][0])
        skin1 = (50, 89, 136)
        skin2 = (231, 147, 181)
        mask = cv2.inRange(image_ycrcb, skin1, skin2)
        result = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow("Color", result)
        # cv2.imshow("Mask", mask) # 240 x 320 white or black image

        # find highest concentration of values in mask matrix closest to bottom of image
        r_region_counters = []
        c_region_counters = []
        region_sums = []
        south_deltas = []
        # find all regions with pixel count larger than 270 000
        for r_counter in range(0,320,5):
            for c_counter in range(0,240,5):
                region_sum = np.sum(mask[r_counter:r_counter+60,c_counter:c_counter+80]) # sum all white pixels in proposal region
                if(region_sum > 100000): # if more white pixels here than average
                    south_delta = abs(r_counter-240)
                    south_deltas.append(south_delta)
                    region_sums.append(region_sum)
                    r_region_counters.append(r_counter)
                    c_region_counters.append(c_counter)
        # find southmost
        if(len(south_deltas) ==0):
            return
        southmost_delta = np.amin(south_deltas)
        # find largest region within 50px of southmost point
        largest_south_index = 0
        for current_region in range(len(region_sums)):
            # print("south_deltas[current_region] - southmost_delta: ", south_deltas[current_region] - southmost_delta)
            if(region_sums[current_region] > region_sums[largest_south_index]):
                if( abs(south_deltas[current_region] - southmost_delta) < 50):
                    largest_south_index = current_region
        moveCube((c_region_counters[largest_south_index]+10)/320, (r_region_counters[largest_south_index]+20)/240)
        

def drawCube(background_image,top_left_coordinates, width_cube):
    # BGR colors
    RED = (0, 0, 255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    ORANGE = (34,156,228)
    front_face_pts = [top_left_coordinates, (top_left_coordinates[0]+width_cube, top_left_coordinates[1]), (top_left_coordinates[0]+width_cube, top_left_coordinates[1]+width_cube), (top_left_coordinates[0],top_left_coordinates[1]+width_cube) ]
    top_face_pts = [top_left_coordinates,(int(top_left_coordinates[0]+width_cube/2),int(top_left_coordinates[1]-width_cube/2)),(int(top_left_coordinates[0]+width_cube*1.5),int(top_left_coordinates[1]-width_cube/2)),(int(top_left_coordinates[0]+width_cube), top_left_coordinates[1])]
    side_face_pts = [(int(top_left_coordinates[0]+width_cube), top_left_coordinates[1]),(int(top_left_coordinates[0]+width_cube*1.5),int(top_left_coordinates[1]-width_cube/2)),(int(top_left_coordinates[0]+width_cube*1.5),int(top_left_coordinates[1]+width_cube/2)),(top_left_coordinates[0]+width_cube, top_left_coordinates[1]+width_cube)]

    cv2.fillPoly(background_image, np.array([front_face_pts]), ORANGE)
    cv2.fillPoly(background_image, np.array([top_face_pts]), GREEN)
    cv2.fillPoly(background_image, np.array([side_face_pts]), BLUE)

def check_space_occupied(depth_data, proposed_x, proposed_y, proposed_z):
    # check depthInput at pixels to see if object there
    depth_z = int(650 -10*(proposed_z-3))
    for x in range(proposed_x-20, proposed_x+20):
        for y in range(proposed_y-20, proposed_y+20):
            if(proposed_x >=0 and proposed_x < 640 and proposed_y >= 0 and proposed_y < 480):
                # if pixel trying to move to has depth value close to that of object - collision!
                if( depth_data[proposed_x][proposed_y] >= depth_z-50 and depth_data[proposed_x][proposed_y] <= depth_z+50):
                    return True
    return False

while True:
    # read in from Kinect
    depthInput = freenect.sync_get_depth()[0]
    videoInput = freenect.sync_get_video()[0]
    background_image = cv2.cvtColor(np.array(videoInput) , cv2.COLOR_BGR2RGB)
    image = cv2.resize(background_image, (320,240), interpolation = cv2.INTER_AREA)
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
    # print(image_ycrcb[0][0])
    skin1 = (50, 89, 136)
    skin2 = (231, 147, 181)
    # skin1 = (89, 119, 140)
    # skin2 = (255, 179, 187)
    mask = cv2.inRange(image_ycrcb, skin1, skin2)
    result = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("Color", result)
    cv2.imshow("Mask", mask) # 240 x 320 white or black image
    # hand tracking
    # find highest concentration of values in mask matrix closest to bottom of image
    r_region_counters = []
    c_region_counters = []
    region_sums = []
    left_deltas = []
    # find all regions with pixel count larger than 270 000
    for r_counter in range(0,320,5):
        for c_counter in range(0,240,5):
            region_sum = np.sum(mask[r_counter:r_counter+60,c_counter:c_counter+80]) # sum all white pixels in proposal region
            if(region_sum > 1000): # if more white pixels here than average
                left_delta = abs(c_counter-120 + r_counter-160) # distance from centre
                left_deltas.append(left_delta)
                region_sums.append(region_sum)
                r_region_counters.append(r_counter)
                c_region_counters.append(c_counter)
    # find leftmost
    if(len(left_deltas) !=0):
        leftmost_delta = np.amin(left_deltas)
        # find largest region within 50px of leftmost point
        largest_left_index = 0
        for current_region in range(len(region_sums)):
            # print("left_deltas[current_region] - leftmost_delta: ", left_deltas[current_region] - leftmost_delta)
            if(region_sums[current_region] > region_sums[largest_left_index]):
                if( abs(left_deltas[current_region] - leftmost_delta) < 30):
                    largest_left_index = current_region
    cv2.imshow("result", cv2.rectangle(result, (c_region_counters[largest_left_index],r_region_counters[largest_left_index]), (c_region_counters[largest_left_index]+80,r_region_counters[largest_left_index]+60), (0,0,255), 2))
    drawCube(background_image,[c_region_counters[largest_left_index]+50,r_region_counters[largest_left_index]+70],100) # draw Cube over Kinect feed
    cv2.imshow('Cube Image', background_image)
    print("Current coords: x: ", top_left_coordinates[0], ", y: ", top_left_coordinates[1], ", z: ", cube_depth," on z axis or ", int(650 -10*(cube_depth-3)), " on depth scale")
    if cv2.waitKey(1) == ord("w"):
        top_left_coordinates[1] -=10
    if cv2.waitKey(1) == ord("s"):
        top_left_coordinates[1] +=10
    if cv2.waitKey(1) == ord("a"):
        object_collision = check_space_occupied(depthInput,top_left_coordinates[0]-10,top_left_coordinates[1])
        if(object_collision == False):
            top_left_coordinates[0] -=10
    if cv2.waitKey(1) == ord("d"):
        object_collision = check_space_occupied(depthInput,top_left_coordinates[0]+10,top_left_coordinates[1])
        if(object_collision == False):
            top_left_coordinates[0] +=10
    if cv2.waitKey(1) == ord("i"):
        width_cube += 10
        cube_depth +=1
    if cv2.waitKey(1) == ord("o"):
        width_cube -= 10
        cube_depth -= 1
cv2.destroyAllWindows()