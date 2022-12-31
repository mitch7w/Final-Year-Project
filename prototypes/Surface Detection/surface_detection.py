# go through depth data left-to-right and find where depth data more or less the same (horizontal surface)
    # array of surfaces with xy start, xy end as well as depth data
    # do this with a empty 

# cube has x,y coord and depth data

# when hand x,y,depth coords match cube x,y depth coords and fist is made then move cube to hand x,y, depth coords while fist is still made (grab)

# have box around hand with depth data of hand the min of inside depth values. Check depth values around perimeter of box and if depth values are similar to hand's then collision occurs and stop moving cube.



# TODO

Rotate cube based on currently identified gesture - have a "next gesture" that are currently moving towards - smoother movement

Consider training different right/left hand classifiers for less noisey classifications for other gestures

Classifier - down, side upwards facing

Classifier - open fist/closed

Classifier - left/right

Classifier - x and y plane rotation

Get xy coords of cube to work with moveCube API and growing/shrinking cube - in another file somewhere

Set up hand coords and cube coords 

Calibration mode with empty frame: find surface and add to surfaces array. xy coords with depth Values

Connect cube scaling with current depth value of hand

Get collision boundary depth values working - object collision, stop moving cube

When moving cube, check current coords against surface matrix - if colliding, stop moving



