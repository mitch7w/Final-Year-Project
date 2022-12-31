import matplotlib.pyplot as plt

distanceMeasurements =[4,4,4,5,5,6,6,8,8,9,10,10,12,12,13,14,16,17,18,20,21,23,25,27,27,29,30,33,34,36,37,39,41,42,44,47,48,49,50,50,51,51,54,56,56,57,59,60,62,63,63,64,65,68,70,71,74,75,77,81,85,87,88,89,91,93,94,96,97,101,102,104,106,108,109,110,112,113,117,120,125,128,129,130,132,137,138,143,146,146,150,153,153,158,161,162,165,167,171,183,189,189,193,197,201,204,208,203,219,212]
plt.xlabel("Measurement Number")
plt.ylabel("Distance from sensor (cm)")
plt.title("Ultrasonic Sensor Depth Distance Prototype")
plt.plot(distanceMeasurements)
plt.show()