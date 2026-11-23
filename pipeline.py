
#1. keypoints.py detect keypoints

from keypoints import N_KPTS, _EMPTY, detect
from config import root, model

#2.(optional) perspective rectify keypoints, and number classification within region of interest to classify plate's belonging



#3. color vision ever 3rd frame to update each plate object's classification as enemy or friend 

from colorDetect import classify_color

#4. uses estimation.py such as perspective and point to calculate approximate 3d position of each plate
#inputs: boxes and keypoints
#outputs: x, y, z, yaw

import estimation

#5. smoothen.py to smoothen x,y,z, yaw data to prevent sporadic updates of movement 
#inputs: x, y, z, yaw of every plate center
#outputs: new x, y, z, yaw based on previous data


#6. output plates and their location
#

#7. export 