

import numpy as np
import cv2 as cv

from skimage.morphology import area_closing, skeletonize, flood_fill

WHITE = 255

class Network():
    '''
    Make network from specific color marker in a 2D image of vasculature  
    '''

    def __init__(self, img, name, colors, tol):
        self.img = img
        self.name = name
        self.colors = colors # hsv color codes for the part of the vasculature we want to extract
        self.tol = tol # color hsv tolerance

        self.num_branches = 0

    def extract_skeleton(self, dilation_iterations):
        '''
        Extract 1D skeleton network
        '''

        # Extract vasculature as whichever pixels are in the 
        # given tolerance        
        hsv_colors_min = np.min( self.colors, axis=0) - self.tol 
        hsv_colors_max = np.max( self.colors, axis=0) + self.tol

        vasculature = cv.inRange(hsv, hsv_colors_min, hsv_colors_max)

        # Image is a bit patchy, so we dilate a given amount of times 
        # before skeletonizing so the network is more connected
        kernel = np.ones((3,3)) 
        for i in range(0, dilation_iterations):
            vasculature = cv.dilate(vasculature, kernel)

        # Remove holes so we don't get extra loops in the 1D skeleton
        mask = area_closing( vasculature, area_threshold=200, connectivity=2)
        vasculature[np.where(mask)] = WHITE

        # Reduce to 1D using skeletonize
        vasculature = skeletonize(np.asarray(vasculature, dtype=bool))
        vasculature = np.asarray(vasculature, dtype=int)*WHITE # TODO: simplify this?

        self.img = vasculature
    
    def connect_skeleton(self, connection_kernel):
        '''
        Connect skeleton segments using dilation and erosion with
        specific kernel
        '''
        
        vasc = self.img
        
        # We dilate and erode again in order to fill in gaps
        vasc = cv.dilate(np.asarray(vasc, dtype=np.uint8), connection_kernel, iterations=1)
        vasc = cv.erode(vasc, connection_kernel, iterations=1) 
        vasc = skeletonize(np.asarray(vasc, dtype=bool))
        vasc = np.asarray(vasc, dtype=int)*WHITE

        self.img = vasc

    def color_segments(self, min_branch_length):
        '''
        Color disconnected segments of the network
        This is done by a sequential flood-filling
        '''
        
        vasc = self.img
        white_pixels = np.where(vasc == WHITE)

        num_branches = 0

        fill_value = 100
        
        while len(white_pixels[0])>0:
            white_pixel = [white_pixels[0][0], white_pixels[1][0]]
            vasc = flood_fill(vasc, seed_point=tuple(white_pixel), new_value=fill_value)
            
            branch_length = len(np.where(vasc==fill_value)[0])
            if branch_length < min_branch_length:
                vasc[np.where(vasc==fill_value)] = 0

            white_pixels = np.where(vasc == WHITE)
            fill_value  += 1
            num_branches += 1

        self.img = vasc
        self.num_branches = num_branches
        

    def __str__(self):

        output_str = f'Network with {self.num_branches} branches \n'
        output_str += f'Image saved as {self.name}.jpg'

        cv.imwrite(f'{self.name}.jpg', self.img)

        return output_str

            

    
## The arteries are marked in "red" (actually orange) and the veins in "blue"
# The idea is that we extract the red and blue pixels separately, and skeletize the
# domain to make it 1D

BLUE = [113, 144, 141] # sampled rgb code for venous markers

RED1 = [186, 131, 106]  # sampled rgb codes for arterial markers
RED2 = [197, 164, 141]
RED3 = [188, 154, 131]

REDS = [RED1, RED2, RED3]

BACKGROUND = [217, 210, 194] # sampled rgb code for background color

# opencv uses bgr ordering
[l.reverse() for l in [RED1, RED2, RED3, BLUE, BACKGROUND]]



# Read in image of frog tongue vasculature
img = cv.imread('frog-tongue.png')

# It's easier to extract colored objects using HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Convert the marker colors to HSV as well
def rgb_to_gbrimage(clr):
    return np.uint8([[clr]])

hsv_reds = [cv.cvtColor(rgb_to_gbrimage(clr), cv.COLOR_BGR2HSV) for clr in REDS]
hsv_blue = cv.cvtColor(rgb_to_gbrimage(BLUE), cv.COLOR_BGR2HSV)


# The blue veins are relatively easy to identify as their hsv value differs substantially from veins and background
blue_tol = [90, 50, 40]

veins = Network(img, 'veins', [hsv_blue], blue_tol)
veins.extract_skeleton(dilation_iterations=2)
veins.color_segments(40)
print(veins)


# The red arteries are trickier as their hsv value is similar to the background
# so we construct thresholds from the three samples we took and add in an tolerance
tol = [5, 30, 5] # saturation value of red differs the most
hsv_red_min = np.min( hsv_reds, axis=0) - tol 
hsv_red_max = np.max( hsv_reds, axis=0) + tol

arteries = Network(img, 'arteries', hsv_reds, tol)
arteries.extract_skeleton(dilation_iterations=4)

# The arteries are missing some parts because they lie underneath the veins
# Most of these missing connections are horizontal, vertical or at 45 degrees

kernel = np.ones((1,30), np.uint8)  # connect lines using horizontal kernel
arteries.connect_skeleton(kernel)

kernel = np.ones((30,1), np.uint8)  # connect lines using vertical kernel
arteries.connect_skeleton(kernel)

# After filling in as much as possible we color the branches
arteries.color_segments(40)
print(arteries)


# Make bgr image of total vasculature (arteries + veins)
imshape = list(arteries.img.shape)
imshape.append(3)

vasculature = np.zeros(imshape)

vasculature[np.where(arteries.img)]=[20, 1, 170]
vasculature[np.where(veins.img)]=[255, 150, 0]
cv.imwrite('vasculature.jpg', vasculature)
