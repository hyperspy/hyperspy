import math

import numpy as np
import numpy.linalg
import scipy.signal
from scipy.ndimage.morphology import binary_erosion

from hyperspy._signals.image import Image



def estimate_1D_Gaussian_parameters(data, axis):
    center = np.sum(axis * data) / np.sum(data)
    sigma = np.sqrt(np.abs(np.sum((axis - center) ** 2 * data) / np.sum(data)))
    height = data.max()
    return center, sigma, height


class HighResADF(Image):
    _signal_type = "HighResADF"

    def __init__(self, *args, **kwards):
        Image.__init__(self, *args, **kwards)
        # Attributes defaults
        self.metadata.Signal.binned = True


    def peak_find_ranger(self,
                         best_size = "auto",
                         refine_positions=False,
                         show_progress=False,
                         sensitivity_threshold=0.34,
                         start_search=3,
                         end_search="auto"):
        
        """
        
        Parameters
        ----------
        refine_position : bool
            ddf
            
        """
        # Removes 'DC offset' from image to simplify Gaussian fitting.
        inputOffset = (self.data - self.data.min()).astype("float32")

        im_dim = self.axes_manager.signal_shape[::-1]
        m, n = im_dim
        if end_search== "auto":    
            big = 2 * math.floor(( float(np.min(im_dim)) / 8) / 2) - 1
        else:
            big = end_search
            
            
        trialSize = best_size
        # Create blank arrays.
        vert_offset = np.zeros(im_dim)                               
        horz_offset = np.zeros(im_dim) 
        peak        = np.zeros(im_dim) 
        spread      = np.zeros(im_dim)
        
        # Half of the trial size, equivalent to the border that will not be inspected.
        test_box_padding = int(( trialSize - 1 ) / 2.)

        # Coordinate set for X and Y fitting.  
        baseAxis = np.arange( -test_box_padding , test_box_padding , 1 )    
        # Followed by the restoration progress bar:
        # h2=waitbar(0,'Identifying Image Peaks...','Name',version_string)
        # hw2=findobj(h2,'Type','Patch')
        # set(hw2,'EdgeColor',[0 0 0],'FaceColor',[1 0 0]) # Changes the color to red.
        
        for i in np.arange( test_box_padding + 1 , m - ( test_box_padding + 1 ) , 1 ):
            
            currentStrip = inputOffset[ i - test_box_padding : i + test_box_padding , : ] 
            
            for j in np.arange( test_box_padding + 1 , n - ( test_box_padding + 1 ) , 1 ):
                I = currentStrip[ : ,  j - test_box_padding : j + test_box_padding ]
                
                x, sx, hx = estimate_1D_Gaussian_parameters(np.sum(I, 0), baseAxis) # The horizontal offset refinement.
                y, sy, hy = estimate_1D_Gaussian_parameters(np.sum(I, 1), baseAxis) # The vertical offset refinement.
                
                horz_offset[i,j] = x
                vert_offset[i,j] = y
                
                peak[i,j] = (hx+hy) /(2*trialSize)  # Calculates the height of the fitted Gaussian.
                spread[i,j] =   2.3548 * math.sqrt(sx ** 2 + sy ** 2)  # 2D FWHM
                
                          
#                percentageRefined = ( ((trialSize-3.)/2.) / ((big-1.)/2.) ) +   ( ( (i-test_box_padding) / (m - 2*test_box_padding) ) / (((big-1)/2)))  # Progress metric when using a looping peak-finding waitbar.
#                waitbar(percentageRefined,h2) 
    
        del x, y, sx, sy, hx, hy
        return horz_offset, vert_offset, peak, spread
#        # delete (h2)
#        # Feature identification section:
        normalisedPeak = peak / ( np.max(inputOffset) - np.min(inputOffset) )  # Make use of peak knowledge:
        normalisedPeak[normalisedPeak < 0] = 0  # Forbid negative (concave) Gaussians.
        spread = spread / trialSize          # Normalise fitted Gaussian widths.
        offsetRadius = np.sqrt( (horz_offset)**2 + (vert_offset)**2 )  # Calculate offset radii.
        offsetRadius = offsetRadius / trialSize 
        offsetRadius[offsetRadius == 0] = 0.001  # Remove zeros values to prevent division error later.
        # Create search metric and screen impossible peaks:
        search_record = normalisedPeak / offsetRadius
        search_record[search_record > 1] = 1 
        search_record[search_record < 0] = 0 
        search_record[spread < 0.05] = 0       # Invalidates negative Gaussian widths.
        search_record[spread > 1] = 0          # Invalidates Gaussian widths greater than a feature spacing.
        search_record[offsetRadius > 1] = 0    # Invalidates Gaussian widths greater than a feature spacing.
        kernel = int(np.round(trialSize/3))
        if kernel % 2 == 0:
            kernel += 1
        search_record = scipy.signal.medfilt2d(search_record, kernel)  # Median filter to strip impossibly local false-positive features.
        sensitivityThreshold = 0.34            # This is an Admin tunable parameter that is defined here within the core file.
        search_record[search_record < sensitivityThreshold ] = 0   # Collapse improbable features to zero likelyhood.
        search_record[search_record >= sensitivityThreshold ] = 1  # Round likelyhood of genuine features to unity.
               
        # Erode regions of likely features down to points.
        search_record = binary_erosion(search_record, iterations=-1 )
#
#        
#        # [point_coordinates(:,2),point_coordinates(:,1)] = np.where(search_record == 1)  # Extract the locations of the identified features.
#        
#        return search_record
    
    
    
    
    
    
    
    
    
