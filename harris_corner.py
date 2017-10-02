# <Your name>
# COMP 776, Fall 2017
# Assignment: Feature Extraction

import numpy as np

from scipy.ndimage.filters import gaussian_filter, maximum_filter, sobel

#-------------------------------------------------------------------------------

class HarrisCornerFeatureDetector:
    def __init__(self, args):
        self.gaussian_sigma = args.gaussian_sigma
        self.maxfilter_window_size = args.maxfilter_window_size
        self.harris_corner_k = args.harris_corner_k
        self.max_num_features = args.max_num_features


    #---------------------------------------------------------------------------

    # detect corner features in an input image
    # inputs:
    # - image: a grayscale image
    # returns:
    # - keypoints: N x 2 array of keypoint (x,y) pixel locations in the image,
    #   assumed to be integer coordinates
    def __call__(self, image):
        corner_response = self.compute_corner_response(image)
        keypoints = self.get_keypoints(corner_response)

        return keypoints

    #---------------------------------------------------------------------------

    # compute the Harris corner response function for each point in the image
    #   R(x, y) = det(M(x, y) - k * tr(M(x, y))^2
    # where
    #             [      I_x(x, y)^2        I_x(x, y) * I_y(x, y) ]
    #   M(x, y) = [ I_x(x, y) * I_y(x, y)        I_y(x, y)^2      ] * G
    #
    # with "* G" denoting convolution with a 2D Gaussian.
    #
    # inputs:
    # - image: a grayscale image
    # returns:
    # - R: transformation of the input image to reflect "cornerness"
    def compute_corner_response(self, image):
        # TODO: Compute the Harris corner response
        # https://www.mathworks.com/matlabcentral/answers/65593-what-is-wrong-with-my-code-harris-corner-detector?requestedDomain=www.mathworks.com
        # Compute gradients
        image_x = sobel(image, axis=1)
        image_y = sobel(image, axis=0)

        image_x_2 = image_x * image_x
        image_y_2 = image_y * image_y
        image_x_y = image_x * image_y

        # Smooth the gradient images
        image_x_2_smooth = gaussian_filter(image_x_2, self.gaussian_sigma)
        image_y_2_smooth = gaussian_filter(image_y_2, self.gaussian_sigma)
        image_x_y_smooth = gaussian_filter(image_x_y, self.gaussian_sigma)

        # Compute M and R
        det = image_x_2_smooth * image_y_2_smooth - image_x_y_smooth * image_x_y_smooth
        trace = image_x_2_smooth + image_y_2_smooth
        R = det - self.harris_corner_k * (trace * trace)

        # Non-maximal suppression
        R_nms = maximum_filter(R, (self.maxfilter_window_size, self.maxfilter_window_size))

        return R_nms

    #---------------------------------------------------------------------------

    # find (x,y) pixel coordinates of maxima in a corner response map
    # inputs:
    # - R: Harris corner response map
    # returns:
    # - keypoints: N x 2 array of keypoint (x,y) pixel locations in the corner
    #   response map, assumed to be integer coordinates
    def get_keypoints(self, R):
        # TODO: apply non-maximum suppression and obtain the up-to-K strongest
        # features having R(x,y) > 0

        return np.empty((0, 2))
