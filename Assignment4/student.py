# Please place imports here.
# BEGIN IMPORTS
import cv2
import numpy as np
import scipy
from scipy import ndimage, linalg
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Columns are normalized and are to be
                  interpreted as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.  Images are height x width x channels arrays, all
                  with identical dimensions.
    Output:
        albedo -- float32 height x width x channels image with dimensions
                  matching the input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """

    '''This flattens the image into a numpy array of....N x Pixels'''
    pixelNum = images[0].shape[0]*images[0].shape[1]
    chanNum = images[0].shape[2]
    imagesFlatColors = np.reshape(images[0], (1,pixelNum,chanNum))
    for image in range(1,len(images)):
        newImage = np.reshape(images[image], (1,pixelNum,chanNum))
        imagesFlatColors = np.concatenate((imagesFlatColors, newImage), axis=0)
    imagesFlat = np.average(imagesFlatColors, axis=2)  #RGB --> Grayscale

    LLT = np.dot(lights, np.transpose(lights))
    LLTinv = linalg.inv(LLT)
    ILT = np.dot(np.transpose(imagesFlat),np.transpose(lights))
    G_gray = np.dot(ILT,LLTinv)

    '''mask for the normals'''
    mask = np.linalg.norm(G_gray,axis = 1)>1e-7
    mask = np.transpose(np.array([mask, mask, mask]))
    b = np.zeros((pixelNum,3))
    G_gray = np.choose(mask, (b,G_gray))
    denom = np.sqrt((G_gray ** 2).sum(-1))[..., np.newaxis]
    denom[denom<1e-7] = 1e-7
    normals = G_gray/denom
    normals = np.reshape(normals, (images[0].shape[0],images[0].shape[1],3))

    '''getting the RGB version of G so we can back out the kd values'''
    G_RGB = np.empty((chanNum, pixelNum, 3))
    imagesFlatColors = np.swapaxes(imagesFlatColors,0,2)
    imagesFlatColors = np.swapaxes(imagesFlatColors,1,2)
    for channel in range(chanNum):
        ILT = np.dot(np.transpose(imagesFlatColors[channel]),np.transpose(lights))
        G_RGB[channel] = np.dot(ILT,LLTinv)

    '''Now we want to find the albedos, which will be the magnitude of the normal for each Red pixel, etc'''
    albedo = np.linalg.norm(G_RGB,axis = 2)
    albedo = np.swapaxes(albedo,0,1)
    height = images[0].shape[0]
    width = images[0].shape[1]
    albedo = np.reshape(albedo, (height,width,chanNum))

    return albedo, normals


def pyrdown_impl(image):
    """
    Prefilters an image with a gaussian kernel and then downsamples the result
    by a factor of 2.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/16 [ 1 4 6 4 1 ]

    Functions such as cv2.GaussianBlur and
    scipy.ndimage.filters.gaussian_filter are prohibited.  You must implement
    the separable kernel.  However, you may use functions such as cv2.filter2D
    or scipy.ndimage.filters.correlate to do the actual
    correlation / convolution.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Downsampling should take the even-numbered coordinates with coordinates
    starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        down -- ceil(height/2) x ceil(width/2) [x channels] image of type
                float32.
    """
    # Creating kernel, K = 1/16 [ 1 4 6 4 1 ]
    gauss_filter = np.zeros((1, 5))
    gauss_filter[0, 0] = 1
    gauss_filter[0, 1] = 4
    gauss_filter[0, 2] = 6
    gauss_filter[0, 3] = 4
    gauss_filter[0, 4] = 1

    Ky = 1.0/16 * gauss_filter
    Kx = np.transpose(Ky)

    #Filter image in x and y directions using gaussian kernel
    filtered_img = cv2.filter2D(image, -1, Ky, borderType=cv2.BORDER_REFLECT_101)
    filtered_img = cv2.filter2D(filtered_img, -1, Kx, borderType=cv2.BORDER_REFLECT_101)

    #Downsampling filtered image
    down = filtered_img[::2, ::2]

    return down


def pyrup_impl(image):
    """
    Upsamples an image by a factor of 2 and then uses a gaussian kernel as a
    reconstruction filter.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/8 [ 1 4 6 4 1 ]
    Note: 1/8 is not a mistake.  The additional factor of 4 (applying this 1D
    kernel twice) scales the solution according to the 2x2 upsampling factor.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Upsampling should produce samples at even-numbered coordinates with
    coordinates starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        up -- 2 height x 2 width [x channels] image of type float32.
    """
    #Upsampling given image
    up_height = image.shape[0] * 2
    up_width = image.shape[1] * 2
    if (len(image.shape) == 3):
        up_channels = image.shape[2]
        unfiltered_up = np.zeros((up_height, up_width, up_channels))
    else:
        unfiltered_up = np.zeros((up_height, up_width))

    unfiltered_up[::2, ::2] = image

    #Creating gaussian kernel, K = 1/8 [ 1 4 6 4 1 ]
    gauss_filter = np.zeros((1, 5))
    gauss_filter[0, 0] = 1
    gauss_filter[0, 1] = 4
    gauss_filter[0, 2] = 6
    gauss_filter[0, 3] = 4
    gauss_filter[0, 4] = 1

    Ky = 1.0/8 * gauss_filter
    Kx = np.transpose(Ky)

    #Filter image in x and y directions using gaussian kernel
    up = cv2.filter2D(unfiltered_up, -1, Ky, borderType=cv2.BORDER_REFLECT_101)
    up = cv2.filter2D(up, -1, Kx, borderType=cv2.BORDER_REFLECT_101)

    return up


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    If the point has a depth < 1e-7 from the camera or is located behind the
    camera, then set the projection to [np.nan, np.nan].

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    proj_height = points.shape[0]
    proj_width = points.shape[1]
    projections = np.ndarray((proj_height, proj_width, 2))

    # Calculating 2D projections
    for row in range(proj_height):
        for col in range(proj_width):
            # Change points to 3D array
            coord = np.ones((1, 4))
            coord[0, :3] = points[row, col, :3]
            
            # xp = K * Rt * X
            X = np.transpose(coord)
            xp = np.dot(K, np.dot(Rt, X))

            if(xp[2, 0] < 1e-7):
                projections[row, col, :2] = np.nan
            else:
                xp /= xp[2, 0]
                projections[(row, col)] = xp[:2, 0]

    return projections


def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Undo camera projection given a calibrated camera and the depth for each
    corner of an image.

    The output points array is a 2x2x3 array arranged for these image
    coordinates in this order:

     (0, 0)      |  (width, 0)
    -------------+------------------
     (0, height) |  (width, height)

    Each of these contains the 3 vector for the corner's corresponding
    point in 3D.

    Tutorial:
      Say you would like to unproject the pixel at coordinate (x, y)
      onto a plane at depth z with camera intrinsics K and camera
      extrinsics Rt.

      (1) Convert the coordinates from homogeneous image space pixel
          coordinates (2D) to a local camera direction (3D):
          (x', y', 1) = K^-1 * (x, y, 1)
          the camera center.  Multiply it by z to get the point at depth z
          from the camera center.
          (z * x', z * y', z) = z * (x', y', 1)
      (3) Use the inverse of the extrinsics matrix, Rt, to move this point
          from the local camera coordinate system to a world space
          coordinate.
          Note:
            | R t |^-1 = | R' -R't |
            | 0 1 |      | 0   1   |

          p = R' * (z * x', z * y', z, 1) - R't

    Input:
        K -- camera intrinsics calibration matrix
        width -- camera width
        height -- camera height
        depth -- depth of plane with respect to camera
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D points
    """
    # (1) 2D -> 3D, (x', y', 1) = K^-1 * (x, y, 1)
    corners = np.ones((3, 4))
    corners[:2, 0] = 0
    corners[0, 1] = width
    corners[1, 1] = 0
    corners[0, 2] = 0
    corners[1, 2] = height
    corners[0, 3] = width
    corners[1, 3] = height

    K_inv = np.linalg.inv(K)
    cam_dir = np.dot(K_inv, corners)
    
    # (2) Include depth, (z * x', z * y', z) = z * (x', y', 1)
    cam_dir = depth * cam_dir

    # (3) Camera coord. system -> world space coord., p = R' * (z * x', z * y', z, 1) - R't
    Rt_matrix = np.zeros((4 , 4))
    Rt_matrix[:3, :] = Rt
    Rt_matrix[3, 3] = 1
    Rt_inv = np.linalg.inv(Rt_matrix)

    cam_pts = np.ones((4, 4))
    cam_pts[:3, :] = cam_dir
    p = np.dot(Rt_inv, cam_pts)

    points = np.zeros((2, 2, 3))
    points[0, 0] = p[:3, 0]
    points[0, 1] = p[:3, 1]
    points[1, 0] = p[:3, 2]
    points[1, 1] = p[:3, 3]
    
    return points


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches of shape channels x height x width (e.g. 3 x ncc_size x ncc_size)
    are to be flattened into vectors with the default numpy row major order.
    For example, given the following 2 (channels) x 2 (height) x 2 (width)
    patch, here is how the output vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.  ncc_size
                    will always be odd.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    halfPatch = np.floor(ncc_size/2)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    normalized = np.zeros((height, width, ncc_size, ncc_size, channels))
    for row in range(int(np.floor(ncc_size/2)), int(image.shape[0]-np.floor(ncc_size/2))):
        for col in range(int(np.floor(ncc_size/2)), int(image.shape[1]-np.floor(ncc_size/2))):
            normalized[row][col] = np.copy(image[row-halfPatch:row+halfPatch+1,col-halfPatch:col+halfPatch+1,:])
    '''check rows and columns here'''
    normalized = np.reshape(normalized,(height, width, ncc_size**2, channels))
    normalized = np.swapaxes(normalized,2,3)
    normalized -= np.mean(normalized, axis = 3, keepdims = True)
    norms = np.linalg.norm(normalized,axis = (2,3))
    norms = np.reshape(norms, (height,width,1,1))
    normalized /= np.maximum(1e-6,norms)
    normalized = np.reshape(normalized, (height, width, channels*ncc_size**2))
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    vectors = image1*image2
    ncc = np.sum(vectors, axis = 2)
    return ncc
