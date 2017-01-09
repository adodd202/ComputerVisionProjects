"""
                     PA5 STUDENT IMPLEMENATION
                     -------------------------
"""

# Answer TODO 1 as a comment here:
############################ TODO 1 BEGIN #################################
# When AlexNet is calculating FC6 and FC7, it is performing several operations on the layers
# once they are intially calculated (by the inner product) and then reassigning the values of FC6 and 7. These operations
# include ReLU and dropout. ReLU effectively makes all values of FC 6 and 7 either positive or zeros.
# However, FC 8 has no ReLU function after it, meaning it is possible to still have negative values from the inner product calculation
# that was performed from FC7 to FC8.
#
#
############################ TODO 1 END #################################


# Add imports here
import numpy as np


def convert_ilsvrc2012_probs_to_dog_vs_food_probs(probs_ilsvrc):
    """
    Convert from 1000-class ILSVRC probabilities to 2-class "dog vs food"
    incices.  Use the variables "dog_indices" and "food_indices" to map from
    ILSVRC2012 classes to our classes.

    HINT:
    Compute "probs" by first estimating the probability of classes 0 and 1,
    using probs_ilsvrc.  Stack together the two probabilities along axis 1, and
    then normalize (along axis 1).

    :param probs_ilsvrc: shape (N, 1000) probabilities across 1000 ILSVRC classes

    :return probs: shape (N, 2): probabilities of each of the N items as being
        either dog (class 0) or food (class 1).
    """
    # in the ILSVRC2012 dataset, indices 151-268 are dogs and index 924-969 are foods
    dog_indices = range(151, 269)
    food_indices = range(924, 970)
    N, _ = probs_ilsvrc.shape
    probs = np.zeros((N, 2)) # placeholder
    ############################ TODO 2 BEGIN #################################
    dog_prob = np.reshape(np.sum(probs_ilsvrc[: , dog_indices], axis=1), (N, 1))
    food_prob = np.reshape(np.sum(probs_ilsvrc[: , food_indices], axis=1), (N, 1))
    probs = np.hstack((dog_prob, food_prob))
    
    for row in range(probs.shape[0]):
        norm_sum = probs[row, 0] + probs[row, 1]
        probs[row] /= norm_sum
    ############################ TODO 2 END #################################
    return probs


def get_prediction_descending_order_indices(probs, cidx):
    """
    Returns the ordering of probs that would sort it in descending order

    :param probs: (N, 2) probabilities (computed in TODO 2)
    :param cidx: class index (0 or 1)

    :return list of N indices that sorts the array in descending order
    """
    order = range(probs.shape[0]) # placeholder
    ############################ TODO 3 BEGIN #################################
    cidx_probs = probs[: , cidx].tolist()
    new_order = []
    while (len(cidx_probs) != 0):
        max_index = np.argmax(cidx_probs)
        new_order.append(order[max_index])
        cidx_probs.remove(cidx_probs[max_index])
        order.remove(order[max_index])

    order = new_order
    ############################ TODO 3 END #################################
    return order


def compute_dscore_dimage(net, data, class_idx):
    """
    Returns the gradient of s_y (the score at index class_idx) with respect to
    the image (data), ds_y / dI.  Note that this is the unnormalized class
    score "s", not the probability "p".

    :param data: (3, 227, 227) array, input image
    :param class_idx: class index in range [0, 999] indicating which class
    :param net: a caffe Net object

    :return grad: (3, 227, 227) array, gradient ds_y / dI
    """
    grad = np.zeros_like(data) # placeholder
    ############################ TODO 4 BEGIN #################################
    net.blobs['fc8'].diff[0, :] = 0
    net.blobs['fc8'].diff[0, class_idx] = 1
    net.backward(start = 'fc8', end = 'data')
    grad = np.copy(net.blobs['data'].diff[0])
    ############################ TODO 4 END #################################
    assert grad.shape == (3, 227, 227) # expected shape
    return grad


def normalized_sgd_with_momentum_update(data, grad, velocity, momentum, learning_rate):
    """
    THIS IS SLIGHTLY DIFFERENT FROM NORMAL SGD+MOMENTUM; READ THE NOTEBOOK :)

    Update the image using normalized SGD+Momentum.  To make learning more
    stable, normalize the gradient before using it in the update rule.

    :param data: shape (3, 227, 227) the current solution
    :param grad: gradient of tthe loss with respect to the image
    :param velocity: momentum vector "V"
    :param momentum: momentum parameter "mu"
    :param learning_rate: learning rate "alpha"

    :return: the updated image and momentum vector (data, velocity)
    """
    ############################ TODO 5a BEGIN #################################
    velocity = momentum*velocity - learning_rate*(grad/np.linalg.norm(grad))
    data = data + velocity
    ############################ TODO 5a END #################################
    return data, velocity


def fooling_image_gradient(net, orig_data, data, target_class, regularization):
    """
    Compute the gradient for make_fooling_image (dL / dI).

    :param net: a caffe Net object
    :param orig_data: shape (3, 227, 227) the original image
    :param target_class: ILSVRC class in range [0, 999]
    :param data: shape (3, 227, 227) the current solution
    :param regularization: weight (lambda) applied to the regularizer.
    """
    grad = np.zeros_like(data) # placeholder
    ############################ TODO 5b BEGIN #################################
    sy = compute_dscore_dimage(net, data, target_class)
    dR = regularization * (data - orig_data)
    grad = -sy + dR
    ############################ TODO 5b END #################################
    assert grad.shape == (3, 227, 227) # expected shape
    return grad


def class_visualization_gradient(net, data, target_class, regularization):
    """
    Compute the gradient for make_class_visualization (dL / dI).

    :param net: a caffe Net object
    :param target_class: ILSVRC class in range [0, 999]
    :param data: shape (3, 227, 227) the current solution
    :param regularization: weight (lambda) applied to the regularizer.
    """
    grad = np.zeros_like(data) # placeholder
    ############################ TODO 6 BEGIN #################################
    sy = compute_dscore_dimage(net, data, target_class)
    grad = -sy + (regularization * data)
    ############################ TODO 6 END #################################
    assert grad.shape == (3, 227, 227) # expected shape
    return grad


def feature_inversion_gradient(net, data, blob_name, target_feat, regularization):
    """
    Compute the gradient for make_feature_inversion (dL / dI).

    :param net: a caffe Net object
    :param target_feat: target feature
    :param data: shape (3, 227, 227) the current solution
    :param regularization: weight (lambda) applied to the regularizer.
    """
    grad = np.zeros_like(data) # placeholder
    ############################ TODO 7a BEGIN #################################
    M = target_feat.size
    # Find the backprop of delta_phi(I)-phi(I')
    net.blobs[blob_name].diff[...] = 1.0/M * (net.blobs[blob_name].data - target_feat)
    net.backward(start = blob_name, end = 'data')
    new_data_diff = net.blobs['data'].diff[0]
    # dL/dI gradient of image
    grad = new_data_diff + regularization*data
    ############################ TODO 7a END #################################
    assert grad.shape == (3, 227, 227) # expected shape
    return grad


# Answer TODO 7b as a comment here:
############################ TODO 7b BEGIN #################################
#
# (a) The quality of the reconstruction of the original image is lower when reconstructing from higher layers. 
# The computation of each layer results in some data loss because of the nonlinearities of many of the layers.
# For example, ReLU gets rid of negative values, making them all zero, dropout is nonlinear, pooling is 
# nonlinear. It suggests that the representation becomes more generalized with increasing layer number.
# This is good because a more generalized image will be easier to classify in a category. For example,
# if an image has a picture of the back of a jeep, the ConvNet will generalize this out to vehicle perhaps.
#
# (b) Because each layer has different values, each layer will respond to parameters differently. In the case
# of regularization, if lambda is too low, this will lead to data overfitting, and if lambda is too high this
# will lead to data underfitting. To expand on this, if lambda is too high, the image just stays as a gray image
# (nothing changes). If lambda is too low, there is more noise due to overfitting. A different value for each
# layer wouldn't be needed if the objective function was modified so that the gradient of the data has greater
# weight than that of the regularizer. This could be done by changing the constant in front of the gradient so
# that the derivative of the regularizer would not matter in the gradient of the loss function.
#
############################ TODO 7b END #################################
