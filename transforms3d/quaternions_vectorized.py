'''Numpy vectorized functions to operate on, or return, quaternions.

Quaternions here consist of 4 values ``w, x, y, z``, where ``w`` is the
real (scalar) part, and ``x, y, z`` are the complex (vector) part.

Note - rotation matrices here apply to column vectors, that is,
they are applied on the left of the vector.  For example:

>>> import numpy as np
>>> q = [0, 1, 0, 0] # 180 degree rotation around axis 0
>>> M = quat2mat(q) # from this module
>>> vec = np.array([1, 2, 3]).reshape((3,1)) # column vector
>>> tvec = np.dot(M, vec)

Terms used in function names:

* *mat* : array shape (3, 3) (3D non-homogenous coordinates)
* *aff* : affine array shape (4, 4) (3D homogenous coordinates)
* *quat* : quaternion shape (4,)
* *axangle* : rotations encoded by axis vector and angle scalar
'''

import math
import numpy as np

_MAX_FLOAT = np.maximum_sctype(np.float)
_FLOAT_EPS = np.finfo(np.float).eps


def qmult(q1, q2):
    ''' Batch multiply two sets of quaternions

    Parameters
    ----------
    q1 : shape (4,n) array of quaternions
    q2 : shape (4,n) array of quaternions

    Returns
    -------
    q12 : array shape (4,n) array of quaternions

    Notes
    -----
    See : http://en.wikipedia.org/wiki/Quaternions#Hamilton_product
    '''
    q1 = np.array(q1)
    q2 = np.array(q2)

    w1 = q1[0]
    x1 = q1[1]
    y1 = q1[2]
    z1 = q1[3]
    w2 = q2[0]
    x2 = q2[1]
    y2 = q2[2]
    z2 = q2[3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.c_[w, x, y, z].T


def qconjugate(q):
    ''' Batch conjugate of quaternion

    Parameters
    ----------
    q : shape (4,n)
       w, i, j, k of quaternions

    Returns
    -------
    conjq : array shape (4,n)
       w, i, j, k of conjugates of `q`
    '''
    q = np.array(q)
    return q * np.array([[1.0, -1, -1, -1]]*np.shape(q)[1]).T


def rotate_vector(v, q):
    ''' Batch apply transformations in quaternions `q` to vectors `v`

    Parameters
    ----------
    v : shape (3, n) element sequence
       3 dimensional vectors
    q : shape (4, n) element sequence
       w, i, j, k of quaternions

    Returns
    -------
    vdash : array shape (3, n)
       `v` rotated by quaternions `q`

    Notes
    -----
    See: http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Describing_rotations_with_quaternions
    '''
    v = np.array(v)
    q = np.array(q)
    varr = np.r_[np.zeros((1, np.shape(v)[1])), v]
    return qmult(q, qmult(varr, qconjugate(q)))[1:]
