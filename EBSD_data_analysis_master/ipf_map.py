#!/usr/bin/env python
# coding: utf-8
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import time
from sklearn.cluster import KMeans
import pandas as pd
import sympy

'''
通过欧拉角画反极图；
通过ipf map画反极图；
'''


def OrientationMatrix2Euler(g):
    """
    Compute the Euler angles from the orientation matrix.

    This conversion follows the paper of Rowenhorst et al. :cite:`Rowenhorst2015`.
    In particular when :math:`g_{33} = 1` within the machine precision,
    there is no way to determine the values of :math:`\phi_1` and :math:`\phi_2`
    (only their sum is defined). The convention is to attribute
    the entire angle to :math:`\phi_1` and set :math:`\phi_2` to zero.

    :param g: The 3x3 orientation matrix
    :return: The 3 euler angles in degrees.
    """
    eps = np.finfo('float').eps
    (phi1, Phi, phi2) = (0.0, 0.0, 0.0)
    # treat special case where g[2, 2] = 1
    if np.abs(g[2, 2]) >= 1 - eps:
        if g[2, 2] > 0.0:
            phi1 = np.arctan2(g[0][1], g[0][0])
        else:
            phi1 = -np.arctan2(-g[0][1], g[0][0])
            Phi = np.pi
    else:
        Phi = np.arccos(g[2][2])
        zeta = 1.0 / np.sqrt(1.0 - g[2][2] ** 2)
        phi1 = np.arctan2(g[2][0] * zeta, -g[2][1] * zeta)
        phi2 = np.arctan2(g[0][2] * zeta, g[1][2] * zeta)
    # ensure angles are in the range [0, 2*pi]
    if phi1 < 0.0:
        phi1 += 2 * np.pi
    if Phi < 0.0:
        Phi += 2 * np.pi
    if phi2 < 0.0:
        phi2 += 2 * np.pi
    return np.degrees([phi1, Phi, phi2])

def OrientationMatrix2Rodrigues(g):
    """
    Compute the rodrigues vector from the orientation matrix.

    :param g: The 3x3 orientation matrix representing the rotation.
    :returns: The Rodrigues vector as a 3 components array.
    """
    t = g.trace() + 1
    if np.abs(t) < np.finfo(g.dtype).eps:
        print('warning, returning [0., 0., 0.], consider using axis, angle '
              'representation instead')
        return np.zeros(3)
    else:
        r1 = (g[1, 2] - g[2, 1]) / t
        r2 = (g[2, 0] - g[0, 2]) / t
        r3 = (g[0, 1] - g[1, 0]) / t
    return np.array([r1, r2, r3])

def OrientationMatrix2Quaternion(g, P=1):
    q0 = 0.5 * np.sqrt(1 + g[0, 0] + g[1, 1] + g[2, 2])
    q1 = P * 0.5 * np.sqrt(1 + g[0, 0] - g[1, 1] - g[2, 2])
    q2 = P * 0.5 * np.sqrt(1 - g[0, 0] + g[1, 1] - g[2, 2])
    q3 = P * 0.5 * np.sqrt(1 - g[0, 0] - g[1, 1] + g[2, 2])

    if g[2, 1] < g[1, 2]:
        q1 = q1 * -1
    elif g[0, 2] < g[2, 0]:
        q2 = q2 * -1
    elif g[1, 0] < g[0, 1]:
        q3 = q3 * -1

    q = np.array([q0, q1, q2, q3])
    return q

def Rodrigues2OrientationMatrix(rod):
    """
    Compute the orientation matrix from the Rodrigues vector.

    :param rod: The Rodrigues vector as a 3 components array.
    :returns: The 3x3 orientation matrix representing the rotation.
    """
    r = np.linalg.norm(rod)
    I = np.diagflat(np.ones(3))
    if r < np.finfo(r.dtype).eps:
        # the rodrigues vector is zero, return the identity matrix
        return I
    theta = 2 * np.arctan(r)
    n = rod / r
    omega = np.array([[0.0, n[2], -n[1]],
                      [-n[2], 0.0, n[0]],
                      [n[1], -n[0], 0.0]])
    g = I + np.sin(theta) * omega + (1 - np.cos(theta)) * omega.dot(omega)
    return g

def Rodrigues2Axis(rod):
    """
    Compute the axis/angle representation from the Rodrigues vector.

    :param rod: The Rodrigues vector as a 3 components array.
    :returns: A tuple in the (axis, angle) form.
    """
    r = np.linalg.norm(rod)
    axis = rod / r
    angle = 2 * np.arctan(r)
    return axis, angle

def Axis2OrientationMatrix(axis, angle):
    """
    Compute the (passive) orientation matrix associated the rotation defined by the given (axis, angle) pair.

    :param axis: the rotation axis.
    :param angle: the rotation angle (degrees).
    :returns: the 3x3 orientation matrix.
    """
    omega = np.radians(angle)
    c = np.cos(omega)
    s = np.sin(omega)
    g = np.array([[c + (1 - c) * axis[0] ** 2,
                   (1 - c) * axis[0] * axis[1] + s * axis[2],
                   (1 - c) * axis[0] * axis[2] - s * axis[1]],
                  [(1 - c) * axis[0] * axis[1] - s * axis[2],
                   c + (1 - c) * axis[1] ** 2,
                   (1 - c) * axis[1] * axis[2] + s * axis[0]],
                  [(1 - c) * axis[0] * axis[2] + s * axis[1],
                   (1 - c) * axis[1] * axis[2] - s * axis[0],
                   c + (1 - c) * axis[2] ** 2]])
    return g

def Euler2Axis(euler):
    """Compute the (axis, angle) representation associated to this (passive)
    rotation expressed by the Euler angles.

    :param euler: 3 euler angles (in degrees).
    :returns: a tuple containing the axis (a vector) and the angle (in radians).
    """
    (phi1, Phi, phi2) = np.radians(euler)
    t = np.tan(0.5 * Phi)
    s = 0.5 * (phi1 + phi2)
    d = 0.5 * (phi1 - phi2)
    tau = np.sqrt(t ** 2 + np.sin(s) ** 2)
    alpha = 2 * np.arctan2(tau, np.cos(s))
    if alpha > np.pi:
        axis = np.array([-t / tau * np.cos(d), -t / tau * np.sin(d), -1 / tau * np.sin(s)])
        angle = 2 * np.pi - alpha
    else:
        axis = np.array([t / tau * np.cos(d), t / tau * np.sin(d), 1 / tau * np.sin(s)])
        angle = alpha
    return axis, angle



def Euler2Rodrigues(euler):
    """Compute the rodrigues vector from the 3 euler angles (in degrees).

    :param euler: the 3 Euler angles (in degrees).
    :return: the rodrigues vector as a 3 components numpy array.
    """
    (phi1, Phi, phi2) = np.radians(euler)
    a = 0.5 * (phi1 - phi2)
    b = 0.5 * (phi1 + phi2)
    r1 = np.tan(0.5 * Phi) * np.cos(a) / np.cos(b)
    r2 = np.tan(0.5 * Phi) * np.sin(a) / np.cos(b)
    r3 = np.tan(b)
    return np.array([r1, r2, r3])


def Euler2OrientationMatrix(euler):
    """Compute the orientation matrix :math:`\mathbf{g}` associated with
    the 3 Euler angles :math:`(\phi_1, \Phi, \phi_2)`.

    The matrix is calculated via (see the `euler_angles` recipe in the
    cookbook for a detailed example):

    .. math::

       \mathbf{g}=\\begin{pmatrix}
       \cos\phi_1\cos\phi_2 - \sin\phi_1\sin\phi_2\cos\Phi &
       \sin\phi_1\cos\phi_2 + \cos\phi_1\sin\phi_2\cos\Phi &
       \sin\phi_2\sin\Phi \\\\
       -\cos\phi_1\sin\phi_2 - \sin\phi_1\cos\phi_2\cos\Phi &
       -\sin\phi_1\sin\phi_2 + \cos\phi_1\cos\phi_2\cos\Phi &
       \cos\phi_2\sin\Phi \\\\
       \sin\phi_1\sin\Phi & -\cos\phi_1\sin\Phi & \cos\Phi \\\\
       \end{pmatrix}

    :param euler: The triplet of the Euler angles (in degrees).
    :return g: The 3x3 orientation matrix.
    """
    (rphi1, rPhi, rphi2) = np.radians(euler)
    c1 = np.cos(rphi1)
    s1 = np.sin(rphi1)
    c = np.cos(rPhi)
    s = np.sin(rPhi)
    c2 = np.cos(rphi2)
    s2 = np.sin(rphi2)

    # rotation matrix g
    g11 = c1 * c2 - s1 * s2 * c
    g12 = s1 * c2 + c1 * s2 * c
    g13 = s2 * s
    g21 = -c1 * s2 - s1 * c2 * c
    g22 = -s1 * s2 + c1 * c2 * c
    g23 = c2 * s
    g31 = s1 * s
    g32 = -c1 * s
    g33 = c
    g = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
    return g


def euler_to_g(a,b,c): #Bunge Euler angles,input radians
    a11= (np.cos(c))*(np.cos(a))-(np.cos(b))*(np.sin(a))*(np.sin(c))
    a12=(np.cos(c))*(np.sin(a))+(np.cos(b))*(np.cos(a))*(np.sin(c))
    a13=(np.sin(c))*(np.sin(b))
    a21=-(np.sin(c))*(np.cos(a))-(np.cos(b))*(np.sin(a))*(np.cos(c))
    a22=-(np.sin(c))*(np.sin(a))+(np.cos(b))*(np.cos(a))*(np.cos(c))
    a23=(np.cos(c))*(np.sin(b))
    a31=(np.sin(b))*(np.sin(a))
    a32=-(np.sin(b))*(np.cos(a))
    a33=(np.cos(b))
    g=np.array(([a11,a12,a13],[a21,a22,a23],[a31,a32,a33])) #rotation matrix
    return g


def Hexagonal_622_Sym():
    """
    This is a utility function that returns the rotation matrices
    corresponding to the 12 Crystal Symmetry elements O[622].
    #Inputs
    None
    #Output:
    Sym_crys: Crystal symmetry operators for Hexagonal O[622] rotations
    Applying 12 Crystal Symmetry elements O[622] to the orientation matrix
    Expressed in ortho-hexagonal coordinates
    Has been comparied with pymicro in Dec.16,2020
    """

    a = np.sqrt(3) / 2.0

    Sym_crys = np.zeros((3, 3, 12))
    # 1
    Sym_crys[0, 0, 0] = 1
    Sym_crys[1, 1, 0] = 1
    Sym_crys[2, 2, 0] = 1
    # 2
    Sym_crys[0, 0, 1] = 0.5
    Sym_crys[1, 1, 1] = 0.5
    Sym_crys[2, 2, 1] = 1
    Sym_crys[0, 1, 1] = a
    Sym_crys[1, 0, 1] = -a
    # 3
    Sym_crys[0, 0, 2] = -0.5
    Sym_crys[1, 1, 2] = -0.5
    Sym_crys[2, 2, 2] = 1
    Sym_crys[0, 1, 2] = a
    Sym_crys[1, 0, 2] = -a
    # 4
    Sym_crys[0, 0, 3] = -0.5
    Sym_crys[1, 1, 3] = -0.5
    Sym_crys[2, 2, 3] = 1
    Sym_crys[0, 1, 3] = -a
    Sym_crys[1, 0, 3] = a
    # 5
    Sym_crys[0, 0, 4] = -1
    Sym_crys[1, 1, 4] = -1
    Sym_crys[2, 2, 4] = 1
    # 6
    Sym_crys[0, 0, 5] = 0.5
    Sym_crys[1, 1, 5] = 0.5
    Sym_crys[2, 2, 5] = 1
    Sym_crys[0, 1, 5] = -a
    Sym_crys[1, 0, 5] = a
    # 7
    Sym_crys[0, 0, 6] = 0.5
    Sym_crys[1, 1, 6] = -0.5
    Sym_crys[2, 2, 6] = -1
    Sym_crys[0, 1, 6] = a
    Sym_crys[1, 0, 6] = a
    # 8
    Sym_crys[0, 0, 7] = 1
    Sym_crys[1, 1, 7] = -1
    Sym_crys[2, 2, 7] = -1
    # 9
    Sym_crys[0, 0, 8] = -0.5
    Sym_crys[1, 1, 8] = 0.5
    Sym_crys[2, 2, 8] = -1
    Sym_crys[0, 1, 8] = a
    Sym_crys[1, 0, 8] = a
    # 10
    Sym_crys[0, 0, 9] = -0.5
    Sym_crys[1, 1, 9] = 0.5
    Sym_crys[2, 2, 9] = -1
    Sym_crys[0, 1, 9] = -a
    Sym_crys[1, 0, 9] = -a
    # 11
    Sym_crys[0, 0, 10] = -1
    Sym_crys[1, 1, 10] = 1
    Sym_crys[2, 2, 10] = -1
    # 12
    Sym_crys[0, 0, 11] = 0.5
    Sym_crys[1, 1, 11] = -0.5
    Sym_crys[2, 2, 11] = -1
    Sym_crys[0, 1, 11] = -a
    Sym_crys[1, 0, 11] = -a
    return Sym_crys

def Hexagonal_622_Sym1():
    sym = np.zeros((12, 3, 3), dtype=np.float64)
    s60 = np.sin(60 * np.pi / 180)
    sym[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    sym[1] = np.array([[0.5, s60, 0.], [-s60, 0.5, 0.], [0., 0., 1.]])
    sym[2] = np.array([[-0.5, s60, 0.], [-s60, -0.5, 0.], [0., 0., 1.]])
    sym[3] = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    sym[4] = np.array([[-0.5, -s60, 0.], [s60, -0.5, 0.], [0., 0., 1.]])
    sym[5] = np.array([[0.5, -s60, 0.], [s60, 0.5, 0.], [0., 0., 1.]])
    sym[6] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    sym[7] = np.array([[0.5, s60, 0.], [s60, -0.5, 0.], [0., 0., -1.]])
    sym[8] = np.array([[-0.5, s60, 0.], [s60, 0.5, 0.], [0., 0., -1.]])
    sym[9] = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    sym[10] = np.array([[-0.5, -s60, 0.], [-s60, 0.5, 0.], [0., 0., -1.]])
    sym[11] = np.array([[0.5, -s60, 0.], [-s60, -0.5, 0.], [0., 0., -1.]])
    return sym



def calc_IPF_position(euler_1, euler_2, euler_3, ND=np.array([0, 0, 1])):
    """
    This function projects the sample direction (ND) into the
    inverse pole figure (IPF) reference frame, applies crystal rotations
    and antipodal symmetry and returns the position(x,y) of ND
    in the fundamental zone of hexagonal symmetry

    Packages needed: import numpy as np
    Dependent on Functions: euler_to_g() and Hexagonal_622_Sym()

    Usage: Calc_IPF_feature(90,10,20,np.array([0,0,1]))

    #Inputs
     euler_1, euler_2, euler_3 : Bunge Euler angles
     ND : Sample direction to be projected

    #Output:
    fzND: a array of 1_pos and 2_pos

    """
    epsilon = 0.5  # Degree Precision
    g = Rodrigues2OrientationMatrix((euler_1, euler_2, euler_3))  # 3*3的数组
    Sym_crys = Hexagonal_622_Sym1()
    ND = ND / np.linalg.norm(ND)  # Making unit vector，devieded by its norm
    # print(ND)

    Sym_len = Sym_crys.shape[0]  # (3,3,12)故该长度就是12
    # print(Sym_len)

    # Apply crystal Symmetry operator to direction and transform it to crystal frame {h} = {Ocrystal}.(g).
    # {Osample}.h

    # Apply crystal Symmetry
    for i in range(Sym_len):  # ND is in sample frame

        v_sym = np.dot(Sym_crys[i], (g.dot(ND)))  # 这里ND不是110时，np.tensordot(ND,
        # g,axes=1)得到的是一个一维数组，像一个列表。这里得到的ND_crys是一个（12,3）数组
        # 3.Convert rotated direction to spherical angles Theta=acos(h'z),Phi=atan2(h'y.h'x)
        if v_sym[2] < 0:  # If z component of ND is negative (southern hemisphere) make ND=-ND, antipodal symmetry
            v_sym = -v_sym
        if v_sym[1] < 0 or v_sym[0] < 0:
            continue
        elif v_sym[1] / v_sym[0] > np.tan(np.pi / 6):
            continue
        else:
            break
    # print(v_sym)
    axis_rot = v_sym
    # print(axis_rot)
    if axis_rot[2] < 0:
        axis_rot *= -1
    c = axis_rot + ND
    # print(c)
    if axis_rot[2] < 0.000001:
        print(axis_rot)
    c /= c[2]
    # print(c)
    ipf_x1 = c[0]
    ipf_y1 = c[1]

    fzND = np.array([ipf_x1, ipf_y1])
    # print(fzND)
    return (fzND)

def eular2rgb_new(euler_1, euler_2, euler_3, axis=np.array([0., 0., 1.])):
    gmatrix = Euler2OrientationMatrix((euler_1, euler_2, euler_3))
    Sym_crys = Hexagonal_622_Sym1()
    axis = axis / np.linalg.norm(axis)

    for sym in Sym_crys:
        Osym = np.dot(sym, gmatrix)
        Vc = np.dot(Osym, axis)
        if Vc[2] < 0:
            Vc *= -1.  # using the upward direction
        uvw = np.array([Vc[2] - Vc[1], Vc[1] - Vc[0], Vc[0]])
        uvw /= np.linalg.norm(uvw)
        uvw /= max(uvw)
        if (uvw[0] >= 0. and uvw[0] <= 1.0) and (uvw[1] >= 0. and uvw[1] <= 1.0) and (
                uvw[2] >= 0. and uvw[2] <= 1.0):
            # print('found sym for sst')
            break
    return uvw


def eular2rgb(euler_1, euler_2, euler_3, axis=np.array([0., 0., 1.]) ):
    # euler_1 = np.radians(euler_1)
    # euler_2 = np.radians(euler_2)
    # euler_3 = np.radians(euler_3)
    # gmatrix = euler_to_g(euler_1, euler_2, euler_3)  # 3*3的数组
    rmatrix = Euler2OrientationMatrix((euler_1, euler_2, euler_3))
    # print(rmatrix)
    Sym_crys = Hexagonal_622_Sym1()
    axis = axis / np.linalg.norm(axis)
    syms = np.concatenate((Sym_crys, -Sym_crys))
    Vc = np.dot(rmatrix, axis)
    # print(rmatrix)
    # print(Vc)
    Vc_syms = np.dot(syms, Vc)

    # phi: rotation around 001 axis, from 100 axis to Vc vector, projected on (100,010) plane
    Vc_phi = np.arctan2(Vc_syms[:, 1], Vc_syms[:, 0]) * 180 / math.pi
    # chi: rotation around 010 axis, from 001 axis to Vc vector, projected on (100,001) plane
    Vc_chi = np.arctan2(Vc_syms[:, 0], Vc_syms[:, 2]) * 180 / math.pi
    # psi : angle from 001 axis to Vc vector
    Vc_psi = np.arccos(Vc_syms[:, 2]) * 180 / math.pi

    angleR = 90 - Vc_psi  # red color proportional to (90 - psi)
    # print(angleR)
    minAngleR = 0
    maxAngleR = 90
    angleB = Vc_phi  # blue color proportional to phi
    # print(angleB)
    minAngleB = 0
    maxAngleB = 30
    fz_list = ((angleR >= minAngleR) & (angleR < maxAngleR) &
               (angleB >= minAngleB) & (angleB < maxAngleB)).tolist()
    # print(fz_list)
    if not fz_list.count(True) == 1:
        raise (ValueError('problem moving to the fundamental zone'))
        return None
    i_SST = fz_list.index(True)

    # print(Vc_syms[i_SST])
    r = angleR[i_SST] / maxAngleR
    g = (maxAngleR - angleR[i_SST]) / maxAngleR * (maxAngleB - angleB[i_SST]) / maxAngleB
    b = (maxAngleR - angleR[i_SST]) / maxAngleR * angleB[i_SST] / maxAngleB
    rgb = np.array([r, g, b])
    # print('11',rgb)
    rgb = rgb / rgb.max()

    return rgb

def rgb2rmatrix(rgb_001, rgb_010 , rgb_100, verbose = 1 ):
    Vc = np.ones([3,3])

    rgb_001 = np.array(rgb_001)
    rgb_001 = 255 * rgb_001 / rgb_001.max()
    r_001 = rgb_001[0]
    g_001 = rgb_001[1]
    b_001 = rgb_001[2]
    angleB_001 = 30 * b_001 / (g_001 + b_001)
    angleR_001 = 255 * 90 / (g_001 + b_001 + 255)
    Vc_phi_001 = np.tan(angleB_001 * math.pi / 180)
    Vc_psi_001 = np.cos((90 - angleR_001) * math.pi / 180)
    Vc0_001 = random.random()

    Vc[:, 2] = np.array([Vc0_001, Vc0_001 * Vc_phi_001, Vc_psi_001])


    rgb_010 = np.array(rgb_010)
    rgb_010 = 255 * rgb_010 / rgb_010.max()
    r_010 = rgb_010[0]
    g_010 = rgb_010[1]
    b_010 = rgb_010[2]
    angleB_010 = 30 * b_010 / (255 + b_010)
    angleR_010 = (90 * 30 * r_010 - 90 * r_010 * angleB_010) / (30*255 + 30* r_010 - r_010 * angleB_010)
    Vc_phi_010 = np.tan(angleB_010 * math.pi / 180)
    Vc_psi_010 = np.cos((90 - angleR_010) * math.pi / 180)
    Vc0_010 = random.random()

    Vc[:, 1] = np.array([Vc0_010, Vc0_010 * Vc_phi_010, Vc_psi_010])

    # else:
    #     print("there is no rgb_010")


    rgb_100 = np.array(rgb_100)
    rgb_100 = 255 * rgb_100 / rgb_100.max()
    r_100 = rgb_100[0]
    g_100 = rgb_100[1]
    b_100 = rgb_100[2]
    angleB_100 = 30 * 255 / (255 + g_100)
    angleR_100 = 90 - 90 * (g_100 + 255) / (r_100 + g_100 + 255)
    Vc_phi_100 = np.tan(angleB_100 * math.pi / 180)
    Vc_psi_100 = np.cos((90 - angleR_100) * math.pi / 180)
    Vc0_100 = random.random()

    Vc[:, 0] = np.array([Vc0_100, Vc0_100 * Vc_phi_100, Vc_psi_100])

    # else:
    #     print("there is no rgb_100")


    # print(angleR_001, angleB_001)
    # print(angleR_010, angleB_010)
    # print(angleR_100, angleB_100)
    Sym_crys = Hexagonal_622_Sym1()
    syms = np.concatenate((Sym_crys, -Sym_crys))
    Vc_syms = np.dot(syms, Vc)

    if verbose == 1:
        random_num = np.random.randint(0,23,size=1)
        # print(random_num)

        eular = OrientationMatrix2Euler(Vc_syms[random_num][0])
    else:
        eular_list = []
        for i in Vc_syms:
            eular1 = OrientationMatrix2Euler(i)
            eular_list.append(eular1)
            eular = np.array(eular_list)

    return eular

def rgb2rmatrix_new(rgb_001, verbose = 1):
    Vc = np.ones([3,3])

    rgb_001 = np.array(rgb_001)
    rgb_001 = 255 * rgb_001 / rgb_001.max()
    r_001 = rgb_001[0]
    g_001 = rgb_001[1]
    b_001 = rgb_001[2]
    angleB_001 = 30 * b_001 / (g_001 + b_001)
    angleR_001 = 255 * 90 / (g_001 + b_001 + 255)
    Vc_phi_001 = np.tan(angleB_001 * math.pi / 180)
    Vc_psi_001 = np.cos((90 - angleR_001) * math.pi / 180)
    Vc0_001 = random.random()

    Vc[:, 0] = np.array([random.random(), random.random(), random.random()])
    Vc[:, 1] = np.array([random.random(), random.random(), random.random()])
    Vc[:, 2] = np.array([Vc0_001, Vc0_001 * Vc_phi_001, Vc_psi_001])

    Sym_crys = Hexagonal_622_Sym1()
    syms = np.concatenate((Sym_crys, -Sym_crys))
    Vc_syms = np.dot(syms, Vc)

    if verbose == 1:
        random_num = np.random.randint(0, 23, size=1)
        # print(random_num)

        eular = OrientationMatrix2Euler(Vc_syms[random_num][0])


    else:
        eular_list = []
        for i in Vc_syms:
            eular1 = OrientationMatrix2Euler(i)
            eular_list.append(eular1)
            eular = np.array(eular_list)

    return eular


def plot_line_between_crystal_dir(c1, c2, ax=None, steps=11):
    """Plot a curve between two crystal directions.

    The curve is actually composed of several straight lines segments to
    draw from direction 1 to direction 2.

    :param c1: vector describing crystal direction 1
    :param c2: vector describing crystal direction 2
    :param ax: a reference to a pyplot ax to draw the line
    :param int steps: number of straight lines composing the curve
        (11 by default)

    """
    z = np.array([0., 0., 1.])
    path = np.zeros((steps, 2), dtype=float)
    for j, i in enumerate(np.linspace(0., 1., steps)):
        ci = i * c1 + (1 - i) * c2
        ci /= np.linalg.norm(ci)

        ci += z
        ci /= ci[2]
        path[j, 0] = ci[0]
        path[j, 1] = ci[1]
    ax.plot(path[:, 0], path[:, 1], color="black", markersize=10, linewidth=2)


def hexagonal(a, c):
    '''
    Create a hexagonal Lattice unit cell with length parameters a and c.
    :param float a: first lattice length parameter.
    :param float b: second lattice length parameter.
    :param float c: third lattice length parameter.
    :param float alpha: first lattice angle parameter.
    :param float beta: second lattice angle parameter.
    :param float gamma: third lattice angle parameter.
    :param bool x_aligned_with_a: flag to control the convention used to define the Cartesian frame.
    '''
    alpha_r = 90*np.pi/180
    beta_r = 90*np.pi/180
    gamma_r = 120*np.pi/180
    b=a
    vector_a = a * np.array([1, 0, 0])
    vector_b = b * np.array([np.cos(gamma_r), np.sin(gamma_r), 0])
    c1 = c * np.cos(beta_r)
    c2 = c * (np.cos(alpha_r) - np.cos(gamma_r) * np.cos(beta_r)) / np.sin(gamma_r)
    vector_c = np.array([c1, c2, np.sqrt(c ** 2 - c1 ** 2 - c2 ** 2)])
    m=np.array([vector_a, vector_b, vector_c],dtype=np.float64).reshape((3,3))

    return m

def reciprocal_lattice(lattice_matrix):
    '''Compute the reciprocal lattice.

    The reciprocal lattice defines a crystal in terms of vectors that
    are normal to a plane and whose lengths are the inverse of the
    interplanar spacing. This method computes the three reciprocal
    lattice vectors defined by:

    .. math::

     * a.a^* = 1
     * b.b^* = 1
     * c.c^* = 1
    '''
    [a, b, c] = lattice_matrix
    V=abs(np.dot(np.cross(lattice_matrix[0], lattice_matrix[1]), lattice_matrix[2]))

    astar = np.cross(b, c) / V
    bstar = np.cross(c, a) / V
    cstar = np.cross(a, b) / V
    return [astar, bstar, cstar]

def scattering_vector(hkl, lattice_matrix):
    '''Calculate the scattering vector of this `HklPlane`.

    The scattering vector (or reciprocal lattice vector) is normal to
    this `HklPlane` and its length is equal to the inverse of the
    interplanar spacing. In the cartesian coordinate system of the
    crystal, it is given by:

    ..math

      G_c = h.a^* + k.b^* + l.c^*

    :returns: a numpy vector expressed in the cartesian coordinate system of the crystal.'''
    [astar, bstar, cstar] = reciprocal_lattice(lattice_matrix)
        # express (h, k, l) in the cartesian crystal CS
    Gc = hkl[0] * astar + hkl[1] * bstar + hkl[2] * cstar
    return Gc

def normal(n):
    '''Returns the unit vector normal to the plane.

    We use of the repiprocal lattice to compute the normal to the plane
    and return a normalised vector.
    '''
    return n / np.linalg.norm(n)

"""Convert four to three index representation of a slip plane"""
def four_to_three_indices(U, V, T, W):
    """Convert four to three index representation of a slip plane (used for hexagonal crystal lattice)."""
    #return (6 * h / 5. - 3 * k / 5., 3 * h / 5. + 6 * k / 5., l)
    return U, V, W


def three_to_four_indices(u, v, w):
    """Convert three to four index representation of a slip plane (used for hexagonal crystal lattice)."""
    return u, v, -(u + v), w


def calc_mis(arr1,arr2):
    """input: two arrays with 3 euler angles
    output: misorientation value"""

    misori_list=[]
    gA=euler_to_g(arr1[0],arr1[1],arr1[2])
    gB=euler_to_g(arr2[0],arr2[1],arr2[2])
    Sym_crys=Hexagonal_622_Sym()
    for (g1,g2) in [(gA,gB),(gB,gA)]:
        for i in range(Sym_crys.shape[2]):
            o1=np.dot(Sym_crys[:,:,i],g1)
            for j in range(Sym_crys.shape[2]):
                o2=np.dot(Sym_crys[:,:,j],g2)
                o12=np.dot(o2,np.linalg.inv(o1))
                value=0.5*(o12.trace()-1)
                if value >1. and value-1 <10 *np.finfo('float32').eps:
                    value=1
                omega=np.arccos(value)
                if omega<np.pi:
                    mis_angle=np.degrees(omega)
                misori_list.append(mis_angle)
    return min(misori_list)


def create_pf_contour( poles, g, ang_step=10):

    ang_step *= np.pi / 180  # change to radians
    n_phi = int(1 + 2 * np.pi / ang_step)
    n_psi = int(1 + 0.5 * np.pi / ang_step)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    psis = np.linspace(0, np.pi / 2, n_psi)
    xv, yv = np.meshgrid(phis, psis)
    values = np.zeros((n_psi, n_phi), dtype=int)

    gt = g.transpose()

    for hkl_plane in poles:
        c = hkl_plane
        c_rot = gt.dot(c)
        # handle poles pointing down
        if c_rot[2] < 0:
            c_rot *= -1  # make unit vector have z>0
        if c_rot[1] >= 0:
            phi = np.arccos(c_rot[0] / np.sqrt(c_rot[0] ** 2 +
                                               c_rot[1] ** 2))
        else:
            phi = 2 * np.pi - np.arccos(c_rot[0] /
                                        np.sqrt(c_rot[0] ** 2 +
                                                c_rot[1] ** 2))
        psi = np.arccos(c_rot[2])  # since c_rot is normed
        i_phi = int((phi + 0.5 * ang_step) / ang_step) % n_phi
        j_psi = int((psi + 0.5 * ang_step) / ang_step) % n_psi
        values[j_psi, i_phi] += 1

    x = (2 * yv / np.pi) * np.cos(xv)
    y = (2 * yv / np.pi) * np.sin(xv)

    values[:, -1] = values[:, 0]
    # self.plot_pf_contour(ax, x, y, values)
    return x, y, values

def draw_background(axs):
    lattice_matrix = hexagonal(a=3.214295, c=5.215406)
    sst_poles = [(0, 0, 1), (2, -1, 0), (1, 0, 0)]  # for hexagonal

    A = scattering_vector(sst_poles[0], lattice_matrix)
    A_n = normal(A)
    B = scattering_vector(sst_poles[1], lattice_matrix)
    B_n = normal(B)
    C = scattering_vector(sst_poles[2], lattice_matrix)
    C_n = normal(C)

    plot_line_between_crystal_dir(A, B, axs)
    plot_line_between_crystal_dir(B, C, axs)
    plot_line_between_crystal_dir(C, A, axs)

    poles = [A_n, B_n, C_n]  # 顶点位置

    v_align = ['top', 'top', 'bottom']

    z = np.array([0., 0., 1.])

    for i in range(3):
        hkl = sst_poles[i]
        c_dir = poles[i] + z
        c_dir /= c_dir[2]  # SP'/SP = r/z with r=1
        pole_str = '%d%d%d%d' % three_to_four_indices(*hkl)
        axs.annotate(pole_str, (c_dir[0], c_dir[1] - (2 * (i < 2) - 1) * 0.01), xycoords='data',
                     fontsize=28, horizontalalignment='center', verticalalignment=v_align[i])

    axs.axis('off')
    plt.xlim(0, 1)
    plt.ylim(-0.1, 0.6)



def angle(v1, v2):
    dy = v1[1] - v2[1]
    dx = v1[0] - v2[0]
    dx = abs(dx)
    dy = abs(dy)
    angle = math.atan2(dy, dx)
    angle = angle * 180. / math.pi
    return angle


def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return x, y


class calc_postion:

    def __init__(self):
        C_x = 1.0 * math.cos(30. / 180 * math.pi)
        C_y = 1.0 * math.sin(30. / 180 * math.pi)
        P_x = 0.5 * math.cos(15. / 180 * math.pi)
        P_y = 0.5 * math.sin(15. / 180 * math.pi)

        fai1 = angle([C_x, C_y], [P_x, P_y])
        x1 = C_y / math.tan(fai1 / 180 * math.pi)
        Cc_x = C_x - x1

        Bb_x, Bb_y = cross_point([0., 0., C_x, C_y], [1.0, 0., P_x, P_y])

        Aa_x = 1 * math.cos(15 / 180 * math.pi)
        Aa_y = 1 * math.sin(15 / 180 * math.pi)

        self.C_x = C_x
        self.C_y = C_y
        self.P_x = P_x
        self.P_y = P_y
        self.fai1 = fai1
        self.Cc_x = Cc_x
        self.Bb_x = Bb_x
        self.Bb_y = Bb_y
        self.Aa_x = Aa_x
        self.Aa_y = Aa_y

    def gb2xy(self, g, b):
        g = int(g) * 1.0
        b = int(b) * 1.0

        distanceP2O = math.sqrt((self.P_x - 0) ** 2 + (self.P_y - 0) ** 2)

        if g > b:

            C0_x = distanceP2O / 255 * g * math.cos(15 / 180 * math.pi)
            C0_y = distanceP2O / 255 * g * math.sin(15 / 180 * math.pi)

            C1_x = self.Cc_x
            C1_y = self.Cc_x * math.tan(15 / 180 * math.pi)
            distanceC12O = math.sqrt((C1_x - 0) ** 2 + (C1_y - 0) ** 2)
            G = 255 * distanceC12O / distanceP2O

            distanceCc2C0 = math.sqrt((C0_x - self.Cc_x) ** 2 + (C0_y - 0) ** 2)
            fai2 = angle([C0_x, C0_y], [self.Cc_x, 0])
            distanceCc2XY = b * distanceCc2C0 / g

            if g <= G:
                X = self.Cc_x - distanceCc2XY * math.cos(fai2 / 180 * math.pi)
                Y = distanceCc2XY * math.sin(fai2 / 180 * math.pi)

            else:
                X = self.Cc_x + distanceCc2XY * math.cos(fai2 / 180 * math.pi)
                Y = distanceCc2XY * math.sin(fai2 / 180 * math.pi)

            # print(X,Y)

        elif b > g:

            B0_x = distanceP2O / 255 * b * math.cos(15 / 180 * math.pi)
            B0_y = distanceP2O / 255 * b * math.sin(15 / 180 * math.pi)
            distanceB02O = math.sqrt((B0_x - 0) ** 2 + (B0_y - 0) ** 2)

            distanceB02Bb = math.sqrt((B0_x - self.Bb_x) ** 2 + (B0_y - self.Bb_y) ** 2)
            B1_x = self.Bb_x
            B1_y = self.Bb_x * math.tan(15 / 180 * math.pi)
            distanceB12O = math.sqrt((B1_x - 0) ** 2 + (B1_y - 0) ** 2)
            B = 255 * distanceB12O / distanceP2O

            fai3 = angle([self.Bb_x, self.Bb_y], [B0_x, B0_y])
            distanceBb2XY = g * distanceB02Bb / b

            if b <= B:
                X = self.Bb_x - distanceBb2XY * math.cos(fai3 / 180 * math.pi)
                Y = self.Bb_y - distanceBb2XY * math.sin(fai3 / 180 * math.pi)

            else:
                X = self.Bb_x + distanceBb2XY * math.cos(fai3 / 180 * math.pi)
                Y = self.Bb_y - distanceBb2XY * math.sin(fai3 / 180 * math.pi)

        else:
            distanceP2O = math.sqrt((self.P_x - 0) ** 2 + (self.P_y - 0) ** 2)
            X = distanceP2O / 255 * b * math.cos(15 / 180 * math.pi)
            Y = distanceP2O / 255 * b * math.sin(15 / 180 * math.pi)

        return X, Y

    def rb2xy(self, r, b):
        r = int(r)
        b = int(b)

        theta0 = 15
        theta = theta0 / 255 * b

        A0_x = 1 * math.cos(theta / 180 * math.pi)
        A0_y = 1 * math.sin(theta / 180 * math.pi)

        Q_x, Q_y = cross_point([0, 0, A0_x, A0_y], [self.Cc_x, 0, self.P_x, self.P_y])

        distanceQ2A0 = math.sqrt((Q_x - A0_x) ** 2 + (Q_y - A0_y) ** 2)

        fai3 = angle([0, 0], [A0_x, A0_y])

        X = A0_x - distanceQ2A0 / 255 * r * math.cos(fai3 / 180 * math.pi)
        Y = A0_y - distanceQ2A0 / 255 * r * math.sin(fai3 / 180 * math.pi)

        return X, Y

    def rg2xy(self, r, g):
        r = int(r)
        g = int(g)

        theta0 = 15
        theta1 = 30 - (theta0 / 255) * g

        A1_x = 1 * math.cos(theta1 / 180 * math.pi)
        A1_y = 1 * math.sin(theta1 / 180 * math.pi)

        Q1_x, Q1_y = cross_point([0, 0, A1_x, A1_y], [1, 0, self.P_x, self.P_y])

        distanceQ12A1 = math.sqrt((A1_x - Q1_x) ** 2 + (A1_y - Q1_y) ** 2)
        fai4 = angle([A1_x, A1_y], [Q1_x, Q1_y])

        X = A1_x - distanceQ12A1 / 255 * r * math.cos(fai4 / 180 * math.pi)
        Y = A1_y - distanceQ12A1 / 255 * r * math.sin(fai4 / 180 * math.pi)

        return X, Y

    def rgb2xy(self, r, g, b):
        r = int(r)
        g = int(g)
        b = int(b)

        if  r == 255 :
            # print("11")
            X, Y = self.gb2xy(g, b)

        elif g == 255 :
            # print("22")
            X, Y = self.rb2xy(r, b)

        elif b == 255 :
            # print("33")
            X, Y = self.rg2xy(r, g)

        else:
            X = 0
            Y = 0

        return X, Y

    def array2xy(self, array):
        pos_x = []
        pos_y = []
        pos_xy = []
        # print(len(r))
        for i in array:
            x, y = self.rgb2xy(r=i[2], g=i[1], b=i[0])
            x, y = float(x), float(y)
            pos_x.append(x)
            pos_y.append(y)
            xy = (x, y)
            pos_xy.append(xy)

        pos_xy = np.array(pos_xy)
        return pos_x, pos_y, pos_xy

def read_img(path, flag = 3, verbose = 1):

    image = cv2.imread(path, flags=flag)

    if verbose == 1:
        return image

    if verbose == 0:
        b, g, r = cv2.split(image)

        height = image.shape[0]
        width = image.shape[1]

        b = np.reshape(b, (height * width, 1))
        g = np.reshape(g, (height * width, 1))
        r = np.reshape(r, (height * width, 1))

        return b, g, r

def image2parts(path, flag = 3):
    image = cv2.imread(path, flags=3)
    height = image.shape[1]
    weight = image.shape[0]

    image = np.reshape(image, (height * weight, 3))

    r_part = []
    g_part = []
    b_part = []
    unindexed_part = []

    for i in image:
        r = int(i[2])
        g = int(i[1])
        b = int(i[0])
        if r == 255:
            r_part.append(i)
        elif g == 255:
            g_part.append(i)
        elif b == 255:
            b_part.append(i)
        else:
            unindexed_part.append(i)

    return r_part, g_part, b_part

def xy2cluster(pos_xy, k = 1):

    kmeans = KMeans(n_clusters=1, init='k-means++')
    y_pred = kmeans.fit_predict(pos_xy)

    return kmeans.cluster_centers_

def Rodrigues2OrientationMatrix(rod):
    """
    Compute the orientation matrix from the Rodrigues vector.

    :param rod: The Rodrigues vector as a 3 components array.
    :returns: The 3x3 orientation matrix representing the rotation.
    """
    r = np.linalg.norm(rod)
    I = np.diagflat(np.ones(3))
    if r < np.finfo(r.dtype).eps:
        # the rodrigues vector is zero, return the identity matrix
        return I
    theta = 2 * np.arctan(r)
    n = rod / r
    omega = np.array([[0.0, n[2], -n[1]],
                      [-n[2], 0.0, n[0]],
                      [n[1], -n[0], 0.0]])
    g = I + np.sin(theta) * omega + (1 - np.cos(theta)) * omega.dot(omega)
    return g



# if __name__ == '__main__':
#     lattice_matrix = hexagonal(a=3.214295, c=5.215406)
#     sst_poles = [(0, 0, 1), (2, -1, 0), (1, 0, 0)]  # for hexagonal
#
#     A = scattering_vector(sst_poles[0], lattice_matrix)
#     A_n = normal(A)
#     B = scattering_vector(sst_poles[1], lattice_matrix)
#     B_n = normal(B)
#     C = scattering_vector(sst_poles[2], lattice_matrix)
#     C_n = normal(C)
#
#     fig, axs = plt.subplots(figsize=(10, 7))
#
#     plot_line_between_crystal_dir(A, B, axs)
#     plot_line_between_crystal_dir(B, C, axs)
#     plot_line_between_crystal_dir(C, A, axs)
#
#     poles = [A_n, B_n, C_n]  # 顶点位置
#
#     v_align = ['top', 'top', 'bottom']
#
#     z = np.array([0., 0., 1.])
#
#     for i in range(3):
#         hkl = sst_poles[i]
#         c_dir = poles[i] + z
#         c_dir /= c_dir[2]  # SP'/SP = r/z with r=1
#         pole_str = '%d%d%d%d' % three_to_four_indices(*hkl)
#         axs.annotate(pole_str, (c_dir[0], c_dir[1] - (2 * (i < 2) - 1) * 0.01), xycoords='data',
#                      fontsize=28, horizontalalignment='center', verticalalignment=v_align[i])
#
#     img = cv2.imread('001.bmp')
#     img = cv2.resize(img, (300, 240))
#
#     height = img.shape[0]
#     width = img.shape[1]
#
#     img_arr = np.reshape(img, (height * width, 3))
#     # print(img_arr)
#     xy_list = []
#     for i in range(height * width):
#         xy = calc_IPF_position(img_arr[i][2], img_arr[i][1],img_arr[i][0])
#         xy_list.append(xy)
#
#     for i in xy_list:
#         axs.scatter(i[0], i[1], c = "BLACK", s = 3)
#
#     axs.axis('off')
#
#     plt.show()



#     pixel = np.loadtxt('az62_pixels.txt', dtype=np.float32)
#     pixel = pd.read_csv("az62_pixels.txt", delim_whitespace=True, header=16,
#                         names=['phi1','phi','phi2','x', 'y','IQ','CI','Fit','Grain ID', 'Edge', 'Phase'],
#                         usecols=['phi1','phi','phi2'])
#     # print(pixel.iloc[-1:])
#     pixel_arr = np.array(pixel)
#
#     pixel_arr_new = np.zeros_like(pixel_arr)
#     for i in range(pixel_arr.shape[0]):
#         phi1 = pixel_arr[i][0]
#         phi = pixel_arr[i][1]
#         phi2 = pixel_arr[i][2]
#         rgb = eular2rgb(np.degrees(phi1),np.degrees(phi),np.degrees(phi2), axis=[0, 0, 1])
#         pixel_arr_new[i] = rgb
#
#     pixel_arr_new = np.reshape(pixel_arr_new, (517, 689, 3))
#     # pixel_arr_new = cv2.cvtColor(pixel_arr_new, cv2.COLOR_BGR2RGB)
#
#     plt.imshow(pixel_arr_new)
#     plt.show()





if __name__ == '__main__':
    a = np.array([309.4, 0.0, 0.0])
    b = np.array([272.1, 180.0, 0.0])
    print(calc_mis(a, b))
    print(eular2rgb((309.4, 0.0, 0.0)))
    print(eular2rgb((272.1, 180.0, 0.0)))

    # eular_list = []
    # for i in r:
    #     eular = np.array(i, dtype=float)
    #     eular = np.reshape(eular, (3, 3)).T
    #     print(eular)
    #     eular = OrientationMatrix2Euler(eular)
    #     eular_list.append(eular)
    #
    # print(eular_list)
    # print(len(eular_list))

#     start = time.time()
#     fig, axs = plt.subplots(figsize=(10, 7))
#     draw_background(axs)
#
#     a = calc_postion()
#     # b = a.rgb2xy(r = 255, g = 200, b = 200)
#     # print(b)
#     path = '001.bmp'
#     # image = read_img(path, verbose=1)
#     b, g, r = image2parts(path)
#     # print(image.shape)
#     # print(r.shape)
#
#     pos_x1, pos_y1, pos_xy1 = a.array2xy(r)
#     axs.scatter(pos_x1, pos_y1, c="b", s=2)
#     pos_x2, pos_y2, pos_xy2 = a.array2xy(g)
#     axs.scatter(pos_x2, pos_y2, c="g", s=2)
#     pos_x3, pos_y3, pos_xy3 = a.array2xy(b)
#     axs.scatter(pos_x3, pos_y3, c="r", s=2)
#
#     center1 = xy2cluster(pos_xy1)
#     center2 = xy2cluster(pos_xy2)
#     center3 = xy2cluster(pos_xy3)
#
#     # np.save("pos_xy1.npy", pos_xy1)
#     # np.save("pos_xy2.npy", pos_xy2)
#     # np.save("pos_xy3.npy", pos_xy3)
#     # np.save("center1.npy", center1)
#     # np.save("center2.npy", center2)
#     # np.save("center3.npy", center3)
#
#     axs.scatter(center1[0][0],center1[0][1], c="black", s=100, marker='*')
#     axs.scatter(center2[0][0], center2[0][1], c="black", s=100, marker='*')
#     axs.scatter(center3[0][0], center3[0][1], c="black", s=100, marker='*')
#
#
#     end = time.time()
#     print("Time : ",end - start)
#     plt.show()