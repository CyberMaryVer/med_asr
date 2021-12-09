# -*- coding: utf-8 -*-
# Detectron2 & Mediapipe implementation

import numpy as np


def get_angle(p1, p2, p3):
    """
    returns angle for tensor points
    """
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.round(cosine_angle, 4)))
    return angle


def get_text_coords(keypoint_coords, scale):
    # (x - 18, y + 8) for 3
    # (x - 10, y + 5) for 1
    x, y = keypoint_coords
    x, y = x - int(8 * scale), y + int(4 * scale)
    return x, y


def get_intersect(x1, y1, x2, y2):
    """
    Returns the point of intersection of the lines or None if lines are parallel
    Ex. p1=(x1,x2)... line_intersection((p1,p2), (p3,p4))
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([x1, y1, x2, y2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return None, None
    return x / z, y / z


def get_line_coords(img_shape, p1, p2):
    """
    returns the coordinates of a line passing through two specified points
    """
    x1, y1 = [int(p) for p in p1]
    x2, y2 = [int(p) for p in p2]
    div = 1.0 * (x2 - x1) if x2 != x1 else .00001
    a = (1.0 * (y2 - y1)) / div
    b = -a * x1 + y1
    y1_, y2_ = 0, img_shape[1]
    x1_ = int((y1_ - b) / a)
    x2_ = int((y2_ - b) / a)
    return (x1_, y1_), (x2_, y2_)


def triangle_centroid(p1, p2, p3):
    x = int((p1[0] + p2[0] + p3[0]) / 3)
    y = int((p1[1] + p2[1] + p3[1]) / 3)

    return x, y


def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


if __name__ == "__main__":
    pass
