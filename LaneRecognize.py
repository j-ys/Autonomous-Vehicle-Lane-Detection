#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image


# 캐니 에지 디텍션
def Canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 250, 300)
    return canny

def InterestRegion(frame, width, height):
    frame = np.array(frame, dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lower_white = np.array([150,150,150])
    upper_white = np.array([255,255,255])

    mask_white = cv2.inRange(rgb, lower_white, upper_white)
    res = cv2.bitwise_and(frame, frame, mask = mask_white)

    area = np.array([[(width*0.5,(height*0.4)),(0,(height*0.65)),(0,height), (width,height),(width,(height*0.6))]], np.int32) # Area 지정
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, area, (255,255,255))
    interestarea = cv2.bitwise_and(res, mask)
    cv2.imshow('mask',interestarea)
    cv2.waitKey(1)
    return interestarea
#=======================================================================================

def Hough_lines(interestregion, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(interestregion,rho,theta,threshold,np.array([]),minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def weighted_img(img, initial_img, a=1, b=1., l=0.):
    return cv2.addWeighted(initial_img, a, img, b, l)

def draw_lines(frame, lines, color=[255, 255, 255], thickness=5):
    cv2.line(frame, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def get_representative(frame, lines):
    lines = np.squeeze(lines)
    lines = lines.reshape(lines.shape[0] * 2, 2)
    rows, cols = frame.shape[:2]
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((frame.shape[0] - 1) - y) / vy * vx + x), frame.shape[0] - 1
    x2, y2 = int(((frame.shape[0] / 2 + 100) - y) / vy * vx + x), int(frame.shape[0] / 2 + 100)

    result = [x1, y1, x2, y2]
    return result

def bird_eyes_view(img, width, height):
    IMAGE_H = 500 #ROI할 곳 자른거
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[269, IMAGE_H], [360, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    img = img[320:(320+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    warped_img2 = warped_img[0:315, 0:640]
    cv2.imshow('wraped_img',warped_img2)
    cv2.waitKey(1)
    return warped_img2