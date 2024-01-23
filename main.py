import numpy as np
import mayavi.mlab as mlab
import torch
import json
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from utils.camera_utils import Calibration
from utils.gt_utils import compute_box_3d,read_label,draw_gt_boxes3d
from utils.imu_utils import load_pointsclouds

def draw_pointscloud(root_dir, file_index,gt = True):
    pointcloud = np.fromfile(os.path.join(root_dir,'Data','test_00','velodyne','{}.bin'.format(file_index)), dtype=np.float32, count=-1).reshape([-1, 4])
    
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    print(pointcloud.shape)
    #print(pointcloud[pointcloud[:,0]<15].shape)
    
    print(x.max(),x.min(),y.max(),y.min(),z.max(),z.min())
    
    r = pointcloud[:, 3]  # reflectance value of point
    print(r.max(),r.min())
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    
    degr = np.degrees(np.arctan(z / d))
    
    vals = 'height'
    if vals == "height":
        col = z#(z-z.min())/(z.max()-z.min())
    else:
        col = d
    
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    nodes = mlab.points3d(x, y, z,
                        col,  # Values used for Color
                        mode="point",
                        colormap='spectral',  # 'bone', 'copper', 'gnuplot','spectral'
                        # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                        #resolution = 20,
                        scale_mode='none',
                        figure=fig,
                        )
    if gt: 
        label_filename = os.path.join(root_dir,'Data','label_2', '{}.txt'.format(file_index))
        object_gt = read_label(label_filename)
        calib_filename = os.path.join(root_dir,'Data','calib', '{}.txt'.format(file_index))
        calib = Calibration(calib_filename)
        for obj in object_gt:
            if obj.type in ["DontCare"]:
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            #print("box3d_pts_3d_velo:")
            #print(box3d_pts_3d_velo)

            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0,0,1))

    mlab.show()
    #mlab.savefig('abc.png', figure=mlab.gcf(), magnification=2)

def show_image_with_boxes(root_dir, file_index, depth=None):
    """ Show image with 2D bounding boxes """
    img_filename = os.path.join(root_dir,'Data','image_2', '{}.png'.format(file_index))
    img = cv2.imread(img_filename)  
    label_filename = os.path.join(root_dir,'Data','label_2', '{}.txt'.format(file_index))
    objects = read_label(label_filename)
    calib_filename = os.path.join(root_dir,'Data','calib', '{}.txt'.format(file_index))
    calib = Calibration(calib_filename)
    img2 = np.copy(img)  # for 3d bbox
    #TODO: change the color of boxes
    for obj in objects:
        if obj.type == "DontCare":
            continue
        box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        if obj.type in ["Car","Van"]:
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0, 0, 255))
        elif obj.type == "Pedestrian":
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0, 0, 255))
        elif obj.type == "Cyclist":
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0, 0, 255))

        
    show3d = True
    if show3d:
        # print("img2:",img2.shape)
        cv2.imshow("3dbox", img2)
        
    if depth is not None:
        cv2.imshow("depth", depth)
        
    key = cv2.waitKey(0)
    return img2

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def get_lidar_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=1.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax-1)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax-1)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo

def show_lidar_on_image(root_dir, file_index):
    """ Project LiDAR points to image """
    img_filename = os.path.join(root_dir,'Data','test_00','image_2', '{}.png'.format(file_index))
    img = cv2.imread(img_filename)
    pointcloud = np.fromfile(os.path.join(root_dir,'Data','test_00','velodyne','{}.bin'.format(file_index)), dtype=np.float32, count=-1).reshape([-1, 4])
    pc_velo = pointcloud[:,:3]
    calib_filename = os.path.join(root_dir,'Data','test_00','calib', '{}.txt'.format(file_index))
    calib = Calibration(calib_filename)
    img_height, img_width, _ = img.shape
    
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    #cmap = plt.cm.get_cmap("hsv", 256)
    cmap = matplotlib.colormaps.get_cmap('viridis')
    #print(cmap(111))
    #dddd 
    cmap = np.array([cmap(i*6) for i in range(256)])[:, :3] * 255
    print(imgfov_pts_2d[:, 1].min(),imgfov_pts_2d[:, 1].max())

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            1,
            color=tuple(color),
            thickness=-1,
        )
    cv2.imshow("projection", img)
    key = cv2.waitKey(0)
    return img


def lidar_color_ini(root_dir, file_index):
    """ Project LiDAR points to image """
    img_filename = os.path.join(root_dir,'Data','image_2', '{}.png'.format(file_index))
    img = cv2.imread(img_filename)
    pointcloud = np.fromfile(os.path.join(root_dir,'Data','velodyne','{}.bin'.format(file_index)), dtype=np.float32, count=-1).reshape([-1, 4])
    pc_velo = pointcloud[:,:3]
    calib_filename = os.path.join(root_dir,'Data','calib', '{}.txt'.format(file_index))
    calib = Calibration(calib_filename)
    img_height, img_width, _ = img.shape
    
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(np.int32)
    
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth = imgfov_pc_rect[:, 2]
    
    #print(imgfov_pts_2d[:,1].min(),imgfov_pts_2d[:,1].max())
    color_pcd = img[imgfov_pts_2d[:,1],imgfov_pts_2d[:,1]]
    
    print(imgfov_pts_2d.shape)
    print(img.shape)
    print(color_pcd.shape)

#'''
if __name__=="__main__":
    idx = 0
    root_dir = os.path.dirname(os.path.abspath(__file__))

    #draw_pointscloud(root_dir,"%06d" % (idx), gt = False)
       
    #lidar_color_ini(root_dir,"%06d" % (idx))
    load_pointsclouds(root_dir,[idx+i for i in range(0,450,50)])
#'''

