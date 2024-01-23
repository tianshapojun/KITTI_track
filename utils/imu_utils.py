import numpy as np
import os
from .camera_utils import Calibration
import mayavi.mlab as mlab

er = 6378137. # average earth radius at the equator

def latlonToMercator(lat,lon,scale):
    ''' converts lat/lon coordinates to mercator coordinates using mercator scale '''

    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log( np.tan((90+lat) * np.pi / 360) )
    return mx,my

def latToScale(lat):
    ''' compute mercator scale from latitude '''
    scale = np.cos(lat * np.pi / 180.0)
    return scale

def convertOxtsToPose(oxts):
    ''' converts a list of oxts measurements into metric poses,
    starting at (0,0,0) meters, OXTS coordinates are defined as
    x = forward, y = right, z = down (see OXTS RT3000 user manual)
    afterwards, pose{i} contains the transformation which takes a
    3D point in the i'th frame and projects it into the oxts
    coordinates with the origin at a lake in Karlsruhe. '''
    
    # origin in OXTS coordinate
    origin_oxts = [48.9843445, 8.4295857] # lake in Karlsruhe
    #origin_oxts = oxts[0,:2]
    
    # compute scale from lat value of the origin
    scale = latToScale(origin_oxts[0])
    
    # origin in Mercator coordinate
    ox,oy = latlonToMercator(origin_oxts[0],origin_oxts[1],scale)
    origin = np.array([ox, oy, 0])
    
    pose = []
    
    # for all oxts packets do
    for i in range(len(oxts)):
        
        # if there is no data => no pose
        if not len(oxts[i]):
            pose.append([])
            continue
    
        # translation vector
        tx, ty = latlonToMercator(oxts[i,0],oxts[i,1],scale)
        t = np.array([tx, ty, oxts[i,2]])
    
        # rotation matrix (OXTS RT3000 user manual, page 71/92)
        rx = oxts[i,3] # roll
        ry = oxts[i,4] # pitch
        rz = oxts[i,5] # heading 
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]]) # base => nav  (level oxts => rotated oxts)
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]]) # base => nav  (level oxts => rotated oxts)
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]]) # base => nav  (level oxts => rotated oxts)
        R  = np.matmul(np.matmul(Rz, Ry), Rx)
        
        # normalize translation
        t = t-origin
            
        # add pose
        pose.append(np.vstack((np.hstack((R,t.reshape(3,1))),np.array([0,0,0,1]))))
    
    return pose

def postprocessPoses (poses_in):

    R = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
    
    poses  = []
    
    for i in range(len(poses_in)):
        P = poses_in[i]
        poses.append( np.matmul(R, P.T).T )
    
    return poses

def read_pointcloud(root_dir, file_index, calib, pose):
    pointcloud = np.fromfile(os.path.join(root_dir,'Data','test_00','velodyne','{}.bin'.format(file_index)), dtype=np.float32, count=-1).reshape([-1, 4])
    pointcloud = pointcloud[:,:3]
    n = pointcloud.shape[0]
    pcd_hom = np.hstack((pointcloud, np.ones((n, 1))))
    pcd_imu = np.dot(pcd_hom, np.transpose(calib.V2I))
    pcd_hom = np.hstack((pcd_imu, np.ones((n, 1))))
    pcd_fin = np.dot(pcd_hom, np.transpose(pose))
    pcd_fin = np.concatenate((pcd_fin, int(file_index[-3:])*np.ones((n, 1))),axis=1)
    return pcd_fin

def load_pointsclouds(root_dir, idxs):
    file_index_1 = "%06d" % (idxs[0])
    
    calib_filename = os.path.join(root_dir,'Data','test_00','calib', '{}.txt'.format('000000'))
    calib = Calibration(calib_filename)
    
    oxts_dir = os.path.join(root_dir,'Data','test_00','oxts','{}.txt'.format('000000'))
    oxts = np.loadtxt(oxts_dir)
    poses = convertOxtsToPose(oxts)
    # convert coordinate system from
    #   x=forward, y=right, z=down 
    # to
    #   x=forward, y=left, z=up
    # poses = postprocessPoses(poses)
    
    pcd_concate= read_pointcloud(root_dir, file_index_1, calib, poses[idxs[0]][:3,:4])
    for i in range(1,len(idxs)): 
        file_index_2 = "%06d" % (idxs[i])
        pcd_2 = read_pointcloud(root_dir, file_index_2, calib, poses[idxs[i]][:3,:4])
        pcd_concate = np.concatenate((pcd_concate, pcd_2), axis=0)
    
    x = pcd_concate[:, 0]  # x position of point
    y = pcd_concate[:, 1]  # y position of point
    z = pcd_concate[:, 2]  # z position of point
    print(pcd_concate.shape)
    
    print(x.max(),x.min(),y.max(),y.min(),z.max(),z.min())

    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    
    degr = np.degrees(np.arctan(z / d))
    
    vals = 'height'
    if vals == "height":
        col = pcd_concate[:, 3] 
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
    mlab.show()
    
