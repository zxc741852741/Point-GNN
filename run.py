"""This file implement an inference pipeline for Point-GNN on KITTI dataset"""

import math
import os
import time
import argparse
import multiprocessing
from functools import partial

import ros_numpy
import rospy
from sensor_msgs.msg import Image , PointCloud2 , PointField
from visualization_msgs.msg import Marker ,MarkerArray 
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2 
from std_msgs.msg import Header

import pcl
import numpy as np
import tensorflow as tf
import open3d
from tqdm import tqdm

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

print(sys.path)
import cv2

from dataset.kitti_dataset import KittiDataset, Points
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, \
                          get_encoding_len
from models import preprocess
from models import nms
from util.config_util import load_config, load_train_config
from util.summary_util import write_summary_scale
from util.ros_utils import *







parser = argparse.ArgumentParser(description='Point-GNN inference on KITTI')
parser.add_argument('checkpoint_path', type=str,
                   help='Path to checkpoint')
parser.add_argument('-l', '--level', type=int, default=0,
                   help='Visualization level, 0 to disable,'+
                   '1 to nonblocking visualization, 2 to block.'+
                   'Default=0')
parser.add_argument('--test', dest='test', action='store_true',
                    default=False, help='Enable test model')
parser.add_argument('--no-box-merge', dest='use_box_merge',
                    action='store_false', default='True',
                   help='Disable box merge.')
parser.add_argument('--no-box-score', dest='use_box_score',
                    action='store_false', default='True',
                   help='Disable box score.')
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                   help='Path to KITTI dataset. Default="../dataset/kitti/"')
parser.add_argument('--dataset_split_file', type=str,
                    default='',
                   help='Path to KITTI dataset split file.'
                   'Default="DATASET_ROOT_DIR/3DOP_splits/val.txt"')
parser.add_argument('--output_dir', type=str,
                    default='',
                   help='Path to save the detection results'
                   'Default="CHECKPOINT_PATH/eval/"')
args = parser.parse_args()
VISUALIZATION_LEVEL = args.level
IS_TEST = args.test
USE_BOX_MERGE = args.use_box_merge
USE_BOX_SCORE = args.use_box_score
DATASET_DIR = args.dataset_root_dir
if args.dataset_split_file == '':
    DATASET_SPLIT_FILE = os.path.join(DATASET_DIR, './3DOP_splits/val.txt')
else:
    DATASET_SPLIT_FILE = args.dataset_split_file
if args.output_dir == '':
    OUTPUT_DIR = os.path.join(args.checkpoint_path, './eval/')
else:
    OUTPUT_DIR = args.output_dir
CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH = os.path.join(CHECKPOINT_PATH, 'config')
assert os.path.isfile(CONFIG_PATH), 'No config file found in %s'
config = load_config(CONFIG_PATH)


##set_ros_data#################
import rospy
from sensor_msgs.msg import PointCloud2,PointField ,PointCloud
import sys
sys.path.insert(0,'/usr/local/lib/python3.7/dist-packages')  
#print(sys.path)
import pcl
import sensor_msgs.point_cloud2 as pcl2 
sys.path.insert(0,'/home/user/anaconda3/lib/python3.7/site-packages/')  
import ros_numpy
import math


def lidar_callback(data):
    pc = ros_numpy.numpify(data)
    #print(pc)
    points=np.zeros((pc.shape[0],5))
    rotation_points = np.zeros((pc.shape[0],4))
    #print('x:{}'.format(pc['x']))
    '''print('y:{}'.format(pc['y']))
    print('-x:{}'.format(-pc['x']))
    print('-y:{}'.format(-pc['y']))'''
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    #intensity = pc['I']
    intensity = pc['intensity'] #/255
    points[:,3]=intensity
    points[:,4] = np.zeros(pc.shape[0])
    #print('points = {}'.format(points))
    #points.astype(np.int32)
    return points
def points_rotation(points,yaw):
    yaw = yaw*math.pi/180
    R = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    out_points = points.copy()
    x_y_points = points.copy()
    x_y_points = x_y_points[:,:2]
    rotation_points = np.dot(x_y_points, R.T)
    out_points[:,:2] = rotation_points
    return out_points


#msg = rospy.wait_for_message("/points_raw", PointCloud2, timeout=None) #lab_data

'''pt = lidar_callback(msg)
points = pt.copy()
ros_lidar = points
print(points)'''

### set_ros_data ########################


# setup dataset ===============================================================
if IS_TEST:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/testing/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/testing/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/testing/calib/'),
        '',
        num_classes=config['num_classes'],
        is_training=False)
else:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        DATASET_SPLIT_FILE,
        num_classes=config['num_classes'])
NUM_TEST_SAMPLE = dataset.num_files
NUM_CLASSES = dataset.num_classes
# occlusion score =============================================================
def occlusion(label, xyz):
    if xyz.shape[0] == 0:
        return 0
    normals, lower, upper = dataset.box3d_to_normals(label)
    projected = np.matmul(xyz, np.transpose(normals))
    x_cover_rate = (np.max(projected[:, 0])-np.min(projected[:, 0]))\
        /(upper[0] - lower[0])
    y_cover_rate = (np.max(projected[:, 1])-np.min(projected[:, 1]))\
        /(upper[1] - lower[1])
    z_cover_rate = (np.max(projected[:, 2])-np.min(projected[:, 2]))\
        /(upper[2] - lower[2])
    return x_cover_rate*y_cover_rate*z_cover_rate
# setup model =================================================================


BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])
if config['input_features'] == 'irgb':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 3])
elif config['input_features'] == '0000':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'i000':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'i':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 1])
elif config['input_features'] == '0':
    t_initial_vertex_features = tf.placeholder(
        dtype=tf.float32, shape=[None, 1])
t_vertex_coord_list = [tf.placeholder(dtype=tf.float32, shape=[None, 3])]
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_vertex_coord_list.append(
        tf.placeholder(dtype=tf.float32, shape=[None, 3]))
t_edges_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_edges_list.append(
        tf.placeholder(dtype=tf.int32, shape=[None, 2]))
t_keypoint_indices_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_keypoint_indices_list.append(
        tf.placeholder(dtype=tf.int32, shape=[None, 1]))
t_is_training = tf.placeholder(dtype=tf.bool, shape=[])
model = get_model(config['model_name'])(num_classes=NUM_CLASSES,
    box_encoding_len=BOX_ENCODING_LEN, mode='test', **config['model_kwargs'])
t_logits, t_pred_box,tfeatures_list,t_edges,t_vertex_coordinates,dis_2,dis_1 = model.predict(
    t_initial_vertex_features, t_vertex_coord_list, t_keypoint_indices_list,
    t_edges_list,
    t_is_training)
t_probs = model.postprocess(t_logits)
t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)
# optimizers ==================================================================
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
'''coco_1 = t_vertex_coordinates[t_edges[0][0]]
coco_2 = t_vertex_coordinates[t_edges[1][0]]
de = t_vertex_coordinates[t_edges[0][1]]

dis_1 = (coco_1 - de)**2
dis_1 = np.sqrt(dis_1)
dis_2 = (coco_2 - de)**2
dis_2 = np.sqrt(dis_2)'''

fetches = {
    'step': global_step,
    'predictions': t_predictions,
    'probs': t_probs,
    'pred_box': t_pred_box,
    'tfeatures_list': tfeatures_list,
    't_edges': t_edges,
    't_vertex_coordinates': t_vertex_coordinates,

    #'coco_1': t_vertex_coordinates[t_edges[0][0]],
    #'coco_2': t_vertex_coordinates[t_edges[1][0]],

    #'de': t_vertex_coordinates[t_edges[0][1]],
    'dis_1':dis_1,
    'dis_2':dis_2
    #'dis_1': np.sqrt(( t_vertex_coordinates[t_edges[0][0]]-t_vertex_coordinates[t_edges[0][1]])**2),
    #'dis_2': np.sqrt(( t_vertex_coordinates[t_edges[1][0]]-t_vertex_coordinates[t_edges[0][1]])**2)
    }


# runing network ==============================================================
time_dict = {}
saver = tf.train.Saver()
graph = tf.get_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)

# setup Visualizer ============================================================
if VISUALIZATION_LEVEL == 1:
    print("Configure the viewpoint as you want and press [q]")
    calib = dataset.get_calib(0)
    cam_points_in_img_with_rgb = dataset.get_cam_points_in_image_with_rgb(0,
        calib=calib)
    vis = open3d.Visualizer()
    vis.create_window()
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(cam_points_in_img_with_rgb.xyz)
    pcd.colors = open3d.Vector3dVector(cam_points_in_img_with_rgb.attr[:,1:4])
    line_set = open3d.LineSet()
    graph_line_set = open3d.LineSet()
    box_corners = np.array([[0, 0, 0]])
    box_edges = np.array([[0,0]])
    line_set.points = open3d.Vector3dVector(box_corners)
    line_set.lines = open3d.Vector2iVector(box_edges)
    graph_line_set.points = open3d.Vector3dVector(box_corners)
    graph_line_set.lines = open3d.Vector2iVector(box_edges)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(graph_line_set)
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 3141.0, 0)
    vis.run()
color_map = np.array([(211,211,211), (255, 0, 0), (255,20,147), (65, 244, 101),
    (169, 244, 65), (65, 79, 244), (65, 181, 244), (229, 244, 66)],
    dtype=np.float32)
color_map = color_map/255.0
gt_color_map = {
    'Pedestrian': (0,255,255),
    'Person_sitting': (218,112,214),
    'Car': (154,205,50),
    'Truck':(255,215,0),
    'Van': (255,20,147),
    'Tram': (250,128,114),
    'Misc': (128,0,128),
    'Cyclist': (255,165,0),
}
#global theta
def publish_edges(edge_pub, corners_3d_velos,types):
    marker_array = MarkerArray()
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()
        a = types[i]
        #a = 1
        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_LIST
         
        r,g,b = DETECTION_COLOR_DICT[str(a)]
        marker.color.r = r/255.0
        marker.color.g = g/255.0
        marker.color.b = b/255.0

        marker.color.a = 1.0
        
        marker.scale.x = 0.15

        marker.points = []
        
        for l in LINES:
            p1x = corners_3d_velo[0][l[0]]
            p1y = corners_3d_velo[1][l[0]]
            p1z = corners_3d_velo[2][l[0]]
            marker.points.append(Point(p1x,p1y,p1z))
            p2x = corners_3d_velo[0][l[1]]
            p2y = corners_3d_velo[1][l[1]]
            p2z = corners_3d_velo[2][l[1]]
            marker.points.append(Point(p2x,p2y,p2z))
        #print(marker)
        marker_array.markers.append(marker)
    box3d_pub.publish(marker_array)
#rospy.init_node('listener', anonymous=True)
class sub_pt_label_detection:
    def __init__ (self):

        self.box3d_pub = rospy.Publisher('nuscenes_lidar_label', MarkerArray, queue_size=1000)
        self.pcl_pub = rospy.Publisher('/pointcloud',PointCloud2,queue_size=1000)
        self.edge_pub = rospy.Publisher('/edges', MarkerArray, queue_size=1000)


        #self.lidar_sub = rospy.Subscriber("/nuscenes_lidar", PointCloud2, self.callback,queue_size = 1000)   #nuscenes_bag
        self.lidar_sub = rospy.Subscriber("/lidar", PointCloud2, self.callback,queue_size = 1000)             #kitti_bag
        self.label_sub = rospy.Subscriber("/lidar_label", MarkerArray, self.label_callback,queue_size = 3000)
        self.ori_label_list = []
        self.gt_marker_array_list = []
        self.count_scan_num = 0

        self.label_timestamp_list = []
    def label_callback(self,msg):
        label_timestamp = msg.markers[0].header.stamp.secs + msg.markers[0].header.stamp.nsecs * 0.000000001
        self.label_timestamp_list.append(label_timestamp)
        
        self.ori_label_list.append(msg)
        gt_bbox , gt_bbox_cube_format = get_gt_label(msg)
        calib = dataset.get_calib(0)
        #gt_marker_array_list = []
        label_mk_array  = MarkerArray()
        for i,box in enumerate(gt_bbox):

            eight_points = compute_3d_box_cam2(*box,1)
            #eight_points.T
            cam_xyz = np.matmul(eight_points.T,
            np.transpose(calib['velo_to_cam'])[:3,:3].astype(np.float32))
            cam_xyz += np.transpose(
                calib['velo_to_cam'])[[3], :3].astype(np.float32)


            cam_xyz1 = np.hstack([cam_xyz, np.ones([cam_xyz.shape[0],1])])
            velo_xyz = np.matmul(cam_xyz1, np.transpose(calib['cam_to_velo']))[:,:3]
            corners_3d_velo = velo_xyz.T

            h = np.sum((corners_3d_velo[:,0] - corners_3d_velo[:,4])**2)**0.5
            w = np.sum((corners_3d_velo[:,0] - corners_3d_velo[:,1])**2)**0.5
            l = np.sum((corners_3d_velo[:,0] - corners_3d_velo[:,3])**2)**0.5
            x = ((corners_3d_velo[:,0] + corners_3d_velo[:,6])/2)[0]
            y = ((corners_3d_velo[:,0] + corners_3d_velo[:,6])/2)[1]
            z = ((corners_3d_velo[:,0] + corners_3d_velo[:,6])/2)[2]
            #yaw = math.atan(((corners_3d_velo[:,0] + corners_3d_velo[:,1])/2)[1]/((corners_3d_velo[:,0] + corners_3d_velo[:,1])/2)[0])
            #yaw = math.pi - math.atan(abs(corners_3d_velo[:,0][1])/abs(corners_3d_velo[:,0][0]))
            point = ((corners_3d_velo[:,0] + corners_3d_velo[:,1])/2)

            intercept_x = ((corners_3d_velo[:,0] + corners_3d_velo[:,1])/2)[0]
            intercept_y = ((corners_3d_velo[:,0] + corners_3d_velo[:,1])/2)[1]
            #print('point = {}'.format(point))
            #print('intercept_x = {}'.format(intercept_x))
            #print('intercept_y = {}'.format(intercept_y))

            yaw = math.atan2(((corners_3d_velo[:,0] + corners_3d_velo[:,1])/2)[1]-y,((corners_3d_velo[:,0] + corners_3d_velo[:,1])/2)[0]-x)
            #print('yaw = {}'.format(yaw))
            #yaw = 45
            result_gt_box = [l,w,h,x,y,z,yaw]
            #cam_rgb_points = dataset.cam_points_to_velo(cam_rgb_points,calib)

            #cam_points = dataset.velo_points_to_cam(eight_points, calib)
            
            marker = publish_cube_label(*result_gt_box,i)
            label_mk_array.markers.append(marker)
        self.gt_marker_array_list.append(label_mk_array)
        #sself.box3d_pub.publish(label_mk_array)
        #self.gt_list.append(gt_bbox_cube_format)
        #publish_cube_cubinput(self.box3d_pub, gt_bbox_cube_format)

        #print('sdfkjsdlkfjdslkfjkl')
    def callback(self,data):
        #np.set_printoptions(threshold=sys.maxsize)
        lidar_time = data.header.stamp.secs + data.header.stamp.nsecs * 0.000000001


        #min_time_diff = 80000000000000000000000
        #self.label_timestamp_list[0]
        #print(self.label_timestamp_list)
        if not self.label_timestamp_list:
            self.label_timestamp_list.append(0)
            lidar_time = 0
        time_diff = abs(self.label_timestamp_list[0]-lidar_time)

        #print('self.label_timestamp_list[0] = {}'.format(self.label_timestamp_list[0]))
        #print('lidar_time = {}'.format(lidar_time))

        '''for index,label_timestamp_number in enumerate(self.label_timestamp_list):
            time_diff = abs(label_timestamp_number-lidar_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                iwant_index = index'''
        #self.label_timestamp_list[iwant_index]
        #matched_gt = self.gt_marker_array_list[iwant_index]
        #del self.label_timestamp_list[:iwant_index]
        #del self.gt_marker_array_list[:iwant_index]
        if time_diff<1:
            del self.label_timestamp_list[0]
        #if self.count_scan_num %10==0:
            with tf.Session(graph=graph,
                config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(tf.variables_initializer(tf.global_variables()))
                sess.run(tf.variables_initializer(tf.local_variables()))
                model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
                print('Restore from checkpoint %s' % model_path)
                saver.restore(sess, model_path)
                previous_step = sess.run(global_step)
                #for frame_idx in tqdm(range(0, NUM_TEST_SAMPLE)):

                #global theta
                pt = lidar_callback(data)
                points = pt.copy()
                ros_lidar = points
                #print(points)
                print(pt)
                frame_idx = 0
                start_time = time.time()
                if VISUALIZATION_LEVEL == 2:
                    pcd = open3d.PointCloud()
                    line_set = open3d.LineSet()
                    graph_line_set = open3d.LineSet()
            # provide input ======================================================
            #cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx,
            #    config['downsample_by_voxel_size'])
                calib = dataset.get_calib(frame_idx)
                
                #calib = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) 
                #calib['velo_to_cam']= np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) 
                #calib['cam_to_velo'] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) 
                #print('calib = {}'.format(calib))
                theta = -90 * math.pi/180
                #print('theta = {}'.format(theta))       
                modify_calib_R = dataset.eulerAnglesToRotationMatrix([0,0,theta])
                #print('calib_velo_to_cam_1 = {}'.format(calib['velo_to_cam']))
                v_to_c = calib['velo_to_cam'].copy()
                calib_rotation = calib['velo_to_cam'][:3,:3].copy()        
                v_to_c[:3,:3] = calib_rotation.dot(modify_calib_R)
                #calib['velo_to_cam'] = np.concatenate((modify_calib_R.dot(calib['velo_to_cam'][:3,:3]),v_to_c[:,])),axis=0)
                #calib['velo_to_cam'] = v_to_c.copy()
                #print('calib_velo_to_cam_2 = {}'.format(calib['velo_to_cam']))

                cam_rgb_points = dataset.ros_data_pre(frame_idx,ros_lidar,
                    config['downsample_by_voxel_size'],calib)
                '''in_modify_calib_R = dataset.eulerAnglesToRotationMatrix([-theta,0,0])
                c_to_v = calib['cam_to_velo'].copy()
                c_to_v[:3,:3] = calib['cam_to_velo'][:3,:3].dot(in_modify_calib_R)
                calib['cam_to_velo'] = c_to_v.copy()'''


                image = dataset.get_image(frame_idx)
                if not IS_TEST:
                    box_label_list = dataset.get_label(frame_idx)
                input_time = time.time()
                time_dict['fetch input'] = time_dict.get('fetch input', 0) \
                    + input_time - start_time 
                graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
                (vertex_coord_list, keypoint_indices_list, edges_list) = \
                    graph_generate_fn(
                        cam_rgb_points.xyz, **config['runtime_graph_gen_kwargs'])
                print('keypoint_indices_list = {}'.format(keypoint_indices_list[0]))
                graph_time = time.time()
                time_dict['gen graph'] = time_dict.get('gen graph', 0) \
                    + graph_time - input_time
                if config['input_features'] == 'irgb':
                    input_v = cam_rgb_points.attr
                elif config['input_features'] == '0rgb':
                    input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)),
                        cam_rgb_points.attr[:, 1:]])
                elif config['input_features'] == '0000':
                    input_v = np.zeros_like(cam_rgb_points.attr)
                elif config['input_features'] == 'i000':
                    input_v = np.hstack([cam_rgb_points.attr[:, [0]], np.zeros(
                        (cam_rgb_points.attr.shape[0], 3))])
                elif config['input_features'] == 'i':
                    #print('input_v = {}'.format(cam_rgb_points))
                    #print('dlksdlkdslkvslkdv')
                    input_v = cam_rgb_points.attr[:, [0]]
                elif config['input_features'] == '0':
                    input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))
                last_layer_graph_level = \
                    config['model_kwargs']['layer_configs'][-1]['graph_level']
                last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
                if config['label_method'] == 'yaw':
                    label_map = {'Background': 0, 'Car': 1, 'Pedestrian': 3,
                        'Cyclist': 5,'DontCare': 7}
                if config['label_method'] == 'Car':
                    label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}
                if config['label_method'] == 'Pedestrian_and_Cyclist':
                    label_map = {'Background': 0, 'Pedestrian': 1, 'Cyclist':3,
                        'DontCare': 5}
                # run forwarding =====================================================
            
                feed_dict = {
                    t_initial_vertex_features: input_v,
                    t_is_training: True,
                }
                feed_dict.update(dict(zip(t_edges_list, edges_list)))
                feed_dict.update(
                    dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
                feed_dict.update(dict(zip(t_vertex_coord_list, vertex_coord_list)))
                results = sess.run(fetches, feed_dict=feed_dict)
                #print('results = {}'.format(results))
                gnn_time = time.time()
                time_dict['gnn inference'] = time_dict.get('gnn inference', 0) \
                    + gnn_time - graph_time
                # box decoding =======================================================
                box_probs = results['probs']
                box_labels = np.tile(np.expand_dims(np.arange(NUM_CLASSES), axis=0),
                    (box_probs.shape[0], 1))
                box_labels = box_labels.reshape((-1))
                raw_box_labels = box_labels
                box_probs = box_probs.reshape((-1))
                pred_boxes = results['pred_box'].reshape((-1, 1, BOX_ENCODING_LEN))
                #print('pred_boxes = {}'.format(pred_boxes))
                last_layer_points_xyz = np.tile(
                    np.expand_dims(last_layer_points_xyz, axis=1), (1, NUM_CLASSES, 1))
                last_layer_points_xyz = last_layer_points_xyz.reshape((-1, 3))
                boxes_centers = last_layer_points_xyz
                decoded_boxes = box_decoding_fn(np.expand_dims(box_labels, axis=1),
                    boxes_centers, pred_boxes, label_map)
                box_mask = (box_labels > 0)*(box_labels < NUM_CLASSES-1)
                box_mask = box_mask*(box_probs > 1./NUM_CLASSES)
                box_indices = np.nonzero(box_mask)[0]
                decode_time = time.time()
                time_dict['decode box'] = time_dict.get('decode box', 0) \
                    + decode_time - gnn_time
                #print('box_indices = {}'.format(box_indices))
                if box_indices.size != 0:
                    box_labels = box_labels[box_indices]
                    box_probs = box_probs[box_indices]
                    box_probs_ori = box_probs
                    decoded_boxes = decoded_boxes[box_indices, 0]
                    box_labels[box_labels==2]=1
                    box_labels[box_labels==4]=3
                    box_labels[box_labels==6]=5
                    detection_scores = box_probs
                    # nms ============================================================
                    if USE_BOX_MERGE and USE_BOX_SCORE:
                        (class_labels, detection_boxes_3d, detection_scores,
                        nms_indices) = nms.nms_boxes_3d_uncertainty(
                            box_labels, decoded_boxes, detection_scores,
                            overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                            overlapped_thres=config['nms_overlapped_thres'],
                            appr_factor=100.0, top_k=-1,
                            attributes=np.arange(len(box_indices)))
                    if USE_BOX_MERGE and not USE_BOX_SCORE:
                        (class_labels, detection_boxes_3d, detection_scores,
                        nms_indices) = nms.nms_boxes_3d_merge_only(
                            box_labels, decoded_boxes, detection_scores,
                            overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                            overlapped_thres=config['nms_overlapped_thres'],
                            appr_factor=100.0, top_k=-1,
                            attributes=np.arange(len(box_indices)))
                    if not USE_BOX_MERGE and USE_BOX_SCORE:
                        (class_labels, detection_boxes_3d, detection_scores,
                        nms_indices) = nms.nms_boxes_3d_score_only(
                            box_labels, decoded_boxes, detection_scores,
                            overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                            overlapped_thres=config['nms_overlapped_thres'],
                            appr_factor=100.0, top_k=-1,
                            attributes=np.arange(len(box_indices)))
                    if not USE_BOX_MERGE and not USE_BOX_SCORE:
                        (class_labels, detection_boxes_3d, detection_scores,
                        nms_indices) = nms.nms_boxes_3d(
                            box_labels, decoded_boxes, detection_scores,
                            overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                            overlapped_thres=config['nms_overlapped_thres'],
                            appr_factor=100.0, top_k=-1,
                            attributes=np.arange(len(box_indices)))
                    box_probs = detection_scores
                    #------------------------------------------
                    last_layer_points_color = np.zeros(
                            (last_layer_points_xyz.shape[0], 3), dtype=np.float32)
                    last_layer_points_color[:, :] =  color_map[raw_box_labels, :]
                    cam_points_color = cam_rgb_points.attr[:, 1:]
                    pointsss = open3d.Vector3dVector(np.vstack(
                        [last_layer_points_xyz[box_indices][nms_indices],
                        last_layer_points_xyz, cam_rgb_points.xyz]))
                    np_pointss = np.asarray(pointsss)

                    #print('np_pointss = {}'.format(np_pointss))
                    '''pcd.points = open3d.Vector3dVector(np.vstack(
                        [last_layer_points_xyz[box_indices][nms_indices],
                        last_layer_points_xyz, cam_rgb_points.xyz]))
                    pcd.colors = open3d.Vector3dVector(np.vstack(
                        [last_layer_points_color[box_indices][nms_indices],
                        np.tile([(1,0.,200./255)],
                            (last_layer_points_color.shape[0], 1)),
                        cam_points_color]))'''
                    output_points_idx = box_indices[nms_indices]//NUM_CLASSES
                    edge_mask = np.isin(
                        edges_list[last_layer_graph_level][:, 1],
                        output_points_idx)
                    all_edges = np.hstack(
                        [edges_list[1][:, [0]],
                        keypoint_indices_list[-1][edges_list[
                        1][:, 1]]])
                    #print('all_edges = {}'.format(all_edges))
                    #print('keypoint_indices_list = {}'.format(keypoint_indices_list))
                    #print('keypoint_indices_list[-1] = {}'.format(keypoint_indices_list[-1]))
                    last_layer_edges = np.hstack(
                        [edges_list[last_layer_graph_level][:, [0]][edge_mask],
                        keypoint_indices_list[-1][edges_list[
                        last_layer_graph_level][:, 1][edge_mask]]])
                    colors = last_layer_points_color[edges_list[
                        last_layer_graph_level][:, 1][edge_mask]]
                    #print('last_layer_edges = {}'.format(last_layer_edges))
                    #print('[last_layer_edges, 0] = {}'.format(last_layer_edges))
                    #print('try = {}'.format(keypoint_indices_list[0][all_edges[0][0]]))
                    for i in range(len(keypoint_indices_list)-2, -1, -1):
                        #print('len(keypoint_indices_list) = {}'.format(len(keypoint_indices_list)))
                        #print(i)
                        last_layer_edges = \
                            keypoint_indices_list[i][last_layer_edges, 0]
                        #print('keypoint_indices_list[i] = {}'.format(keypoint_indices_list[i]))
                        all_edges = \
                            keypoint_indices_list[i][all_edges, 0]
                    #print('all_edges_2 = {}'.format(all_edges))
                    #print('all_edges_2 = {}'.format(all_edges))
                    lines = last_layer_edges.copy()
                    last_layer_edges += len(box_indices[nms_indices])
                    last_layer_edges += last_layer_points_xyz.shape[0]
                    
                    #print('lines = {}'.format(lines))
                    #----------------------------------------------
                    # visualization ===================================================

                    if VISUALIZATION_LEVEL > 0:
                        last_layer_points_color = np.zeros(
                            (last_layer_points_xyz.shape[0], 3), dtype=np.float32)
                        last_layer_points_color[:, :] =  color_map[raw_box_labels, :]
                        cam_points_color = cam_rgb_points.attr[:, 1:]
                        pcd.points = open3d.Vector3dVector(np.vstack(
                            [last_layer_points_xyz[box_indices][nms_indices],
                            last_layer_points_xyz, cam_rgb_points.xyz]))
                        pcd.colors = open3d.Vector3dVector(np.vstack(
                            [last_layer_points_color[box_indices][nms_indices],
                            np.tile([(1,0.,200./255)],
                                (last_layer_points_color.shape[0], 1)),
                            cam_points_color]))
                        output_points_idx = box_indices[nms_indices]//NUM_CLASSES
                        edge_mask = np.isin(
                            edges_list[last_layer_graph_level][:, 1],
                            output_points_idx)
                        last_layer_edges = np.hstack(
                            [edges_list[last_layer_graph_level][:, [0]][edge_mask],
                            keypoint_indices_list[-1][edges_list[
                            last_layer_graph_level][:, 1][edge_mask]]])
                        colors = last_layer_points_color[edges_list[
                            last_layer_graph_level][:, 1][edge_mask]]
                        for i in range(len(keypoint_indices_list)-2, -1, -1):
                            last_layer_edges = \
                                keypoint_indices_list[i][last_layer_edges, 0]
                        last_layer_edges += len(box_indices[nms_indices])
                        last_layer_edges += last_layer_points_xyz.shape[0]
                        lines = last_layer_edges
                        graph_line_set.points = pcd.points
                        graph_line_set.lines = open3d.Vector2iVector(lines)
                        graph_line_set.colors = open3d.Vector3dVector(colors)
                    # convert to KITTI ================================================
                    #print('detection_boxes_3d = {}'.format(detection_boxes_3d))
                    detection_boxes_3d_corners = nms.boxes_3d_to_corners(
                        detection_boxes_3d)
                    pred_labels = []
                    #print('detection_boxes_3d_corners = {}'.format(detection_boxes_3d_corners))
                    #print('detection_boxes_3d_corners = {}'.format(detection_boxes_3d_corners))
                    for i in range(len(detection_boxes_3d_corners)):
                        detection_box_3d_corners = detection_boxes_3d_corners[i]
                        corners_cam_points = Points(
                            xyz=detection_box_3d_corners, attr=None)
                        corners_img_points = dataset.cam_points_to_image(
                            corners_cam_points, calib)
                        corners_xy = corners_img_points.xyz[:, :2]
                        if config['label_method'] == 'yaw':
                            all_class_name = ['Background', 'Car', 'Car', 'Pedestrian',
                                'Pedestrian', 'Cyclist', 'Cyclist', 'DontCare']
                        if config['label_method'] == 'Car':
                            all_class_name = ['Background', 'Car', 'Car', 'DontCare']
                        if config['label_method'] == 'Pedestrian_and_Cyclist':
                            all_class_name = ['Background', 'Pedestrian', 'Pedestrian',
                                'Cyclist', 'Cyclist', 'DontCare']
                        if config['label_method'] == 'alpha':
                            all_class_name = ['Background', 'Car', 'Car', 'Pedestrian',
                                'Pedestrian', 'Cyclist', 'Cyclist', 'DontCare']
                        class_name = all_class_name[class_labels[i]]
                        xmin, ymin = np.amin(corners_xy, axis=0)
                        xmax, ymax = np.amax(corners_xy, axis=0)
                        clip_xmin = max(xmin, 0.0)
                        clip_ymin = max(ymin, 0.0)
                        clip_xmax = min(xmax, 1242.0)
                        clip_ymax = min(ymax, 375.0)
                        height = clip_ymax - clip_ymin
                        truncation_rate = 1.0 - (clip_ymax - clip_ymin)*(
                            clip_xmax - clip_xmin)/((ymax - ymin)*(xmax - xmin))
                        if truncation_rate > 0.4:
                            continue
                        x3d, y3d, z3d, l, h, w, yaw = detection_boxes_3d[i]
                        assert l > 0, str(i)
                        score = box_probs[i]
                        if USE_BOX_SCORE:
                            tmp_label = {"x3d": x3d, "y3d" : y3d, "z3d": z3d,
                            "yaw": yaw, "height": h, "width": w, "length": l}
                            # Rescore or not ===========================================
                            inside_mask = dataset.sel_xyz_in_box3d(tmp_label,
                                last_layer_points_xyz[box_indices])
                            points_inside = last_layer_points_xyz[
                                box_indices][inside_mask]
                            score_inside = box_probs_ori[inside_mask]
                            score = (1+occlusion(tmp_label, points_inside))*score
                        pred_labels.append((class_name, -1, -1, 0,
                            clip_xmin, clip_ymin, clip_xmax, clip_ymax,
                            h, w, l, x3d, y3d, z3d, yaw, score))
                        if VISUALIZATION_LEVEL > 0:
                            cv2.rectangle(image,
                                (int(clip_xmin), int(clip_ymin)),
                                (int(clip_xmax), int(clip_ymax)), (0, 255, 0), 2)
                            if class_name == "Pedestrian":
                                cv2.putText(image, '{:s} | {:.3f}'.format('P', score),
                                    (int(clip_xmin), int(clip_ymin)-int(clip_xmin/10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0) ,1)
                            else:
                                cv2.putText(image, '{:s} | {:.3f}'.format('C', score),
                                    (int(clip_xmin), int(clip_ymin)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) ,1)
                    nms_time = time.time()
                    time_dict['nms'] = time_dict.get('nms', 0) + nms_time - decode_time
                    # output ===========================================================
                    filename = OUTPUT_DIR+'/data/'+dataset.get_filename(
                        frame_idx)+'.txt'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w") as f:
                        for pred_label in pred_labels:
                            for field in pred_label:
                                f.write(str(field)+' ')
                            f.write('\n')
                        f.write('\n')
                    if VISUALIZATION_LEVEL > 0:
                        if not IS_TEST:
                            gt_boxes = []
                            gt_colors = []
                            for label in box_label_list:
                                if label['name'] in gt_color_map:
                                    gt_boxes.append([
                                        label['x3d'], label['y3d'], label['z3d'],
                                        label['length'], label['height'],
                                        label['width'], label['yaw']])
                                    gt_colors.append(gt_color_map[label['name']])
                            gt_boxes = np.array(gt_boxes)
                            gt_colors = np.array(gt_colors)/255.
                            gt_box_corners, gt_box_edges, gt_box_colors = \
                                dataset.boxes_3d_to_line_set(gt_boxes,
                                boxes_color=gt_colors)
                            if gt_box_corners is None or gt_box_corners.size<1:
                                gt_box_corners = np.array([[0, 0, 0]])
                                gt_box_edges = np.array([[0, 0]])
                                gt_box_colors =  np.array([[0, 0, 0]])
                        box_corners, box_edges, box_colors = \
                            dataset.boxes_3d_to_line_set(detection_boxes_3d)
                else:
                    if VISUALIZATION_LEVEL > 0:
                        last_layer_points_color = np.zeros(
                            (last_layer_points_xyz.shape[0], 3), dtype=np.float32)
                        last_layer_points_color[:, :] =  color_map[raw_box_labels, :]
                        cam_points_color = cam_rgb_points.attr[:, 1:]
                        box_corners = np.array([[0, 0, 0]])
                        box_edges = np.array([[0, 0]])
                        box_colors =  np.array([[0, 0, 0]])
                        pcd.points = open3d.Vector3dVector(np.vstack([
                            last_layer_points_xyz, cam_rgb_points.xyz]))
                        pcd.colors = open3d.Vector3dVector(np.vstack([np.tile(
                            [(128./255,0.,128./255)],
                            (last_layer_points_color.shape[0], 1)), cam_points_color]))
                        graph_line_set.points = open3d.Vector3dVector(
                            np.array([[0, 0, 0]]))
                        graph_line_set.lines = open3d.Vector2iVector(
                            [[0, 0]])
                        graph_line_set.colors = open3d.Vector3dVector(
                            np.array([[0, 0, 0]]))
                        if not IS_TEST:
                            gt_boxes = []
                            gt_colors = []
                            no_gt = True
                            for label in box_label_list:
                                if label['name'] in gt_color_map:
                                    gt_boxes.append([label['x3d'], label['y3d'],
                                    label['z3d'], label['length'], label['height'],
                                    label['width'], label['yaw']])
                                    gt_colors.append(gt_color_map[label['name']])
                                    no_gt = False
                            gt_boxes = np.array(gt_boxes)
                            gt_colors = np.array(gt_colors)/255.
                            gt_box_corners, gt_box_edges, gt_box_colors = \
                                dataset.boxes_3d_to_line_set(gt_boxes,
                                    boxes_color=gt_colors)
                            if gt_box_corners is None or gt_box_corners.size<1:
                                gt_box_corners = np.array([[0, 0, 0]])
                                gt_box_edges = np.array([[0, 0]])
                                gt_box_colors =  np.array([[0, 0, 0]])
                    filename = OUTPUT_DIR+'/data/'+dataset.get_filename(
                        frame_idx)+'.txt'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w") as f:
                        f.write('\n')

                if VISUALIZATION_LEVEL > 0:
                    #cv2.imshow('image', image)
                    #cv2.waitKey(10)
                    if not IS_TEST:
                        box_edges += gt_box_corners.shape[0]
                        line_set.points = open3d.Vector3dVector(np.vstack(
                            [gt_box_corners, box_corners]))
                        line_set.lines = open3d.Vector2iVector(np.vstack(
                            [gt_box_edges, box_edges]))
                        line_set.colors = open3d.Vector3dVector(np.vstack(
                            [gt_box_colors, box_colors]))
                    else:
                        line_set.points = open3d.Vector3dVector(np.vstack(
                            [box_corners]))
                        line_set.lines = open3d.Vector2iVector(np.vstack(
                            [box_edges]))
                        line_set.colors = open3d.Vector3dVector(np.vstack(
                            [box_colors]))
                if VISUALIZATION_LEVEL == 1:
                    #vis.destroy_window()
                    '''vis.update_geometry()
                    vis.poll_events()
                    vis.update_renderer()'''
                if VISUALIZATION_LEVEL == 2:
                    print("Configure the viewpoint as you want and press [q]")
                    def custom_draw_geometry_load_option(geometry_list):
                        vis = open3d.Visualizer()
                        vis.create_window()
                        for geometry in geometry_list:
                            vis.add_geometry(geometry)
                        ctr = vis.get_view_control()
                        ctr.rotate(0.0, 3141.0, 0)
                        vis.run()
                        vis.destroy_window()
                    custom_draw_geometry_load_option([pcd, line_set, graph_line_set])
                total_time = time.time()
                time_dict['total'] = time_dict.get('total', 0) \
                    + total_time - start_time

                #cam_rgb_points
                #print('cam_rgb_points.attr[:,0] = {}'.format(cam_rgb_points.attr[:,[0]]))
            cam_rgb_points = dataset.cam_points_to_velo(cam_rgb_points,calib)
            pub_points =  np.concatenate((cam_rgb_points.xyz, cam_rgb_points.attr[:,[0]]), axis=1)



            #print('cam_rgb_points = {}'.format(pub_points))
            
            #print('new_detection_boxes_3d_corners = {}'.format(detection_boxes_3d_corners))
            def publish_3dbox(box3d_pub, corners_3d_velos,types,__FRAME_ID__='/nuscenes_lidar'):
                marker_array = MarkerArray()
                #list(types)
                header = Header()
                header.stamp = rospy.Time.now()
                model_name = 'graph'
                if model_name == 'PVRCNN':
                    DETECTION_COLOR_DICT = {'truck':(160,32,240) , 'barrier':(255,30,0),'motorcycle':(0,255,0),'car':(0,0,255) , 'pedestrian':(255,0,255),'trailer':(255,255,255),'bus':(0,255,255),'bicycle':(255,153,18),'traffic_cone':(56,94,15),'construction_vehicle':(160,32,240),'Car':(0,0,255),'Cyclist':(0,255,0),'Pedestrian':(255,0,255)}
                else:
                    DETECTION_COLOR_DICT = {'truck':(160,32,240) , 'barrier':(255,30,0),'motorcycle':(0,255,0),'car':(0,0,255) , 'pedestrian':(255,0,255),'trailer':(255,255,255),'bus':(0,255,255),'bicycle':(255,153,18),'traffic_cone':(56,94,15),'construction_vehicle':(160,32,240),'Car':(0,0,255),'Cyclist':(0,255,0),'Pedestrian':(255,0,255)}
                #DETECTION_COLOR_DICT = {'1':(255,0,0) , '2':(255,30,0),'3':(0,255,0),'4':(0,0,255) , '5':(255,0,255),'6':(255,255,255),'0':(0,255,255),'7':(255,153,18),'8':(56,94,15),'9':(160,32,240)}
                LINES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[4,0],[5,1],[6,2],[7,3]]#,[4,1],[5,0]]
                LIFETIME = 600
                FRAME_ID = '/nuscenes_lidar'
                FRAME_ID = __FRAME_ID__
                for i, corners_3d_velo in enumerate(corners_3d_velos):
                    corners_3d_velo = corners_3d_velo.T
                    marker = Marker()
                    marker.header.frame_id = FRAME_ID
                    marker.header.stamp = rospy.Time.now()
                    #a = types[i]
                    #a = 1
                    a = 'car'
                    marker.id = i

                    #marker.ns = types[i]
                    '''if types[i] == 'car':
                        marker.ns = 'car'
                    if types[i] != 'car':
                        marker.ns = "others"'''
                    '''marker.ns = "small vehicles"
                    marker.ns = "big vehicles"
                    marker.ns = "pedestrian"
                    marker.ns = "motorcyclist and bicyclist"
                    marker.ns = "traffic cones"'''
                    
                    marker.action = Marker.ADD
                    marker.lifetime = rospy.Duration(LIFETIME)
                    marker.type = Marker.LINE_LIST
                    
                    r,g,b = DETECTION_COLOR_DICT[str(a)]
                    marker.color.r = r/255.0
                    marker.color.g = g/255.0
                    marker.color.b = b/255.0

                    marker.color.a = 1.0
                    
                    marker.scale.x = 0.15

                    marker.points = []
                    #print(corners_3d_velo[0][1])
                    '''for l in LINES:
                        p1 = corners_3d_velo[l[0]]
                        marker.points.append(Point(p1[0],p1[1],p1[2]))
                        print(p1)
                        p2 = corners_3d_velo[l[1]]
                        marker.points.append(Point(p2[0],p2[1],p2[2]))'''
                    
                    for l in LINES:
                        p1x = corners_3d_velo[0][l[0]]
                        p1y = corners_3d_velo[1][l[0]]
                        p1z = corners_3d_velo[2][l[0]]
                        marker.points.append(Point(p1x,p1y,p1z))
                        p2x = corners_3d_velo[0][l[1]]
                        p2y = corners_3d_velo[1][l[1]]
                        p2z = corners_3d_velo[2][l[1]]
                        marker.points.append(Point(p2x,p2y,p2z))
                    #print(marker)
                    marker_array.markers.append(marker)
                box3d_pub.publish(marker_array)
            def publish_point_cloud(pcl_pub,point_cloud):
                FRAME_ID = '/nuscenes_lidar'
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = FRAME_ID
                #pcl_pub.publish(pcl2.create_cloud_xyz32(header,point_cloud[:, :3]))
                #print('intensity = {}'.format(point_cloud[:,3]))
                #new_intensity = []
                #pc_intensity = point_cloud[0][:,3]
                #new_pc_intensity = pc_intensity*255
                '''for j in pc_intensity:
                    #a = struct.pack('i', j)
                    a = j*255
                    new_pc_intensity.append(a)
                new_pc_intensity = np.array(new_pc_intensity)'''

                #point_cloud[0][:,3] = new_pc_intensity
                fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    #PointField('rgb', 12, PointField.UINT32, 1),
                    PointField('rgba', 12, PointField.FLOAT32, 1),
                    ]
            
                #print('point_cloud = {}'.format(point_cloud[0]))
                #print(type(point_cloud[0]))
                pcl_pub.publish(pcl2.create_cloud(header,fields,point_cloud[:,:4]))
            def delete_marker():
                reset_marker_array = MarkerArray()
                for i in range(100):     
                    pub_empty_marker = rospy.Publisher('nuscenes_lidar_label', MarkerArray, queue_size=10)
                    reset_marker = Marker()
                    reset_marker.header.frame_id = '/nuscenes_lidar'
                    reset_marker.header.stamp = rospy.Time()
                    reset_marker.action = 3
                    reset_marker_array.markers.append(reset_marker)
                #pub_empty_marker.publish(reset_marker)
                return pub_empty_marker.publish(reset_marker_array)
            #-----------------
            '''pt = lidar_callback(data)
            points = pt.copy()
            ros_lidar = points
            pub_points = ros_lidar'''
            #---------------
            types = np.ones(100)
            delete_marker()
            import random

            

            #num_of_group = len(edges_list[0][:,1])
            dest = edges_list[0][:,1]
            num_of_group = np.amax(edges_list[0][:,1])+1

            #print('num_of_group = {}'.format(num_of_group))

            #numpy.where(x == 0)[0]
            '''for i in range(num_of_group):
                rand_num = random.randint(0,255)
                group_member_idx = np.where(dest == i)[0]
                #print('number of point = {}'.format(len(group_member_idx)))
                pub_points[edges_list[0][group_member_idx,0],3] = rand_num #/num_of_group * 255'''
            pub_points[keypoint_indices_list[0],3] = 0.0
            mask = np.ones(len(pub_points), np.bool)
            mask[keypoint_indices_list[0]] = 0
            mask = np.where(mask == 1)[0]
            pub_points[mask,3]= 255.0
            #print('pub_points = {}'.format(pub_points[mask]))
            #print('key_points_number = {}'.format(len(keypoint_indices_list[0])))
            #pub_points[edges_list[0],3] = 255
            #pub_points[edges_list[0],3] = 255
            #print('t_edges_list = {}'.format(t_edges_list))
            #print('edges_list = {}'.format(edges_list))
            #keypoint_indices_list

            publish_point_cloud(self.pcl_pub,pub_points)
            
            #publish_point_cloud(pcl_pub,pt)
            #if len(detection_boxes_3d_corners) != 0:
            #np.set_printoptions(threshold=sys.maxsize)
            #pred_boxes = np.squeeze(pred_boxes)
            #print('pred_boxes = {}'.format(pred_boxes))
            #rospy.sleep(1)
            #detection_boxes_3d_corners = nms.boxes_3d_to_corners(
            #            pred_boxes)
            if box_indices.size != 0:
                detection_boxes_3d_corners = dataset.cam_points_to_velo_array(detection_boxes_3d_corners,calib)
                
                #print('detection_boxes_3d_corners = {}'.format(detection_boxes_3d_corners))
                publish_3dbox(self.box3d_pub, detection_boxes_3d_corners,types)
            #self.gt_marker_array_list.append(label_mk_array)

            
            if self.gt_marker_array_list:
                self.box3d_pub.publish(self.gt_marker_array_list[0])
                del self.gt_marker_array_list[0]
            #self.box3d_pub.publish(self.ori_label_list[0])
            #draw_ros_edge(self.box3d_pub,lines,pub_points)
            #edges_list[0][:10,:]
            #print('edges_list = {}'.format(all_edges))
            #print('lines = {}'.format(lines))
            #print('edges_list_len = {}'.format(len(edges_list[1])))


            choose = np.random.random_integers(0, 700000, 10000)
            #draw_ros_edge(self.box3d_pub,all_edges,pub_points)              #draw_all_edges
            #draw_ros_edge(self.box3d_pub,all_edges[choose,:],pub_points)   #draw_random_edges
            #draw_ros_edge(self.box3d_pub,lines,pub_points)                  #draw_last_layer_edges
            #keypoint_indices_list[0]
            #draw_ros_sphere(self.box3d_pub,all_edges,pub_points)           #draw_all_sphere

            #draw_ros_sphere(self.box3d_pub,lines,pub_points)               #draw_last_layer_sphere


            #publish_cube_cubinput(self.box3d_pub, self.gt_list[0])
            
        #del self.ori_label_list[0]
        self.count_scan_num+=1
        print('------------------------------------------------LL')
def ros_main():

    rospy.init_node('listener', anonymous=True)

    ss = sub_pt_label_detection()
    rospy.spin()
ros_main()
#lidar_sub = rospy.Subscriber("/nuscenes_lidar", PointCloud2, callback,queue_size = 100000000000)
#rospy.spin()