import numpy as np
import math
import rospy
from sensor_msgs.msg import Image , PointCloud2 , PointField
from visualization_msgs.msg import Marker ,MarkerArray 
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2 
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation as R
import random

def publish_cube_cubinput(box3d_pub,gt_bbox_cube_format):
    marker_array = MarkerArray()
    header = Header()
    header.stamp = rospy.Time.now()
    for i ,gt_bbox in enumerate(gt_bbox_cube_format):
        marker = Marker()
        marker.header.frame_id = '/nuscenes_lidar'

        #時間戳
        #marker.header.stamp.secs = secs
        #marker.header.stamp.nsecs = nsecs
        marker.header.stamp = rospy.Time.now()


        marker.ns = 'GT'
        #Marker的id號
        marker.id = i

        #Marker的類型，有ARROW，CUBE等
        marker.type = Marker.CUBE

        #Marker的尺寸，單位是m
        marker.scale.x = gt_bbox[3]
        marker.scale.y = gt_bbox[4]
        marker.scale.z = gt_bbox[5]

        #Marker的動作類型有ADD，DELETE等
        marker.action = Marker.ADD

        #Marker的位置姿態
        marker.pose.position.x = gt_bbox[0]
        marker.pose.position.y = gt_bbox[1]
        marker.pose.position.z = gt_bbox[2]

        marker.pose.orientation.x = gt_bbox[6]
        marker.pose.orientation.y = gt_bbox[7]
        marker.pose.orientation.z = gt_bbox[8]
        marker.pose.orientation.w = gt_bbox[9]

        #Marker的顏色和透明度
        #r,g,b = DETECTION_COLOR_DICT[str(label_type)]
        marker.color.r = 0/255.0
        marker.color.g = 255/255.0
        marker.color.b = 0/255.0
        marker.color.a = 0.6

        #Marker被自動銷毀之前的存活時間，rospy.Duration()意味着在程序結束之前一直存在
        marker.lifetime = rospy.Duration(600)
        marker_array.markers.append(marker)
    box3d_pub.publish(marker_array)


def publish_cube_label(l,w,h,x,y,z,yaw,label_id):
    #定義一個marker對象，並初始化各種Marker的特性
    #marker_array = MarkerArray()
    rot = R.from_euler('xzy', [0, yaw*180/math.pi, 0], degrees=True)
    rot_quat = rot.as_quat()
    #print(x,y,z,l,w,h,yaw,label_type,label_id,secs,nsecs)
    #指定Marker的參考框架
    marker = Marker()
    marker.header.frame_id = '/nuscenes_lidar'

    #時間戳
    #marker.header.stamp.secs = secs
    #marker.header.stamp.nsecs = nsecs
    marker.header.stamp = rospy.Time.now()

    marker.ns = 'GT000'

    #Marker的id號
    marker.id = label_id

    #Marker的類型，有ARROW，CUBE等
    marker.type = Marker.CUBE

    #Marker的尺寸，單位是m
    marker.scale.x = l
    marker.scale.y = w
    marker.scale.z = h

    #Marker的動作類型有ADD，DELETE等
    marker.action = Marker.ADD

    #Marker的位置姿態
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z

    marker.pose.orientation.x = rot_quat[0]
    marker.pose.orientation.y = rot_quat[1]
    marker.pose.orientation.z = rot_quat[2]
    marker.pose.orientation.w = rot_quat[3]

    #Marker的顏色和透明度
    #r,g,b = DETECTION_COLOR_DICT[str(label_type)]
    marker.color.r = 0/255.0
    marker.color.g = 255/255.0
    marker.color.b = 0/255.0
    marker.color.a = 0.1

    #Marker被自動銷毀之前的存活時間，rospy.Duration()意味着在程序結束之前一直存在
    marker.lifetime = rospy.Duration(600)

    return marker
def publish_3dbox(box3d_pub, corners_3d_velos,types,model_name,__FRAME_ID__='velodyne'):
            marker_array = MarkerArray()
            #list(types)
            header = Header()
            header.stamp = rospy.Time.now()
            if model_name == 'PVRCNN':
                DETECTION_COLOR_DICT = {'truck':(160,32,240) , 'barrier':(255,30,0),'motorcycle':(0,255,0),'car':(0,0,255) , 'pedestrian':(255,0,255),'trailer':(255,255,255),'bus':(0,255,255),'bicycle':(255,153,18),'traffic_cone':(56,94,15),'construction_vehicle':(160,32,240),'Car':(255,0,0),'Cyclist':(0,255,0),'Pedestrian':(255,0,255),'GT':(255,0,0)}
            else:
                DETECTION_COLOR_DICT = {'truck':(160,32,240) , 'barrier':(255,30,0),'motorcycle':(0,255,0),'car':(0,0,255) , 'pedestrian':(255,0,255),'trailer':(255,255,255),'bus':(0,255,255),'bicycle':(255,153,18),'traffic_cone':(56,94,15),'construction_vehicle':(160,32,240),'Car':(255,0,0),'Cyclist':(0,255,0),'Pedestrian':(255,0,255),'GT':(255,0,0)}
            #DETECTION_COLOR_DICT = {'1':(255,0,0) , '2':(255,30,0),'3':(0,255,0),'4':(0,0,255) , '5':(255,0,255),'6':(255,255,255),'0':(0,255,255),'7':(255,153,18),'8':(56,94,15),'9':(160,32,240)}
            LINES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[4,0],[5,1],[6,2],[7,3]]#,[4,1],[5,0]]
            LIFETIME = 0.3
            FRAME_ID = 'velodyne'
            FRAME_ID = __FRAME_ID__
            for i, corners_3d_velo in enumerate(corners_3d_velos):
                marker = Marker()
                marker.header.frame_id = FRAME_ID
                marker.header.stamp = rospy.Time.now()
                a = types[i]
                #a = 1
                marker.id = i

                marker.ns = types[i]
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
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R
def compute_3d_box_cam2(l,w,h,x,y,z,yaw,nuscenes):
#def compute_3d_box_cam2(x,y,z,l,w,h,yaw,pitch,roll):      
#def compute_3d_box_cam2(x,y,z,l,w,h,yaw,roll,pitch):
#def compute_3d_box_cam2(x,y,z,l,w,h,roll,pitch,yaw):    
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]

    if nuscenes==0:
        #z_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [0,0,0,0,h,h,h,h]
        #z_corners = [-h,-h,-h,-h,0,0,0,0]
    else:
        z_corners = [-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2]



    R = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners]))
    z_corners = np.array(z_corners)
    corners_3d_cam2 = np.row_stack([corners_3d_cam2,z_corners])

    #R = eulerAnglesToRotationMatrix([roll*180/math.pi,pitch*180/math.pi,yaw*180/math.pi])
    #corners_3d_cam2 = np.dot(R,np.vstack([x_corners,y_corners,z_corners]))
    #R = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    #corners_3d_cam2 = np.dot(R,np.vstack([x_corners,y_corners,z_corners]))


    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2
def publish_point_cloud(pcl_pub,point_cloud):
    FRAME_ID = 'velodyne'
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
def points_rotation(points,yaw):
    yaw = yaw*math.pi/180
    R = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    out_points = points.copy()
    x_y_points = points.copy()
    x_y_points = x_y_points[:,:2]
    rotation_points = np.dot(x_y_points, R.T)
    out_points[:,:2] = rotation_points
    return out_points
def publish_distance(distance_pub,corners_3d_velos):
    marker_array = MarkerArray()
    #list(types)
    DETECTION_COLOR_DICT = {'1':(255,0,0) , '2':(255,30,0),'3':(0,255,0),'4':(0,0,255) , '5':(255,0,255),'6':(255,255,255),'0':(0,255,255),'7':(255,153,18),'8':(56,94,15),'9':(160,32,240)}
    FRAME_ID = 'velodyne'
    LIFETIME = 0.3
    #print(corners_3d_velos)
    for i, each_box in enumerate(corners_3d_velos):
        marker = Marker()
        
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()
        #a = types[i]
        a = 3
        marker.id = i
        '''marker.ns = "small vehicles"
        marker.ns = "big vehicles"
        marker.ns = "pedestrian"
        marker.ns = "motorcyclist and bicyclist"
        marker.ns = "traffic cones"
        marker.ns = "others"'''
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.TEXT_VIEW_FACING
        
        r,g,b = DETECTION_COLOR_DICT[str(a)]
        marker.color.r = r/255.0
        marker.color.g = g/255.0
        marker.color.b = b/255.0

        marker.color.a = 1.0
        
        marker.scale.x = 1.6
        marker.scale.y = 1.6
        marker.scale.z = 1.6

        marker.points = []
        x = (each_box[0][4]+each_box[0][6])/2
        y = (each_box[1][4] + each_box[1][6])/2
        z = each_box[2][6]
        marker.pose.position.x = each_box[0][5]
        marker.pose.position.y = each_box[1][5]
        marker.pose.position.z = each_box[2][6]
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        d = (each_box[0][0]**2 + each_box[1][1]**2 + each_box[2][2]**2) ** 0.5
        #num ** 0.5
        d = round(d, 1)
        d = '%s'%d + 'm'

        marker.text = d
        
        marker_array.markers.append(marker)
    distance_pub.publish(marker_array)
import sys
sys.path.insert(0,'/home/user/anaconda3/lib/python3.7/site-packages/')  
import ros_numpy
def ros_msg_to_pointsss(data):
    pc = ros_numpy.numpify(data)
    #print(pc)
    points=np.zeros((pc.shape[0],4))
    #print('fun_points = {}'.format(points))
    rotation_points = np.zeros((pc.shape[0],4))
    #print('x:{}'.format(pc['x']))
    '''print('y:{}'.format(pc['y']))
    print('-x:{}'.format(-pc['x']))
    print('-y:{}'.format(-pc['y']))'''
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    #intensity = pc['I']
    intensity = pc['rgba'] #/31539.20
    #intensity.flags.writeable = True
    #print(type(intensity))
    new_intensity = []
    for idx,i in enumerate(intensity):
        if i>=1:
            #intensity[idx]=0
            new_intensity.append(0)
            continue
        new_intensity.append(i)
    #intensity = intensity/31539.0   
    #intensity = np.zeros(len(pc['z']))
    new_intensity = np.array(new_intensity)
    points[:,3]=new_intensity

    #print('points = {}'.format(points))
    #points[:,4] = np.zeros(pc.shape[0])
    #print('ahfashfjlsahfkajsh')
    #zero_array = 
    #print('points = {}'.format(points))
    #points.astype(np.int32)
    #print('fun_points_2 = {}'.format(points))
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


def get_gt_label(gt_msg):
    gt_bbox = []
    gt_bbox_cube_format = []
    for marker in gt_msg.markers:
        #print(marker.ns)
        if marker.ns == 'vehicle.car':
            #print(marker)
            #print('marker.ns = {}'.format(marker.ns))
            car_label = np.empty(7)
            car_label = []
            x = marker.pose.position.x
            y = marker.pose.position.y
            z = marker.pose.position.z
            
            length = marker.scale.x
            width = marker.scale.y
            height = marker.scale.z

            orientation_x = marker.pose.orientation.x
            orientation_y = marker.pose.orientation.y
            orientation_z = marker.pose.orientation.z
            orientation_w = marker.pose.orientation.w
            
            r = R.from_quat([orientation_x, orientation_y, orientation_z, orientation_w])
            yaw = r.as_rotvec()[2]
            gt_bbox.append([length,width,height,x,y,z,yaw])
            gt_bbox_cube_format.append([x,y,z,length,width,height,orientation_x,orientation_y,orientation_z,orientation_w])
    return gt_bbox,gt_bbox_cube_format


def draw_ros_edge(box3d_pub,edges_list,pub_points):
    marker_array = MarkerArray()
    #list(types)
    header = Header()
    header.stamp = rospy.Time.now()
   
    #DETECTION_COLOR_DICT = {'1':(255,0,0) , '2':(255,30,0),'3':(0,255,0),'4':(0,0,255) , '5':(255,0,255),'6':(255,255,255),'0':(0,255,255),'7':(255,153,18),'8':(56,94,15),'9':(160,32,240)}
    LINES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[4,0],[5,1],[6,2],[7,3]]#,[4,1],[5,0]]
    LIFETIME = 600
    FRAME_ID = '/nuscenes_lidar'
    #FRAME_ID = __FRAME_ID__
    for i, edge in enumerate(edges_list):
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()
        #a = types[i]
        #a = 1
        marker.id = i

        marker.ns = 'edge'
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_LIST
        
        #r,g,b = DETECTION_COLOR_DICT[str(a)]
        marker.color.r = 255/255.0
        marker.color.g = 255/255.0
        marker.color.b = 0/255.0

        marker.color.a = 1.0
        
        marker.scale.x = 0.01

        marker.points = []
    
        [p1x,p1y,p1z,in1] = pub_points[edge[0]]
        [p2x,p2y,p2z,in2] = pub_points[edge[1]]

        marker.points.append(Point(p1x,p1y,p1z))
        marker.points.append(Point(p2x,p2y,p2z))
        marker_array.markers.append(marker)
    box3d_pub.publish(marker_array)


def draw_ros_sphere(box3d_pub,lines,pub_points):
    marker_array = MarkerArray()
    #list(types)
    header = Header()
    header.stamp = rospy.Time.now()
   
    #DETECTION_COLOR_DICT = {'1':(255,0,0) , '2':(255,30,0),'3':(0,255,0),'4':(0,0,255) , '5':(255,0,255),'6':(255,255,255),'0':(0,255,255),'7':(255,153,18),'8':(56,94,15),'9':(160,32,240)}
    LINES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[4,0],[5,1],[6,2],[7,3]]#,[4,1],[5,0]]
    LIFETIME = 600
    FRAME_ID = '/nuscenes_lidar'
    #FRAME_ID = __FRAME_ID__
    #print('keypoint_indices_list = {}'.format(lines))
    points = pub_points[lines[:,0]]
    for i, point in enumerate(points):
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()
        #a = types[i]
        #a = 1
        marker.id = i

        marker.ns = 'sphere'
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.SPHERE
        
        #r,g,b = DETECTION_COLOR_DICT[str(a)]
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        marker.color.r = r/255.0
        marker.color.g = g/255.0
        marker.color.b = b/255.0

        marker.color.a = 0.5
        
        marker.scale.x = 2.0
        marker.scale.y = 2.0
        marker.scale.z = 2.0
        #marker.points = []
        #print(point)
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]
        #[p1x,p1y,p1z,in1] = pub_points[edge[0]]
        #[p2x,p2y,p2z,in2] = pub_points[edge[1]]

        #marker.points.append(Point(p1x,p1y,p1z))
        #marker.points.append(Point(p2x,p2y,p2z))
        marker_array.markers.append(marker)
    box3d_pub.publish(marker_array)