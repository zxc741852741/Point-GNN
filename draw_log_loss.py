import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import numpy as np
import os 
color_list = ['green','red','blue','purple']
each_line_name = ['Complete model','Add attention']#'Reduced two layers','Surrounding feature','Edge feature']

all_layer_888_path = 'my_training_model/han_car_auto_T3_train_epoch_808'
cut_layer_path = 'my_training_model/car_auto_T3_train_-2_layer'
edgetozero_path = 'my_training_model/car_auto_T3_train_all_layers_edge2zero'
featuretozero_path = 'my_training_model/car_auto_T3_train_all_layers_feature2zero'
dev_attetion_path = 'my_training_model/dev'
path_list = [all_layer_888_path,dev_attetion_path] #,cut_layer_path,edgetozero_path,featuretozero_path]
#path_list = [all_layer_888_path,cut_layer_path]
save_path = 'my_training_model/dev'
draw_epoch = 179 #366
# 設置放大區間
zone_left = 140
zone_right = 178
def get_draw_list(log_path):
    with open (os.path.join(log_path,'log.txt'),'r') as f:
        lines = f.readlines()
    #print(lines)
    epoch_list = []
    cls_loss_list = [] 
    loc_loss_list = []
    for i in range(0,draw_epoch*2,2):
        epoch = lines[i].strip().split(',')[1]
        cls_loss = lines[i+1].strip().split(',')[0]
        loc_loss = lines[i+1].strip().split(',')[1]
        total_loss = lines[i+1].strip().split(',')[3]
        print(epoch[12:])
        epoch_list.append(int(epoch[12:]))
        cls_loss_list.append(float(cls_loss[4:]))
        loc_loss_list.append(float(loc_loss[5:]))
        print(epoch)
        print(cls_loss)
    return epoch_list,cls_loss_list,loc_loss_list

def draw_picture(x_axis,y_axis_cls_list,mode = 'cls'):

    fig, ax = plt.subplots(1, 1)
    for i,y_axis_cls in enumerate(y_axis_cls_list):
        ax.plot(x_axis,y_axis_cls, color=color_list[i], label=each_line_name[i])
    #ax.plot(x_axis,alayer_cls_loss_list, color='green', label='All layers')
    #ax.plot(x_axis,cutlayer_cls_loss_list, color='red', label='Reduced layers')
    #ax.plot(edgetozero_epoch_list,edgetozero_cls_loss_list, color='blue', label='edgetozero layers')
    ax.set_xlabel("epoch")
    ax.set_ylabel(mode + " loss")
    ax.set_title(mode + " loss")
    
    axins = inset_axes(ax, width="50%", height="40%", loc='lower left',
                    bbox_to_anchor=(0.45, 0.3, 1, 1), 
                    bbox_transform=ax.transAxes)
    for i,y_axis_cls in enumerate(y_axis_cls_list):
        #ax.plot(x_axis,y_axis_cls, color=color_list[i], label=each_line_name[i])
        axins.plot(x_axis,y_axis_cls, color=color_list[i], label=each_line_name[i])
    #axins.plot(x_axis,alayer_cls_loss_list, color='green', label='All layers')
    #axins.plot(x_axis,cutlayer_cls_loss_list, color='red', label='Reduced layers')
    #axins.plot(edgetozero_epoch_list,edgetozero_cls_loss_list, color='blue', label='edgetozero layers')



    # 座標軸的擴展比例（根據實際數據調整）
    x_ratio = 0  # x軸顯示範圍的擴展比例
    y_ratio = 0.05  # y軸顯示範圍的擴展比例

    # X軸的顯示範圍
    xlim0 = x_axis[zone_left]-(x_axis[zone_right]-x_axis[zone_left])*x_ratio
    xlim1 = x_axis[zone_right]+(x_axis[zone_right]-x_axis[zone_left])*x_ratio

    # Y軸的顯示範圍
    
    #y = np.hstack(([[123],[123]]))
    #print(y)
    small_y_axis = [y_axis_cls[zone_left:zone_right] for y_axis_cls in y_axis_cls_list]
    y = np.hstack((small_y_axis))
    #y = np.hstack((alayer_cls_loss_list[zone_left:zone_right], cutlayer_cls_loss_list[zone_left:zone_right],
    #            edgetozero_cls_loss_list[zone_left:zone_right]))
    
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    # 調整子座標系的顯示範圍
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)



    # 原圖中畫方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    ax.plot(sx,sy,"black")

    # 畫兩條線
    xy = (xlim0,ylim0)
    xy2 = (xlim0,ylim1)
    con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",
            axesA=axins,axesB=ax)
    axins.add_artist(con)

    xy = (xlim1,ylim0)
    xy2 = (xlim1,ylim1)
    con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",
            axesA=axins,axesB=ax)
    axins.add_artist(con)



    ax.legend()
    
    fig.savefig(os.path.join(save_path,mode + '_loss.png'),format="png",dpi=800)





y_axis_cls_list = []
y_axis_loc_list = []
for i,path in enumerate(path_list):
    x_axis, y_axis_cls,y_axis_loc = get_draw_list(path)
    if i == 0:
        Denominator  = y_axis_cls[0] - y_axis_cls[-1]
    if i == 1:
        molecular = y_axis_cls[-1]- y_axis_cls_list[0][-1]
    y_axis_cls_list.append(y_axis_cls)
    y_axis_loc_list.append(y_axis_loc)
print(molecular/Denominator)
draw_picture(x_axis,y_axis_cls_list,'cls')
draw_picture(x_axis,y_axis_loc_list,'loc')
