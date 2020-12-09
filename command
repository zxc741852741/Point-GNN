Train : python3 train.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root /data3/KITTI_GNN 記得去config改model儲存位置和訓練epoch
Test : python3 run.py my_training_model/car_auto_T3_train/ --test --dataset_root_dir /data3/KITTI_GNN --output_dir tmp_result

conda activate pointgnn

cd /data2/Point-GNN_train/Point-GNN

conda activate DS_HW1
cd /data2/CV_HW3/HW3
python main.py --train_bs 128 --test_bs 128

