python train_xuez.py --train_csv_path /mnt/sda/gbw/HRCenterNet_/datasets/xxshanzi/train_250.csv \
                --train_data_dir /mnt/sda/gbw/HRCenterNet_/datasets/xxshanzi/train_250/ \
                --val_csv_path /mnt/sda/gbw/HRCenterNet_/datasets/xxshanzi/val_50.csv \
                --val_data_dir /mnt/sda/gbw/HRCenterNet_/datasets/xxshanzi/val_50/ --val \
                --weight_dir /mnt/sda/gbw/HRCenterNet_/weights/xxshanzi_1/ \
                --log_dir /mnt/sda/gbw/HRCenterNet_/xxs_best.pth.tar \
                --batch_size 8 \
                --epoch 300 
                # --device cuda:1