python GCN_reg/train_main.py --lr 1e-4 --batch_size 8 --num_epoch 60 --data_path ../dataset/CUB --load_dir pretrained/CUB/step2/h4f2_b0.1_86.5

python GCN_reg/alternate_with_GCN.py --lr 1e-4 --batch_size 8 --num_epoch 100 --data_path ../dataset/CUB