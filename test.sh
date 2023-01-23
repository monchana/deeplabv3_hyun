python main.py --model deeplabv3_resnet101 --enable_vis --vis_port 28333 --gpu_id 0,1,2 --year 2012 \
    --data_root /home/monchana/data/dataset/diffaug/coco_voc_8000/data \
    --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 \
    --ckpt /home/monchana/weights/semantic_segmentation/deeplabv3/best_deeplabv3_resnet101_voc_os16.pth --test_only 
    