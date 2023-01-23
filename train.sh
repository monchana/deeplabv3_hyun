
python main_syn.py --model deeplabv3_resnet101 --enable_vis --gpu_id 0,1,2 \
    --year 2012 --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 \
    --data_root /home/monchana/data/dataset/diffaug/coco_voc_8000/data \
    --save_dir /home/monchana/data/playgrounds/semantic_segmentation/deeplabv3 \
    --save_val_results

curl -d "Puck on Diffaug Training complete" ntfy.sh/monchana