CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/instructblip/caption_coco_vicuna7b_train.yaml
