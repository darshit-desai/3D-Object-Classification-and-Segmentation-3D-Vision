 #  1. Code Running Instructions

 For referring the detailed report and results follow this webpage : https://darshit-desai.github.io/3D-Object-Classification-and-Segmentation-3D-Vision/

## For Problem 1 and 2 training

```bash
python train.py --task cls --num_workers 12
python train.py --task seg --num_workers 12 
```
The models will be stored in the checkpoint folder

## For Running evaluation for Problem 1, 2 and 3

Run the below bash file in the terminal

```bash
./run_seg.sh
./run_cls.sh
```
The results will be stored in the results/ folder

