 ##  1. Code Running Instructions

### For Problem 1 and 2 training

```bash
python train.py --task cls --num_workers 12
python train.py --task seg --num_workers 12 
```
The models will be stored in the checkpoint folder

###  1.4. For Running evaluation for Problem 1, 2 and 3

Run the below bash file in the terminal

```bash
./run_seg.sh
./run_cls.sh
```
The results will be stored in the results/ folder

