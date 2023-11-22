## Common evaluation for both robustness checks
python eval_seg.py --load_checkpoint best_model
wait

## Rotation angle robustness evaluation
python eval_seg.py --RotationXYZ 120.0 120.0 120.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationXYZ 180.0 180.0 180.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationXYZ 90.0 90.0 90.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationXYZ 45.0 45.0 45.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationXYZ 30.0 30.0 30.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationXYZ 15.0 15.0 15.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationX 15.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationX 30.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationX 45.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationX 90.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationX 120.0 --load_checkpoint best_model
wait
python eval_seg.py --RotationX 180.0 --load_checkpoint best_model
wait

## Number of points varying robustness check evaluation
python eval_seg.py --num_points 10 --load_checkpoint best_model
wait
python eval_seg.py --num_points 50 --load_checkpoint best_model
wait
python eval_seg.py --num_points 250 --load_checkpoint best_model
wait
python eval_seg.py --num_points 1000 --load_checkpoint best_model
wait
python eval_seg.py --num_points 2500 --load_checkpoint best_model
wait

