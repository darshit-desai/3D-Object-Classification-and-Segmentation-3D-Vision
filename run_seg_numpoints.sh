#python eval_seg.py --num_points 10 --load_checkpoint best_model
#wait
#python eval_seg.py --num_points 50 --load_checkpoint best_model
#wait
#python eval_seg.py --num_points 250 --load_checkpoint best_model
#wait
python eval_seg.py --num_points 1000 --load_checkpoint best_model
wait
python eval_seg.py --num_points 2500 --load_checkpoint best_model
wait
#python eval_seg.py --load_checkpoint best_model
#wait


