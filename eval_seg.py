import numpy as np
import argparse
import os
import csv
import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg
from model_robustness import rotate_x, rotate_y, rotate_z

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")
    parser.add_argument('--task', type=str, default="seg", help='The task: cls or seg')
    parser.add_argument('--cluster', type=bool, default=False, help="If true then it will assume that you're in cluster and run the program accordingly")
    # Rotation arguments
    parser.add_argument('--RotationX', type=float, default=0.0, help='Input the amount of rotation along x-axis')
    parser.add_argument('--RotationXYZ', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Input the amount of rotation along x, y, and z axes')
    parser.add_argument('--RotationXY', type=float, nargs=3, default=[0.0, 0.0], help='Input the amount of rotation along x, y, and z axes')
    parser.add_argument('--RotationYZ', type=float, nargs=2, default=[0.0, 0.0], help='Input the amount of rotation along y, and z axes')
    parser.add_argument('--RotationXZ', type=float, nargs=2, default=[0.0, 0.0], help='Input the amount of rotation along x, and z axes')
    args = parser.parse_args()
    if (args.cluster == True):
        parser.add_argument('--test_data', type=str, default='/fs/class-projects/fall2023/cmsc848f/c848f010/data/seg/data_test.npy')
        parser.add_argument('--test_label', type=str, default='/fs/class-projects/fall2023/cmsc848f/c848f010/data/seg/label_test.npy')
        parser.add_argument('--output_dir', type=str, default='./output/seg')
        parser.add_argument('--main_dir', type=str, default='/fs/class-projects/fall2023/cmsc848f/c848f010/data/')
    else:    
        parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
        parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
        parser.add_argument('--output_dir', type=str, default='./output/seg')
        parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--num_workers', type=int, default=12, help='The number of threads to use for the DataLoader.')
    return parser

def apply_rotations(batch_data, args):
    rotated_data = batch_data.clone()
    if args.RotationX:
        rotated_data = rotate_x(rotated_data, args.RotationX, args)
    elif args.RotationXYZ:
        rotated_data = rotate_x(rotated_data, args.RotationXYZ[0], args)
        rotated_data = rotate_y(rotated_data, args.RotationXYZ[1], args)
        rotated_data = rotate_z(rotated_data, args.RotationXYZ[2], args)
    elif args.RotationXY:
        rotated_data = rotate_x(rotated_data, args.RotationXY[0], args)
        rotated_data = rotate_y(rotated_data, args.RotationXY[1], args)
    elif args.RotationYZ:
        rotated_data = rotate_y(rotated_data, args.RotationYZ[0], args)
        rotated_data = rotate_z(rotated_data, args.RotationYZ[1], args)
    elif args.RotationXZ:
        rotated_data = rotate_x(rotated_data, args.RotationXZ[0], args)
        rotated_data = rotate_z(rotated_data, args.RotationXZ[1], args)
    return rotated_data


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)
    
    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(args, args.num_seg_class)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval().to(args.device)
    print ("successfully loaded checkpoint from {}".format(model_path))
    test_dataloader = get_data_loader(args=args, train=False)
    correct_obj = 0
    num_obj = 0
    accuracies = []
    rotation_flag = args.RotationX or args.RotationXYZ or args.RotationXY or args.RotationYZ or args.RotationXZ
    if rotation_flag:
        print ("Evaluating Rotated data")
    else:
        print("Evaluating Normal data")
    # label_dict = {0: 'chair', 1: 'vases', 2: 'lamps'}
    for i, batch in enumerate(test_dataloader):
        batch_data, batch_label = batch
        batch_data = batch_data.to(args.device)
        batch_label = batch_label.to(args.device)
        rotated_data = apply_rotations(batch_data, args)

        # Generate dynamic output directory based on rotations, angle values, and class only if any rotation argument is passed
        if rotation_flag:
            if args.RotationX:
                output_rot_dir = f'RotationX_{args.RotationX}/'
            elif args.RotationXYZ:
                output_rot_dir = f'RotationXYZ_{args.RotationXYZ[0]}_{args.RotationXYZ[1]}_{args.RotationXYZ[2]}/'
            elif args.RotationXY:
                output_rot_dir = f'RotationXY_{args.RotationXY[0]}_{args.RotationXY[1]}/'
            elif args.RotationYZ:
                output_rot_dir = f'RotationYZ_{args.RotationYZ[0]}_{args.RotationYZ[1]}/'
            elif args.RotationXZ:
                output_rot_dir = f'RotationXZ_{args.RotationXZ[0]}_{args.RotationXZ[1]}/'
            # ------ TO DO: Make Prediction ------
            with torch.no_grad():
                pred_label = model(rotated_data.to(args.device))
                pred_label = pred_label.max(dim=-1)[1]
            # print("Pred label size= ", pred_label.shape)
            # print("batchlabel data size = ", batch_label.data.shape)
            correct_obj += pred_label.eq(batch_label.data).cpu().sum().item()
            # num_obj += batch_label.size()[0]
            num_obj += batch_label.view([-1,1]).size()[0]
            # Calculate and print accuracy for each object
            accuracy = pred_label.eq(batch_label.data).float().mean(dim=1)
            accuracies.extend(accuracy.cpu().numpy())
        else:
            with torch.no_grad():
                pred_label = model(batch_data.to(args.device))
                pred_label = pred_label.max(dim=-1)[1]
            # print("Pred label size= ", pred_label.shape)
            # print("batchlabel data size = ", batch_label.data.shape)
            correct_obj += pred_label.eq(batch_label.data).cpu().sum().item()
            # num_obj += batch_label.size()[0]
            num_obj += batch_label.view([-1,1]).size()[0]
            # Calculate and print accuracy for each object
            accuracy = pred_label.eq(batch_label.data).float().mean(dim=1)
            accuracies.extend(accuracy.cpu().numpy())
        for idx, acc in enumerate(accuracy):
            print(f'Object {i * args.batch_size + idx}: Accuracy {acc.item()}')
            # Save visualizations for objects with accuracy >= 50%
            if acc.item() >= 0.5:
                if rotation_flag:
                    success_dir = args.output_dir + output_rot_dir + 'success/'
                else:
                    success_dir = os.path.join(args.output_dir, 'success')
                create_dir(success_dir)
                viz_seg(batch_data[idx], pred_label[idx], os.path.join(success_dir, f"pred_{i}_{idx}.gif"), args.device, args)
                viz_seg(batch_data[idx], batch_label[idx], os.path.join(success_dir, f"gt_{i}_{idx}.gif"), args.device, args)
            # Save visualizations for objects with accuracy < 50%
            else:
                if rotation_flag:
                    fail_dir = args.output_dir + output_rot_dir + 'fail/'
                else:
                    fail_dir = os.path.join(args.output_dir, 'fail')
                create_dir(fail_dir)
                viz_seg(batch_data[idx], pred_label[idx], os.path.join(fail_dir, f"pred_{i}_{idx}.gif"), args.device, args)
                viz_seg(batch_data[idx], batch_label[idx], os.path.join(fail_dir, f"gt_{i}_{idx}.gif"), args.device, args)




        # for idx, acc in enumerate(accuracy):
        #     print(f'Object {i * args.batch_size + idx}: Accuracy {acc.item()}')
        # for idx in range(batch_data.shape[0]):
        #     # if (pred_class[idx]=='chair'):
        #     #     path_class = 'chair/'
        #     # elif (pred_class[idx]=='vases'):
        #     #     path_class = 'vases/'
        #     # elif (pred_class[idx]=='lamps'):
        #     #     path_class = 'lamps/'
        #     # new_output_dir = args.output_dir + '/seg/'
        #     # print(new_output_dir)
        #     # viz_seg(batch_data[idx], batch_label[idx], "{}/gt_"+idx+".gif".format(args.output_dir, args.exp_name), args.device)
        #     viz_seg(batch_data[idx], pred_label[idx], "{}/pred_{}".format(args.output_dir, idx)+str(i)+".gif", args.device, args)
        #     viz_seg(batch_data[idx], batch_label[idx], "{}/gt_{}".format(args.output_dir, idx)+str(i)+".gif", args.device, args)
        #     # viz_seg(batch_data[idx], pred_label[idx], "{}/pred_"+idx+".gif".format(args.output_dir, args.exp_name), args.device)
        

    # Sample Points per Object
    # ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy((np.load(args.test_label))[:,ind])


    # test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    test_accuracy = correct_obj / num_obj
    print ("test accuracy: {}".format(test_accuracy))
    if rotation_flag:
        csv_file_p = args.output_dir + output_rot_dir
        csv_file_path = os.path.join(csv_file_p, 'accuracies.csv')
    else:
        csv_file_path = os.path.join(args.output_dir, 'accuracies.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Object Index', 'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, acc in enumerate(accuracies):
            writer.writerow({'Object Index': i, 'Accuracy': acc})

    print(f'Accuracies exported to {csv_file_path}')

    # Visualize Segmentation Result (Pred VS Ground Truth)
    # viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    # viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
