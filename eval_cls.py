import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_classify
from data_loader import get_data_loader
from model_robustness import rotate_x, rotate_y, rotate_z

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=None, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--num_workers', type=int, default=12, help='The number of threads to use for the DataLoader.')

    # Rotation arguments
    parser.add_argument('--RotationX', type=float, default=None, help='Input the amount of rotation along x-axis')
    parser.add_argument('--RotationXYZ', type=float, nargs=3, default=None, help='Input the amount of rotation along x, y, and z axes')
    parser.add_argument('--RotationXY', type=float, nargs=3, default=None, help='Input the amount of rotation along x, y, and z axes')
    parser.add_argument('--RotationYZ', type=float, nargs=2, default=None, help='Input the amount of rotation along y, and z axes')
    parser.add_argument('--RotationXZ', type=float, nargs=2, default=None, help='Input the amount of rotation along x, and z axes')

    # Cluster argument
    parser.add_argument('--cluster', type=bool, default=False, help="If true then it will assume that you're in a cluster and run the program accordingly")
    args = parser.parse_args()
    # Additional arguments based on cluster
    if args.cluster:
        parser.add_argument('--test_data', type=str, default='/fs/class-projects/fall2023/cmsc848f/c848f010/data/cls/data_test.npy')
        parser.add_argument('--test_label', type=str, default='/fs/class-projects/fall2023/cmsc848f/c848f010/data/cls/label_test.npy')
        parser.add_argument('--output_dir', type=str, default='./output/cls/')
        parser.add_argument('--main_dir', type=str, default='/fs/class-projects/fall2023/cmsc848f/c848f010/data/')
    else:
        parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
        parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
        parser.add_argument('--output_dir', type=str, default='./output/cls/')
        parser.add_argument('--main_dir', type=str, default='./data/')

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
    if args.num_points is not None:
        args.output_dir = args.output_dir+f'/num_points_{args.num_points}/'
    elif args.num_points == None:
        args.num_points = 10000
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(num_classes=args.num_cls_class)
    model.to(args.device)
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))
    test_dataloader = get_data_loader(args=args, train=False)
    correct_obj = 0
    num_obj = 0
    label_dict = {0: 'chair', 1: 'vases', 2: 'lamps'}
    rotation_flag = args.RotationX or args.RotationXYZ or args.RotationXY or args.RotationYZ or args.RotationXZ
    if rotation_flag:
        print ("Evaluating Rotated data")
    else:
        print("Evaluating Normal data")
    for i, batch in enumerate(test_dataloader):
        batch_data, batch_label = batch
        batch_data = batch_data.to(args.device)
        batch_label = batch_label.to(args.device)
        # print(batch_data.shape)
        # print(batch_label.shape)
        selected_indices = torch.randint(0, batch_data.shape[1],(args.num_points,))
        # print(selected_indices)
        # Update batch_data and batch_label with selected points
        batch_data = batch_data[:, selected_indices, :]
        # batch_label = batch_label[:, selected_indices]
        # print(batch_data.shape)
        # print(batch_label.shape)
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
                pred_label = pred_label.max(dim=1)[1]
            pred_class = [label_dict[label.item()] for label in pred_label]
            GT_class = [label_dict[label.item()] for label in batch_label]
            correct_obj += pred_label.eq(batch_label.data).cpu().sum().item()
            num_obj += batch_label.size()[0]
        else:
            with torch.no_grad():
                pred_label = model(batch_data.to(args.device))
                pred_label = pred_label.max(dim=1)[1]
            pred_class = [label_dict[label.item()] for label in pred_label]
            GT_class = [label_dict[label.item()] for label in batch_label]
            correct_obj += pred_label.eq(batch_label.data).cpu().sum().item()
            num_obj += batch_label.size()[0]
        
        for idx in range(batch_data.shape[0]):
            # if (GT_class[idx]=='chair'):
            #     gt_path_class = 'chair/'
            # elif (GT_class[idx]=='vases'):
            #     gt_path_class = 'vases/'
            # elif (GT_class[idx]=='lamps'):
            #     gt_path_class = 'lamps/'
            # if (pred_class[idx]!=GT_class[idx]):
            #     pred_path_class = gt_path_class + 'fail/'
            # new_output_dir = args.output_dir+pred_path_class
            # create_dir(new_output_dir)
            # # print(new_output_dir)
            if args.RotationX or args.RotationXYZ or args.RotationXY or args.RotationYZ or args.RotationXZ:
                if GT_class[idx] == pred_class[idx]:
                    pred_path_class = output_rot_dir + GT_class[idx] + '/'
                else:
                    pred_path_class = output_rot_dir + GT_class[idx] + '/fail/'
            else:
                if GT_class[idx] == pred_class[idx]:
                    pred_path_class = GT_class[idx] + '/'
                else:
                    pred_path_class = GT_class[idx] + '/fail/'

            new_output_dir = args.output_dir + pred_path_class
            create_dir(new_output_dir)
            viz_classify(rotated_data[idx], batch_label[idx], new_output_dir+'batchidx'+str(i)+'cls'+str(idx)+'Pred_'+str(pred_class[idx])+'_GT_'+str(GT_class[idx])+'.gif', args.device, args)
    # # Sample Points per Object
    # ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy(np.load(args.test_label))

    
    # Compute Accuracy
    test_accuracy = correct_obj / num_obj
    print ("test accuracy: {}".format(test_accuracy))

