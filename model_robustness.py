import torch
import numpy as np
def rotate_x(input_tensor, angle, args):
    """
    Rotate the input tensor around the x-axis.

    Parameters:
    - input_tensor: torch.Tensor or numpy array, input data to be rotated
    - angle: float, rotation angle in degrees

    Returns:
    - rotated_tensor: torch.Tensor, rotated input data
    """
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor).to(args.device)

    angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32)).to(args.device)
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_rad), -torch.sin(angle_rad)],
        [0, torch.sin(angle_rad), torch.cos(angle_rad)]
    ], dtype=torch.float32).to(args.device)

    rotated_tensor = torch.matmul(input_tensor, rotation_matrix)
    return rotated_tensor

def rotate_y(input_tensor, angle, args):
    """
    Rotate the input tensor around the y-axis.

    Parameters:
    - input_tensor: torch.Tensor or numpy array, input data to be rotated
    - angle: float, rotation angle in degrees

    Returns:
    - rotated_tensor: torch.Tensor, rotated input data
    """
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor).to(args.device)

    angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32)).to(args.device)
    rotation_matrix = torch.tensor([
        [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
        [0, 1, 0],
        [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
    ], dtype=torch.float32).to(args.device)

    rotated_tensor = torch.matmul(input_tensor, rotation_matrix)
    return rotated_tensor

def rotate_z(input_tensor, angle, args):
    """
    Rotate the input tensor around the z-axis.

    Parameters:
    - input_tensor: torch.Tensor or numpy array, input data to be rotated
    - angle: float, rotation angle in degrees

    Returns:
    - rotated_tensor: torch.Tensor, rotated input data
    """
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor).to(args.device)

    angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32)).to(args.device)
    rotation_matrix = torch.tensor([
        [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
        [torch.sin(angle_rad), torch.cos(angle_rad), 0],
        [0, 0, 1]
    ], dtype=torch.float32).to(args.device)

    rotated_tensor = torch.matmul(input_tensor, rotation_matrix)
    return rotated_tensor