import torch
import numpy as np


def rotate(input, degrees=90):
    """(..., H, W) input expected"""
    if degrees == 90:
        return input.transpose(-2, -1).flip(-2)
    if degrees == 180:
        return input.flip(-2).flip(-1)
    if degrees == 270:
        return input.transpose(-2, -1).flip(-1)


def transpose(input):
    """(..., H, W) input expected"""
    return input.transpose(-2, -1)


def apply_rotate_transpose(input, rot90, rot180, rot270, transpose):
    transformed: torch.Tensor = input.clone()
    to_rot90 = rot90.to(input.device)
    transformed[to_rot90] = rotate(input[to_rot90], degrees=90)
    to_rot180 = rot180.to(input.device)
    transformed[to_rot180] = rotate(input[to_rot180], degrees=180)
    to_rot270 = rot270.to(input.device)
    transformed[to_rot270] = rotate(input[to_rot270], degrees=270)
    to_transpose = transpose.to(input.device)
    transformed[to_transpose] = transformed[to_transpose].transpose(-2, -1)
    return transformed


def gpu_da(x_data, y_data, gpu_da_params):
    with torch.no_grad():
        bs = y_data.size(0)

        no_dihedral_p = gpu_da_params[0]
        transpose, rot90, rot180, rot270 = get_transpose_rot_boolean_lists(bs, no_dihedral_p)

        transpose, rot90, rot180, rot270 = (
            torch.tensor(transpose),
            torch.tensor(rot90),
            torch.tensor(rot180),
            torch.tensor(rot270),
        )
        debug_show = False
        if debug_show:
            raise NotImplementedError()
        else:
            x_data = apply_rotate_transpose(x_data, rot90, rot180, rot270, transpose)
            y_data = apply_rotate_transpose(y_data, rot90, rot180, rot270, transpose)
            return x_data, y_data


def get_transpose_rot_boolean_lists(bs, no_dihedral_p):
    """
    In no_dihedral_p % do nothing, in (1-no_dihedral_p) % / 7 do one of the 7 possible transpose/rot combinations.
    """
    transpose, rot90, rot180, rot270 = [False] * bs, [False] * bs, [False] * bs, [False] * bs
    perc_for_each_combination = (1 - no_dihedral_p) / 7
    for k in range(bs):
        rand_float = np.random.random()
        if rand_float < perc_for_each_combination:
            rot90[k] = True
        elif rand_float < 2 * perc_for_each_combination:
            rot180[k] = True
        elif rand_float < 3 * perc_for_each_combination:
            rot270[k] = True
        elif rand_float < 4 * perc_for_each_combination:
            rot90[k] = True
            transpose[k] = True
        elif rand_float < 5 * perc_for_each_combination:
            rot180[k] = True
            transpose[k] = True
        elif rand_float < 6 * perc_for_each_combination:
            rot270[k] = True
            transpose[k] = True
        elif rand_float < 7 * perc_for_each_combination:
            transpose[k] = True
        else:
            pass
    # print(f"transpose: {sum(transpose)}/{bs}, 90degree rot: {sum(rot90)}/{bs}, 180degree rot: {sum(rot180)}/{bs}, 270degree rot: {sum(rot270)}/{bs}")
    return transpose, rot90, rot180, rot270
