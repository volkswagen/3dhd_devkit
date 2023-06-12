""" Utility functions.
"""
import torch


def example_convert_to_torch(example, dtype=torch.float32, device=None) -> dict:
    """ Converts an example provided by data_loader.MddDataset to torch tensors.
    Args:
        example (dict): contains tensors to convert
        dtype (torch.dtype): target data type of torch tensors
        device (str): device where to store tensors (e.g., 'cpu', 'cuda:0, ...)

    Returns:
        example_torch (dict): example with torch tensors
    """
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels",
        "map_fm"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "num_points"]:
            example_torch[k] = torch.as_tensor(v, dtype=torch.int32, device=device)
        elif k in ['target_dict']:
            example_torch['target_dict'] = {}
            # Convert tensors in target_dict
            for element_type, sub_target_dict in example['target_dict'].items():
                sub_dict = {}
                for key, data in sub_target_dict.items():
                    sub_dict[key] = torch.as_tensor(data, dtype=torch.float32, device=device)

                example_torch['target_dict'][element_type] = sub_dict
    return example_torch
