""" Provides weights for loss computation.
"""


def get_loss_norm_by_elements(gt_elements_batched):
    """ Counts map elements to compute weights for loss computation.

    Args:
        gt_elements_batched (list[dict:list]): contains lists of map elements per element type (list of batches)

    Returns:
        norm_factors (dict:int): number of map elements per element type
    """

    norm_factors = {}

    for elements_batch in gt_elements_batched:
        for element_type, elements in elements_batch.items():
            if element_type not in norm_factors:
                norm_factors[element_type] = len(elements)
            else:
                norm_factors[element_type] += len(elements)

    for element_type, factor in norm_factors.items():
        if factor == 0:
            norm_factors[element_type] = 1

    return norm_factors
