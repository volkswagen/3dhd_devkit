""" Provides the combined loss for both object and deviation detection tasks.
"""

import torch
import torch.nn as nn

# Class definitions ####################################################################################################
########################################################################################################################


class CombinedLoss(nn.Module):
    """ Combined loss function comprising both detection (or classification) and regression loss.
    """
    def __init__(self, weight_dict, weight_dict_classes=None, use_class_specific_weights=False):
        """
        For detection and classification, we apply a focal loss (see our article on map deviation detection).
        The regression loss is implemented as Huber loss.

        Args:
            weight_dict (dict:float): weight dictionary provided by config.ini
            use_class_specific_weights (bool): True if weights in weight_dict are to be applied
        """
        super(CombinedLoss, self).__init__()
        self.a_o = weight_dict['element']  # a: alpha, o: object
        self.a_n = weight_dict['no_element']  # n: no_object

        # Apply class (evaluation state) specific weights
        self.use_class_specific_weights = use_class_specific_weights
        if use_class_specific_weights:
            # classes: VER, INS, DEL, SUB
            # Order of VER, INS, DEL, SUB is defined in dataset.map_deviation.py (DeviationTypes)
            weight_dev_classes = [
                weight_dict_classes['VER'],
                weight_dict_classes['INS'],
                weight_dict_classes['DEL'],
                weight_dict_classes['SUB'],
            ]
            self.weight_dict_classes = {
                'lights': weight_dev_classes,
                'poles': weight_dev_classes,
                'signs': weight_dev_classes,
            }

        self.g = 2.0            # gamma
        self.l_dtc = 1          # weight of detection focal loss
        self.l_reg = 2          # weight of regression loss

        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.regression_features = {
            'poles': ['x', 'y', 'z', 'd'],
            'signs': ['x', 'y', 'z', 'width', 'height', 'yaw_sin', 'yaw_cos'],
            'lights': ['x', 'y', 'z', 'width', 'height', 'yaw_sin', 'yaw_cos']
        }

    def forward(self, out_dict, target_dict, norm_factors=None):
        loss_dict = {
            'poles': {},
            'signs': {},
            'lights': {},
        }

        focal2d_dict = {
            'poles': None,
            'signs': None,
            'lights': None,
        }

        # Iterate through each map element
        total_loss = 0
        for element_type, _ in out_dict.items():
            # Detection focal loss
            # mask: 1 if loss is to be evaluated, [B, 1, L, X, Y], 0 for "don't care" (see anchor generation)
            # o: object, n: no object, r: regression, y: target, p: prediction
            # B: batch, n_classes: number of classes, L: vertical layers (z dimension)
            mask = target_dict[element_type]['mask']
            y_o = target_dict[element_type]['task']     # [B, n_classes, L, X, Y]
            y_r = target_dict[element_type]['reg']
            y_n = (1 - y_o).float()
            p_o = out_dict[element_type]['task']
            p_r = out_dict[element_type]['reg']
            p_n = 1 - p_o
            a_o = self.a_o
            a_n = self.a_n
            g = self.g
            s = 10 ** -6    # for robust log computation
            batch_size, num_classes, num_layers, num_x, num_y = y_o.shape
            num_reg_features = y_r.shape[1]

            # Reshape output of NN to extract layer-based information
            # p_o: [B, features, N_x, N_y] -> [B, num_classes, N_l, N_x, N_y]
            p_o = p_o.view(batch_size, num_classes, num_layers, num_x, num_y)
            p_n = p_n.view(batch_size, num_classes, num_layers, num_x, num_y)
            p_r = p_r.view(batch_size, num_reg_features, num_layers, num_x, num_y)

            # Repeat mask to match number of classes
            if num_classes > 1:
                mask = mask.repeat(1, num_classes, 1, 1, 1)

            # Create weight matrices
            if self.use_class_specific_weights:
                weight_vec = self.weight_dict[element_type]
                a_o = torch.ones(y_o.shape, device=y_o.device)
                a_n = torch.ones(y_o.shape, device=y_o.device)     # background class
                for i_cls, w in enumerate(weight_vec):
                    a_o[:, i_cls, :, :, :] *= w * self.a_o
                    a_n[:, i_cls, :, :, :] *= w * self.a_n

            # Focal loss 3d: [B, C, L, X, Y]
            focal_loss3d = mask * y_o * -a_o * torch.pow((1 - y_o * p_o), g) * torch.log(p_o + s) + \
                           mask * y_n * -a_n * torch.pow((1 - y_n * p_n), g) * torch.log(p_n + s)

            focal_loss = torch.sum(focal_loss3d)
            loss_dict[element_type]['focal'] = focal_loss

            # Use max value over classes for vis
            focal_loss3d = torch.max(focal_loss3d, dim=1, keepdim=True)[0]      # collapse the C dimension
            focal_loss2d = torch.max(focal_loss3d, dim=2, keepdim=False)[0]     # collapse the L dimension
            focal2d_dict[element_type] = focal_loss2d

            # Regression loss L2
            # Get all regression loss contributions
            # obj: flag indicating if object is present in class (max over all classes)
            obj = torch.max(y_o, dim=1, keepdim=True)[0]
            features = self.regression_features[element_type]

            for i, feature in enumerate(features):
                pred, target = p_r[:, i, :, :, :], y_r[:, i, :, :, :]
                loss_dict[element_type][feature] = torch.sum(mask[:, 0, :, :, :] * obj[:, 0, :, :, :] *
                                                             self.smooth_l1_loss(pred, target))

            # Compute total regression loss
            reg_loss = 0
            for key, loss in loss_dict[element_type].items():
                if key != 'focal':
                    reg_loss += loss
            loss_dict[element_type]['reg'] = reg_loss

            # Compute total element loss
            loss_dict[element_type]['element'] = self.l_dtc * focal_loss + self.l_reg * reg_loss

            # Norm by num elements
            if norm_factors:
                loss_dict[element_type]['element'] *= 1/norm_factors[element_type]

            # Normalize by batch size
            for key, loss in loss_dict[element_type].items():
                loss_dict[element_type][key] = loss / batch_size

            # total_loss += focal_loss
            total_loss += loss_dict[element_type]['element']

        return total_loss, loss_dict, focal2d_dict


# Module functions #####################################################################################################
########################################################################################################################


def get_criterion(loss_type, weight_dict, weight_dict_classes, use_class_specific_weights=False):
    """ Returns the criterion.

    Args:
        loss_type (str): loss type to construct
        weight_dict (dict:float): focal loss weights (element and no_element)
        weight_dict_classes (dict:float): weights for specific classes (only for deviation detection implemented)
        use_class_specific_weights (bool): apply class specific weights

    Returns:
        criterion (nn.Module): loss object
    """
    if loss_type == "CombinedLoss":
        criterion = CombinedLoss(weight_dict, weight_dict_classes, use_class_specific_weights)
    else:
        raise LookupError(f"Undefined loss type '{loss_type}'.")

    return criterion
