""" Visualization used for evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
from dataset.map_deviation import MapDeviation
from utility import transformations

# Module functions #####################################################################################################
########################################################################################################################


def vis_focal_loss_2d(focal_loss_2d, show_plot=True):
    """ Visualizes the focal loss as 2D heat map.

    Args:
        focal_loss_2d (float tensor): focal loss in 2D as [B, C, X, Y] ([batch, channel, num_x, num_y])
        show_plot (bool): turn on immediate displaying of plot (false if images are only logged)

    Returns:
        plt (matplotlib.pyplot): matplotlib object
    """
    if not show_plot:
        plt.ioff()  # turn off interactive mode to only show figures when calling # plt.show()

    focal_loss_2d = focal_loss_2d[0, 0, :, :]  # [B, C, X, Y]
    focal_loss_2d = np.swapaxes(focal_loss_2d, 0, 1)

    cmap = plt.get_cmap('plasma')
    cmap.colors[0] = [0, 0, 0]

    plt.imshow(focal_loss_2d, cmap=cmap)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # if show_plot:
    #     plt.show()

    return plt


def vis_prediction_target_loss(output_dict, target_dict, focal_loss_2d, show_plot=True):
    """ Visualizes predicted task, task target, and mask tensors as 2D heatmaps.

    Task tensor is shaped as: [B, C, Z, X, Y] ([batch, channel, num_z, num_x, num_y]).

    Args:
        output_dict (dict:tensor): dict of tensors (task, mask, anc, reg) for a specific element type
        target_dict (dict:tensor): dict of respective target tensors
        focal_loss_2d (float tensor): 2D focal loss
        show_plot (bool): turn on immediate displaying of plot (false if images are only logged)

    Returns:
        fig (matplotlib figure): created 2D figure
    """
    if not show_plot:
        plt.ioff()  # turn off interactive mode to only show figures when calling # plt.show()

    focal_loss_2d = focal_loss_2d[0, 0, :, :]  # [B, C, X, Y]
    # focal_loss_2d = np.swapaxes(focal_loss_2d, 0, 1)

    task = output_dict['task']
    task_target = target_dict['task']
    mask = target_dict['mask']

    # task: [B, C, Z, X, Y]
    task = task[0, :, :, :, :]  # -> [C, Z, X, Y]
    task_target = task_target[0, :, :, :, :]
    mask = mask[0, :, :, :, :]

    # Collapse class dimension
    task = np.max(task, axis=0, keepdims=False)
    task_target = np.max(task_target, axis=0, keepdims=False)
    mask = np.squeeze(mask, axis=0)

    # Collapse L dimension
    task = np.max(task, axis=0, keepdims=False)
    task_target = np.max(task_target, axis=0, keepdims=False)
    mask = np.min(mask, axis=0, keepdims=False)

    plt.rcParams['figure.constrained_layout.use'] = True
    fig, axs = plt.subplots(1, 4)

    cmap = plt.get_cmap('plasma')
    cmap.colors[0] = [0, 0, 0]

    axs[0].imshow(task_target, cmap=cmap)
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)

    axs[1].imshow(task, cmap=cmap)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    axs[2].imshow(mask, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

    axs[3].imshow(focal_loss_2d, cmap=cmap)
    axs[3].get_xaxis().set_visible(False)
    axs[3].get_yaxis().set_visible(False)

    # if show_plot:
    #     plt.show()

    return fig


def vis_nms_evaluation(predictions_lists, analysis, element_type, fm_extent, show_plot=True):
    """ Visualizes predictions (obtained form NMS) according association state (TP, FP, FN) as 2D images.

    Args:
        predictions_lists (dict:list): contains a list of predictions per element type (keys)
        analysis (dict:dict): analysis result provided by evaluation per element type
        element_type (str): element type for which to generate the 2D visualization
        fm_extent (list[list]): [[xmin, xmax], [ymin, ymax]] as floats
        show_plot (bool): turn on immediate displaying of plot (false if images are only logged)

    Returns:
        fig (matplotlib figure): created 2D figure
    """

    x_min = fm_extent[0][0]
    x_max = fm_extent[0][1]
    y_min = fm_extent[1][0]
    y_max = fm_extent[1][1]

    # Plot ground truth vs. classification
    plt.rcParams['figure.constrained_layout.use'] = True
    if not show_plot:
        plt.ioff()  # turn off interactive mode to only show figures when calling # plt.show()

    fig, axis = plt.subplots(1, 1, figsize=(16, 6), sharex=True, sharey=True)
    gt_elements = analysis[element_type]['gt_objects']
    elements_eval = analysis[element_type]['predictions']

    if len(elements_eval) > 0 and isinstance(elements_eval[0], MapDeviation):
        elements_eval = [e.get_most_recent() for e in elements_eval]
    if len(gt_elements) > 0 and isinstance(gt_elements[0], MapDeviation):
        gt_elements = [e.get_most_recent() for e in gt_elements]

    # Mapped gt_elements in blue, tp in green, fp in yellow, fn in red
    gt_elements_mapped = [p for p in gt_elements if 'mapped_flag' in p.keys()]  # blue
    gt_elements_fn = [p for p in gt_elements if 'mapped_flag' not in p.keys()]  # red

    elements_tp = [p for p in elements_eval if p['eval_class'] == 'TP']  # green
    elements_fp = [p for p in elements_eval if p['eval_class'] == 'FP']  # cyan

    predictions = predictions_lists[element_type]

    plot_data = [predictions, gt_elements_mapped, gt_elements_fn, elements_tp, elements_fp]
    colors = [[.8, .8, .8], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1]]

    for obj_list, color in zip(plot_data, colors):
        if not obj_list:
            continue

        # Plot poles
        if element_type == 'poles':
            x_p = [p['x_vrf'] for p in obj_list]
            y_p = [p['y_vrf'] for p in obj_list]
            d_p = [p['diameter'] for p in obj_list]

            offsets = list(zip(x_p, y_p))
            axis.add_collection(EllipseCollection(widths=d_p, heights=d_p, angles=0, units='xy',
                                                  facecolors='none', edgecolors=color, offsets=offsets,
                                                  transOffset=axis.transData))

        # Plot signs
        if element_type == 'signs':
            for obj in obj_list:
                x = obj['x_vrf']
                y = obj['y_vrf']
                width = obj['width']
                yaw = obj['yaw_vrf']

                a = np.array([x - width / 2, y, 0])
                b = np.array([x + width / 2, y, 0])
                seg_points = np.column_stack((a, b))  # 2 x 3
                centroid = (x, y, 0)
                # seg_points: [3, N(points)]
                seg_points = transformations.rotate_points(seg_points, yaw, centroid=centroid)
                seg_points = seg_points[0:2, :]  # take only x and y coordinates
                axis.plot(seg_points[0, :], seg_points[1, :], linewidth=1, color=color)

        if element_type == 'lights':
            for obj in obj_list:
                x_center = obj['x_vrf']
                y_center = obj['y_vrf']
                size = obj['width']
                yaw = -obj['yaw_vrf'] + 90  # adjust VRF rotation to matplotlib

                # Define corner points of bounding cube
                points_list = []
                x_coords = [size / 2, -size / 2]
                y_coords = [size / 2, -size / 2]
                points_list.append(np.array([x_coords[0], y_coords[0], 0]))
                points_list.append(np.array([x_coords[0], y_coords[1], 0]))
                points_list.append(np.array([x_coords[1], y_coords[1], 0]))
                points_list.append(np.array([x_coords[1], y_coords[0], 0]))
                # Get center front point (used for arrow)
                points_list.append(np.array([x_coords[0], 0, 0]))

                # Rotate (in VRF), transpose and shift points
                points_norm = np.column_stack(points_list)
                points_norm = transformations.rotate_points(points_norm, yaw)
                points_norm = points_norm.T
                points = points_norm + np.array([x_center, y_center, 0])

                # Plot rectangle
                first_point = np.reshape(points[0], (1, 3))
                rectangle = np.append(points[:4], first_point, axis=0)  # add first point to close rectangle
                axis.plot(rectangle[:, 0], rectangle[:, 1], linewidth=1, color=color)

                # Plot arrow
                pos = [x_center, y_center]
                scale = 1.25
                ori = [x * scale for x in points_norm[-1]]
                axis.arrow(pos[0], pos[1], ori[0], ori[1], width=0.05, linewidth=1, color=color)

    # Set limits and plot
    axis.set_xlim((x_min, x_max))
    axis.set_ylim((y_min, y_max))
    axis.set_aspect('equal', adjustable='box')

    if show_plot:
        plt.show()

    return fig
