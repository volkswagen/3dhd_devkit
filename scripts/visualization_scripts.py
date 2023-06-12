""" Collection of methods and classes to configure and plot PR curves. """

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from dataset import helper
from deep_learning.experiments.run_evaluation import run_get_best_setting
from deep_learning.util import config_parser
from utility.system_paths import get_system_paths
from visualization.visualization_basics import convert_color_string_to_rgb


@dataclass
class PRCPlotRunSettings:
    # Settings for a single plot 'line'
    experiment_name: str
    element_type: str           # lights, poles, or signs
    color: Union[str, Tuple]    # e.g. 'blue', '#FFFFFF', (0, 0, 1)
    style: str = '-'            # '-', '--', ':', '-.'
    lw: float = 1.5             # line width
    dataset: str = 'val'        # val, test, or train
    prc_class: str = 'All'      # deviation class (MDD) or element type subclass (object detection)
    net_path = 'checkpoint'
    show_best_f1: bool = True
    plot_metrics: list = list   # f1, precision, recall (only relevant if plot_prc is False)
    plot_styles: dict = dict    # keys: f1, precision, recall (only relevant if plot_prc is False)
    f1_shift_x: float = 0.01    # 0.01 to display f1 above curve, -0.07 to display it below
    f1_shift_y: float = 0.01    # 0.01 to display f1 above curve, -0.07 to display it below


@dataclass
class PRCPlotSettings:
    # Settings for the whole plot
    plot_name: str                  # used as filename
    experiments: List[PRCPlotRunSettings]
    save_path: Path
    title: str = ''                 # title to display at the top
    save_type: str = 'svg'          # e.g., jpg, png, pdf, svg
    export: bool = True             # save as file
    plot_prc: bool = True           # (will plot plot_metric against threshold instead of pr-curve)
    compress: bool = False          # slice curve data for compressed, but less detailed plots
    x_min: float = 0.5              # x-axis lower bound
    x_max: float = 1.0              # x-axis upper bound
    y_min: float = 0.5              # y-axis lower bound
    y_max: float = 1.0              # y-axis upper bound
    height: float = 2               # Figure height in inches, approximately A4-height - 2*1.25in margin
    width: float = 3                # Figure width in inches, approximately A4-width - 2*1.25in margin
    show_x_label: bool = True       # includes "Recall" and x-axis labelling
    show_y_label: bool = True       # includes "Precision" and y-axis labelling
    font_size: int = 12             # base font size

    def plot(self):
        run_plot_prc_curves(self)


# Script section #######################################################################################################
########################################################################################################################


def run_plot_prc_curves(settings: PRCPlotSettings):
    """
    This script plots the prc curves for specified experiments. run_inference and run_evaluation required previously.
    """

    log_dir = get_system_paths()['log_dir']

    plt.rcParams.update({
        'figure.figsize': (settings.width, settings.height),
        'font.size': settings.font_size,
        'axes.labelsize': settings.font_size,  # -> axis labels
        'legend.fontsize': settings.font_size,  # -> legends
        'font.family': 'serif',  # 'lmodern',
        'text.usetex': False if settings.save_type == 'svg' and settings.export else True,
        'svg.fonttype': 'none',
    })

    fig, axis = plt.subplots(1, 1)
    plt.grid(linewidth=.2)
    axis.spines[['right']].set_visible(False)

    # Load prc data
    for run_idx, run in enumerate(settings.experiments):
        exp_name = run.experiment_name
        config_path = log_dir / exp_name / 'config.ini'
        configs = config_parser.parse_config(config_path, False)
        configs['system']['experiment'] = exp_name  # experiment name in config might differ

        file_path = log_dir / exp_name / run.dataset / run.net_path / 'prc.pickle'
        prc_data = helper.load_data_from_pickle(file_path)

        prc_metrics = prc_data[run.element_type][run.prc_class]

        recall = prc_metrics['recall']
        precision = prc_metrics['precision']
        f1 = prc_metrics['f1']
        thresholds = prc_data['thresholds']

        # Remove invalid entries
        valid = []
        for i in range(0, len(recall)):
            status = True
            if recall[i] is None or precision[i] is None:
                status = False
            valid.append(status)
        recall = [r for i, r in enumerate(recall) if valid[i]]
        precision = [p for i, p in enumerate(precision) if valid[i]]
        f1 = [f for i, f in enumerate(f1) if valid[i]]
        thresholds = [t for i, t in enumerate(thresholds) if valid[i]]

        # Color must be RGB tuple, so convert from string if necessary
        col = convert_color_string_to_rgb(run.color) if isinstance(run.color, str) else run.color
        print(col)

        best_setting = run_get_best_setting(configs, 'val', run.net_path)
        thres_best = best_setting[run.element_type][run.prc_class]['threshold']
        idx = thresholds.index(thres_best)

        f1_max = f1[idx]
        rec_max = recall[idx]
        pre_max = precision[idx]

        print(f"experiment: {exp_name}, max f1 threshold: {thres_best:.2f}, f1: {f1_max:06.3f}, rec: {rec_max:06.3f}, "
              f"pre: {pre_max:06.3f}")

        if run.show_best_f1 and settings.plot_prc:
            axis.scatter(rec_max, pre_max, marker='.', color=col, s=run.lw * 30)
            f1_text = r"$\mathrm{F}_1=" + f"{f1_max:5.2f}$"
            f1_fontsize = settings.font_size - 1
            if settings.save_type == 'svg' and settings.export:
                f1_text = f1_text.replace('$', '\$')
                f1_text = r"{\small " + f1_text + "}"
                f1_fontsize = 1
            axis.text(rec_max + run.f1_shift_x, pre_max + run.f1_shift_y, f1_text, color=col, fontsize=f1_fontsize)

        if settings.compress:
            thresholds = thresholds[::5]
            recall = recall[::5]
            precision = precision[::5]

        if settings.plot_prc:
            axis.plot(recall, precision, run.style, color=col, lw=run.lw, label=exp_name)
        else:
            metrics = {
                'f1': f1,
                'recall': recall,
                'precision': precision,
            }
            styles = {
                'f1': run.plot_styles['f1'],
                'recall': run.plot_styles['recall'],
                'precision': run.plot_styles['precision']
            }
            for metric in run.plot_metrics:
                axis.plot(thresholds, metrics[metric], styles[metric], color=col, lw=run.lw, label=exp_name)

    axis.set_xticks(np.arange(.1, 1.1, 0.1))
    axis.set_yticks(np.arange(.1, 1.1, 0.1))
    axis.set_xlim([settings.x_min, settings.x_max])
    axis.set_ylim([settings.y_min, settings.y_max])

    if settings.plot_prc:
        if settings.show_x_label:
            axis.set_xlabel('Recall')
        else:
            axis.set_xticklabels([])
        if settings.show_y_label:
            axis.set_ylabel('Precision')
        else:
            axis.set_yticklabels([])
    else:
        axis.set_xlabel('Threshold')
        axis.set_ylabel("f1(-) / recall (--) / precision (:)")

    if settings.title != '':
        if settings.save_type == 'svg' and settings.export:
            plt.title(r"{" + settings.title + '}')
        else:
            plt.title(settings.title)

    plt.show(block=True if not settings.export else False)

    if settings.export:
        file_name = f"{settings.plot_name}.{settings.save_type}"
        print(f"Saving fig as {file_name} ...")
        plt.savefig(
            os.path.join(settings.save_path, file_name),
            dpi=1000,  # Simple recommendation for publication plots
            bbox_inches='tight',  # Plot will occupy a maximum of available space
            )
        print("Saving fig... done.")
    plt.close('all')


# Run section ##########################################################################################################
########################################################################################################################


def run_generate_all_prc():
    """ Plots and saves PR curves from basic MDD runs ('dd-{s1,s2,s3}-60m') for all deviation classes. """
    # Settings
    export = False  # enables the export of PRC curves as used in our paper
    save_path = Path(r"C:\Workspace")  # path where to save created PRC curves as svg

    stages = ['s1', 's2', 's3']
    e_types = ['lights', 'signs', 'poles']
    colors = {
        's1': 'darkblue',
        's2': 'green',
        's3': 'red'
    }
    plot_styles = {
        's1': ':',
        's2': '--',
        's3': '-'
    }

    # DEVIATING PLOTS
    for e_type in e_types:
        cls = 'Deviating'
        f_shifts = {
            'lights': [(0.01, .02), (0.03, 0.01), (0.01, 0.01)],  # s1, s2, s3
            'poles': [(0.01, 0.01), (0.02, 0.013), (0.01, 0.01)],
            'signs': [(-0.02, 0.03), (-0.08, 0.04), (0.01, 0.01)]
        }
        experiments = []
        for i, stage in enumerate(stages):
            color = colors[stage]
            f_shift = f_shifts[e_type][i]
            experiments.append(PRCPlotRunSettings(
                f'dd-{stage}-60m', e_type, color, prc_class=cls, f1_shift_x=f_shift[0], f1_shift_y=f_shift[1],
                style=plot_styles[stage]))
        PRCPlotSettings(
            plot_name=f"prc-dd-default-{e_type}-{cls}",
            experiments=experiments,
            title=f"{e_type.capitalize()}" + " \\texttt{DEV}",
            y_min=0.35,
            export=export,
            save_path=save_path
        ).plot()

    # DELETION PLOTS
    for e_type in e_types:
        cls = 'DEL'
        f_shifts = {
            'lights': [(0.01, 0.03), (0.03, 0.01), (0.01, 0.01)],  # r, g, b
            'poles': [(-0.2, -0.02), (0.01, 0.01), (-0.01, 0.03)],
            'signs': [(-0.02, -0.15), (-0.06, 0.08), (0.01, 0.01)]
        }
        experiments = []
        for i, stage in enumerate(stages):
            color = colors[stage]
            f_shift = f_shifts[e_type][i]
            experiments.append(PRCPlotRunSettings(
                f'dd-{stage}-60m', e_type, color, prc_class=cls, f1_shift_x=f_shift[0], f1_shift_y=f_shift[1],
                style=plot_styles[stage]))
        PRCPlotSettings(
            plot_name=f"prc-dd-default-{e_type}-{cls}",
            experiments=experiments,
            title=f"{e_type.capitalize()}" + " \\texttt{DEL}",
            y_min=0.35,
            export=export,
            save_path=save_path
        ).plot()

    # INSERTION PLOTS
    for e_type in e_types:
        cls = 'INS'
        f_shifts = {
            'lights': [(-.2, 0.01), (-0.15, 0.09), (-0.15, -0.07)],  # s1, s2, s3
            'poles': [(-.18, 0.00), (-0.2, -0.02), (-0.1, -0.07)],
            'signs': [(-.18, -0.01), (-0.13, 0.12), (-0.11, -0.08)]
        }
        experiments = []
        for i, stage in enumerate(stages):
            color = colors[stage]
            f_shift = f_shifts[e_type][i]
            experiments.append(
                PRCPlotRunSettings(
                    f'dd-{stage}-60m', e_type, color, prc_class=cls, f1_shift_x=f_shift[0], f1_shift_y=f_shift[1],
                    style=plot_styles[stage]))
        PRCPlotSettings(
            plot_name=f"prc-dd-default-{e_type}-{cls}",
            experiments=experiments,
            title=f"{e_type.capitalize()}" + " \\texttt{INS}",
            y_min=0.35,
            export=export,
            save_path=save_path
        ).plot()

    # SUBSTITUTION PLOTS
    for e_type in e_types:
        cls = 'SUB'
        if e_type != 'poles':
            f_shifts = {
                'lights': [(-0.07, -0.1), (-0.04, -0.15), (-0.08, -0.18)],  # r, g, b
                'signs': [(-.1, -0.15), (-.03, -0.15), (-0.17, -0.08)]
            }
            experiments = []
            for i, stage in enumerate(stages):
                color = colors[stage]
                f_shift = f_shifts[e_type][i]
                experiments.append(
                    PRCPlotRunSettings(f'dd-{stage}-60m', e_type, color, prc_class=cls, f1_shift_x=f_shift[0],
                                       f1_shift_y=f_shift[1], style=plot_styles[stage]))
            PRCPlotSettings(
                plot_name=f"prc-dd-default-{e_type}-{cls}",
                experiments=experiments,
                title=f"{e_type.capitalize()}" + " \\texttt{SUB}",
                export=export,
                save_path=save_path
            ).plot()

    # VERIFICATION PLOTS
    for e_type in e_types:
        cls = 'VER'
        f_shifts = {
            'lights': [(-.1, .01), (-.11, -.03), (-0.09, 0.013)],  # s1, s2, s3
            'poles': [(-.1, .01), (-.11, -.03), (-0.11, 0.02)],
            'signs': [(-0.05, -.035), (-0.1, 0.015), (-0.095, 0.015)]
        }
        experiments = []
        for i, stage in enumerate(stages):
            color = colors[stage]
            f_shift = f_shifts[e_type][i]
            experiments.append(PRCPlotRunSettings(
                f'dd-{stage}-60m', e_type, color, prc_class=cls, f1_shift_x=f_shift[0], f1_shift_y=f_shift[1],
                style=plot_styles[stage]))
        PRCPlotSettings(
            plot_name=f"prc-dd-default-{e_type}-VER",
            experiments=experiments,
            title=f"{e_type.capitalize()}" + " \\texttt{VER}",
            y_min=0.75, y_max=1.04, x_min=0.7,
            export=export,
            save_path=save_path
        ).plot()


def main():
    run_generate_all_prc()


if __name__ == "__main__":
    main()
