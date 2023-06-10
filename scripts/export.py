
import numpy as np
from dataset import helper
from deep_learning.util import config_parser
from utility.system_paths import get_system_paths


def run_get_results_specific_experiments(experiment_names=None, partition='test', net_dir='checkpoint'):
    log_dir = get_system_paths()['log_dir']
    delimiter = '&'  # include whitespace if necessary (e.g., use '\t' for Excel format, or ' & ' for latex)
    for exp_name in experiment_names:
        print('*' * 50)
        print(exp_name)
        # print metrics and errors
        src_path = log_dir / exp_name / partition / net_dir
        metrics_data = helper.load_data_from_pickle(src_path / 'metrics.pickle')
        errors_data = helper.load_data_from_pickle(src_path / 'errors.pickle')
        # pprint.pprint(metrics_data)
        # pprint.pprint(errors_data)
        for e_type in metrics_data.keys():
            print(e_type.capitalize())
            metrics = metrics_data[e_type]
            classes = ['All']
            if len(list(metrics.keys())) == 1:  # deviation detection: skip class level
                metrics = metrics['All']
                classes = ['VER', 'Deviating', 'DEL', 'INS', 'SUB', 'All']
                if e_type == 'poles':
                    classes.remove('SUB')
            for cls in classes:
                # print(f"\t{cls}")
                text_keys = f"\t\tF1, RE, PR"
                text_values = [f"{metrics[cls][m]:.2f}" for m in ['f1', 'recall', 'precision']]
                if cls in errors_data[e_type].keys():
                    errors = errors_data[e_type][cls]
                    err_keys = [k for k in list(errors.keys()) if k not in ['x_vrf', 'y_vrf', 'z_vrf']]
                    for err_key in err_keys:
                        text_keys += f", {err_key}"
                        text_values += [f"{errors[err_key]:.1f}"]
                # print(text_keys)
                # print(f"\t\t{delimiter.join(text_values)}")
                print(f"{delimiter.join(text_values)}")

# return metrics_data, errors_data


def run_get_results_default(partition='test', net_dir='checkpoint'):
    """ "quick and dirty" function to list metrics/errors of several experiments in a table-like fashion for a
    convenient transfer to excel, latex, etc.
    """
    delimiter = '\t'  # include whitespace if necessary (e.g., use '\t' for Excel format, or ' & ' for latex)
    latex = False
    highlight = False
    errors = False
    log_dir = get_system_paths()['log_dir']

    study = 'pd'  # no_mods, inf_probs, pd, occ, extent, train_probs
    # MDD
    stages = ['s1', 's2', 's3']
    pc_mods = {
        'no_mods': [''],
        'pd': ['', '-pd-50', '-pd-25', '-pd-10'],
        'occ': ['', '-occ-25', '-occ-50', '-occ-75']
    }
    inf_probs = ['2-2-1', '10-10-5', '30-30-15']
    train_probs = ['10-10-5', '20-20-10', '30-30-15']
    extents = ['28m', '40m', '60m', '70m']

    if study == 'no_mods':
        # base_exp_name = '3dhd-dd-#1#2-probs-20-20-10-inf-10-10-5-60m-bs1-fix'
        base_exp_name = 'dd-#1#2-60m'
        mods1 = pc_mods[study]  # listed in rows
        mods2 = stages  # listed in columns
    elif study in ['pd', 'occ']:
        base_exp_name = 'dd-#2-30m#1'
        mods1 = pc_mods[study]  # listed in rows
        mods2 = stages  # listed in columns
    elif study == 'inf_probs':
        base_exp_name = 'f-dd-#1-probs-20-20-10-inf-#2'
        mods1 = inf_probs
        mods2 = stages
    elif study == 'extent':
        base_exp_name = 'f-dd-s3-bs1-extent-#1#2'
        mods1 = extents
        mods2 = ['', '-no-matching']
    elif study == 'train_probs':
        base_exp_name = 'f-dd-#2-probs-#1-inf-10-10-5'
        mods1 = train_probs
        mods2 = ['s2', 's3']
    else:
        raise Exception(f"Unspecified study '{study}'. Maybe a typo?")

    e_types = ['lights', 'poles', 'signs']
    dev_classes = ['VER', 'Deviating', 'DEL', 'INS', 'SUB']
    metrics = ['f1', 'recall', 'precision']
    metrics_data = {}

    for mod1 in mods1:
        metrics_data[mod1] = {}
        for mod2 in mods2:
            final_exp_name = base_exp_name.replace('#1', mod1).replace('#2', mod2)
            # exp_name = f'f-dd-{stage}{mod}-probs-20-20-10-inf-10-10-5'
            src_path = log_dir / final_exp_name / partition / net_dir
            print(f"Loading: {src_path}")
            metrics_data[mod1][mod2] = helper.load_data_from_pickle(src_path / 'metrics.pickle')
    y_labels = []
    for metric in metrics:
        for mod2 in mods2:
            y_labels.append(f'{metric} {mod2}')
    print(', '.join(y_labels))
    for e_type in e_types:
        if latex:
            print("\\midrule")
            print(f"\\multicolumn{{10}}{{c}}{{{{{e_type.capitalize()}}}}}\\\\")
            print("\\midrule")
        else:
            print(e_type.capitalize())
        for cls in dev_classes:
            if e_type == 'poles' and cls == 'SUB':
                continue
            # print(f"\t{cls}")
            for mod1 in mods1:  # inf-probs
                # print(f"\t{mod1}")
                # Find out max values in row
                f1_values = []
                for mod2 in mods2:
                    f1_values.append(metrics_data[mod1][mod2][e_type]['All'][cls]['f1'])
                i_max = np.argsort(f1_values)[-1]
                i_max_2nd = np.argsort(f1_values)[-2]

                row_values = []
                for metric in metrics:
                    for mod2 in mods2:
                        row_values.append(f1_values.append(metrics_data[mod1][mod2][e_type]['All'][cls][metric]))

                table_row = []
                i_ctr = 0
                for metric in metrics:
                    for mod2 in mods2:
                        i_ctr += 1
                        e_types_mean = e_types if cls != 'SUB' else ['lights', 'signs']
                        # value = metrics_data[mod1][mod2][e_type]['All'][cls][metric]
                        value = np.mean([metrics_data[mod1][mod2][e]['All'][cls][metric] for e in e_types_mean])
                        if latex:
                            # Table row
                            if i_ctr == 1:
                                if cls == 'VER':
                                    table_row.append(f"\\multicolumn{{1}}{{l|}}{{\\texttt{{VER}}}}")
                                elif cls == 'Deviating':
                                    table_row.append(f"\\multicolumn{{1}}{{l|}}{{\\texttt{{DEV}}}}")
                                elif cls == 'DEL':
                                    table_row.append(f"\\multicolumn{{1}}{{l|}}{{├ \\texttt{{DEL}}}}")
                                elif cls == 'INS':
                                    if e_type == 'poles':
                                        table_row.append(f"\\multicolumn{{1}}{{l|}}{{└ \\texttt{{INS}}}}")
                                    else:
                                        table_row.append(f"\\multicolumn{{1}}{{l|}}{{├ \\texttt{{INS}}}}")
                                elif cls == 'SUB':
                                    table_row.append(f"\\multicolumn{{1}}{{l|}}{{└ \\texttt{{SUB}}}}")

                            # Max val highlight
                            if np.mod(i_ctr, 3) == 0 and (i_ctr) != len(row_values):
                                if highlight:
                                    if i_ctr-1 == i_max:
                                        table_row.append(f"\\multicolumn{{1}}{{r|}}{{\\textbf{{{value:.2f}}}}}")
                                    elif i_ctr-1 == i_max_2nd:
                                        table_row.append(f"\\multicolumn{{1}}{{r|}}{{\\underline{{{value:.2f}}}}}")
                                    else:
                                        table_row.append(f"\\multicolumn{{1}}{{r|}}{{{value:.2f}}}")
                                else:
                                    table_row.append(f"\\multicolumn{{1}}{{r|}}{{{value:.2f}}}")
                            else:
                                if highlight:
                                    if i_ctr-1 == i_max:
                                        table_row.append(f"\\textbf{{{value:.2f}}}")
                                    elif i_ctr-1 == i_max_2nd:
                                        table_row.append(f"\\underline{{{value:.2f}}}")
                                    else:
                                        table_row.append(f"{value:.2f}")
                                else:
                                    table_row.append(f"{value:.2f}")
                        else:
                            table_row.append(f"{value:.2f}")

                row_str = f"{delimiter.join(table_row)}"
                if latex:
                    row_str += "\\\\"
                print(row_str)

    if not errors:
        return
    errors_data = {}
    # errors = ['x_vrf', 'y_vrf', 'z_vrf', 'distance', 'diameter', 'width', 'height', 'yaw_vrf']
    errors = ['distance', 'diameter', 'width', 'height', 'yaw_vrf']
    print(errors)
    for stage in stages:
        final_exp_name = f'dd-{stage}-60m'
        src_path = log_dir / final_exp_name / partition / net_dir
        print(f"Loading: {src_path}")
        errors_data[stage] = helper.load_data_from_pickle(src_path / 'errors.pickle')
    for e_type in e_types:
        print(e_type.capitalize())
        # for cls in dev_classes:
        # if e_type == 'poles' and cls == 'SUB':
        #     continue
        # print(f"\t{cls}")
        for stage in stages:
            # print(f"\t{stage}")
            table_row = []
            for error in errors:
                # print(errors_data[stage][e_type])
                if error not in errors_data[stage][e_type]['All'].keys():
                    continue
                value = errors_data[stage][e_type]['All'][error]
                # e_types_mean = e_types if cls != 'SUB' else ['lights', 'signs']
                # value = np.mean([metrics_data[stage][e]['All'][cls][metric] for e in e_types_mean])
                table_row.append(f"{value:.1f}")
            print(f"{delimiter.join(table_row)}")


def run_export_results():
    # Default settings
    cfg = {
        'experiment_names': [
            # '3dhd-dd-s1-probs-20-20-10-inf-10-10-5-60m-bs1',
            # '3dhd-dd-s2-probs-20-20-10-inf-10-10-5-60m-bs1',
            # '3dhd-dd-s3-probs-20-20-10-inf-10-10-5-60m-bs1'
                ],
        'partitions': ['val', 'test'],
        'net_dirs': ['checkpoint']
    }
    # Override settings with any given command-line parameters
    parser = config_parser.create_argparser_from_dict(cfg)
    args = parser.parse_args()
    config_parser.update_dict_from_args(args, cfg)
    #################

    run_get_results_default()


def main():
    run_export_results()

if __name__ == "__main__":
    main()