""" This module contains auxiliary tools to manage experiment logs in general. """

import sys

sys.path.append("..")

import shutil

import yaml

from deep_learning.util import config_parser
from utility.system_paths import get_system_paths


def create_experiment_copy(experiment_to_copy, new_name, clear_mlflow_eval_logs=True, new_settings=None):
    """ Creates a copy of an experiment, including the log folder, mlflow logs, and underlying config files. If
        specified, the settings of the experiment copy can be modified, which is useful for conducting a range of
        experiments that only vary during inference.

    Args:
        experiment_to_copy (str): name of the original experiment
        new_name (str): name of the experiment copy
        clear_mlflow_eval_logs (bool): if true, removes evaluation-related mlflow logs (bc they are clumsy to override)
        new_settings (list[tuple] | None): list of new config settings to apply on the experiment copy; every new
                                           setting is a 3-tuple (str, str, Any), referring to the [0] sub-config
                                           (train, model, system), [1] config key, and [2] new value
    """

    print(f"Copying experiment '{experiment_to_copy}' as '{new_name}' ...")

    log_dir = get_system_paths()['log_dir']
    old_exp_dir = log_dir / experiment_to_copy
    new_exp_dir = log_dir / new_name

    assert old_exp_dir.exists(), f"Unknown experiment '{experiment_to_copy}'. Maybe a typo? (searched in {log_dir})"
    assert not new_exp_dir.exists(), f"Experiment '{new_name}' already exists! Please choose another name."

    # Copy experiment dir
    print("- Copying experiment directory ...")
    shutil.copytree(old_exp_dir, new_exp_dir)

    # Read config file
    print("- Loading and updating new config ...")
    config_path = new_exp_dir / 'config.ini'
    configs = config_parser.parse_config(config_path, False)
    system_config, train_config = configs['system'], configs['train']

    # Adjust configs to new experiment name
    system_config['experiment'] = new_name
    system_config['log_file_path'] = system_config['log_file_path'].replace(experiment_to_copy, new_name)

    # Apply new setting
    if new_settings is not None:
        for new_setting in new_settings:
            print(f"- Applying new setting {new_setting} ...")
            configs[new_setting[0]][new_setting[1]] = new_setting[2]

    # If mlflow is used: copy mlflow dir (find new available run_id) and adjust meta-files
    if 'mlflow' in train_config['tracking_backend']:
        mlflow_exp_dir = log_dir / 'mlruns' / system_config['mlflow_experiment_id']
        mlflow_old_dir = mlflow_exp_dir / system_config['mlflow_run_id']

        idx = 0
        new_run_id = system_config['mlflow_run_id'][:-1] + str(idx)
        mlflow_new_dir = mlflow_exp_dir / new_run_id
        while mlflow_new_dir.exists():
            print(f"- [mlflow new run] {mlflow_new_dir} already exists!")
            idx += 1
            new_run_id = system_config['mlflow_run_id'][:-len(str(idx))] + str(idx)
            mlflow_new_dir = mlflow_exp_dir / new_run_id

        print("- New mlflow run id: {new_run_id}")
        print("- Copying mlflow directory ...")
        shutil.copytree(mlflow_old_dir, mlflow_new_dir)
        system_config['mlflow_run_id'] = new_run_id

        # Adjust mlflow files
        print("- Updating new mlflow run ...")
        meta_file = mlflow_new_dir / 'meta.yaml'
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            data['artifact_uri'] = data['artifact_uri'].replace(experiment_to_copy, new_name)
            data['mlflow_run_id'] = new_run_id
            data['run_uuid'] = new_run_id
            with open(meta_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True)

        mlf_run_name_log = mlflow_new_dir / 'tags' / 'mlflow.runName'
        with open(mlf_run_name_log, 'w', encoding='utf-8') as f:
            f.write(new_name)
        mlf_run_name_log = mlflow_new_dir / 'params' / 'system' / 'experiment'
        with open(mlf_run_name_log, 'w', encoding='utf-8') as f:
            f.write(new_name)
        mlf_run_id_log = mlflow_new_dir / 'params' / 'system' / 'mlflow_run_id'
        with open(mlf_run_id_log, 'w', encoding='utf-8') as f:
            f.write(new_run_id)

        if new_settings is not None:
            for new_setting in new_settings:
                mlf_new_setting = mlflow_new_dir / 'params' / new_setting[0] / new_setting[1]
                with open(mlf_new_setting, 'w', encoding='utf-8') as f:
                    f.write(str(new_setting[2]))

        if clear_mlflow_eval_logs:
            for partition in ['train', 'val', 'test']:
                mlf_eval_dir = mlflow_new_dir / 'metrics' / partition / 'eval'
                if mlf_eval_dir.exists():
                    print(f"- Removing {mlf_eval_dir} ...")
                    shutil.rmtree(mlf_eval_dir)

    # Adjust config and save it
    print("- Saving updated config ...")
    config_parser.save_config(config_path, configs)

    print(f"Successfully copied experiment '{experiment_to_copy}' as '{new_name}'")


def rename_experiment(old_name, new_name):
    """ Renames an experiment, including the log folder, mlflow logs, and underlying config files.

    Args:
        old_name: current name of the experiment
        new_name: preferred name of the experiment
    """

    print(f"Renaming experiment '{old_name}' to '{new_name}' ...")

    log_dir = get_system_paths()['log_dir']
    old_exp_dir = log_dir / old_name
    new_exp_dir = log_dir / new_name

    assert old_exp_dir.exists(), f"Unknown experiment '{old_name}'. Maybe a typo? (searched in {log_dir})"
    assert not new_exp_dir.exists(), f"Experiment '{new_name}' already exists! Please choose another name."

    # Rename experiment dir
    print("- Renaming experiment directory ...")
    shutil.move(old_exp_dir, new_exp_dir)

    # Read config file
    print("- Loading and updating config ...")
    config_path = new_exp_dir / 'config.ini'
    configs = config_parser.parse_config(config_path, False)
    system_config, train_config = configs['system'], configs['train']

    # Adjust configs to new experiment name
    system_config['experiment'] = new_name
    system_config['log_file_path'] = system_config['log_file_path'].replace(old_name, new_name)

    # If mlflow: adjust meta-files
    if 'mlflow' in train_config['tracking_backend']:
        mlflow_exp_dir = log_dir / 'mlruns' / system_config['mlflow_experiment_id']
        run_id = system_config['mlflow_run_id']
        mlflow_run_dir = mlflow_exp_dir / run_id

        # Adjust mlflow files
        print("- Updating mlflow run ...")
        meta_file = mlflow_run_dir / 'meta.yaml'
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            data['artifact_uri'] = data['artifact_uri'].replace(old_name, new_name)
            with open(meta_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True)

        mlf_run_name_log = mlflow_run_dir / 'tags' / 'mlflow.runName'
        with open(mlf_run_name_log, 'w', encoding='utf-8') as f:
            f.write(new_name)
        mlf_run_name_log = mlflow_run_dir / 'params' / 'system' / 'experiment'
        with open(mlf_run_name_log, 'w', encoding='utf-8') as f:
            f.write(new_name)

    # Adjust config and save it
    print("- Saving updated config ...")
    config_parser.save_config(config_path, configs)

    print(f"Successfully renamed experiment '{old_name}' to '{new_name}'")


def remove_experiment(experiment_name):
    """ Completely removes an experiment from the log archives, including the log folder and mlflow logs.

    Args:
        experiment_name: name of the experiment to delete
    """

    print(f"Removing experiment '{experiment_name}' ...")

    log_dir = get_system_paths()['log_dir']
    experiment_dir = log_dir / experiment_name

    assert experiment_dir.exists(), f"Unknown experiment '{experiment_name}'. Maybe a typo? (searched in {log_dir})"

    # Read config file
    print("- Loading config ...")
    config_path = experiment_dir / 'config.ini'
    configs = config_parser.parse_config(config_path, False)
    system_config, train_config = configs['system'], configs['train']

    print(f"- Removing {experiment_dir} ...")
    shutil.rmtree(experiment_dir)

    # If mlflow: adjust meta-files
    if 'mlflow' in train_config['tracking_backend']:
        mlflow_exp_dir = log_dir / 'mlruns' / system_config['mlflow_experiment_id']
        run_id = system_config['mlflow_run_id']
        mlflow_run_dir = mlflow_exp_dir / run_id

        if mlflow_run_dir.exists():
            print(f"- Removing {mlflow_run_dir} ...")
            shutil.rmtree(mlflow_run_dir)

    print(f"Successfully removed experiment '{experiment_name}'")


def main():
    # Example calls
    create_experiment_copy('dd-s3-30m', 'dd-s3-30m-copy', new_settings=[('train', 'occlusion_prob', 0.75)])
    rename_experiment('dd-s3-30m-copy', 'dd-s3-30m-copy-renamed')
    remove_experiment('dd-s3-30m-copy-renamed')


if __name__ == "__main__":
    main()
