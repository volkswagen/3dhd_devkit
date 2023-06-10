""" This module contains auxiliary tools to manage mlflow log archives. """

import glob
import shutil
from pathlib import Path

import yaml

from utility.system_paths import get_system_paths
from deep_learning.util.tracking import clean_mlflow_path


def update_artifact_paths(test_run=True):
    """
    Updates the artifact paths, which are stored in the metafiles of mlflow runs, to the current system.
    This is necessary after moving mlflow logs to another location, e.g. from cluster to PC
    """

    log_dir = get_system_paths()['log_dir']
    mlflow_dir = log_dir / 'mlruns'
    run_metas = glob.glob(str(mlflow_dir / '[0-9]*' / '*' / 'meta.yaml'))
    print(f"Run metas: found {len(run_metas)} files")
    count = 0

    print(f"{'=' * 50}\nConverting run metas ...")
    for meta_file in run_metas:
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        old_artifact_path = clean_mlflow_path(data['artifact_uri'])
        run_name = old_artifact_path.name
        new_artifact_path = log_dir / run_name
        artifact_uri = f"file:{str(new_artifact_path)}"
        if artifact_uri != data['artifact_uri']:
            print(f"Updating {meta_file} ({run_name})...")
            data['artifact_uri'] = artifact_uri
            if not test_run:
                with open(meta_file, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, allow_unicode=True)
            count += 1
    print('-' * 50)
    print(f"Fixed {count} meta files in total!")
    if test_run:
        print("...not really though, because this was just a test run.")
    print('=' * 50)


def remove_unfinished_runs(test_run=True):
    """
    Removes mlflow run folders that do not have a matching log/artifact folder
    """

    log_dir = get_system_paths()['log_dir']
    mlflow_dir = log_dir / 'mlruns'
    run_metas = glob.glob(str(mlflow_dir / '[0-9]*' / '*' / 'meta.yaml'))
    print(f"Run metas: found {len(run_metas)} files")
    count = 0

    print(f"{'=' * 50}\nDeleting incomplete mlflow runs ...")
    for meta_file in run_metas:
        run_folder = Path(meta_file).parent
        with open(run_folder / 'tags' / 'mlflow.runName', 'r', encoding='utf-8') as f:
            run_name = f.readline()

        # check metafile status
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data['status'] == 1 or data['lifecycle_stage'] == 'deleted':
            print(f"{run_folder}")
            print(f"[Status {data['status']}] {run_name}")
            if not test_run:
                shutil.rmtree(run_folder)
            count += 1
            continue

        # check if eval results were logged
        eval_folder = run_folder / 'metrics' / 'val' / 'eval'
        if not eval_folder.exists():
            print(f"{run_folder}")
            print(f"[Status {data['status']}] {run_name}")
            if not test_run:
                shutil.rmtree(run_folder)
            count += 1

    print(f"Deleted {count} runs")
    if test_run:
        print("... not really though, because this was just a test run.")


def main():
    # update_artifact_paths(test_run=False)
    remove_unfinished_runs()


if __name__ == "__main__":
    main()
