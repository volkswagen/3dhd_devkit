""" Dataset, deviation, and occlusion data loading.
"""

import json
import os

from dataset.deviation_exporter import load_generated_deviations_and_occlusions


# Class definitions ####################################################################################################
########################################################################################################################


class MddDatasetSplitter:
    """ Loader for dataset, deviation, and occlusion files.
    """

    def __init__(self, dataset_dir, partitions=None, version=''):
        """
        Args:
            dataset_dir (str): directory of dataset files (train, val, test).json
            partitions (list[str]: partitions to load
            version (str): dataset versions
        """

        self.partitions_to_load = partitions
        if self.partitions_to_load is None:
            self.partitions_to_load = ['train', 'val', 'test']
        self.partition = {p: [] for p in self.partitions_to_load}
        self.version = version
        if len(self.version) > 0 and self.version[0] != '_':  # add leading underscore if it's not there yet
            self.version = f"_{self.version}"
        self.dataset_dir = dataset_dir

    def load_partitions(self):
        """ Loads train, val, and test JSON files containing sample data. """

        files = os.listdir(self.dataset_dir)
        for partition in self.partitions_to_load:
            filename = f'{partition}{self.version}.json'
            if filename not in files:
                raise FileNotFoundError(f"Couldn't find {filename} dataset in {self.dataset_dir}")
            else:
                with open(os.path.join(self.dataset_dir, filename), encoding='utf-8') as jf:
                    samples = json.load(jf)

                for sample in samples:
                    sample['partition'] = partition

                self.partition[partition] = samples

    def load_generated_deviations_and_occlusions(self, deviation_setting, occlusion_prob):
        """ Loads previously generated deviation and occlusion assignments.
        Respective assignments are added to the sample information from (train,val,test).json.

        Args:
            deviation_setting (str): deviation setting file to load
            occlusion_prob (float): occlusion probability file to load
        """

        partitions = [p for p in self.partitions_to_load if p in ['val', 'test']]
        for partition in partitions:
            partition_name = f"{partition}{self.version}"
            self.partition[partition] = load_generated_deviations_and_occlusions(
                self.partition[partition], partition_name, self.dataset_dir, deviation_setting, occlusion_prob)


def main():
    pass


if __name__ == "__main__":
    main()
