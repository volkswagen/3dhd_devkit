import os

from dataset import helper
from dataset_api import map_reader


def run_debug_load_lvl():
    """
    This script validates 'online' and 'generated' map elements to detect differences between.
    -> Result: all good.
    """
    # Check map elements
    dataset_dir = r"C:\Workspace\datasets\3DHD_CityScenes\Dataset"
    map_base_dir = r"C:\Workspace\datasets\3DHD_CityScenes\HD_Map"
    partitions = ['train', 'val', 'test']
    e_types = ["Lights", "Poles", "Signs"]

    map_r = map_reader.MapReader(map_base_dir, e_types)
    map_r.read_map_json_files()

    samples = []
    for part in partitions:
        samples.extend(helper.load_from_json(os.path.join(dataset_dir, part+'.json')))

    num_samples = len(samples)
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"Processing sample: {i} / {num_samples}")
        # generated
        file_name = f"{str(sample['ID']).zfill(7)}_elements.json"
        abs_path_map_elements = os.path.join(dataset_dir, sample['run_name'], 'MapElements', file_name)
        map_elements_gen = helper.load_from_json(abs_path_map_elements)

        # online
        global_pose_wgs = (sample['loc_data']['Latitude_deg'],
                           sample['loc_data']['Longitude_deg'],
                           sample['loc_data']['Yaw_deg'])
        map_elements_on = map_r.filter_map_elements_by_location(global_pose_wgs, radius=100)

        # filter
        map_elements_gen = [e for e in map_elements_gen if 'two_sided' not in e.keys()]
        map_elements_on = [e for e in map_elements_on if 'two_sided' not in e.keys()]

        for e in map_elements_gen:
            if 'two_sided' in e.keys():
                print("Two sided elements in generated elements found!")

        if len(map_elements_gen) != len(map_elements_on):
            print(f"[Error] sample {sample['ID']}. Length mismatch. gen: {len(map_elements_gen)}, on: {len(map_elements_on)}")

        for e in map_elements_on:
            es = [ef for ef in map_elements_gen if e['id'] == ef['id']]
            if len(es) == 0:
                print(f" [Error] element not found in generated list: {e}")
            elif len(es) == 1:
                pass
            elif len(es) == 2:
                print(f"[Error] too many elements match ID: {es}")
                # raise Exception(f"Too many elements match ID: {e_gen}")

        for e in map_elements_gen:
            es = [ef for ef in map_elements_on if e['id'] == ef['id']]
            if len(es) == 0:
                print(f"[Error] element not found in online list: {e}")
            elif len(es) == 1:
                pass
            elif len(es) == 2:
                print(f"[Error] too many elements match ID: {es}")


def main():
    run_debug_load_lvl()


if __name__ == "__main__":
    main()