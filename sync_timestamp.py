import glob
import os

def get_file_timestamp(files):
    if len(files) == 0:
        return None, None
    file_path = files.pop(0)
    name = os.path.splitext(os.path.basename(file_path))[0]
    return file_path, float(name)


def main():
    data_dir_1 = r'D:\\data\\free_way_2022-06-14-16-36-06_0\\pointcloud.benewake_horn_x2_ros_node.benewake_pointcloud'
    data_dir_2 = r'D:\\data\\free_way_2022-06-14-16-36-06_0\\pointcloud.tanwaylidar_pointcloud'
    threshold = 0.05
    file_ext = '.pcd'

    files_1 = sorted(glob.glob(f'{data_dir_1}/*{file_ext}'))
    files_2 = sorted(glob.glob(f'{data_dir_2}/*{file_ext}'))
    file_1, timestamp_1 = get_file_timestamp(files_1)
    file_2, timestamp_2 = get_file_timestamp(files_2)

    remove = []
    matched = []
    while True:
        if timestamp_1 - timestamp_2 > threshold:
            remove.append(file_2)
            file_2, timestamp_2 = get_file_timestamp(files_2)
        elif timestamp_2 - timestamp_1 > threshold:
            remove.append(file_1)
            file_1, timestamp_1 = get_file_timestamp(files_1)
        else:
            matched.append((file_1, file_2))
            file_1, timestamp_1 = get_file_timestamp(files_1)
            file_2, timestamp_2 = get_file_timestamp(files_2)
        if file_1 is None or file_2 is None:
            break

    remove.extend(files_1)
    remove.extend(files_2)

    for file_name in remove:
        os.remove(file_name)


if __name__ == '__main__':
    main()
