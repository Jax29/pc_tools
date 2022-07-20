import argparse
import glob
import os
import random
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from numpy.typing import DTypeLike
from open3d.visualization import VisualizerWithKeyCallback

"""
python pc_vis.py /path/to/point/cloud/dir/or/file

按键说明：
- 鼠标左键拖拽：调整视角
- 鼠标中键拖拽：平移点云
- 空格：暂停/播放
- A：前一帧
- D：后一帧
- S：保存图像，默认保存到当前目录/saved/
- Q：退出

"""


class PointCloudVisualizer:
    def __init__(
        self,
        pc_path: str,
        ext: Optional[str] = None,
        num_samples: Optional[int] = None,
        save_dir: str = './saved/'
    ):
        if os.path.isfile(pc_path):
            self.pc_files = [pc_path]
        else:
            self.pc_files = load_files(pc_path, ext)
        if num_samples is not None:
            self.pc_files = random.sample(self.pc_files, num_samples)
        self.num_files = len(self.pc_files)
        if self.num_files == 0:
            raise ValueError(f"No supported point cloud file in {pc_path}.")

        self.save_dir = save_dir

        self.vis = VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord(" "), self.on_pause)
        self.vis.register_key_callback(ord("Q"), self.on_stop)
        self.vis.register_key_callback(ord("A"), self.on_prev)
        self.vis.register_key_callback(ord("D"), self.on_next)
        self.vis.register_key_callback(ord("S"), self.on_save_image)

        self.pcd_buffer = read_open3d(self.pc_files[0])

        self.play = True
        self.exit = False
        self.idx = 0

    # noinspection PyUnusedLocal
    def on_pause(self, vis: VisualizerWithKeyCallback):
        self.play = not self.play

    # noinspection PyUnusedLocal
    def on_stop(self, vis: VisualizerWithKeyCallback):
        self.exit = True

    # noinspection PyUnusedLocal
    def on_next(self, vis: VisualizerWithKeyCallback):
        if self.idx < self.num_files - 1:
            self.idx += 1
            self.play = False
            self.render_pc_file(self.pc_files[self.idx])

    # noinspection PyUnusedLocal
    def on_prev(self, vis: VisualizerWithKeyCallback):
        if self.idx > 0:
            self.idx -= 1
            self.play = False
            self.render_pc_file(self.pc_files[self.idx])

    # noinspection PyUnusedLocal
    def on_save_image(self, vis: VisualizerWithKeyCallback):
        self.play = False
        file_path = os.path.join(self.save_dir, f'{self.idx}.jpg')
        os.makedirs(self.save_dir, exist_ok=True)
        self.vis.capture_screen_image(file_path)
        print(f"Saved to {file_path}.")

    def run(
        self,
        front: Tuple[int] = (-5, 0, 5),
        lookat: Tuple[int] = (5, 0, 0),
        up: Tuple[int] = (0, 0, 1),
        zoom: float = 0.5
    ):
        self.vis.create_window()
        self.vis.get_render_option().background_color = np.zeros(3)
        self.vis.get_render_option().point_size = 1.0

        self.render_pc_file(self.pc_files[0], first=True)
        self.vis.get_view_control().set_front(front)
        self.vis.get_view_control().set_lookat(lookat)
        self.vis.get_view_control().set_up(up)
        self.vis.get_view_control().set_zoom(zoom)

        self.idx = 0
        self.play = True
        self.exit = False

        while self.idx < len(self.pc_files):
            self.vis.poll_events()
            if self.exit:
                break
            if (not self.play) or (self.idx == self.num_files - 1):
                continue

            self.idx += 1
            self.render_pc_file(self.pc_files[self.idx])

        self.vis.destroy_window()

    def render_pc_file(self, pc_file: str, first: bool = False):
        pcd = read_open3d(pc_file)
        self.pcd_buffer.points = pcd.points
        self.pcd_buffer.paint_uniform_color(np.ones(3))
        if first:
            self.vis.add_geometry(self.pcd_buffer)
        else:
            self.vis.update_geometry(self.pcd_buffer)
        self.vis.update_renderer()


def load_files(pc_dir: str, ext: Optional[str] = None):
    if ext is not None:
        pc_files = glob.glob(os.path.join(pc_dir, '*' + ext))
    else:
        pc_files = glob.glob(os.path.join(pc_dir, '*.bin'))
        pc_files.extend(glob.glob(os.path.join(pc_dir, '*.pcd')))
    pc_files = sorted(pc_files)
    print(f"Found {len(pc_files)} files.")
    return pc_files


def read_open3d(file_path: str, bin_dtype: DTypeLike = np.float32, bin_num_columns: int = 4) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(file_path)[1]
    if ext == '.pcd':
        return o3d.io.read_point_cloud(file_path)
    elif ext == '.bin':
        data = np.fromfile(file_path, dtype=bin_dtype).reshape((-1, bin_num_columns))
        points = data[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd


def main():
    args = parse_args()
    vis = PointCloudVisualizer(
        pc_path=args.pc_path
    )
    vis.run()


def parse_args():
    parser = argparse.ArgumentParser(description='Point cloud visualizer.')
    parser.add_argument(dest='pc_path', type=str, default=None, help="Point cloud directory or file path.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
