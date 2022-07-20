from abc import ABC, abstractmethod
import os
from tokenize import Pointfloat
from turtle import st
import numpy as np
import open3d as o3d
from numpy.typing import NDArray, DTypeLike
import glob
from tqdm import tqdm


class PointCloud(ABC):
    @property
    @abstractmethod
    def points(self) -> NDArray:
        pass

    @property
    @abstractmethod
    def intensity(self) -> NDArray:
        pass

    @property
    @abstractmethod
    def has_intensity(self) -> bool:
        pass

    def transform_points(self, matrix: NDArray) -> NDArray:
        points = self.points
        points_h = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
        transformed = np.matmul(matrix, points_h.T)
        return transformed[:3].T

    def to_open3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        return pcd

    def to_pcd(self, file_path: str, write_ascii: str = False, compressed: str = False):
        pcd = self.to_open3d()
        o3d.io.write_point_cloud(file_path, pcd, write_ascii=write_ascii, compressed=compressed)

    def to_bin(
        self,
        file_path: str,
        with_intensity: bool = True,
        normalize_intensity: bool = True,
        dtype: DTypeLike = np.float32
    ):
        data = [self.points]
        if with_intensity:
            intensity = self.intensity
            if normalize_intensity:
                intensity /= 255.0
            data.append(intensity.reshape(-1, 1))
        data = np.hstack(data).astype(dtype)
        data.tofile(file_path)

    def __repr__(self) -> str:
        return f"PointCloud with {len(self.points)} points."

    @staticmethod
    def from_file(file_path: str) -> 'PointCloud':
        if file_path.endswith('.bin'):
            return PointCloud.from_bin(file_path)
        elif file_path.endswith('.pcd'):
            return PointCloud.from_pcd(file_path)
        else:
            raise ValueError(f"Unrecognized file format: {file_path}.")

    @staticmethod
    def from_pcd(file_path: str) -> 'PointCloud':
        pc = PcdPointCloud.from_pcd(file_path)
        return pc

    @staticmethod
    def from_bin(file_path: str) -> 'PointCloud':
        pc = BinPointCloud.from_bin(file_path)
        return pc

class BinPointCloud(PointCloud):
    def __init__(self, data: NDArray):
        self.data = data

    @property
    def points(self) -> NDArray:
        return self.data[:, :3].copy()

    @property
    def intensity(self) -> NDArray:
        return self.data[:, 3].copy()

    @property
    def has_intensity(self) -> bool:
        return self.data.shape[1] >= 4

    @staticmethod
    def from_bin(file_path: str, num_columns: int = 4, dtype: DTypeLike = np.float32) -> 'BinPointCloud':
        data = np.fromfile(file_path, dtype=dtype).reshape(-1, num_columns)
        return BinPointCloud(data)

######################################
import struct
from dataclasses import dataclass
from enum import Enum
from typing import List
from typing import Tuple

import lzf
import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing.io import BinaryIO

NP_PCD_TYPE_MAPPINGS = [
    (np.float32, ('F', 4)), (np.float64, ('F', 8)),
    (np.uint8, ('U', 1)), (np.uint16, ('U', 2)), (np.uint32, ('U', 4)), (np.uint64, ('U', 8)),
    (np.int16, ('I', 2)), (np.int32, ('I', 4)), (np.int64, ('I', 8))
]
NP_TYPE_TO_PCD_TYPE = dict(NP_PCD_TYPE_MAPPINGS)
PCD_TYPE_TO_NP_TYPE = dict((q, p) for (p, q) in NP_PCD_TYPE_MAPPINGS)


class DataType(Enum):
    BINARY = 1
    BINARY_COMPRESSED = 2
    ASCII = 3


@dataclass
class Metadata:
    version: str
    fields: List[str]
    size: List[int]
    type: List[str]
    count: List[int]
    width: int
    height: int
    viewpoint: List[float]
    points: int
    data: DataType


class PcdPointCloud(PointCloud):
    def __init__(self, metadata: Metadata, data: NDArray):
        self.metadata = metadata
        self.data = data

    @property
    def points(self) -> NDArray:
        columns = [self.data['x'], self.data['y'], self.data['z']]
        return np.vstack(columns).T

    @property
    def has_intensity(self) -> bool:
        return 'intensity' in self.data.dtype.names

    @property
    def intensity(self) -> NDArray:
        if not self.has_intensity:
            raise ValueError("Point cloud does not have intensity.")
        return self.data['intensity'].copy()

    @staticmethod
    def from_pcd(file_path: str) -> 'PcdPointCloud':
        metadata, data = read_point_cloud(file_path)
        return PcdPointCloud(metadata, data)


def read_point_cloud(file_path: str) -> Tuple[Metadata, NDArray]:
    headers = []
    with open(file_path, 'rb') as f:
        while True:
            line = f.readline()
            line = str(line, encoding='utf8')
            line = line.strip()
            headers.append(line)
            if line.startswith('DATA'):
                break

        metadata = parse_headers(headers)
        if metadata.data == DataType.BINARY:
            data = parse_binary_data(f, metadata)
        elif metadata.data == DataType.ASCII:
            data = parse_ascii_data(f, metadata)
        elif metadata.data == DataType.BINARY_COMPRESSED:
            data = parse_compressed_binary_data(f, metadata)
        else:
            raise IOError(f"Unsupported format: {metadata.data}")

    return metadata, data


def parse_headers(headers: List[str]) -> Metadata:
    metadata = {}
    for header in headers:
        if header.startswith('#'):
            continue
        splits = header.split()
        key, values = splits[0].lower(), splits[1:]
        if key == 'version':
            metadata[key] = values[0]
        elif key in {'fields', 'type'}:
            metadata[key] = values
        elif key in {'size', 'count'}:
            metadata[key] = list(map(int, values))
        elif key in {'width', 'height', 'points'}:
            metadata[key] = int(values[0])
        elif key == 'viewpoint':
            metadata[key] = list(map(float, values))
        elif key == 'data':
            metadata[key] = DataType[values[0].strip().upper()]
    return Metadata(**metadata)


def build_dtype(metadata: Metadata) -> DTypeLike:
    fields, types = [], []
    for i, (f, c, t, s) in enumerate(zip(metadata.fields, metadata.count, metadata.type, metadata.size)):
        np_type = PCD_TYPE_TO_NP_TYPE[(t, s)]
        if c == 1:
            fields.append(f)
            types.append(np_type)
        else:
            fields.extend([f'{f}_{i}_{j}' for j in range(c)])
            types.extend([np_type] * c)
    dtype = np.dtype(list(zip(fields, types)))
    return dtype


def parse_binary_data(f: BinaryIO, metadata: Metadata) -> np.ndarray:
    dtype = build_dtype(metadata)
    size = metadata.points * dtype.itemsize
    raw = f.read(size)
    data = np.frombuffer(raw, dtype=dtype)
    return data


def parse_ascii_data(f: BinaryIO, metadata: Metadata) -> np.array:
    dtype = build_dtype(metadata)
    data = np.loadtxt(f, dtype=dtype, delimiter=' ')
    return data


def parse_compressed_binary_data(f: BinaryIO, metadata: Metadata) -> np.array:
    compressed_size = struct.unpack('I', f.read(struct.calcsize('I')))[0]
    uncompressed_size = struct.unpack('I', f.read(struct.calcsize('I')))[0]
    compressed_data = f.read(compressed_size)

    buf = lzf.decompress(compressed_data, uncompressed_size)
    if len(buf) != uncompressed_size:
        raise IOError(f"Failed to decompressed data, size mismatch: {len(buf)} != {uncompressed_size}.")

    dtype = build_dtype(metadata)
    data = np.zeros(metadata.width, dtype=dtype)

    offset = 0
    for i in range(len(dtype)):
        dt = dtype[i]
        size = dt.itemsize * metadata.points
        column = np.frombuffer(buf[offset:(offset + size)], dt)
        data[dtype.names[i]] = column
        offset += size

    return data

def convert_pcd_to_bin(dir_path: str, output_dir: str):
    files = sorted(glob.glob(os.path.join(dir_path, '*.pcd')))
    print(f"Found {len(files)} files.")
    num = 0
    for file in tqdm(files):
        # file_name = os.path.splitext(os.path.basename(file))[0] + '.bin'
        file_name = str(num).zfill(6) + '.bin'
        bin_path = os.path.join(output_dir, file_name)
        PointCloud.from_file(file).to_bin(bin_path)
        num += 1

if __name__ == '__main__':
    dir_path = r'D:\\data\\free_way_2022-06-14-16-36-06_0\\pointcloud.benewake_horn_x2_ros_node.benewake_pointcloud'
    output_dir = r'C:\\Users\\ptrgu\\Downloads\\pc-tools-master-pc_tools\\pc-tools-master-pc_tools\\pc_tools\\output_benewake'
    convert_pcd_to_bin(dir_path, output_dir)

