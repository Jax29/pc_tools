import numpy as np
import open3d as o3d
import os
import glob
from tqdm import tqdm

def compute_transform_matrix(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.1,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    seed: int = 47,
):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        seed=seed
    )
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=4,
            max_nn=300
        )
    )
    normals = np.asarray(inlier_cloud.normals)
    normal = normals.mean(axis=0)

    before = normal / np.sqrt(np.sum(normal ** 2))
    after = np.array([0, 0, 1])
    rotation = compute_rotation_matrix(before, after)
    rotation[2][3] = d
    print(rotation)
    return rotation


def compute_rotation_matrix(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    angle = np.arccos(before.dot(after))
    p_rotate = np.cross(before, after)
    p_rotate = p_rotate / np.sqrt(np.sum(p_rotate ** 2))

    rotation = np.eye(4)
    rotation[0][0] = np.cos(angle) + p_rotate[0] * p_rotate[0] * (1 - np.cos(angle))
    rotation[0][1] = p_rotate[0] * p_rotate[1] * (1 - np.cos(angle) - p_rotate[2] * np.sin(angle))
    rotation[0][2] = p_rotate[1] * np.sin(angle) + p_rotate[0] * p_rotate[2] * (1 - np.cos(angle))
    rotation[1][0] = p_rotate[2] * np.sin(angle) + p_rotate[0] * p_rotate[1] * (1 - np.cos(angle))
    rotation[1][1] = np.cos(angle) + p_rotate[1] * p_rotate[1] * (1 - np.cos(angle))
    rotation[1][2] = -p_rotate[0] * np.sin(angle) + p_rotate[1] * p_rotate[2] * (1 - np.cos(angle))
    rotation[2][0] = -p_rotate[1] * np.sin(angle) + p_rotate[0] * p_rotate[2] * (1 - np.cos(angle))
    rotation[2][1] = p_rotate[0] * np.sin(angle) + p_rotate[1] * p_rotate[2] * (1 - np.cos(angle))
    rotation[2][2] = np.cos(angle) + p_rotate[2] * p_rotate[2] * (1 - np.cos(angle))

    return rotation


def calib_bin(file_path: str, mat: np.ndarray, output_dir: str):

    files = sorted(glob.glob(os.path.join(file_path, '*.bin')))
    print(f"Found: {len(files)} files!")
    for file in tqdm(files):
        data = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
        points = data[:, :3]
        points_h = np.hstack([points, np.ones((len(points), 1))])
        transformed = np.matmul(mat, points_h.T).T
        data[:, :3] = transformed[:, :3]

        os.makedirs(output_dir, exist_ok=True)
        write_path = os.path.join(output_dir, os.path.basename(file))
        data.tofile(write_path)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    # o3d.io.write_point_cloud('ground_calib.pcd', pcd)



if __name__ == '__main__':
    PCD_PATH = r'C:\\Users\\ptrgu\Downloads\\pc-tools-master-pc_tools\\pc-tools-master-pc_tools\\pc_tools\\1655195766.643371246.pcd'
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    mat = compute_transform_matrix(pcd)

    dir_path = r'C:\\Users\\ptrgu\\Downloads\\pc-tools-master-pc_tools\\pc-tools-master-pc_tools\\pc_tools\\output_tanway'
    calib_bin(dir_path, mat, '../ground_calib_tanway')
