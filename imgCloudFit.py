import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor

def load_stl_obj(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    return mesh

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def compute_scaling_factor(point_cloud, mesh_vertices):
    pc_min = point_cloud.min(axis=0)
    pc_max = point_cloud.max(axis=0)
    pc_size = pc_max - pc_min

    mesh_min = mesh_vertices.min(axis=0)
    mesh_max = mesh_vertices.max(axis=0)
    mesh_size = mesh_max - mesh_min

    scaling_factor = mesh_size / pc_size
    return scaling_factor

def scale_point_cloud(point_cloud, scaling_factor):
    scaled_point_cloud = point_cloud * scaling_factor
    return scaled_point_cloud

def rotate_point_cloud(point_cloud, rotation_matrix):
    return np.dot(point_cloud, rotation_matrix.T)

def compute_surface_roughness(point_cloud, mesh_vertices):
    tree = KDTree(mesh_vertices)
    distances, _ = tree.query(point_cloud)
    surface_roughness = np.mean(distances)
    return surface_roughness

def refine_alignment_with_icp(source, target, voxel_size):
    threshold = voxel_size * 4
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return icp_result.transformation

def sample_mesh_to_point_cloud(mesh, number_of_points=100000):
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    return pcd

def align_and_scale_point_cloud(point_cloud, mesh):
    # Convert mesh to point cloud for ICP
    mesh_pcd = sample_mesh_to_point_cloud(mesh)

    # Compute centroids of the original point cloud and the mesh
    pc_centroid = np.mean(point_cloud, axis=0)
    mesh_centroid = np.mean(np.asarray(mesh_pcd.points), axis=0)

    # Translate the point cloud to align centroids
    translation = mesh_centroid - pc_centroid
    point_cloud += translation

    # Compute scaling factor and scale the point cloud
    scaling_factor = compute_scaling_factor(point_cloud, np.asarray(mesh_pcd.points))
    scaled_point_cloud = scale_point_cloud(point_cloud, scaling_factor)

    # Apply translation again to ensure centroids remain aligned after scaling
    scaled_pc_centroid = np.mean(scaled_point_cloud, axis=0)
    translation_after_scaling = mesh_centroid - scaled_pc_centroid
    scaled_point_cloud += translation_after_scaling

    # Create a PointCloud object for the scaled and translated points
    scaled_point_cloud_obj = o3d.geometry.PointCloud()
    scaled_point_cloud_obj.points = o3d.utility.Vector3dVector(scaled_point_cloud)

    return scaled_point_cloud_obj, mesh_centroid

def create_rotation_matrix(angles):
    rx, ry, rz = np.deg2rad(angles)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def find_optimal_orientation(point_cloud, mesh_vertices):
    best_surface_roughness = float('inf')
    best_rotation_matrix = None
    best_angles = None

    def compute_surface_roughness_for_angles(angles):
        rotation_matrix = create_rotation_matrix(angles)
        rotated_point_cloud = rotate_point_cloud(point_cloud, rotation_matrix)
        surface_roughness = compute_surface_roughness(rotated_point_cloud, mesh_vertices)
        return angles, surface_roughness, rotation_matrix

    angles_range = np.arange(0, 360, 0.1)
    tasks = [(rx, ry, rz) for rx in angles_range for ry in angles_range for rz in angles_range]

    with ThreadPoolExecutor() as executor:
        results = executor.map(compute_surface_roughness_for_angles, tasks)
        for angles, surface_roughness, rotation_matrix in results:
            if surface_roughness < best_surface_roughness:
                best_surface_roughness = surface_roughness
                best_rotation_matrix = rotation_matrix
                best_angles = angles

    return best_angles, best_rotation_matrix, best_surface_roughness

def create_sphere_at_point(point, radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.translate(point)
    sphere.paint_uniform_color(color)
    return sphere

def visualize_with_transparency(mesh, point_cloud, centroid_mesh_sphere, centroid_pc_sphere):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(centroid_mesh_sphere)
    vis.add_geometry(centroid_pc_sphere)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def main():
    mesh_file_path = 'bag_holder.stl'
    point_cloud_file_path = 'output_point_cloud.ply'

    # Load mesh and point cloud
    with ThreadPoolExecutor() as executor:
        mesh_future = executor.submit(load_stl_obj, mesh_file_path)
        point_cloud_future = executor.submit(load_point_cloud, point_cloud_file_path)
        mesh = mesh_future.result()
        point_cloud = point_cloud_future.result()

    # Align, scale, and translate point cloud
    point_cloud, mesh_centroid = align_and_scale_point_cloud(np.asarray(point_cloud.points), mesh)

    # Find the optimal orientation
    best_angles, best_rotation_matrix, best_surface_roughness = find_optimal_orientation(np.asarray(point_cloud.points), np.asarray(mesh.vertices))

    # Apply the optimal rotation to the point cloud
    optimal_point_cloud = rotate_point_cloud(np.asarray(point_cloud.points), best_rotation_matrix)

    # Create a PointCloud object for the optimal points
    optimal_point_cloud_obj = o3d.geometry.PointCloud()
    optimal_point_cloud_obj.points = o3d.utility.Vector3dVector(optimal_point_cloud)

    # Compute the final centroid
    final_pc_centroid = np.mean(optimal_point_cloud, axis=0)

    # Create centroid points as spheres
    centroid_mesh_sphere = create_sphere_at_point(mesh_centroid, radius=5.0, color=[1, 0, 0])
    centroid_pc_sphere = create_sphere_at_point(final_pc_centroid, radius=5.0, color=[0, 1, 0])

    # Visualization with transparency for the mesh
    visualize_with_transparency(mesh, optimal_point_cloud_obj, centroid_mesh_sphere, centroid_pc_sphere)

    print(f'Optimal Surface Roughness: {best_surface_roughness}')
    print(f'Optimal Angles (degrees): Roll: {best_angles[0]}, Pitch: {best_angles[1]}, Yaw: {best_angles[2]}')

if __name__ == "__main__":
    main()
