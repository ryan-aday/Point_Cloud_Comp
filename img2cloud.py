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

def align_principal_axes(point_cloud, mesh_vertices):
    pca_pc = PCA(n_components=3)
    pca_mesh = PCA(n_components=3)

    pc_pca = pca_pc.fit_transform(point_cloud)
    mesh_pca = pca_mesh.fit_transform(mesh_vertices)

    # Align the principal axes
    R = np.dot(pca_mesh.components_.T, pca_pc.components_)
    aligned_point_cloud = np.dot(point_cloud, R)
    return aligned_point_cloud

def compute_surface_roughness(point_cloud, mesh_vertices):
    tree = KDTree(mesh_vertices)
    distances, _ = tree.query(point_cloud)
    surface_roughness = np.mean(distances)
    return surface_roughness

def create_fake_points(point_cloud, mesh_vertices):
    # Create KDTree for point cloud
    tree = KDTree(point_cloud)

    # Find the nearest neighbors for each point in the mesh
    _, indices = tree.query(mesh_vertices)

    # Select points in the point cloud that are closest to the mesh points
    fake_points = point_cloud[indices]

    return fake_points

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

    # Align principal axes
    point_cloud = align_principal_axes(point_cloud, mesh_pcd)

    # Compute scaling factor and scale the point cloud
    scaling_factor = compute_scaling_factor(point_cloud, mesh_pcd)
    scaled_point_cloud = scale_point_cloud(point_cloud, scaling_factor)

    # Apply translation again to ensure centroids remain aligned after scaling
    scaled_pc_centroid = np.mean(scaled_point_cloud, axis=0)
    mesh_pcd_centroid = snp.mean(mesh_pcd, axis=0)

    translation_after_scaling = scaled_pc_centroid - mesh_pcd_centroid
    scaled_point_cloud += translation_after_scaling
    scaled_pc_centroid = np.mean(scaled_point_cloud, axis=0)

    # Create a PointCloud object for the scaled and translated points
    scaled_point_cloud_obj = o3d.geometry.PointCloud()
    scaled_point_cloud_obj.points = o3d.utility.Vector3dVector(scaled_point_cloud)



    # Refine alignment using ICP
    voxel_size = max(scaled_point_cloud.ptp(axis=0)) / 10  # Define voxel size
    transformation = refine_alignment_with_icp(scaled_point_cloud_obj, mesh_pcd, voxel_size)

    # Apply the refined transformation
    scaled_point_cloud_obj.transform(transformation)

    return scaled_point_cloud_obj, mesh_centroid

def create_sphere_at_point(point, radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.translate(point)
    sphere.paint_uniform_color(color)
    return sphere

def visualize_with_transparency(mesh, point_cloud, fake_pcd, centroid_mesh_sphere, centroid_pc_sphere):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(fake_pcd)
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
    point_cloud, mesh_centroid = align_and_scale_point_cloud(np.asarray(point_cloud.points), np.asarray(mesh.vertices))

    '''
    # Create fake points for surfaces not captured by the point cloud
    fake_points = create_fake_points(np.asarray(point_cloud.points), np.asarray(mesh.vertices))

    fake_pcd = o3d.geometry.PointCloud()
    fake_pcd.points = o3d.utility.Vector3dVector(fake_points)
    '''
    
    # Compute the final centroid after adding fake points
    final_pc_centroid = np.mean(np.asarray(point_cloud.points), axis=0)

    # Compute surface roughness after scaling and adding fake points
    surface_roughness = compute_surface_roughness(np.asarray(point_cloud.points), np.asarray(mesh.vertices))
    print(f'Surface Roughness After Scaling: {surface_roughness}')

    # Create centroid points as spheres
    centroid_mesh_sphere = create_sphere_at_point(mesh_centroid, radius=5.0, color=[1, 0, 0])
    centroid_pc_sphere = create_sphere_at_point(final_pc_centroid, radius=5.0, color=[0, 1, 0])

    # Visualization with transparency for the mesh
    visualize_with_transparency(mesh, point_cloud, fake_pcd, centroid_mesh_sphere, centroid_pc_sphere)

if __name__ == "__main__":
    main()
