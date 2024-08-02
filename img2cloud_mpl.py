import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def compute_depth(image):
    # Use Sobel operator to detect edges
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # Compute magnitude of gradient
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize depth to range [0, 1]
    depth = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    return depth

def create_point_cloud(image, depth):
    height, width = image.shape
    point_cloud = []

    for y in range(height):
        for x in range(width):
            z = depth[y, x]
            point_cloud.append([x, y, z])

    point_cloud = np.array(point_cloud)
    return point_cloud

def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = point_cloud[:, 0]
    ys = point_cloud[:, 1]
    zs = point_cloud[:, 2]

    ax.scatter(xs, ys, zs, c=zs, cmap='viridis', marker='o')
    plt.show()

def save_point_cloud(point_cloud, file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(file_path, pcd)

def main():
    image_file_path = 'img.webp'
    output_ply_file_path = 'output_point_cloud.ply'

    # Load image
    image = load_image(image_file_path)

    # Compute depth from image
    depth = compute_depth(image)

    # Create point cloud
    point_cloud = create_point_cloud(image, depth)

    # Visualize point cloud (optional)
    #visualize_point_cloud(point_cloud)

    # Save point cloud to PLY file
    save_point_cloud(point_cloud, output_ply_file_path)
    print(f'Point cloud saved to {output_ply_file_path}')

if __name__ == "__main__":
    main()
