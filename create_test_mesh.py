"""
Create a simple test cube mesh for testing the simplification algorithm.
"""
import numpy as np
from QEM import PLYWriter

def create_cube_mesh():
    """Create a simple cube mesh with subdivisions."""
    # Define cube vertices (corners of a unit cube)
    vertices = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ], dtype=np.float32)
    
    # Define cube faces (12 triangles, 2 per face)
    faces = np.array([
        # Bottom face (z=0)
        [0, 1, 2],
        [0, 2, 3],
        # Top face (z=1)
        [4, 6, 5],
        [4, 7, 6],
        # Front face (y=0)
        [0, 5, 1],
        [0, 4, 5],
        # Back face (y=1)
        [3, 2, 6],
        [3, 6, 7],
        # Left face (x=0)
        [0, 3, 7],
        [0, 7, 4],
        # Right face (x=1)
        [1, 5, 6],
        [1, 6, 2],
    ], dtype=np.int32)
    
    return vertices, faces


def create_subdivided_cube(subdivisions=2):
    """Create a subdivided cube mesh for more interesting simplification."""
    # Start with base cube
    vertices_list = []
    faces_list = []
    
    # Create grid of vertices
    n = subdivisions + 2  # Number of vertices per edge
    step = 1.0 / (n - 1)
    
    # Generate vertices for each face and add center vertices
    vertex_idx = 0
    vertex_grid = {}
    
    # Generate all vertices in a 3D grid
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Only include surface vertices (at least one coordinate is 0 or max)
                if i == 0 or i == n-1 or j == 0 or j == n-1 or k == 0 or k == n-1:
                    x, y, z = i * step, j * step, k * step
                    vertices_list.append([x, y, z])
                    vertex_grid[(i, j, k)] = vertex_idx
                    vertex_idx += 1
    
    # Generate faces for each surface
    # Bottom face (k=0)
    for i in range(n-1):
        for j in range(n-1):
            if (i, j, 0) in vertex_grid and (i+1, j, 0) in vertex_grid and \
               (i+1, j+1, 0) in vertex_grid and (i, j+1, 0) in vertex_grid:
                v1 = vertex_grid[(i, j, 0)]
                v2 = vertex_grid[(i+1, j, 0)]
                v3 = vertex_grid[(i+1, j+1, 0)]
                v4 = vertex_grid[(i, j+1, 0)]
                faces_list.append([v1, v2, v3])
                faces_list.append([v1, v3, v4])
    
    # Top face (k=n-1)
    for i in range(n-1):
        for j in range(n-1):
            if (i, j, n-1) in vertex_grid and (i+1, j, n-1) in vertex_grid and \
               (i+1, j+1, n-1) in vertex_grid and (i, j+1, n-1) in vertex_grid:
                v1 = vertex_grid[(i, j, n-1)]
                v2 = vertex_grid[(i+1, j, n-1)]
                v3 = vertex_grid[(i+1, j+1, n-1)]
                v4 = vertex_grid[(i, j+1, n-1)]
                faces_list.append([v1, v3, v2])
                faces_list.append([v1, v4, v3])
    
    # Front face (j=0)
    for i in range(n-1):
        for k in range(n-1):
            if (i, 0, k) in vertex_grid and (i+1, 0, k) in vertex_grid and \
               (i+1, 0, k+1) in vertex_grid and (i, 0, k+1) in vertex_grid:
                v1 = vertex_grid[(i, 0, k)]
                v2 = vertex_grid[(i+1, 0, k)]
                v3 = vertex_grid[(i+1, 0, k+1)]
                v4 = vertex_grid[(i, 0, k+1)]
                faces_list.append([v1, v3, v2])
                faces_list.append([v1, v4, v3])
    
    # Back face (j=n-1)
    for i in range(n-1):
        for k in range(n-1):
            if (i, n-1, k) in vertex_grid and (i+1, n-1, k) in vertex_grid and \
               (i+1, n-1, k+1) in vertex_grid and (i, n-1, k+1) in vertex_grid:
                v1 = vertex_grid[(i, n-1, k)]
                v2 = vertex_grid[(i+1, n-1, k)]
                v3 = vertex_grid[(i+1, n-1, k+1)]
                v4 = vertex_grid[(i, n-1, k+1)]
                faces_list.append([v1, v2, v3])
                faces_list.append([v1, v3, v4])
    
    # Left face (i=0)
    for j in range(n-1):
        for k in range(n-1):
            if (0, j, k) in vertex_grid and (0, j+1, k) in vertex_grid and \
               (0, j+1, k+1) in vertex_grid and (0, j, k+1) in vertex_grid:
                v1 = vertex_grid[(0, j, k)]
                v2 = vertex_grid[(0, j+1, k)]
                v3 = vertex_grid[(0, j+1, k+1)]
                v4 = vertex_grid[(0, j, k+1)]
                faces_list.append([v1, v2, v3])
                faces_list.append([v1, v3, v4])
    
    # Right face (i=n-1)
    for j in range(n-1):
        for k in range(n-1):
            if (n-1, j, k) in vertex_grid and (n-1, j+1, k) in vertex_grid and \
               (n-1, j+1, k+1) in vertex_grid and (n-1, j, k+1) in vertex_grid:
                v1 = vertex_grid[(n-1, j, k)]
                v2 = vertex_grid[(n-1, j+1, k)]
                v3 = vertex_grid[(n-1, j+1, k+1)]
                v4 = vertex_grid[(n-1, j, k+1)]
                faces_list.append([v1, v3, v2])
                faces_list.append([v1, v4, v3])
    
    vertices = np.array(vertices_list, dtype=np.float32)
    faces = np.array(faces_list, dtype=np.int32)
    
    return vertices, faces


if __name__ == "__main__":
    import os
    
    output_dir = "/home/runner/work/border/border/demo/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simple cube
    print("Creating simple cube mesh...")
    vertices, faces = create_cube_mesh()
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    output_path = os.path.join(output_dir, "cube_simple.ply")
    PLYWriter.write_ply(output_path, vertices, faces)
    print(f"  Saved to: {output_path}")
    
    # Create subdivided cube
    print("\nCreating subdivided cube mesh...")
    vertices, faces = create_subdivided_cube(subdivisions=4)
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    output_path = os.path.join(output_dir, "cube_subdivided.ply")
    PLYWriter.write_ply(output_path, vertices, faces)
    print(f"  Saved to: {output_path}")
    
    print("\nTest meshes created successfully!")
