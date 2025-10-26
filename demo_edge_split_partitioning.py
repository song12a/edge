"""
Demonstration script for edge splitting with octree partitioning.
This script shows the difference between partitioned and non-partitioned edge splitting.
"""

from edge_split import EdgeSplitter, PLYWriter
import os

# Configuration constants
OUTPUT_DIR = 'demo/output'
NUM_PARTITIONS = 8  # Octree partitioning uses 8 partitions


def create_larger_test_mesh():
    """
    Create a larger test mesh to better demonstrate partitioning effects.
    
    A 6x6x6 grid creates 216 vertices, which provides sufficient spatial 
    distribution for octree subdivision into 8 partitions.
    """
    vertices = []
    faces = []
    
    # Create a 6x6x6 grid of vertices for demonstrating octree partitioning
    grid_size = 6
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                vertices.append([float(i), float(j), float(k)])
    
    # Create faces
    def vertex_index(i, j, k):
        return i * (grid_size * grid_size) + j * grid_size + k
    
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            for k in range(grid_size - 1):
                # Create cube faces
                v000 = vertex_index(i, j, k)
                v001 = vertex_index(i, j, k+1)
                v010 = vertex_index(i, j+1, k)
                v011 = vertex_index(i, j+1, k+1)
                v100 = vertex_index(i+1, j, k)
                v101 = vertex_index(i+1, j, k+1)
                v110 = vertex_index(i+1, j+1, k)
                v111 = vertex_index(i+1, j+1, k+1)
                
                # Front face
                faces.extend([[v000, v100, v110], [v000, v110, v010]])
                # Back face
                faces.extend([[v001, v011, v111], [v001, v111, v101]])
                # Left face
                faces.extend([[v000, v010, v011], [v000, v011, v001]])
                # Right face
                faces.extend([[v100, v101, v111], [v100, v111, v110]])
                # Bottom face
                faces.extend([[v000, v001, v101], [v000, v101, v100]])
                # Top face
                faces.extend([[v010, v110, v111], [v010, v111, v011]])
    
    return vertices, faces


def main():
    print("=" * 70)
    print("Edge Splitting with Octree Partitioning - Demonstration")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create test mesh
    print("\n[1] Creating test mesh...")
    vertices, faces = create_larger_test_mesh()
    print(f"    Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Save original mesh
    writer = PLYWriter()
    writer.write_ply(f"{OUTPUT_DIR}/demo_original.ply", vertices, faces)
    print(f"    Saved: {OUTPUT_DIR}/demo_original.ply")
    
    # Test 1: Edge splitting WITHOUT partitioning (original behavior)
    print("\n" + "=" * 70)
    print("[2] Edge Splitting WITHOUT Partitioning")
    print("=" * 70)
    
    print("\n[2.1] Subremeshing mode (no partition)...")
    splitter1 = EdgeSplitter(use_partitioning=False)
    splitter1.initialize(vertices, faces)
    result1_v, result1_f = splitter1.split_edges(mode="subremeshing", max_iter=1)
    print(f"      Result: {len(result1_v)} vertices (+{len(result1_v) - len(vertices)}), "
          f"{len(result1_f)} faces (+{len(result1_f) - len(faces)})")
    writer.write_ply(f"{OUTPUT_DIR}/demo_subremesh_no_partition.ply", result1_v, result1_f)
    print(f"      Saved: {OUTPUT_DIR}/demo_subremesh_no_partition.ply")
    
    print("\n[2.2] Histogram mode (no partition)...")
    splitter2 = EdgeSplitter(use_partitioning=False)
    splitter2.initialize(vertices, faces)
    result2_v, result2_f = splitter2.split_edges(mode="histogram", max_iter=3)
    print(f"      Result: {len(result2_v)} vertices (+{len(result2_v) - len(vertices)}), "
          f"{len(result2_f)} faces (+{len(result2_f) - len(faces)})")
    writer.write_ply(f"{OUTPUT_DIR}/demo_histogram_no_partition.ply", result2_v, result2_f)
    print(f"      Saved: {OUTPUT_DIR}/demo_histogram_no_partition.ply")
    
    # Test 2: Edge splitting WITH octree partitioning (new behavior)
    print("\n" + "=" * 70)
    print("[3] Edge Splitting WITH Octree Partitioning")
    print("=" * 70)
    print("    (Only interior vertices are split, boundary vertices preserved)")
    
    print("\n[3.1] Subremeshing mode (with partition)...")
    splitter3 = EdgeSplitter(use_partitioning=True, num_partitions=NUM_PARTITIONS)
    splitter3.initialize(vertices, faces)
    result3_v, result3_f = splitter3.split_edges(mode="subremeshing", max_iter=1)
    print(f"      Result: {len(result3_v)} vertices (+{len(result3_v) - len(vertices)}), "
          f"{len(result3_f)} faces (+{len(result3_f) - len(faces)})")
    writer.write_ply(f"{OUTPUT_DIR}/demo_subremesh_with_partition.ply", result3_v, result3_f)
    print(f"      Saved: {OUTPUT_DIR}/demo_subremesh_with_partition.ply")
    
    print("\n[3.2] Histogram mode (with partition)...")
    splitter4 = EdgeSplitter(use_partitioning=True, num_partitions=NUM_PARTITIONS)
    splitter4.initialize(vertices, faces)
    result4_v, result4_f = splitter4.split_edges(mode="histogram", max_iter=3)
    print(f"      Result: {len(result4_v)} vertices (+{len(result4_v) - len(vertices)}), "
          f"{len(result4_f)} faces (+{len(result4_f) - len(faces)})")
    writer.write_ply(f"{OUTPUT_DIR}/demo_histogram_with_partition.ply", result4_v, result4_f)
    print(f"      Saved: {OUTPUT_DIR}/demo_histogram_with_partition.ply")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Original mesh:                    {len(vertices):4d} vertices, {len(faces):4d} faces")
    print()
    print("Without partitioning:")
    print(f"  Subremeshing:                   {len(result1_v):4d} vertices, {len(result1_f):4d} faces")
    print(f"  Histogram:                      {len(result2_v):4d} vertices, {len(result2_f):4d} faces")
    print()
    print("With octree partitioning:")
    print(f"  Subremeshing (boundary-aware):  {len(result3_v):4d} vertices, {len(result3_f):4d} faces")
    print(f"  Histogram (boundary-aware):     {len(result4_v):4d} vertices, {len(result4_f):4d} faces")
    print()
    print("Key differences:")
    print("  - Without partitioning: Splits ALL edges that meet criteria")
    print("  - With partitioning:    Only splits INTERIOR vertices (preserves boundaries)")
    print()
    print(f"All output files saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
