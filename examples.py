"""
Example usage of the mesh simplification algorithm with MDD and LME.

This script demonstrates various ways to use the mesh simplification functionality.
"""

from mesh_simplification_mdd_lme import (
    simplify_mesh_with_partitioning,
    process_ply_file,
    MeshPartitioner,
    LMESimplifier,
    MeshMerger
)
from QEM import PLYReader, PLYWriter, simplify_ply_file
import numpy as np
import os


def example_2_programmatic_usage():
    """
    Example 2: Programmatic usage - batch process all PLY files in a directory.
    """
    print("\n" + "=" * 70)
    print("Example 2: Programmatic Usage (Batch Processing)")
    print("=" * 70)

    # 配置输入输出目录
    input_dir = r"D:\sxl08\rand1\neural-mesh-simplification\neural-mesh-simplification\demo\data"  # 输入PLY文件所在目录
    output_dir = r"D:\sxl08\rand1\neural-mesh-simplification\neural-mesh-simplification\demo\output"  # 输出目录
    target_ratio = 0.4  # 保留40%顶点
    num_partitions = 8  # 八叉树分区数量

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取目录中所有PLY文件
    ply_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.ply')]

    if not ply_files:
        print(f"\nNo PLY files found in {input_dir}")
        return

    print(f"\nFound {len(ply_files)} PLY files in {input_dir}")
    print(f"Processing with target ratio: {target_ratio}, partitions: {num_partitions}\n")

    # 批量处理每个PLY文件
    for filename in ply_files:
        # 构建输入输出路径
        input_path = os.path.join(input_dir, filename)
        output_filename = f"programmatic_{filename}"  # 输出文件名（添加前缀区分）
        output_path = os.path.join(output_dir, output_filename)

        try:
            # 读取网格数据
            vertices, faces = PLYReader.read_ply(input_path)
            print(f"Processing {filename}:")
            print(f"  Original: {len(vertices)} vertices, {len(faces)} faces")

            # 执行简化
            simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
                vertices,
                faces,
                target_ratio=target_ratio,
                num_partitions=num_partitions
            )

            # 保存简化结果
            PLYWriter.write_ply(output_path, simplified_vertices, simplified_faces)
            print(f"  Simplified: {len(simplified_vertices)} vertices, {len(simplified_faces)} faces")
            print(f"  Saved to: {output_path}\n")

        except Exception as e:
            print(f"  Failed to process {filename}: {str(e)}\n")

    print(f"Batch processing complete. All results saved to {output_dir}")




def example_4_different_partition_counts():
    """
    Example 4: Test different numbers of partitions.
    """
    print("\n" + "=" * 70)
    print("Example 4: Different Partition Counts")
    print("=" * 70)
    
    input_file = "demo/data/cube_subdivided.ply"
    vertices, faces = PLYReader.read_ply(input_file)
    
    print(f"\nOriginal mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Try different partition counts
    for num_parts in [8]:  # Could also try [1, 4, 8, 16, 27] for larger meshes
        print(f"\n--- Using {num_parts} partitions ---")
        simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
            vertices, faces, target_ratio=0.5, num_partitions=num_parts
        )
        output_file = f"demo/output/example4_parts_{num_parts}.ply"
        PLYWriter.write_ply(output_file, simplified_vertices, simplified_faces)








def main():
    """Run all examples."""
    
    print("=" * 70)
    print("Mesh Simplification Examples")
    print("MDD (Minimal Simplification Domain) + LME (Local Minimal Edges)")
    print("=" * 70)
    
    # Make sure output directory exists
    os.makedirs("demo/output", exist_ok=True)
    os.makedirs("demo/output/batch", exist_ok=True)
    
    # Run examples

    example_2_programmatic_usage()
    #example_4_different_partition_counts()

    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nCheck the 'demo/output' directory for results.")


if __name__ == "__main__":
    main()
