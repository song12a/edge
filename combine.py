"""
Combine module for mesh processing with octree partitioning.

This module provides functionality to:
1. Partition meshes using octree-based spatial subdivision
2. Apply mesh simplification (from mesh_simplification_mdd_lme.py)
3. Apply edge splitting (from edge_split.py)
"""

import numpy as np
import os
from typing import List, Tuple, Dict, Set
from QEM import PLYReader, PLYWriter
from mesh_simplification_mdd_lme import simplify_mesh_with_partitioning, MeshPartitioner
from edge_split import EdgeSplitter


class CombinedMeshProcessor:
    """
    A unified processor for mesh partitioning, simplification, and edge splitting.
    
    This class uses octree-based partitioning with 2-ring neighborhood support
    (following the approach from mesh_simplification_mdd_lme.py) to ensure both
    simplification and edge splitting operations work on properly extended partitions.
    
    The 2-ring neighborhoods are essential because:
    - For simplification: Provides topological context for QEM calculations
    - For edge splitting: Ensures consistent edge splitting across partition boundaries
    - For both: Maintains mesh quality at partition interfaces
    """
    
    def __init__(self, target_edges_per_partition: int = 200):
        """
        Initialize the combined mesh processor.
        
        Args:
            target_edges_per_partition: Target number of edges per partition for dynamic partitioning
        """
        self.target_edges_per_partition = target_edges_per_partition
        
    def calculate_num_partitions(self, num_edges: int) -> int:
        """
        Calculate optimal number of partitions based on edge count.
        
        Args:
            num_edges: Total number of edges in the mesh
            
        Returns:
            Number of partitions (always power of 2 for octree: 1, 8, 64, etc.)
        """
        if num_edges <= self.target_edges_per_partition:
            return 1
        
        # Calculate how many partitions we need
        needed_partitions = num_edges / self.target_edges_per_partition
        
        # Round to nearest power of 8 (for octree: 1, 8, 64, 512, etc.)
        # Actually, octree subdivision gives us 8^n partitions
        if needed_partitions <= 1:
            return 1
        elif needed_partitions <= 8:
            return 8
        elif needed_partitions <= 64:
            return 64
        else:
            return 512
    
    def count_edges(self, faces: np.ndarray) -> int:
        """
        Count unique edges in the mesh.
        
        Args:
            faces: Face array (M x 3)
            
        Returns:
            Number of unique edges
        """
        edges = set()
        for face in faces:
            v0, v1, v2 = face
            edges.add((min(v0, v1), max(v0, v1)))
            edges.add((min(v1, v2), max(v1, v2)))
            edges.add((min(v2, v0), max(v2, v0)))
        return len(edges)
    
    def partition_mesh(self, vertices: np.ndarray, faces: np.ndarray, 
                      num_partitions: int = None) -> Tuple[MeshPartitioner, List[Dict]]:
        """
        Partition a mesh using octree spatial subdivision with 2-ring neighborhood support.
        Uses the partitioning approach from mesh_simplification_mdd_lme.py.
        
        Args:
            vertices: Vertex array (N x 3)
            faces: Face array (M x 3)
            num_partitions: Number of partitions (if None, calculated from edge count)
            
        Returns:
            Tuple of (partitioner, partitions) where partitions include 2-ring neighborhoods
        """
        if num_partitions is None:
            num_edges = self.count_edges(faces)
            num_partitions = self.calculate_num_partitions(num_edges)
        
        print(f"\n=== Partitioning Mesh with 2-Ring Neighborhoods ===")
        print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
        print(f"Edges: {self.count_edges(faces)}")
        print(f"Target edges per partition: {self.target_edges_per_partition}")
        print(f"Using {num_partitions} partitions (octree subdivision)")
        
        # Use MeshPartitioner from mesh_simplification_mdd_lme.py
        # This automatically includes 2-ring neighborhood computation
        partitioner = MeshPartitioner(vertices, faces, num_partitions)
        partitions = partitioner.partition_octree()
        
        print(f"Created {len(partitions)} non-empty partitions")
        print(f"Border vertices: {len(partitioner.border_vertices)}")
        
        # Print 2-ring neighborhood statistics
        for i, partition in enumerate(partitions):
            core_count = len(partition['core_vertices'])
            total_count = len(partition['vertices'])
            ring_count = total_count - core_count
            print(f"  Partition {i}: {core_count} core + {ring_count} 2-ring = {total_count} total vertices")
        
        return partitioner, partitions
    
    def simplify_mesh(self, vertices: np.ndarray, faces: np.ndarray,
                     target_ratio: float = 0.5, num_partitions: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplify a mesh using partitioned simplification.
        
        Args:
            vertices: Input mesh vertices (N x 3)
            faces: Input mesh faces (M x 3)
            target_ratio: Target simplification ratio (0.5 = keep 50% of vertices)
            num_partitions: Number of partitions (if None, calculated from edge count)
            
        Returns:
            Tuple of (simplified_vertices, simplified_faces)
        """
        if num_partitions is None:
            num_edges = self.count_edges(faces)
            num_partitions = self.calculate_num_partitions(num_edges)
        
        print(f"\n=== Simplifying Mesh ===")
        return simplify_mesh_with_partitioning(vertices, faces, target_ratio, num_partitions)
    
    def split_edges_mesh(self, vertices: List, faces: List, 
                        mode: str = "histogram", max_iter: int = 3,
                        use_partitioning: bool = True, num_partitions: int = None) -> Tuple[List, List]:
        """
        Split edges in a mesh using the same 2-ring neighborhood partitioning as simplification.
        
        When use_partitioning=True, this method uses octree partitioning with 2-ring neighborhoods
        (same approach as mesh_simplification_mdd_lme.py) to ensure edge splitting works on
        properly extended partitions.
        
        Args:
            vertices: Input mesh vertices (list of [x, y, z])
            faces: Input mesh faces (list of [v1, v2, v3])
            mode: Splitting mode ("subremeshing" or "histogram")
            max_iter: Maximum iterations
            use_partitioning: Whether to use octree partitioning with 2-ring neighborhoods
            num_partitions: Number of partitions (if None, calculated from edge count)
            
        Returns:
            Tuple of (split_vertices, split_faces)
        """
        if use_partitioning and num_partitions is None:
            # Convert to numpy for edge counting
            faces_np = np.array(faces, dtype=np.int32)
            num_edges = self.count_edges(faces_np)
            num_partitions = self.calculate_num_partitions(num_edges)
        
        print(f"\n=== Splitting Edges ===")
        if use_partitioning:
            print(f"Using octree partitioning with 2-ring neighborhoods ({num_partitions} partitions)")
            print(f"Note: EdgeSplitter internally uses the same partitioning approach as mesh_simplification_mdd_lme.py")
        
        # EdgeSplitter has its own MeshPartitioner that includes 2-ring neighborhood support
        # This is consistent with mesh_simplification_mdd_lme.py's approach
        splitter = EdgeSplitter(use_partitioning=use_partitioning, 
                               num_partitions=num_partitions if use_partitioning else 8)
        splitter.initialize(vertices, faces)
        return splitter.split_edges(mode=mode, max_iter=max_iter)
    
    def process_mesh_file(self, input_path: str, output_dir: str,
                         operation: str = "simplify", **kwargs):
        """
        Process a single mesh file.
        
        Args:
            input_path: Path to input PLY file
            output_dir: Directory for output files
            operation: Operation to perform ("simplify", "split", or "both")
            **kwargs: Additional arguments for operations
                - For simplify: target_ratio (default 0.5), num_partitions (default None)
                - For split: mode (default "histogram"), max_iter (default 3), 
                            use_partitioning (default True), num_partitions (default None)
        """
        print(f"\n{'='*80}")
        print(f"Processing: {input_path}")
        print(f"Operation: {operation}")
        print(f"{'='*80}")
        
        # Read input mesh
        reader = PLYReader()
        vertices, faces = reader.read_ply(input_path)
        
        # Convert to numpy arrays
        vertices_np = np.array(vertices, dtype=np.float32)
        faces_np = np.array(faces, dtype=np.int32)
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        if operation == "simplify" or operation == "both":
            # Simplification
            target_ratio = kwargs.get('target_ratio', 0.5)
            num_partitions = kwargs.get('num_partitions', None)
            
            simplified_vertices, simplified_faces = self.simplify_mesh(
                vertices_np, faces_np, target_ratio, num_partitions
            )
            
            # Save simplified mesh
            output_path = os.path.join(output_dir, f"{base_name}_simplified.ply")
            writer = PLYWriter()
            writer.write_ply(output_path, simplified_vertices, simplified_faces)
            print(f"\nSaved simplified mesh to: {output_path}")
            
            # If doing both, use simplified mesh for splitting
            if operation == "both":
                vertices = simplified_vertices.tolist()
                faces = simplified_faces.tolist()
        
        if operation == "split" or operation == "both":
            # Edge splitting
            mode = kwargs.get('mode', 'histogram')
            max_iter = kwargs.get('max_iter', 3)
            use_partitioning = kwargs.get('use_partitioning', True)
            num_partitions = kwargs.get('num_partitions', None)
            
            split_vertices, split_faces = self.split_edges_mesh(
                vertices if operation == "both" else vertices_np.tolist(),
                faces if operation == "both" else faces_np.tolist(),
                mode, max_iter, use_partitioning, num_partitions
            )
            
            # Save split mesh
            suffix = "_split" if operation == "split" else "_simplified_split"
            output_path = os.path.join(output_dir, f"{base_name}{suffix}.ply")
            writer = PLYWriter()
            writer.write_ply(output_path, split_vertices, split_faces)
            print(f"\nSaved split mesh to: {output_path}")
    
    def process_directory(self, input_dir: str, output_dir: str,
                         operation: str = "simplify", **kwargs):
        """
        Process all PLY files in a directory.
        
        Args:
            input_dir: Directory containing input PLY files
            output_dir: Directory for output files
            operation: Operation to perform ("simplify", "split", or "both")
            **kwargs: Additional arguments passed to process_mesh_file
        """
        if not os.path.exists(input_dir):
            print(f"Error: Input directory does not exist: {input_dir}")
            return
        
        # Find all PLY files
        ply_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
        
        if not ply_files:
            print(f"No PLY files found in: {input_dir}")
            return
        
        print(f"\n{'='*80}")
        print(f"Processing {len(ply_files)} PLY files from: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}")
        
        for ply_file in ply_files:
            input_path = os.path.join(input_dir, ply_file)
            try:
                self.process_mesh_file(input_path, output_dir, operation, **kwargs)
            except Exception as e:
                print(f"\nError processing {ply_file}: {e}")
                import traceback
                traceback.print_exc()


def main():
    """
    Main function demonstrating usage of the combined mesh processor.
    
    This demonstrates partitioning with 2-ring neighborhoods (following mesh_simplification_mdd_lme.py)
    and applying both simplification and edge splitting to the partitioned mesh.
    """
    # Configuration
    input_folder = "demo/data"
    output_folder = "demo/output"
    target_edges_per_partition = 200
    
    # Create processor
    processor = CombinedMeshProcessor(target_edges_per_partition=target_edges_per_partition)
    
    # Check if input directory exists and has files
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist. Creating it...")
        os.makedirs(input_folder, exist_ok=True)
        print(f"Please place PLY files in '{input_folder}' and run again.")
        
        # Create a sample mesh for demonstration
        print("\nCreating a sample mesh for demonstration...")
        from create_test_mesh import create_subdivided_cube
        vertices, faces = create_subdivided_cube()
        writer = PLYWriter()
        sample_path = os.path.join(input_folder, "sample_mesh.ply")
        writer.write_ply(sample_path, vertices, faces)
        print(f"Created sample mesh: {sample_path}")
    
    # Process all meshes in the input directory
    print("\n" + "="*80)
    print("Combined Mesh Processing Tool")
    print("Using mesh_simplification_mdd_lme.py partitioning approach")
    print("="*80)
    print(f"Target edges per partition: {target_edges_per_partition}")
    print(f"Partitioning: Octree with 2-ring neighborhoods")
    print(f"Input directory: {input_folder}")
    print(f"Output directory: {output_folder}")
    
    # Example 1: Simplify only
    print("\n\n### Example 1: Mesh Simplification ###")
    processor.process_directory(
        input_folder, output_folder,
        operation="simplify",
        target_ratio=0.5
    )
    
    # Example 2: Edge splitting only
    print("\n\n### Example 2: Edge Splitting ###")
    processor.process_directory(
        input_folder, output_folder,
        operation="split",
        mode="histogram",
        max_iter=3,
        use_partitioning=True
    )
    
    # Example 3: Both simplification and splitting
    print("\n\n### Example 3: Simplification + Edge Splitting ###")
    processor.process_directory(
        input_folder, output_folder,
        operation="both",
        target_ratio=0.5,
        mode="histogram",
        max_iter=3,
        use_partitioning=True
    )
    
    print("\n" + "="*80)
    print("Processing Complete!")
    print(f"Results saved to: {output_folder}")
    print("="*80)


if __name__ == "__main__":
    main()
