"""
Mesh Simplification using MDD (Minimal Simplification Domain) and LME (Local Minimal Edges)

This module implements a mesh simplification algorithm that:
1. Partitions a large mesh into smaller sub-meshes (MDD)
2. Simplifies each sub-mesh independently using QEM with LME approach
3. Merges the simplified sub-meshes back into a single output mesh

The implementation is based on the concepts from the paper on out-of-core mesh simplification
and uses the QEM method from QEM.py as the base simplifier.
"""

import numpy as np
import os
from QEM import PLYReader, PLYWriter, QEMSimplifier
from typing import List, Tuple, Dict, Set


class MeshPartitioner:
    """
    Partitions a mesh into smaller sub-meshes based on spatial subdivision.
    Implements the MDD (Minimal Simplification Domain) concept with 2-ring neighborhood support.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, num_partitions: int = 8):
        """
        Initialize the mesh partitioner.

        Args:
            vertices: Array of vertex coordinates (N x 3)
            faces: Array of face indices (M x 3)
            num_partitions: Number of spatial partitions (default: 8 for octree)
        """
        self.vertices = vertices
        self.faces = faces
        self.num_partitions = num_partitions
        self.partitions = []
        self.border_vertices = set()  # Vertices on partition boundaries
        self.vertex_adjacency = None  # Will store vertex-to-vertex connectivity

    def build_vertex_adjacency(self) -> Dict[int, Set[int]]:
        """
        Build vertex-to-vertex adjacency information from faces.

        Returns:
            Dictionary mapping each vertex index to a set of adjacent vertex indices.
        """
        adjacency = {i: set() for i in range(len(self.vertices))}

        for face in self.faces:
            v0, v1, v2 = face
            # Each vertex in a triangle is adjacent to the other two
            adjacency[v0].add(v1)
            adjacency[v0].add(v2)
            adjacency[v1].add(v0)
            adjacency[v1].add(v2)
            adjacency[v2].add(v0)
            adjacency[v2].add(v1)

        return adjacency

    def compute_n_ring_neighborhood(self, vertex_set: Set[int], n: int = 1) -> Set[int]:
        """
        Compute the n-ring neighborhood of a set of vertices.

        Args:
            vertex_set: Initial set of vertex indices
            n: Number of rings to expand (1 for 1-ring, 2 for 2-ring, etc.)

        Returns:
            Set of all vertices within n-ring distance from the initial set.
        """
        if self.vertex_adjacency is None:
            self.vertex_adjacency = self.build_vertex_adjacency()

        current_ring = vertex_set.copy()
        all_vertices = vertex_set.copy()

        for _ in range(n):
            next_ring = set()
            for vertex in current_ring:
                # Add all neighbors of vertices in current ring
                next_ring.update(self.vertex_adjacency[vertex])

            # Add new vertices to the total set
            all_vertices.update(next_ring)
            # Next iteration starts from the newly added vertices
            current_ring = next_ring - vertex_set  # Exclude original vertices to avoid redundant work
            vertex_set = all_vertices.copy()  # Update to include all found so far

        return all_vertices

    def partition_octree(self) -> List[Dict]:
        """
        Partition the mesh using octree spatial subdivision with 2-ring neighborhood support.

        Each partition includes:
        - Core vertices: vertices within the spatial bounds
        - Extended vertices: vertices in the 2-ring neighborhood of core vertices

        Returns:
            List of partition dictionaries, each containing:
                - 'vertices': all vertex indices in this partition (core + 2-ring)
                - 'core_vertices': vertex indices in the spatial bounds only
                - 'faces': face indices in this partition
                - 'is_border': set of border vertices (shared with other partitions)
        """
        # Calculate bounding box
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        center = (min_coords + max_coords) / 2

        # Determine which octant each vertex belongs to (based on spatial position)
        vertex_partitions = np.zeros(len(self.vertices), dtype=np.int32)

        for i, vertex in enumerate(self.vertices):
            octant = 0
            if vertex[0] > center[0]:
                octant += 1
            if vertex[1] > center[1]:
                octant += 2
            if vertex[2] > center[2]:
                octant += 4
            vertex_partitions[i] = octant

        # Initialize partition data structures
        partition_data = [{'core_vertices': set(), 'vertices': set(), 'faces': [], 'is_border': set()}
                         for _ in range(8)]

        # First pass: assign vertices to their core partitions based on spatial position
        for i in range(len(self.vertices)):
            partition_idx = vertex_partitions[i]
            partition_data[partition_idx]['core_vertices'].add(i)

        # Second pass: expand each partition with 2-ring neighborhoods
        print("  Computing 2-ring neighborhoods for each partition...")
        for p_idx, p_data in enumerate(partition_data):
            if len(p_data['core_vertices']) > 0:
                # Compute 2-ring neighborhood of core vertices
                extended_vertices = self.compute_n_ring_neighborhood(p_data['core_vertices'], n=2)
                p_data['vertices'] = extended_vertices
                print(f"    Partition {p_idx}: {len(p_data['core_vertices'])} core vertices -> "
                      f"{len(extended_vertices)} vertices with 2-ring")

        # Third pass: assign faces to partitions
        # A face belongs to a partition's core if all its vertices are in core or immediate neighbors
        # A face is in the extended set if it's needed for 2-ring context
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = face

            # Determine which partition's core this face primarily belongs to
            core_partitions = {vertex_partitions[v0], vertex_partitions[v1], vertex_partitions[v2]}

            # Assign face to partitions where all vertices are in the extended vertex set
            for p_idx, p_data in enumerate(partition_data):
                if v0 in p_data['vertices'] and v1 in p_data['vertices'] and v2 in p_data['vertices']:
                    p_data['faces'].append(face_idx)

            # Determine if any vertex is a border vertex
            # A vertex is on the border if its face spans multiple core partitions
            if len(core_partitions) > 1:
                for v in face:
                    self.border_vertices.add(v)

        # Fourth pass: identify border vertices for each partition
        # Border vertices are those that should not be simplified:
        # 1. Vertices in the 2-ring extension (not in core)
        # 2. Vertices on the boundary between core partitions
        for p_idx, p_data in enumerate(partition_data):
            for v in p_data['vertices']:
                # If vertex is not in this partition's core, it's part of the 2-ring extension
                if v not in p_data['core_vertices']:
                    p_data['is_border'].add(v)
                # If vertex is in the core but belongs to a face that spans multiple core partitions
                elif v in self.border_vertices:
                    p_data['is_border'].add(v)

        # Filter out empty partitions
        self.partitions = [p for p in partition_data if len(p['faces']) > 0]

        return self.partitions

    def extract_submesh(self, partition: Dict) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        """
        Extract a sub-mesh from a partition.

        Args:
            partition: Partition dictionary with vertex and face information

        Returns:
            Tuple of (vertices, faces, vertex_map) where:
                - vertices: array of vertex coordinates in the sub-mesh
                - faces: array of face indices in the sub-mesh (local indexing)
                - vertex_map: mapping from global vertex indices to local indices
        """
        # Create vertex mapping from global to local indices
        vertex_list = sorted(partition['vertices'])
        vertex_map = {global_idx: local_idx for local_idx, global_idx in enumerate(vertex_list)}

        # Extract vertices
        submesh_vertices = self.vertices[vertex_list]

        # Extract and reindex faces
        submesh_faces = []
        for face_idx in partition['faces']:
            face = self.faces[face_idx]
            local_face = [vertex_map[v] for v in face]
            submesh_faces.append(local_face)

        submesh_faces = np.array(submesh_faces, dtype=np.int32)

        return submesh_vertices, submesh_faces, vertex_map


class LMESimplifier:
    """
    Local Minimal Edges (LME) Simplifier that extends QEM to preserve border vertices.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, border_vertices: Set[int]):
        """
        Initialize the LME simplifier.

        Args:
            vertices: Array of vertex coordinates
            faces: Array of face indices
            border_vertices: Set of vertex indices that are on partition borders
        """
        self.base_simplifier = QEMSimplifier(vertices, faces)
        self.border_vertices = border_vertices

    def simplify(self, target_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplify the mesh while preserving border vertices.

        Args:
            target_ratio: Target ratio of vertices to retain

        Returns:
            Tuple of (simplified_vertices, simplified_faces)
        """
        # Calculate target vertex count, ensuring border vertices are preserved
        num_border = len(self.border_vertices)
        num_total = len(self.base_simplifier.vertices)
        num_interior = num_total - num_border

        # We want to keep all border vertices plus a fraction of interior vertices
        target_interior = max(4, int(num_interior * target_ratio))
        target_vertex_count = num_border + target_interior

        print(f"  LME Simplification: {num_total} vertices ({num_border} border, {num_interior} interior)")
        print(f"  Target: {target_vertex_count} vertices ({num_border} border, {target_interior} interior)")

        # Find all valid edges
        edges = self.base_simplifier.find_valid_edges()

        # Create priority queue, excluding edges involving border vertices
        import heapq
        heap = []

        for edge in edges:
            v1, v2 = edge
            # Skip edges involving border vertices
            if v1 in self.border_vertices or v2 in self.border_vertices:
                continue

            if v1 not in self.base_simplifier.valid_vertices or v2 not in self.base_simplifier.valid_vertices:
                continue

            optimal_pos = self.base_simplifier.compute_optimal_position(v1, v2)
            cost = self.base_simplifier.compute_cost(v1, v2, optimal_pos)
            heapq.heappush(heap, (cost, v1, v2, optimal_pos))

        # Perform edge contractions
        contraction_count = 0
        current_vertex_count = len(self.base_simplifier.valid_vertices)

        while current_vertex_count > target_vertex_count and heap:
            cost, v1, v2, optimal_pos = heapq.heappop(heap)

            # Check if vertices are still valid
            if v1 not in self.base_simplifier.valid_vertices or v2 not in self.base_simplifier.valid_vertices:
                continue

            # Double-check border vertices (shouldn't happen but safety check)
            if v1 in self.border_vertices or v2 in self.border_vertices:
                continue

            # Contract edge
            self.base_simplifier.contract_edge(v1, v2, optimal_pos)
            contraction_count += 1
            current_vertex_count = len(self.base_simplifier.valid_vertices)

            if contraction_count % 50 == 0:
                print(f"    Contracted {contraction_count} edges, {current_vertex_count} vertices remaining")

        # Rebuild mesh
        self.base_simplifier.rebuild_mesh()

        print(f"  Simplification complete: {len(self.base_simplifier.vertices)} vertices, "
              f"{len(self.base_simplifier.faces)} faces")

        return self.base_simplifier.vertices, self.base_simplifier.faces


class MeshMerger:
    """
    Merges simplified sub-meshes back into a single mesh.
    """

    def __init__(self):
        self.merged_vertices = []
        self.merged_faces = []
        self.global_vertex_map = {}  # Maps (partition_idx, local_vertex_idx) to global vertex idx

    def merge_submeshes(self, submeshes: List[Dict], original_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge multiple simplified sub-meshes into a single mesh.

        Args:
            submeshes: List of dictionaries containing:
                - 'vertices': vertex array
                - 'faces': face array
                - 'vertex_map': mapping from global to local indices (original mesh)
                - 'reverse_map': mapping from local to global indices (original mesh)
            original_vertices: Original mesh vertices for position-based deduplication

        Returns:
            Tuple of (merged_vertices, merged_faces)
        """
        vertex_global_to_merged = {}  # Maps original global vertex idx to merged mesh idx
        vertex_position_map = {}  # Maps vertex positions to merged indices for tolerance-based matching
        tolerance = 1e-6  # Tolerance for vertex position comparison

        # First pass: collect all faces and determine which vertices are actually used
        temp_faces = []
        used_vertices = set()  # Set of (submesh_idx, local_idx) tuples

        for submesh_idx, submesh in enumerate(submeshes):
            faces = submesh['faces']
            for face in faces:
                temp_faces.append((submesh_idx, face))
                for v in face:
                    used_vertices.add((submesh_idx, v))

        # Second pass: process only used vertices
        for submesh_idx, submesh in enumerate(submeshes):
            vertices = submesh['vertices']
            reverse_map = submesh['reverse_map']

            # Process only vertices that are actually used by faces
            for local_idx in range(len(vertices)):
                if (submesh_idx, local_idx) not in used_vertices:
                    continue  # Skip unused vertices

                vertex = vertices[local_idx]
                original_global_idx = reverse_map.get(local_idx)

                # Check if this vertex was already added from another partition
                if original_global_idx is not None and original_global_idx in vertex_global_to_merged:
                    # Vertex already exists by original index, reuse it
                    merged_idx = vertex_global_to_merged[original_global_idx]
                else:
                    # Check for duplicate by position
                    vertex_key = tuple(np.round(vertex / tolerance).astype(int))

                    if vertex_key in vertex_position_map:
                        # Found duplicate by position
                        merged_idx = vertex_position_map[vertex_key]
                    else:
                        # New vertex, add it
                        merged_idx = len(self.merged_vertices)
                        self.merged_vertices.append(vertex)
                        vertex_position_map[vertex_key] = merged_idx

                        if original_global_idx is not None:
                            vertex_global_to_merged[original_global_idx] = merged_idx

                # Map (submesh_idx, local_idx) to merged_idx
                self.global_vertex_map[(submesh_idx, local_idx)] = merged_idx

        # Third pass: process faces
        for submesh_idx, face in temp_faces:
            merged_face = [self.global_vertex_map[(submesh_idx, v)] for v in face]
            # Check for degenerate faces
            if len(set(merged_face)) == 3:
                self.merged_faces.append(merged_face)

        # Remove duplicate faces (same vertices, regardless of order)
        unique_faces = []
        seen_faces = set()
        for face in self.merged_faces:
            face_tuple = tuple(sorted(face))
            if face_tuple not in seen_faces:
                seen_faces.add(face_tuple)
                unique_faces.append(face)

        merged_vertices_array = np.array(self.merged_vertices, dtype=np.float32)
        merged_faces_array = np.array(unique_faces, dtype=np.int32)

        print(f"Merged {len(submeshes)} submeshes:")
        print(f"  Total vertices in submeshes: {sum(len(s['vertices']) for s in submeshes)}")
        print(f"  Used vertices before deduplication: {len(used_vertices)}")
        print(f"  Unique vertices after deduplication: {len(merged_vertices_array)}")
        print(f"  Total faces before deduplication: {len(self.merged_faces)}")
        print(f"  Unique faces after deduplication: {len(merged_faces_array)}")

        return merged_vertices_array, merged_faces_array


def simplify_mesh_with_partitioning(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_ratio: float = 0.5,
    num_partitions: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplify a mesh using partitioning, local simplification, and merging.

    Args:
        vertices: Input mesh vertices (N x 3)
        faces: Input mesh faces (M x 3)
        target_ratio: Target simplification ratio (0.5 = keep 50% of vertices)
        num_partitions: Number of spatial partitions (default: 8 for octree)

    Returns:
        Tuple of (simplified_vertices, simplified_faces)
    """
    print(f"\n=== Starting Mesh Simplification with MDD/LME ===")
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Target ratio: {target_ratio}")

    # Step 1: Partition the mesh
    print("\n[Step 1] Partitioning mesh...")
    partitioner = MeshPartitioner(vertices, faces, num_partitions)
    partitions = partitioner.partition_octree()
    print(f"Created {len(partitions)} non-empty partitions")
    print(f"Border vertices: {len(partitioner.border_vertices)}")

    # Step 2: Simplify each partition
    print("\n[Step 2] Simplifying partitions...")
    simplified_submeshes = []

    for idx, partition in enumerate(partitions):
        print(f"\nPartition {idx + 1}/{len(partitions)}:")
        print(f"  Core vertices: {len(partition['core_vertices'])}, Total vertices: {len(partition['vertices'])}")
        print(f"  Faces: {len(partition['faces'])}, Border vertices: {len(partition['is_border'])}")

        # Extract submesh
        submesh_vertices, submesh_faces, vertex_map = partitioner.extract_submesh(partition)

        # Create reverse mapping (local to global)
        reverse_map = {local: global_idx for global_idx, local in vertex_map.items()}

        # Identify border vertices in local indices
        local_border_vertices = {vertex_map[v] for v in partition['is_border'] if v in vertex_map}

        # Identify core vertices in local indices (vertices that belong to this partition's core)
        local_core_vertices = {vertex_map[v] for v in partition['core_vertices'] if v in vertex_map}

        # Simplify the submesh using LME
        simplifier = LMESimplifier(submesh_vertices, submesh_faces, local_border_vertices)
        simplified_vertices, simplified_faces = simplifier.simplify(target_ratio)

        # Filter faces to only include those with at least one vertex from the core
        # This ensures we don't duplicate faces from 2-ring extensions
        core_faces = []
        for face in simplified_faces:
            # Check if any vertex in the face maps back to a core vertex
            face_has_core_vertex = False
            for v_idx in face:
                # Try to find which original vertex this simplified vertex came from
                min_dist = float('inf')
                best_orig_local = None
                for orig_local_idx in range(len(submesh_vertices)):
                    if v_idx < len(simplified_vertices):
                        dist = np.linalg.norm(simplified_vertices[v_idx] - submesh_vertices[orig_local_idx])
                        if dist < min_dist:
                            min_dist = dist
                            best_orig_local = orig_local_idx

                if best_orig_local is not None and min_dist < 1e-5 and best_orig_local in local_core_vertices:
                    face_has_core_vertex = True
                    break

            if face_has_core_vertex:
                core_faces.append(face)

        print(f"  Filtered faces: {len(simplified_faces)} -> {len(core_faces)} (core only)")

        # Create a reverse map for simplified vertices
        # We need to track which original vertices each simplified vertex came from
        simplified_reverse_map = {}
        for local_idx in range(len(simplified_vertices)):
            # For now, we'll try to map back based on position matching
            # This is approximate but necessary since simplification changes vertex count
            simplified_reverse_map[local_idx] = None

        # Try to match simplified vertices back to original vertices by position
        for local_idx, simp_vert in enumerate(simplified_vertices):
            min_dist = float('inf')
            best_orig_idx = None
            for orig_local_idx, orig_global_idx in reverse_map.items():
                if orig_local_idx < len(submesh_vertices):
                    dist = np.linalg.norm(simp_vert - submesh_vertices[orig_local_idx])
                    if dist < min_dist:
                        min_dist = dist
                        best_orig_idx = orig_global_idx
            if min_dist < 1e-5:  # Close enough to consider it the same vertex
                simplified_reverse_map[local_idx] = best_orig_idx

        # Store simplified submesh with mappings (use core_faces instead of all faces)
        simplified_submeshes.append({
            'vertices': simplified_vertices,
            'faces': core_faces,  # Only include core faces
            'vertex_map': vertex_map,
            'reverse_map': simplified_reverse_map
        })

    # Step 3: Merge simplified submeshes
    print("\n[Step 3] Merging simplified submeshes...")
    merger = MeshMerger()
    final_vertices, final_faces = merger.merge_submeshes(simplified_submeshes, vertices)

    print(f"\n=== Simplification Complete ===")
    print(f"Output mesh: {len(final_vertices)} vertices, {len(final_faces)} faces")
    print(f"Reduction: {len(vertices)} -> {len(final_vertices)} vertices "
          f"({100 * len(final_vertices) / len(vertices):.1f}%)")
    print(f"Reduction: {len(faces)} -> {len(final_faces)} faces "
          f"({100 * len(final_faces) / len(faces):.1f}%)")

    return final_vertices, final_faces


def process_ply_file(
    input_path: str,
    output_path: str,
    target_ratio: float = 0.5,
    num_partitions: int = 8
) -> bool:
    """
    Process a single PLY file with partitioned mesh simplification.

    Args:
        input_path: Path to input PLY file
        output_path: Path to output PLY file
        target_ratio: Target simplification ratio
        num_partitions: Number of spatial partitions

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(input_path)}")
        print(f"{'='*70}")

        # Read input mesh
        print(f"Reading mesh from: {input_path}")
        vertices, faces = PLYReader.read_ply(input_path)

        # Simplify using partitioning
        simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
            vertices, faces, target_ratio, num_partitions
        )

        # Write output mesh
        print(f"\nWriting simplified mesh to: {output_path}")
        PLYWriter.write_ply(output_path, simplified_vertices, simplified_faces)

        print(f"✓ Successfully processed {os.path.basename(input_path)}")
        return True

    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to process PLY files from input directory to output directory.
    """
    # Default paths (can be modified by user)
    # Note: These are Windows paths from the requirements, but made cross-platform
    default_input_folder = r"D:\sxl08\rand1\neural-mesh-simplification\neural-mesh-simplification\demo\data"
    default_output_folder = r"D:\sxl08\rand1\neural-mesh-simplification\neural-mesh-simplification\demo\output"

    # Use relative paths if absolute paths don't exist
    input_folder = default_input_folder if os.path.exists(default_input_folder) else "./demo/data"
    output_folder = default_output_folder if os.path.exists(default_output_folder) else "./demo/output"

    # Parameters
    simplification_ratio = 0.5  # Keep 50% of vertices
    num_partitions = 8  # Octree partitioning (2x2x2)

    print("="*70)
    print("Mesh Simplification with MDD (Minimal Simplification Domain)")
    print("and LME (Local Minimal Edges)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Input folder:  {input_folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  Simplification ratio: {simplification_ratio}")
    print(f"  Number of partitions: {num_partitions}")

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"\n⚠ Input folder not found: {input_folder}")
        print(f"Please create the folder and add PLY files, or modify the paths in the script.")
        return

    # Process all PLY files in input folder
    ply_files = [f for f in os.listdir(input_folder) if f.endswith('.ply')]

    if not ply_files:
        print(f"\n⚠ No PLY files found in {input_folder}")
        return

    print(f"\nFound {len(ply_files)} PLY file(s) to process")

    # Process each file
    successful = 0
    failed = 0

    for filename in ply_files:
        input_path = os.path.join(input_folder, filename)
        output_filename = f"simplified_{filename}"
        output_path = os.path.join(output_folder, output_filename)

        if process_ply_file(input_path, output_path, simplification_ratio, num_partitions):
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*70}")
    print("Processing Summary")
    print(f"{'='*70}")
    print(f"Total files:     {len(ply_files)}")
    print(f"Successful:      {successful}")
    print(f"Failed:          {failed}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
