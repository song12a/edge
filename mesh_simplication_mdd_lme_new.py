"""
Mesh Simplification using MDD (Minimal Simplification Domain) and LME (Local Minimal Edges)

Implements the iterative simplification strategy from the paper:
1. Iteratively execute: expand 2-ring → extract LME → synchronous contraction
2. After each contraction, re-expand patches and extract new LMEs
3. Synchronize boundary vertex contractions across adjacent partitions
4. Continue until target face count is reached
"""

import numpy as np
import os
from QEM import PLYReader, PLYWriter, QEMSimplifier
from typing import List, Tuple, Dict, Set
from collections import defaultdict


def is_valid_triangle(v1: int, v2: int, v3: int, vertices: np.ndarray, min_area: float = 1e-10) -> bool:
    """Check if triangle is valid (non-degenerate)."""
    if v1 == v2 or v2 == v3 or v1 == v3:
        return False
    if v1 >= len(vertices) or v2 >= len(vertices) or v3 >= len(vertices):
        return False

    p1, p2, p3 = vertices[v1], vertices[v2], vertices[v3]
    edge1 = p2 - p1
    edge2 = p3 - p1
    cross = np.cross(edge1, edge2)
    area = np.linalg.norm(cross) / 2

    return area > min_area


class MeshPartitioner:
    """Partitions mesh into patches with approximately equal edge counts."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, target_edges_per_partition: int = 200):
        self.vertices = vertices
        self.faces = faces
        self.target_edges_per_partition = target_edges_per_partition
        self.partitions = []
        self.vertex_adjacency = None
        self.edges = None
        self.edge_to_faces = None

    def build_vertex_adjacency(self) -> Dict[int, Set[int]]:
        """Build vertex-to-vertex adjacency."""
        adjacency = {i: set() for i in range(len(self.vertices))}
        for face in self.faces:
            v0, v1, v2 = face
            adjacency[v0].update([v1, v2])
            adjacency[v1].update([v0, v2])
            adjacency[v2].update([v0, v1])
        return adjacency

    def build_edges(self):
        """Build edge set and edge-to-faces mapping."""
        edges = set()
        edge_to_faces = {}

        for face_idx, face in enumerate(self.faces):
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                edges.add(edge)
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)

        self.edges = edges
        self.edge_to_faces = edge_to_faces

    def compute_n_ring_neighborhood(self, vertex_set: Set[int], n: int = 1) -> Set[int]:
        """Compute n-ring neighborhood of a set of vertices."""
        if self.vertex_adjacency is None:
            self.vertex_adjacency = self.build_vertex_adjacency()

        current_ring = vertex_set.copy()
        all_vertices = vertex_set.copy()

        for _ in range(n):
            next_ring = set()
            for vertex in current_ring:
                next_ring.update(self.vertex_adjacency[vertex])
            all_vertices.update(next_ring)
            current_ring = next_ring - vertex_set
            vertex_set = all_vertices.copy()

        return all_vertices

    def partition_by_edge_count(self) -> List[Dict]:
        """Partition mesh by edge count using octree."""
        if self.edges is None:
            self.build_edges()

        total_edges = len(self.edges)
        import math
        depth = max(1, int(math.ceil(math.log(total_edges / self.target_edges_per_partition, 8))))

        print(f"  Total edges: {total_edges}, Target per partition: {self.target_edges_per_partition}")
        print(f"  Using octree depth: {depth}")

        return self.partition_octree(depth)

    def partition_octree(self, depth: int = 1) -> List[Dict]:
        """Partition mesh using octree subdivision."""
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)

        vertex_partitions = np.zeros(len(self.vertices), dtype=np.int32)
        for i, vertex in enumerate(self.vertices):
            cell_idx = self._get_octree_cell(vertex, min_coords, max_coords, depth)
            vertex_partitions[i] = cell_idx

        num_partitions = 8 ** depth
        partition_data = []
        for _ in range(num_partitions):
            partition_data.append({
                'core_vertices': set(),
                'vertices': set(),
                'faces': [],
                'owned_faces': [],
                'is_border': set(),
                'edges': set()
            })

        # Assign vertices to core partitions
        for i in range(len(self.vertices)):
            partition_idx = vertex_partitions[i]
            partition_data[partition_idx]['core_vertices'].add(i)

        # Expand to 2-ring neighborhoods
        for p_idx, p_data in enumerate(partition_data):
            if len(p_data['core_vertices']) > 0:
                extended = self.compute_n_ring_neighborhood(p_data['core_vertices'], n=2)
                p_data['vertices'] = extended

        # Assign faces and determine ownership
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = face
            face_centroid = np.mean(self.vertices[face], axis=0)

            # Determine owner by closest partition center
            min_dist = float('inf')
            owner_idx = 0
            for p_idx, p_data in enumerate(partition_data):
                if len(p_data['core_vertices']) > 0:
                    core_verts = [self.vertices[v] for v in p_data['core_vertices']]
                    p_center = np.mean(core_verts, axis=0)
                    dist = np.linalg.norm(face_centroid - p_center)
                    if dist < min_dist:
                        min_dist = dist
                        owner_idx = p_idx

            # Assign to all partitions containing all vertices
            for p_idx, p_data in enumerate(partition_data):
                if v0 in p_data['vertices'] and v1 in p_data['vertices'] and v2 in p_data['vertices']:
                    p_data['faces'].append(face_idx)
                    if p_idx == owner_idx:
                        p_data['owned_faces'].append(face_idx)

        # Identify border vertices
        for p_idx, p_data in enumerate(partition_data):
            for v in p_data['vertices']:
                if v not in p_data['core_vertices']:
                    p_data['is_border'].add(v)

        self.partitions = [p for p in partition_data if len(p['faces']) > 0]
        return self.partitions

    def _get_octree_cell(self, vertex: np.ndarray, min_coords: np.ndarray,
                         max_coords: np.ndarray, depth: int) -> int:
        """Get octree cell index for a vertex."""
        cell_idx = 0
        current_min = min_coords.copy()
        current_max = max_coords.copy()

        for d in range(depth):
            center = (current_min + current_max) / 2
            octant = 0

            if vertex[0] > center[0]:
                octant += 1
                current_min[0] = center[0]
            else:
                current_max[0] = center[0]

            if vertex[1] > center[1]:
                octant += 2
                current_min[1] = center[1]
            else:
                current_max[1] = center[1]

            if vertex[2] > center[2]:
                octant += 4
                current_min[2] = center[2]
            else:
                current_max[2] = center[2]

            cell_idx = cell_idx * 8 + octant

        return cell_idx


class IterativeSimplifier:
    """
    Implements the iterative simplification strategy from the paper:
    Expand → Extract LME → Synchronous Contraction (repeat until target reached)
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, partitions: List[Dict]):
        self.original_vertices = vertices
        self.original_faces = faces
        self.partitions = partitions

        # Current state (updated after each iteration)
        self.current_vertices = vertices.copy()
        self.current_faces = faces.copy()
        self.valid_vertices = set(range(len(vertices)))
        self.valid_faces = set(range(len(faces)))

        # Build data structures
        self.vertex_adjacency = self._build_vertex_adjacency()
        self.edge_to_faces = self._build_edge_to_faces()
        self.vertex_quadrics = self._compute_quadrics()

        # Track boundary edges between partitions
        self.boundary_edges = self._identify_boundary_edges()

    def _build_vertex_adjacency(self) -> Dict[int, Set[int]]:
        """Build vertex adjacency from current faces."""
        adjacency = defaultdict(set)
        for face_idx in self.valid_faces:
            face = self.current_faces[face_idx]
            v0, v1, v2 = face
            if v0 in self.valid_vertices and v1 in self.valid_vertices and v2 in self.valid_vertices:
                adjacency[v0].update([v1, v2])
                adjacency[v1].update([v0, v2])
                adjacency[v2].update([v0, v1])
        return adjacency

    def _build_edge_to_faces(self) -> Dict[Tuple[int, int], List[int]]:
        """Build edge-to-faces mapping."""
        edge_to_faces = defaultdict(list)
        for face_idx in self.valid_faces:
            face = self.current_faces[face_idx]
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                edge_to_faces[edge].append(face_idx)
        return edge_to_faces

    def _compute_quadrics(self) -> Dict[int, np.ndarray]:
        """Compute QEM quadrics for all valid vertices."""
        quadrics = {v: np.zeros((4, 4)) for v in self.valid_vertices}

        for face_idx in self.valid_faces:
            face = self.current_faces[face_idx]
            v0, v1, v2 = face

            if not (v0 in self.valid_vertices and v1 in self.valid_vertices and v2 in self.valid_vertices):
                continue

            p0, p1, p2 = self.current_vertices[v0], self.current_vertices[v1], self.current_vertices[v2]

            # Compute plane equation
            edge1 = p1 - p0
            edge2 = p2 - p0
            normal = np.cross(edge1, edge2)
            length = np.linalg.norm(normal)

            if length > 1e-10:
                normal /= length
            else:
                normal = np.array([0, 0, 1])

            a, b, c = normal
            d = -np.dot(normal, p0)
            plane_eq = np.array([a, b, c, d])

            # Compute Kp matrix
            Kp = np.outer(plane_eq, plane_eq)

            # Add to vertices
            for v in face:
                if v in quadrics:
                    quadrics[v] += Kp

        return quadrics

    def _identify_boundary_edges(self) -> Dict[Tuple[int, int], List[int]]:
        """Identify edges on partition boundaries."""
        boundary_edges = defaultdict(list)

        for p_idx, partition in enumerate(self.partitions):
            border_verts = partition['is_border']
            for face_idx in partition['faces']:
                if face_idx not in self.valid_faces:
                    continue
                face = self.current_faces[face_idx]
                for i in range(3):
                    v1, v2 = face[i], face[(i + 1) % 3]
                    if v1 in border_verts or v2 in border_verts:
                        edge = (min(v1, v2), max(v1, v2))
                        if p_idx not in boundary_edges[edge]:
                            boundary_edges[edge].append(p_idx)

        return boundary_edges

    def expand_2ring(self, partition_idx: int) -> Set[int]:
        """Dynamically expand 2-ring neighborhood for a partition."""
        partition = self.partitions[partition_idx]
        core_vertices = partition['core_vertices'] & self.valid_vertices

        # Compute 2-ring from current valid vertices
        current_ring = core_vertices.copy()
        all_vertices = core_vertices.copy()

        for _ in range(2):
            next_ring = set()
            for vertex in current_ring:
                if vertex in self.vertex_adjacency:
                    next_ring.update(self.vertex_adjacency[vertex] & self.valid_vertices)
            all_vertices.update(next_ring)
            current_ring = next_ring - core_vertices
            core_vertices = all_vertices.copy()

        return all_vertices

    def extract_lme_edges(self, partition_idx: int, extended_vertices: Set[int]) -> List[Tuple[float, int, int, np.ndarray]]:
        """Extract LME edges from the extended region."""
        partition = self.partitions[partition_idx]
        border_verts = partition['is_border'] & self.valid_vertices

        # Find all candidate edges in the extended region
        candidate_edges = []
        for v1 in extended_vertices:
            if v1 not in self.vertex_adjacency:
                continue
            if v1 in border_verts:  # Skip border vertices
                continue

            for v2 in self.vertex_adjacency[v1]:
                if v2 not in extended_vertices or v2 in border_verts:
                    continue
                if v1 >= v2:  # Avoid duplicates
                    continue

                # Compute contraction cost
                optimal_pos = self._compute_optimal_position(v1, v2)
                cost = self._compute_cost(v1, v2, optimal_pos)
                candidate_edges.append((cost, v1, v2, optimal_pos))

        # Sort by cost
        candidate_edges.sort(key=lambda x: x[0])

        # Filter for LME: edge is LME if minimal in its 2-ring neighborhood
        lme_edges = []
        for cost, v1, v2, optimal_pos in candidate_edges:
            if self._is_lme(v1, v2, cost, candidate_edges, extended_vertices):
                lme_edges.append((cost, v1, v2, optimal_pos))

        return lme_edges

    def _compute_optimal_position(self, v1: int, v2: int) -> np.ndarray:
        """Compute optimal position for contracting edge (v1, v2)."""
        Q = self.vertex_quadrics[v1] + self.vertex_quadrics[v2]
        A = Q[:3, :3]
        b = -Q[:3, 3]

        try:
            optimal_pos = np.linalg.solve(A, b)
            if not (np.any(np.isnan(optimal_pos)) or np.any(np.isinf(optimal_pos))):
                return optimal_pos
        except np.linalg.LinAlgError:
            pass

        # Fallback: midpoint
        return (self.current_vertices[v1] + self.current_vertices[v2]) / 2

    def _compute_cost(self, v1: int, v2: int, optimal_pos: np.ndarray) -> float:
        """Compute QEM cost for contracting edge."""
        Q = self.vertex_quadrics[v1] + self.vertex_quadrics[v2]
        homo_pos = np.append(optimal_pos, 1)
        cost = homo_pos @ Q @ homo_pos.T
        return cost

    def _is_lme(self, v1: int, v2: int, cost: float,
                all_edges: List[Tuple[float, int, int, np.ndarray]],
                extended_vertices: Set[int]) -> bool:
        """Check if edge is LME (minimal in 2-ring neighborhood)."""
        # Get 2-ring neighborhood
        neighborhood = self._get_2ring_vertices(v1) | self._get_2ring_vertices(v2)
        neighborhood &= extended_vertices

        # Check if any edge in neighborhood has lower cost
        for e_cost, e_v1, e_v2, _ in all_edges:
            if e_cost >= cost:
                break
            if e_v1 in neighborhood or e_v2 in neighborhood:
                return False

        return True

    def _get_2ring_vertices(self, vertex: int) -> Set[int]:
        """Get 2-ring neighborhood of a vertex."""
        if vertex not in self.vertex_adjacency:
            return set()

        one_ring = self.vertex_adjacency[vertex].copy()
        one_ring.add(vertex)

        two_ring = one_ring.copy()
        for v in one_ring:
            if v in self.vertex_adjacency:
                two_ring.update(self.vertex_adjacency[v])

        return two_ring

    def synchronize_boundary_contractions(self, all_lmes: Dict[int, List]) -> Dict[int, List]:
        """
        Synchronize boundary edge contractions across adjacent partitions.
        For each boundary edge, ensure only one partition contracts it and others lock vertices.
        """
        # Group boundary edges by their partitions
        boundary_operations = {}  # Maps edge -> {'owner': p_idx, 'cost': cost, 'pos': optimal_pos, 'partners': [p_idx, ...]}

        for p_idx, lmes in all_lmes.items():
            for cost, v1, v2, optimal_pos in lmes:
                edge = (min(v1, v2), max(v1, v2))

                # Check if this is a boundary edge
                if edge in self.boundary_edges and len(self.boundary_edges[edge]) > 1:
                    if edge not in boundary_operations:
                        # First partition to propose this boundary edge contraction
                        boundary_operations[edge] = {
                            'owner': p_idx,
                            'cost': cost,
                            'pos': optimal_pos,
                            'partners': self.boundary_edges[edge].copy()
                        }
                    else:
                        # Another partition also wants to contract this edge
                        # Choose the one with lower cost as owner
                        if cost < boundary_operations[edge]['cost']:
                            boundary_operations[edge]['owner'] = p_idx
                            boundary_operations[edge]['cost'] = cost
                            boundary_operations[edge]['pos'] = optimal_pos

        # Update LMEs: remove boundary edges from non-owner partitions
        synchronized_lmes = {}
        for p_idx, lmes in all_lmes.items():
            filtered_lmes = []
            for cost, v1, v2, optimal_pos in lmes:
                edge = (min(v1, v2), max(v1, v2))

                if edge in boundary_operations:
                    # This is a boundary edge
                    if boundary_operations[edge]['owner'] == p_idx:
                        # This partition owns the contraction
                        # Use synchronized position (average of all proposals)
                        filtered_lmes.append((cost, v1, v2, boundary_operations[edge]['pos']))
                    # Else: don't contract, lock vertices
                else:
                    # Interior edge, keep as is
                    filtered_lmes.append((cost, v1, v2, optimal_pos))

            synchronized_lmes[p_idx] = filtered_lmes

        return synchronized_lmes

    def contract_edge(self, v1: int, v2: int, optimal_pos: np.ndarray):
        """Contract edge (v1, v2) by merging v2 into v1."""
        # Update v1 position
        self.current_vertices[v1] = optimal_pos

        # Merge quadrics
        self.vertex_quadrics[v1] += self.vertex_quadrics[v2]

        # Update faces: replace v2 with v1
        faces_to_remove = set()
        for face_idx in list(self.valid_faces):
            face = self.current_faces[face_idx]
            new_face = []
            for v in face:
                if v == v2:
                    new_face.append(v1)
                else:
                    new_face.append(v)

            # Check for degenerate faces
            if len(set(new_face)) < 3:
                faces_to_remove.add(face_idx)
            else:
                self.current_faces[face_idx] = new_face

        # Remove degenerate faces
        self.valid_faces -= faces_to_remove

        # Remove v2
        self.valid_vertices.discard(v2)
        if v2 in self.vertex_quadrics:
            del self.vertex_quadrics[v2]

        # Update adjacency
        self.vertex_adjacency = self._build_vertex_adjacency()

    def simplify_iteratively(self, target_face_count: int, max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iteratively simplify the mesh:
        1. Expand 2-ring neighborhoods for all partitions
        2. Extract LME edges from each partition
        3. Synchronize boundary contractions
        4. Perform contractions
        5. Repeat until target face count is reached
        """
        print(f"\n  Iterative simplification: target {target_face_count} faces")

        iteration = 0
        while len(self.valid_faces) > target_face_count and iteration < max_iterations:
            iteration += 1
            print(f"\n  Iteration {iteration}: {len(self.valid_faces)} faces, {len(self.valid_vertices)} vertices")

            # Step 1: Expand 2-ring for all partitions
            expanded_regions = {}
            for p_idx in range(len(self.partitions)):
                expanded_regions[p_idx] = self.expand_2ring(p_idx)

            # Step 2: Extract LME edges from each partition
            all_lmes = {}
            for p_idx, extended_verts in expanded_regions.items():
                lmes = self.extract_lme_edges(p_idx, extended_verts)
                all_lmes[p_idx] = lmes
                print(f"    Partition {p_idx}: {len(lmes)} LME edges")

            # Step 3: Synchronize boundary contractions
            synchronized_lmes = self.synchronize_boundary_contractions(all_lmes)

            # Step 4: Select independent LMEs (no shared vertices) and contract
            contracted_count = 0
            used_vertices = set()

            # Collect all LMEs and sort by cost
            all_synchronized_lmes = []
            for p_idx, lmes in synchronized_lmes.items():
                for cost, v1, v2, optimal_pos in lmes:
                    all_synchronized_lmes.append((cost, v1, v2, optimal_pos, p_idx))

            all_synchronized_lmes.sort(key=lambda x: x[0])

            # Greedily contract independent edges
            for cost, v1, v2, optimal_pos, p_idx in all_synchronized_lmes:
                if v1 not in used_vertices and v2 not in used_vertices:
                    if v1 in self.valid_vertices and v2 in self.valid_vertices:
                        self.contract_edge(v1, v2, optimal_pos)
                        used_vertices.add(v1)
                        used_vertices.add(v2)
                        contracted_count += 1

            print(f"    Contracted {contracted_count} edges")

            # Check if we've reached the target
            if len(self.valid_faces) <= target_face_count:
                print(f"  Reached target face count: {len(self.valid_faces)}")
                break

            # If no contractions were made, stop
            if contracted_count == 0:
                print(f"  No more contractions possible, stopping")
                break

        # Rebuild final mesh
        return self._rebuild_mesh()

    def _rebuild_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Rebuild final simplified mesh."""
        # Map old vertex indices to new indices
        valid_vertex_list = sorted(self.valid_vertices)
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_vertex_list)}

        # Create new vertex array
        new_vertices = np.array([self.current_vertices[i] for i in valid_vertex_list], dtype=np.float32)

        # Create new face array
        new_faces = []
        for face_idx in self.valid_faces:
            face = self.current_faces[face_idx]
            if all(v in vertex_map for v in face):
                new_face = [vertex_map[v] for v in face]
                if len(set(new_face)) == 3:
                    if is_valid_triangle(new_face[0], new_face[1], new_face[2], new_vertices):
                        new_faces.append(new_face)

        new_faces = np.array(new_faces, dtype=np.int32)

        return new_vertices, new_faces


def simplify_mesh_with_partitioning(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_ratio: float = 0.5,
    target_edges_per_partition: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplify mesh using iterative MDD/LME approach.
    """
    print(f"\n=== Iterative Mesh Simplification with MDD/LME ===")
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Target ratio: {target_ratio}")

    # Step 1: Partition the mesh
    print("\n[Step 1] Partitioning mesh...")
    partitioner = MeshPartitioner(vertices, faces, target_edges_per_partition)
    partitions = partitioner.partition_by_edge_count()
    print(f"Created {len(partitions)} partitions")

    # Step 2: Iterative simplification
    print("\n[Step 2] Iterative simplification...")
    target_face_count = int(len(faces) * target_ratio)
    simplifier = IterativeSimplifier(vertices, faces, partitions)
    simplified_vertices, simplified_faces = simplifier.simplify_iteratively(target_face_count)

    print(f"\n=== Simplification Complete ===")
    print(f"Output mesh: {len(simplified_vertices)} vertices, {len(simplified_faces)} faces")
    print(f"Reduction: {len(vertices)} -> {len(simplified_vertices)} vertices "
          f"({100 * len(simplified_vertices) / len(vertices):.1f}%)")
    print(f"Reduction: {len(faces)} -> {len(simplified_faces)} faces "
          f"({100 * len(simplified_faces) / len(faces):.1f}%)")

    return simplified_vertices, simplified_faces


def process_ply_file(input_path: str, output_path: str, target_ratio: float = 0.5,
                     target_edges_per_partition: int = 200) -> bool:
    """Process a single PLY file."""
    try:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(input_path)}")
        print(f"{'='*70}")

        vertices, faces = PLYReader.read_ply(input_path)
        simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
            vertices, faces, target_ratio, target_edges_per_partition
        )

        PLYWriter.write_ply(output_path, simplified_vertices, simplified_faces)
        print(f"✓ Successfully processed {os.path.basename(input_path)}")
        return True

    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    input_folder = r"demo/data"
    output_folder = r"demo/output"

    if not os.path.exists(input_folder):
        input_folder = "./demo/data"
        output_folder = "./demo/output"

    simplification_ratio = 0.5
    target_edges_per_partition = 200

    print("="*70)
    print("Iterative Mesh Simplification with MDD and LME")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Input folder:  {input_folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  Simplification ratio: {simplification_ratio}")
    print(f"  Target edges per partition: {target_edges_per_partition}")

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"\n⚠ Input folder not found: {input_folder}")
        return

    ply_files = [f for f in os.listdir(input_folder) if f.endswith('.ply')]
    if not ply_files:
        print(f"\n⚠ No PLY files found in {input_folder}")
        return

    print(f"\nFound {len(ply_files)} PLY file(s) to process")

    successful = 0
    failed = 0

    for filename in ply_files:
        input_path = os.path.join(input_folder, filename)
        output_filename = f"simplified_{filename}"
        output_path = os.path.join(output_folder, output_filename)

        if process_ply_file(input_path, output_path, simplification_ratio, target_edges_per_partition):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print("Processing Summary")
    print(f"{'='*70}")
    print(f"Total files:  {len(ply_files)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()