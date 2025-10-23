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


def is_valid_triangle(v1: int, v2: int, v3: int, vertices: np.ndarray, min_area: float = 1e-10) -> bool:
    """
    Check if triangle is valid (non-degenerate).
    
    Args:
        v1, v2, v3: Vertex indices of the triangle
        vertices: Array of vertex coordinates
        min_area: Minimum area threshold for valid triangle
    
    Returns:
        True if triangle is valid, False otherwise
    """
    # Check for duplicate vertices
    if v1 == v2 or v2 == v3 or v1 == v3:
        return False
    
    # Check if indices are valid
    if v1 >= len(vertices) or v2 >= len(vertices) or v3 >= len(vertices):
        return False
    
    p1, p2, p3 = vertices[v1], vertices[v2], vertices[v3]
    
    # Check area
    edge1 = p2 - p1
    edge2 = p3 - p1
    cross = np.cross(edge1, edge2)
    area = np.linalg.norm(cross) / 2
    
    return area > min_area


class MeshPartitioner:
    """
    Partitions a mesh into smaller sub-meshes based on spatial subdivision.
    Implements the MDD (Minimal Simplification Domain) concept with 2-ring neighborhood support.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, 
                 target_edges_per_partition: int = 200, num_partitions: int = None):
        """
        Initialize the mesh partitioner.

        Args:
            vertices: Array of vertex coordinates (N x 3)
            faces: Array of face indices (M x 3)
            target_edges_per_partition: Target number of edges per partition (default: 200)
            num_partitions: (Deprecated) Legacy parameter for backward compatibility
        """
        self.vertices = vertices
        self.faces = faces
        self.target_edges_per_partition = target_edges_per_partition
        self.num_partitions = num_partitions if num_partitions is not None else 8
        self.partitions = []
        self.border_vertices = set()  # Vertices on partition boundaries
        self.vertex_adjacency = None  # Will store vertex-to-vertex connectivity
        self.edges = None  # Will store all edges in the mesh
        self.edge_to_faces = None  # Maps edge to faces containing it

    def get_face_owner(self, face_idx: int, partitions: List[Dict]) -> int:
        """
        Determine which partition owns this face based on centroid.
        
        Args:
            face_idx: Index of the face
            partitions: List of partition dictionaries
        
        Returns:
            Index of the partition that owns this face
        """
        face = self.faces[face_idx]
        centroid = np.mean(self.vertices[face], axis=0)
        
        min_dist = float('inf')
        owner = 0
        for p_idx, partition in enumerate(partitions):
            # Compute partition center from core vertices
            core_verts = [self.vertices[v] for v in partition['core_vertices']]
            if core_verts:
                p_center = np.mean(core_verts, axis=0)
                dist = np.linalg.norm(centroid - p_center)
                if dist < min_dist:
                    min_dist = dist
                    owner = p_idx
        return owner

    def build_edges(self) -> Set[Tuple[int, int]]:
        """
        Build set of all edges in the mesh.
        
        Returns:
            Set of edges as (v1, v2) tuples where v1 < v2
        """
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
        return edges

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

    def partition_by_edge_count(self) -> List[Dict]:
        """
        Partition the mesh adaptively to keep approximately target_edges_per_partition edges in each partition.
        Uses recursive octree subdivision.
        
        Returns:
            List of partition dictionaries
        """
        # Build edges if not already built
        if self.edges is None:
            self.build_edges()
        
        total_edges = len(self.edges)
        estimated_partitions = max(1, total_edges // self.target_edges_per_partition)
        
        # Determine octree depth needed
        import math
        depth = max(1, int(math.ceil(math.log(estimated_partitions, 8))))
        
        print(f"  Total edges: {total_edges}")
        print(f"  Target edges per partition: {self.target_edges_per_partition}")
        print(f"  Using octree depth: {depth} (up to {8**depth} partitions)")
        
        return self.partition_octree_adaptive(depth)
    
    def partition_octree_adaptive(self, depth: int = 1) -> List[Dict]:
        """
        Partition the mesh using adaptive octree subdivision with specified depth.
        
        Args:
            depth: Octree subdivision depth (1 = 8 partitions, 2 = 64 partitions, etc.)
        
        Returns:
            List of partition dictionaries
        """
        # Calculate bounding box
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        
        # Assign each vertex to an octree cell
        vertex_partitions = np.zeros(len(self.vertices), dtype=np.int32)
        
        for i, vertex in enumerate(self.vertices):
            cell_idx = self._get_octree_cell(vertex, min_coords, max_coords, depth)
            vertex_partitions[i] = cell_idx
        
        # Count partitions
        num_partitions = 8 ** depth
        partition_data = [{'core_vertices': set(), 'vertices': set(), 'faces': [], 
                          'owned_faces': [], 'is_border': set(), 'edges': set()}
                         for _ in range(num_partitions)]
        
        # First pass: assign vertices to their core partitions
        for i in range(len(self.vertices)):
            partition_idx = vertex_partitions[i]
            partition_data[partition_idx]['core_vertices'].add(i)
        
        # Build edges for each partition
        for edge in self.edges:
            v1, v2 = edge
            p1, p2 = vertex_partitions[v1], vertex_partitions[v2]
            # Edge belongs to partition if both vertices are in it (or will be after 2-ring)
            if p1 == p2:
                partition_data[p1]['edges'].add(edge)
        
        # Second pass: expand each partition with 2-ring neighborhoods
        print("  Computing 2-ring neighborhoods for each partition...")
        for p_idx, p_data in enumerate(partition_data):
            if len(p_data['core_vertices']) > 0:
                extended_vertices = self.compute_n_ring_neighborhood(p_data['core_vertices'], n=2)
                p_data['vertices'] = extended_vertices
                
                # Count edges in this partition (including 2-ring)
                edge_count = 0
                for edge in self.edges:
                    v1, v2 = edge
                    if v1 in extended_vertices and v2 in extended_vertices:
                        edge_count += 1
                
                print(f"    Partition {p_idx}: {len(p_data['core_vertices'])} core vertices -> "
                      f"{len(extended_vertices)} vertices with 2-ring, ~{edge_count} edges")
        
        # Third pass: assign faces to partitions and determine ownership
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = face
            core_partitions = {vertex_partitions[v0], vertex_partitions[v1], vertex_partitions[v2]}
            
            # Assign face to partitions where all vertices are in the extended vertex set
            for p_idx, p_data in enumerate(partition_data):
                if v0 in p_data['vertices'] and v1 in p_data['vertices'] and v2 in p_data['vertices']:
                    p_data['faces'].append(face_idx)
            
            # Mark border vertices
            if len(core_partitions) > 1:
                for v in face:
                    self.border_vertices.add(v)
        
        # Fourth pass: determine face ownership
        for face_idx in range(len(self.faces)):
            owner_idx = self.get_face_owner(face_idx, partition_data)
            if face_idx in partition_data[owner_idx]['faces']:
                partition_data[owner_idx]['owned_faces'].append(face_idx)
        
        # Fifth pass: identify border vertices for each partition
        for p_idx, p_data in enumerate(partition_data):
            for v in p_data['vertices']:
                if v not in p_data['core_vertices']:
                    p_data['is_border'].add(v)
                elif v in self.border_vertices:
                    p_data['is_border'].add(v)
        
        # Filter out empty partitions
        self.partitions = [p for p in partition_data if len(p['faces']) > 0]
        
        return self.partitions
    
    def _get_octree_cell(self, vertex: np.ndarray, min_coords: np.ndarray, 
                         max_coords: np.ndarray, depth: int) -> int:
        """
        Get octree cell index for a vertex at specified depth.
        
        Args:
            vertex: Vertex coordinates
            min_coords: Minimum coordinates of bounding box
            max_coords: Maximum coordinates of bounding box
            depth: Octree depth
        
        Returns:
            Cell index
        """
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
        partition_data = [{'core_vertices': set(), 'vertices': set(), 'faces': [], 'owned_faces': [], 'is_border': set()}
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

        # Third pass: assign faces to partitions and determine ownership
        # A face belongs to a partition's extended set if all vertices are available (for 2-ring context)
        # A face is OWNED by the partition closest to its centroid (for output)
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = face

            # Determine which partition's core this face primarily belongs to
            core_partitions = {vertex_partitions[v0], vertex_partitions[v1], vertex_partitions[v2]}

            # Assign face to partitions where all vertices are in the extended vertex set (for context)
            for p_idx, p_data in enumerate(partition_data):
                if v0 in p_data['vertices'] and v1 in p_data['vertices'] and v2 in p_data['vertices']:
                    p_data['faces'].append(face_idx)

            # Determine if any vertex is a border vertex
            # A vertex is on the border if its face spans multiple core partitions
            if len(core_partitions) > 1:
                for v in face:
                    self.border_vertices.add(v)
        
        # Fourth pass: determine face ownership based on centroid (after all faces are assigned)
        # Each face is owned by exactly one partition
        for face_idx in range(len(self.faces)):
            owner_idx = self.get_face_owner(face_idx, partition_data)
            # Only add to owned_faces if this face exists in the partition's face list
            if face_idx in partition_data[owner_idx]['faces']:
                partition_data[owner_idx]['owned_faces'].append(face_idx)

        # Fifth pass: identify border vertices for each partition
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
    Implements LME validation according to the paper:
    - An edge is LME if its cost is minimal in its 2-ring neighborhood
    - LME edges form an independent set (no shared vertices)
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, border_vertices: Set[int],
                 vertex_adjacency: Dict[int, Set[int]] = None):
        """
        Initialize the LME simplifier.

        Args:
            vertices: Array of vertex coordinates
            faces: Array of face indices
            border_vertices: Set of vertex indices that are on partition borders
            vertex_adjacency: Pre-computed vertex adjacency (optional)
        """
        self.base_simplifier = QEMSimplifier(vertices, faces)
        self.border_vertices = border_vertices
        # Track vertex merging: maps each vertex to the set of original vertices it represents
        self.vertex_merge_map = {i: {i} for i in range(len(vertices))}
        
        # Build vertex adjacency if not provided
        if vertex_adjacency is None:
            self.vertex_adjacency = self._build_vertex_adjacency()
        else:
            self.vertex_adjacency = vertex_adjacency
    
    def _build_vertex_adjacency(self) -> Dict[int, Set[int]]:
        """Build vertex-to-vertex adjacency from faces."""
        adjacency = {i: set() for i in range(len(self.base_simplifier.vertices))}
        for face in self.base_simplifier.faces:
            v0, v1, v2 = face
            adjacency[v0].update([v1, v2])
            adjacency[v1].update([v0, v2])
            adjacency[v2].update([v0, v1])
        return adjacency
    
    def _get_2ring_neighborhood(self, vertex: int) -> Set[int]:
        """
        Get 2-ring neighborhood of a vertex.
        
        Args:
            vertex: Vertex index
        
        Returns:
            Set of vertex indices in 2-ring neighborhood
        """
        if vertex not in self.vertex_adjacency:
            return set()
        
        # 1-ring
        one_ring = self.vertex_adjacency[vertex].copy()
        one_ring.add(vertex)
        
        # 2-ring
        two_ring = one_ring.copy()
        for v in one_ring:
            if v in self.vertex_adjacency:
                two_ring.update(self.vertex_adjacency[v])
        
        return two_ring
    
    def _is_lme(self, edge: Tuple[int, int], edge_cost: float, 
                all_edges: List[Tuple[float, int, int, np.ndarray]]) -> bool:
        """
        Check if an edge is a Local Minimal Edge (LME).
        An edge is LME if its cost is minimal among all edges in its 2-ring neighborhood.
        
        Args:
            edge: Edge as (v1, v2) tuple
            edge_cost: Cost of this edge
            all_edges: List of all edges with their costs as (cost, v1, v2, optimal_pos)
        
        Returns:
            True if edge is LME, False otherwise
        """
        v1, v2 = edge
        
        # Get 2-ring neighborhood of both vertices
        neighborhood = self._get_2ring_neighborhood(v1) | self._get_2ring_neighborhood(v2)
        
        # Check if any edge in the neighborhood has lower cost
        for cost, e_v1, e_v2, _ in all_edges:
            if cost >= edge_cost:
                # Since edges are sorted, we can stop here
                break
            
            # Check if this edge is in the neighborhood
            if e_v1 in neighborhood or e_v2 in neighborhood:
                # Found a lower-cost edge in neighborhood, so current edge is not LME
                return False
        
        return True
    
    def _select_independent_lmes(self, lme_candidates: List[Tuple[float, int, int, np.ndarray]]) -> List[Tuple[float, int, int, np.ndarray]]:
        """
        Select independent LME edges (no shared vertices).
        Uses greedy selection: process edges in order of increasing cost.
        
        Args:
            lme_candidates: List of LME candidate edges as (cost, v1, v2, optimal_pos)
        
        Returns:
            List of independent LME edges
        """
        independent_lmes = []
        used_vertices = set()
        
        # Edges are already sorted by cost
        for cost, v1, v2, optimal_pos in lme_candidates:
            # Check if vertices are already used
            if v1 not in used_vertices and v2 not in used_vertices:
                independent_lmes.append((cost, v1, v2, optimal_pos))
                used_vertices.add(v1)
                used_vertices.add(v2)
        
        return independent_lmes

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

        # Compute costs for all interior edges and store them
        all_interior_edges = []
        
        for edge in edges:
            v1, v2 = edge
            # Skip edges involving border vertices
            if v1 in self.border_vertices or v2 in self.border_vertices:
                continue

            if v1 not in self.base_simplifier.valid_vertices or v2 not in self.base_simplifier.valid_vertices:
                continue

            optimal_pos = self.base_simplifier.compute_optimal_position(v1, v2)
            cost = self.base_simplifier.compute_cost(v1, v2, optimal_pos)
            all_interior_edges.append((cost, v1, v2, optimal_pos))
        
        # Sort edges by cost
        all_interior_edges.sort(key=lambda x: x[0])
        
        # Identify LME candidates (edges that are minimal in their 2-ring neighborhood)
        print(f"  Identifying LME candidates from {len(all_interior_edges)} interior edges...")
        lme_candidates = []
        
        for i, (cost, v1, v2, optimal_pos) in enumerate(all_interior_edges):
            if self._is_lme((v1, v2), cost, all_interior_edges[:i]):
                lme_candidates.append((cost, v1, v2, optimal_pos))
        
        print(f"  Found {len(lme_candidates)} LME candidates")
        
        # Select independent LMEs (no shared vertices)
        independent_lmes = self._select_independent_lmes(lme_candidates)
        print(f"  Selected {len(independent_lmes)} independent LMEs")
        
        # Create priority queue with LME edges
        import heapq
        heap = list(independent_lmes)
        heapq.heapify(heap)

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

            # Track vertex merging: v1 absorbs v2
            if v1 in self.vertex_merge_map and v2 in self.vertex_merge_map:
                self.vertex_merge_map[v1] = self.vertex_merge_map[v1] | self.vertex_merge_map[v2]
                # Remove v2's entry as it's now part of v1
                del self.vertex_merge_map[v2]

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


class BoundarySynchronizer:
    """
    Synchronizes boundary edge contractions across adjacent partitions.
    Ensures that boundary vertices match exactly after simplification.
    """
    
    def __init__(self, partitions: List[Dict], vertices: np.ndarray, faces: np.ndarray):
        """
        Initialize the boundary synchronizer.
        
        Args:
            partitions: List of partition dictionaries
            vertices: Original mesh vertices
            faces: Original mesh faces
        """
        self.partitions = partitions
        self.vertices = vertices
        self.faces = faces
        self.boundary_edges = {}  # Maps boundary edge -> list of partition indices
        self._identify_boundary_edges()
    
    def _identify_boundary_edges(self):
        """Identify edges that lie on partition boundaries."""
        # Build edges for each partition
        for p_idx, partition in enumerate(self.partitions):
            core_verts = partition['core_vertices']
            border_verts = partition['is_border']
            
            # Check edges in partition faces
            for face_idx in partition['faces']:
                face = self.faces[face_idx]
                for i in range(3):
                    v1, v2 = face[i], face[(i + 1) % 3]
                    edge = (min(v1, v2), max(v1, v2))
                    
                    # Edge is boundary if at least one vertex is border
                    if v1 in border_verts or v2 in border_verts:
                        if edge not in self.boundary_edges:
                            self.boundary_edges[edge] = []
                        if p_idx not in self.boundary_edges[edge]:
                            self.boundary_edges[edge].append(p_idx)
    
    def synchronize_boundary_contractions(self, simplifiers: List[LMESimplifier],
                                         edge_contractions: List[Dict]) -> List[Dict]:
        """
        Synchronize boundary edge contractions across partitions.
        
        Args:
            simplifiers: List of LMESimplifier instances for each partition
            edge_contractions: List of edge contractions for each partition as
                              {'edge': (v1, v2), 'optimal_pos': pos, 'global_v1': gv1, 'global_v2': gv2}
        
        Returns:
            List of synchronized edge contractions for each partition
        """
        # Group boundary edges by their global indices
        boundary_contraction_map = {}  # Maps global edge -> {'pos': optimal_pos, 'partitions': [p_idx, ...]}
        
        for p_idx, contractions in enumerate(edge_contractions):
            for contraction in contractions.get('boundary_edges', []):
                global_edge = contraction['global_edge']
                edge = (min(global_edge), max(global_edge))
                
                if edge in boundary_contraction_map:
                    # Edge already seen from another partition, verify position matches
                    existing_pos = boundary_contraction_map[edge]['pos']
                    new_pos = contraction['optimal_pos']
                    
                    # Use average position for consistency
                    avg_pos = (existing_pos + new_pos) / 2
                    boundary_contraction_map[edge]['pos'] = avg_pos
                    boundary_contraction_map[edge]['partitions'].append(p_idx)
                else:
                    boundary_contraction_map[edge] = {
                        'pos': contraction['optimal_pos'],
                        'partitions': [p_idx]
                    }
        
        # Update contractions with synchronized positions
        synchronized_contractions = []
        for p_idx in range(len(self.partitions)):
            partition_contractions = []
            
            for contraction in edge_contractions[p_idx].get('boundary_edges', []):
                global_edge = contraction['global_edge']
                edge = (min(global_edge), max(global_edge))
                
                if edge in boundary_contraction_map:
                    # Use synchronized position
                    contraction['optimal_pos'] = boundary_contraction_map[edge]['pos']
                
                partition_contractions.append(contraction)
            
            synchronized_contractions.append({
                'boundary_edges': partition_contractions,
                'interior_edges': edge_contractions[p_idx].get('interior_edges', [])
            })
        
        return synchronized_contractions


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

        # Third pass: process faces with validation
        for submesh_idx, face in temp_faces:
            merged_face = [self.global_vertex_map[(submesh_idx, v)] for v in face]
            # Check for degenerate faces
            if len(set(merged_face)) == 3:
                # Validate the triangle
                v1, v2, v3 = merged_face
                if v1 < len(self.merged_vertices) and v2 < len(self.merged_vertices) and v3 < len(self.merged_vertices):
                    vertices_array = np.array(self.merged_vertices)
                    if is_valid_triangle(v1, v2, v3, vertices_array):
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
    num_partitions: int = None,
    target_edges_per_partition: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplify a mesh using partitioning, local simplification, and merging.

    Args:
        vertices: Input mesh vertices (N x 3)
        faces: Input mesh faces (M x 3)
        target_ratio: Target simplification ratio (0.5 = keep 50% of vertices)
        num_partitions: (Deprecated) Number of partitions - kept for backward compatibility
        target_edges_per_partition: Target number of edges per partition (default: 200)

    Returns:
        Tuple of (simplified_vertices, simplified_faces)
    """
    print(f"\n=== Starting Mesh Simplification with MDD/LME ===")
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Target ratio: {target_ratio}")

    # Step 1: Partition the mesh by edge count
    print("\n[Step 1] Partitioning mesh by edge count...")
    partitioner = MeshPartitioner(vertices, faces, target_edges_per_partition)
    partitions = partitioner.partition_by_edge_count()
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

        # Determine which faces are owned by this partition
        # Use the owned_faces list from the partition to filter
        owned_face_indices = set(partition['owned_faces'])
        
        # Create a set of vertices that appear in owned faces (in original submesh space)
        owned_face_vertices = set()
        for global_face_idx in owned_face_indices:
            if global_face_idx in [partition['faces'][i] for i in range(len(partition['faces']))]:
                # Find local face index
                local_idx = partition['faces'].index(global_face_idx)
                if local_idx < len(submesh_faces):
                    owned_face_vertices.update(submesh_faces[local_idx])
        
        # Filter simplified faces: only keep faces where ALL vertices originated from owned face vertices
        # This ensures we only output faces that truly belong to this partition
        owned_simplified_faces = []
        for face in simplified_faces:
            # Check if all vertices in this face originated from vertices in owned faces
            all_from_owned = True
            for v in face:
                if v in simplifier.vertex_merge_map:
                    # Get the original vertices that merged into this vertex
                    orig_verts = simplifier.vertex_merge_map[v]
                    # Check if at least one originated from an owned face vertex
                    if not (orig_verts & owned_face_vertices):
                        all_from_owned = False
                        break
                else:
                    all_from_owned = False
                    break
            
            if all_from_owned:
                # Validate the face is not degenerate
                if len(set(face)) == 3 and is_valid_triangle(face[0], face[1], face[2], simplified_vertices):
                    owned_simplified_faces.append(face)

        print(f"  Filtered faces: {len(simplified_faces)} -> {len(owned_simplified_faces)} (owned only)")

        # Create a reverse map for simplified vertices using the merge map
        # Map simplified vertex indices to their original global vertex indices
        simplified_reverse_map = {}
        for simplified_idx in range(len(simplified_vertices)):
            if simplified_idx in simplifier.vertex_merge_map:
                # Pick one representative original vertex (preferably a border vertex if possible)
                orig_locals = simplifier.vertex_merge_map[simplified_idx]
                # Prefer border vertices as they're more stable
                border_orig = [ol for ol in orig_locals if ol in local_border_vertices]
                if border_orig:
                    representative = border_orig[0]
                else:
                    representative = min(orig_locals)  # Pick the first one
                
                # Map to global index
                if representative in reverse_map:
                    simplified_reverse_map[simplified_idx] = reverse_map[representative]
                else:
                    simplified_reverse_map[simplified_idx] = None
            else:
                simplified_reverse_map[simplified_idx] = None

        # Store simplified submesh with mappings (use owned faces only)
        simplified_submeshes.append({
            'vertices': simplified_vertices,
            'faces': owned_simplified_faces,  # Only include owned faces
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
    target_edges_per_partition: int = 200
) -> bool:
    """
    Process a single PLY file with partitioned mesh simplification.

    Args:
        input_path: Path to input PLY file
        output_path: Path to output PLY file
        target_ratio: Target simplification ratio
        target_edges_per_partition: Target number of edges per partition

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
            vertices, faces, target_ratio, target_edges_per_partition
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
    target_edges_per_partition = 200  # Target edges per partition

    print("="*70)
    print("Mesh Simplification with MDD (Minimal Simplification Domain)")
    print("and LME (Local Minimal Edges)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Input folder:  {input_folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  Simplification ratio: {simplification_ratio}")
    print(f"  Target edges per partition: {target_edges_per_partition}")

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

        if process_ply_file(input_path, output_path, simplification_ratio, target_edges_per_partition):
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
