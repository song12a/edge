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
    Partitions a mesh into smaller sub-meshes based on edge count.
    Implements the MDD (Minimal Simplification Domain) concept with 2-ring neighborhood support.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, target_edges_per_partition: int = 200):
        """
        Initialize the mesh partitioner.

        Args:
            vertices: Array of vertex coordinates (N x 3)
            faces: Array of face indices (M x 3)
            target_edges_per_partition: Target number of edges per partition (default: 200)
        """
        self.vertices = vertices
        self.faces = faces
        self.target_edges_per_partition = target_edges_per_partition
        self.partitions = []
        self.border_vertices = set()  # Vertices on partition boundaries
        self.vertex_adjacency = None  # Will store vertex-to-vertex connectivity
        self.edge_set = None  # Will store all edges in the mesh

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

    def build_edge_set(self) -> Set[Tuple[int, int]]:
        """
        Build a set of all edges in the mesh.

        Returns:
            Set of edges, where each edge is a tuple (min_vertex, max_vertex).
        """
        edges = set()
        for face in self.faces:
            v0, v1, v2 = face
            # Add the three edges of the triangle
            edges.add((min(v0, v1), max(v0, v1)))
            edges.add((min(v1, v2), max(v1, v2)))
            edges.add((min(v2, v0), max(v2, v0)))
        return edges

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

    def partition_bfs(self) -> List[Dict]:
        """
        Partition the mesh using BFS (Breadth-First Search) with target edge count.
        Creates partitions with approximately target_edges_per_partition edges and complete 2-ring neighborhoods.

        According to the paper:
        - Boundary vertices (at intersections between sub-meshes) CAN be simplified
        - 2-ring neighborhood vertices also CAN be simplified (their LME can be simplified)
        - The 2-ring provides topological context for accurate QEM, but doesn't prevent simplification

        Each partition includes:
        - Core vertices: vertices in the partition
        - Extended vertices: vertices in the 2-ring neighborhood of core vertices (MDD)

        Returns:
            List of partition dictionaries, each containing:
                - 'vertices': all vertex indices in this partition (core + 2-ring)
                - 'core_vertices': vertex indices in the core partition
                - 'faces': face indices in this partition
                - 'is_border': set of border vertices (at intersections between partitions)
        """
        # Build adjacency and edge information
        if self.vertex_adjacency is None:
            self.vertex_adjacency = self.build_vertex_adjacency()
        if self.edge_set is None:
            self.edge_set = self.build_edge_set()

        # Build face adjacency (faces sharing edges)
        edge_faces = {}  # Maps edge tuple to list of face indices
        for face_idx, face in enumerate(self.faces):
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(face_idx)

        # Track which faces have been assigned to partitions
        assigned_faces = set()
        partition_data = []

        print(f"  Partitioning mesh using BFS with target {self.target_edges_per_partition} edges per partition...")
        print(f"  Total edges in mesh: {len(self.edge_set)}")

        while len(assigned_faces) < len(self.faces):
            # Find an unassigned face as seed for BFS
            seed_face = None
            for face_idx in range(len(self.faces)):
                if face_idx not in assigned_faces:
                    seed_face = face_idx
                    break

            if seed_face is None:
                break

            # BFS to grow partition
            from collections import deque
            queue = deque([seed_face])
            core_faces = set()
            core_vertices = set()
            assigned_faces.add(seed_face)

            # Helper function to count edges in partition
            def count_edges_in_partition(vertices_set):
                edge_count = 0
                for v1, v2 in self.edge_set:
                    if v1 in vertices_set and v2 in vertices_set:
                        edge_count += 1
                return edge_count

            # BFS expansion
            while queue:
                current_face_idx = queue.popleft()
                current_face = self.faces[current_face_idx]

                # Add face to core
                core_faces.add(current_face_idx)
                core_vertices.update(current_face)

                # Check if we should continue growing
                current_edge_count = count_edges_in_partition(core_vertices)

                # Stop if we've reached target (with tolerance)
                if current_edge_count >= self.target_edges_per_partition * 1.2:
                    break

                # Only continue BFS if below target
                if current_edge_count < self.target_edges_per_partition * 0.8:
                    # Find adjacent faces through shared edges
                    for i in range(3):
                        v1, v2 = current_face[i], current_face[(i + 1) % 3]
                        edge = (min(v1, v2), max(v1, v2))

                        if edge in edge_faces:
                            for neighbor_face_idx in edge_faces[edge]:
                                if neighbor_face_idx not in assigned_faces and neighbor_face_idx != current_face_idx:
                                    assigned_faces.add(neighbor_face_idx)
                                    queue.append(neighbor_face_idx)

            # Create partition with 2-ring neighborhood (MDD requirement)
            extended_vertices = self.compute_n_ring_neighborhood(core_vertices, n=2)

            # Find all faces that are fully contained in extended vertices
            partition_faces = []
            for face_idx, face in enumerate(self.faces):
                if all(v in extended_vertices for v in face):
                    partition_faces.append(face_idx)

            edge_count = count_edges_in_partition(core_vertices)

            partition = {
                'core_vertices': core_vertices,
                'vertices': extended_vertices,
                'faces': partition_faces,
                'is_border': set()  # Will be filled later
            }
            partition_data.append(partition)

            print(f"    Partition {len(partition_data)}: {len(core_vertices)} core vertices, "
                  f"{len(extended_vertices)} total vertices (2-ring), {edge_count} edges, {len(partition_faces)} faces")

        # Identify border vertices according to paper definition:
        # Vertices at the intersection between different sub-meshes (core partitions)
        vertex_partition_count = {i: 0 for i in range(len(self.vertices))}
        for p in partition_data:
            for v in p['core_vertices']:
                vertex_partition_count[v] += 1

        # Border vertices are those in multiple core partitions
        for v, count in vertex_partition_count.items():
            if count > 1:
                self.border_vertices.add(v)

        # Mark border vertices in each partition
        # According to paper: boundary vertices CAN be simplified
        for p_idx, p_data in enumerate(partition_data):
            for v in p_data['vertices']:
                # Vertices not in core are part of 2-ring extension (should NOT be simplified)
                if v not in p_data['core_vertices']:
                    p_data['is_border'].add(v)
                # Vertices in core but shared between partitions are boundary vertices (CAN be simplified)
                # We mark them but they are still simplifiable
                elif v in self.border_vertices:
                    p_data['is_border'].add(v)

        self.partitions = partition_data
        print(f"  Created {len(partition_data)} partitions")
        print(f"  Border vertices (at partition intersections): {len(self.border_vertices)}")

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
    Local Minimal Edges (LME) Simplifier that extends QEM to simplify meshes including boundaries.
    Implements boundary simplification with subsequent boundary alignment as per the paper.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, border_vertices: Set[int],
                 two_ring_extension: Set[int]):
        """
        Initialize the LME simplifier.

        Args:
            vertices: Array of vertex coordinates
            faces: Array of face indices
            border_vertices: Set of vertex indices that are on partition borders (can be simplified)
            two_ring_extension: Set of vertex indices in 2-ring extension (can also be simplified per paper)
        """
        self.base_simplifier = QEMSimplifier(vertices, faces)
        self.border_vertices = border_vertices
        self.two_ring_extension = two_ring_extension

        # Track which original vertices each vertex represents (for border alignment)
        # Key: current vertex index, Value: set of original vertex indices it represents
        self.vertex_lineage = {i: {i} for i in range(len(vertices))}

    def simplify(self, target_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplify the mesh including boundary and 2-ring vertices, following the paper's approach.
        According to the paper, LME (Local Minimal Edges) within the 2-ring neighborhood can be simplified.
        
        This implementation ensures that for each vertex v_i, only the edge e(i,j) with the minimal
        QEM cost among all edges connected to v_i is chosen for collapse.

        Args:
            target_ratio: Target ratio of vertices to retain

        Returns:
            Tuple of (simplified_vertices, simplified_faces, vertex_lineage)
        """
        # Calculate target vertex count
        # Per paper: All vertices can be simplified, including 2-ring extension
        num_total = len(self.base_simplifier.vertices)
        num_border = len(self.border_vertices)
        num_two_ring = len(self.two_ring_extension)
        num_interior = num_total - num_border - num_two_ring
        
        # Calculate target, but ensure we don't over-simplify
        # Minimum should be 4 vertices OR at least keep all border vertices to maintain structure
        min_vertices = max(4, num_border)
        target_vertex_count = max(min_vertices, int(num_total * target_ratio))
        
        # Additional safety: if almost all vertices are border/2-ring, be more conservative
        if num_interior < num_total * 0.2:  # Less than 20% interior vertices
            # Keep more vertices to maintain mesh structure
            target_vertex_count = max(target_vertex_count, int(num_total * 0.7))
            print(f"  ⚠ Warning: Partition has few interior vertices ({num_interior}/{num_total}), "
                  f"adjusting target to {target_vertex_count} to prevent over-simplification")

        print(f"  LME Simplification: {num_total} vertices ({num_two_ring} in 2-ring, "
              f"{num_border} border, {num_interior} interior)")
        print(f"  Target: {target_vertex_count} vertices (minimum: {min_vertices})")

        # Find all valid edges
        edges = self.base_simplifier.find_valid_edges()

        # Helper function to compute edge cost with priority weighting
        def compute_edge_cost(v1, v2):
            """Compute cost for an edge with priority weighting."""
            optimal_pos = self.base_simplifier.compute_optimal_position(v1, v2)
            base_cost = self.base_simplifier.compute_cost(v1, v2, optimal_pos)
            
            # Prioritize interior edges, then boundary edges, then 2-ring extension edges
            if v1 in self.two_ring_extension or v2 in self.two_ring_extension:
                cost = base_cost * 1.2  # Lower priority for 2-ring extension edges
            elif v1 in self.border_vertices or v2 in self.border_vertices:
                cost = base_cost * 1.1  # Lower priority for boundary edges
            else:
                cost = base_cost
            
            return cost, optimal_pos

        # Build vertex-to-edges mapping
        import heapq
        vertex_edges = {v: set() for v in self.base_simplifier.valid_vertices}
        
        for edge in edges:
            v1, v2 = edge
            if v1 in self.base_simplifier.valid_vertices and v2 in self.base_simplifier.valid_vertices:
                vertex_edges[v1].add(edge)
                vertex_edges[v2].add(edge)

        # For each vertex, find the Local Minimal Edge (LME)
        # LME for vertex v_i is the edge with minimal cost among all edges connected to v_i
        def find_vertex_lme(vertex):
            """Find the LME for a given vertex by recomputing costs for all its edges."""
            if vertex not in vertex_edges or vertex not in self.base_simplifier.valid_vertices:
                return None
            
            min_cost = float('inf')
            min_edge = None
            min_optimal_pos = None
            
            for edge in vertex_edges[vertex]:
                v1, v2 = edge
                # Check if both vertices are still valid
                if v1 not in self.base_simplifier.valid_vertices or v2 not in self.base_simplifier.valid_vertices:
                    continue
                
                # Recompute cost for this edge (critical for correctness after collapses)
                cost, optimal_pos = compute_edge_cost(v1, v2)
                
                if cost < min_cost:
                    min_cost = cost
                    min_edge = edge
                    min_optimal_pos = optimal_pos
            
            if min_edge is not None:
                return (min_cost, min_edge, min_optimal_pos)
            return None
        
        # Create priority queue with only LME edges
        # An edge is added to the heap if it's the LME for at least one of its endpoints
        heap = []
        heap_edges = set()  # Track which edges are in heap
        
        # Initialize heap with LME edges
        for vertex in self.base_simplifier.valid_vertices:
            lme_result = find_vertex_lme(vertex)
            if lme_result:
                cost, edge, optimal_pos = lme_result
                if edge not in heap_edges:
                    v1, v2 = edge
                    heapq.heappush(heap, (cost, v1, v2, optimal_pos))
                    heap_edges.add(edge)

        # Track border vertex movements for later alignment
        self.border_vertex_mapping = {v: v for v in self.border_vertices}

        # Perform edge contractions
        contraction_count = 0
        current_vertex_count = len(self.base_simplifier.valid_vertices)

        while current_vertex_count > target_vertex_count and heap:
            cost, v1, v2, optimal_pos = heapq.heappop(heap)
            edge = (min(v1, v2), max(v1, v2))
            heap_edges.discard(edge)

            # Check if vertices are still valid
            if v1 not in self.base_simplifier.valid_vertices or v2 not in self.base_simplifier.valid_vertices:
                continue
            
            # Verify this edge is still the LME for at least one endpoint
            # This prevents using stale heap entries
            is_lme = False
            
            lme_v1 = find_vertex_lme(v1)
            if lme_v1 and lme_v1[1] == edge:
                is_lme = True
            
            lme_v2 = find_vertex_lme(v2)
            if lme_v2 and lme_v2[1] == edge:
                is_lme = True
            
            if not is_lme:
                # This edge is no longer an LME for either endpoint, skip it
                continue

            # Collect all vertices that will be affected by this collapse
            affected_vertices = set()
            for v in [v1, v2]:
                if v in vertex_edges:
                    for e in vertex_edges[v]:
                        affected_vertices.add(e[0])
                        affected_vertices.add(e[1])

            # Before contraction, track vertex lineage (v2 merges into v1)
            if v2 in self.vertex_lineage:
                if v1 not in self.vertex_lineage:
                    self.vertex_lineage[v1] = {v1}
                # v1 now represents all vertices that v2 represented, plus itself
                self.vertex_lineage[v1].update(self.vertex_lineage[v2])
                del self.vertex_lineage[v2]

            # Contract edge - v2 merges into v1 at optimal_pos
            self.base_simplifier.contract_edge(v1, v2, optimal_pos)

            # CRITICAL FIX: Rebuild edge connectivity from actual mesh
            # After collapse, the mesh topology changes - v1 now connects to vertices that
            # were previously only connected to v2. We must rebuild the connectivity graph
            # to reflect the actual current mesh structure.
            
            # Remove v2 from vertex_edges first
            if v2 in vertex_edges:
                del vertex_edges[v2]
            
            # Collect ALL vertices whose edges need to be updated
            # This includes affected_vertices AND any new neighbors discovered during rebuild
            vertices_to_update = affected_vertices.copy()
            vertices_to_update.discard(v2)  # v2 is deleted
            
            # PHASE 1: Clear all old edges for affected vertices
            # This must be done in a separate pass to avoid mixing old and new edges
            for vertex in list(vertices_to_update):
                if vertex not in self.base_simplifier.valid_vertices:
                    continue
                
                # Clear old edges for this vertex
                if vertex in vertex_edges:
                    old_edges = vertex_edges[vertex].copy()
                    # Remove this vertex from all its old edges
                    for e in old_edges:
                        for vx in [e[0], e[1]]:
                            if vx in vertex_edges and vx != vertex:
                                vertex_edges[vx].discard(e)
                    vertex_edges[vertex].clear()
                else:
                    vertex_edges[vertex] = set()
            
            # PHASE 2: Rebuild edges from current mesh faces
            # Discover and clear any additional vertices that weren't in affected_vertices
            new_edges_to_add = []  # Collect edges first
            
            for vertex in list(vertices_to_update):
                if vertex not in self.base_simplifier.valid_vertices:
                    continue
                
                # Collect edges for this vertex from current mesh faces
                if vertex in self.base_simplifier.vertex_faces:
                    for face_idx in self.base_simplifier.vertex_faces[vertex]:
                        if face_idx < len(self.base_simplifier.faces):
                            face = self.base_simplifier.faces[face_idx]
                            # Extract edges from this face
                            for i in range(3):
                                fv1, fv2 = face[i], face[(i + 1) % 3]
                                if fv1 != fv2:  # Skip degenerate edges
                                    edge_tuple = (min(fv1, fv2), max(fv1, fv2))
                                    if fv1 in self.base_simplifier.valid_vertices and fv2 in self.base_simplifier.valid_vertices:
                                        new_edges_to_add.append(edge_tuple)
                                        # If we discover a vertex not in vertices_to_update, clear it now
                                        for vx in [fv1, fv2]:
                                            if vx not in vertices_to_update and vx in vertex_edges:
                                                # Clear this newly discovered vertex
                                                old_edges = vertex_edges[vx].copy()
                                                for e in old_edges:
                                                    for vy in [e[0], e[1]]:
                                                        if vy in vertex_edges and vy != vx:
                                                            vertex_edges[vy].discard(e)
                                                vertex_edges[vx].clear()
                                                vertices_to_update.add(vx)
                                            elif vx not in vertex_edges:
                                                vertex_edges[vx] = set()
                                                vertices_to_update.add(vx)
            
            # PHASE 3: Add all the new edges
            for edge_tuple in new_edges_to_add:
                fv1, fv2 = edge_tuple
                if fv1 in vertex_edges:
                    vertex_edges[fv1].add(edge_tuple)
                if fv2 in vertex_edges:
                    vertex_edges[fv2].add(edge_tuple)
            
            # Recompute LME for all affected vertices and add new LME edges to heap
            for vertex in vertices_to_update:
                if vertex in self.base_simplifier.valid_vertices:
                    lme_result = find_vertex_lme(vertex)
                    if lme_result:
                        new_cost, new_edge, new_optimal_pos = lme_result
                        if new_edge not in heap_edges:
                            nv1, nv2 = new_edge
                            heapq.heappush(heap, (new_cost, nv1, nv2, new_optimal_pos))
                            heap_edges.add(new_edge)

            contraction_count += 1
            current_vertex_count = len(self.base_simplifier.valid_vertices)

            if contraction_count % 50 == 0:
                print(f"    Contracted {contraction_count} edges, {current_vertex_count} vertices remaining")

        # Rebuild mesh
        self.base_simplifier.rebuild_mesh()

        print(f"  Simplification complete: {len(self.base_simplifier.vertices)} vertices, "
              f"{len(self.base_simplifier.faces)} faces")

        # Return vertices, faces, and vertex lineage for boundary alignment
        return self.base_simplifier.vertices, self.base_simplifier.faces, self.vertex_lineage


class BoundaryAligner:
    """
    Aligns boundaries between simplified partitions as per the paper.
    Uses position-based matching with tolerance to handle simplified border vertices.
    """

    @staticmethod
    def align_boundaries(submeshes: List[Dict], original_vertices: np.ndarray,
                        border_vertices_global: Set[int], tolerance: float = 1e-4) -> List[Dict]:
        """
        Align boundaries between partitions after simplification.
        Uses position-based clustering to group border vertices that should be aligned.

        Args:
            submeshes: List of simplified submeshes
            original_vertices: Original mesh vertices
            border_vertices_global: Global border vertices set
            tolerance: Distance tolerance for considering vertices as the same (default: 1e-4)

        Returns:
            List of submeshes with aligned boundaries
        """
        print("  Aligning boundaries between partitions...")

        # Collect all border vertices from all submeshes
        # Format: [(submesh_idx, local_idx, position, set of original_global_indices)]
        all_border_candidates = []

        for submesh_idx, submesh in enumerate(submeshes):
            vertices = submesh['vertices']
            lineage_map = submesh.get('lineage_map', {})
            reverse_map = submesh.get('reverse_map', {})

            for local_idx, vertex in enumerate(vertices):
                # Get all original vertices this simplified vertex represents
                original_indices = lineage_map.get(local_idx, set())
                if len(original_indices) == 0:
                    # Fallback to reverse_map if lineage not available
                    orig_idx = reverse_map.get(local_idx)
                    if orig_idx is not None:
                        original_indices = {orig_idx}

                # Check if any of the represented vertices are border vertices
                border_indices = original_indices & border_vertices_global
                if len(border_indices) > 0:
                    all_border_candidates.append(
                        (submesh_idx, local_idx, vertex.copy(), border_indices)
                    )

        # Use both lineage and position-based clustering to group vertices that should be aligned
        # This handles cases where edge contractions have moved border vertices
        aligned_groups = []
        processed = set()

        for i, (submesh_i, local_i, pos_i, border_set_i) in enumerate(all_border_candidates):
            if i in processed:
                continue

            # Start a new group with this vertex
            group = [(submesh_i, local_i, pos_i)]
            processed.add(i)

            # Find all other vertices that should be aligned with this one
            for j, (submesh_j, local_j, pos_j, border_set_j) in enumerate(all_border_candidates):
                if j in processed:
                    continue

                # Check if vertices share any original border vertices (via lineage)
                # OR are within tolerance distance (spatial proximity)
                shares_lineage = len(border_set_i & border_set_j) > 0
                dist = np.linalg.norm(pos_i - pos_j)
                close_enough = dist < tolerance

                if shares_lineage or close_enough:
                    group.append((submesh_j, local_j, pos_j))
                    processed.add(j)
                    # Update border_set_i to include j's borders for transitive matching
                    border_set_i |= border_set_j

            # Only align if vertex appears in multiple partitions
            unique_submeshes = set(item[0] for item in group)
            if len(unique_submeshes) > 1:
                aligned_groups.append(group)

        # Align each group by computing average position
        alignment_count = 0
        total_vertices_aligned = 0

        for group in aligned_groups:
            # Compute average position
            positions = [item[2] for item in group]
            avg_position = np.mean(positions, axis=0)

            # Update all vertices in the group to use average position
            for submesh_idx, local_idx, _ in group:
                submeshes[submesh_idx]['vertices'][local_idx] = avg_position

            alignment_count += 1
            total_vertices_aligned += len(group)

        print(f"    Aligned {alignment_count} groups ({total_vertices_aligned} vertex instances) across partitions")
        print(f"    Tolerance: {tolerance}")

        return submeshes


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
    target_edges_per_partition: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplify a mesh using partitioning, local simplification, and merging.

    Args:
        vertices: Input mesh vertices (N x 3)
        faces: Input mesh faces (M x 3)
        target_ratio: Target simplification ratio (0.5 = keep 50% of vertices)
        target_edges_per_partition: Target number of edges per partition (default: 200)

    Returns:
        Tuple of (simplified_vertices, simplified_faces)
    """
    print(f"\n=== Starting Mesh Simplification with MDD/LME ===")
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Target ratio: {target_ratio}")
    print(f"Target edges per partition: {target_edges_per_partition}")

    # Step 1: Partition the mesh
    print("\n[Step 1] Partitioning mesh...")
    partitioner = MeshPartitioner(vertices, faces, target_edges_per_partition)
    partitions = partitioner.partition_bfs()
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

        # Identify border vertices in local indices (partition borders, can be simplified)
        local_border_vertices = {vertex_map[v] for v in partition['is_border']
                                if v in vertex_map and v in partitioner.border_vertices}

        # Identify 2-ring extension vertices in local indices (CAN be simplified per paper)
        # These provide topological context but their LME can be simplified
        local_two_ring_extension = {vertex_map[v] for v in partition['is_border']
                                    if v in vertex_map and v not in partition['core_vertices']
                                    and v not in partitioner.border_vertices}

        # Identify core vertices in local indices (vertices that belong to this partition's core)
        local_core_vertices = {vertex_map[v] for v in partition['core_vertices'] if v in vertex_map}

        # Simplify the submesh using LME with boundary simplification enabled
        simplifier = LMESimplifier(submesh_vertices, submesh_faces,
                                   local_border_vertices, local_two_ring_extension)
        simplified_vertices, simplified_faces, vertex_lineage = simplifier.simplify(target_ratio)

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

        # Create a reverse map for simplified vertices using vertex lineage
        # This tracks which original global vertices each simplified vertex represents
        simplified_reverse_map = {}
        simplified_lineage_map = {}  # Maps simplified local idx to set of original global indices

        # After rebuild_mesh, we need to map from the old local indices (used in lineage)
        # to the new local indices in simplified_vertices
        # The rebuild_mesh creates a mapping from old to new indices
        old_to_new_map = {}
        for new_idx in range(len(simplified_vertices)):
            # Find which old index this corresponds to by position matching
            best_old_idx = None
            min_dist = float('inf')
            for old_idx in vertex_lineage.keys():
                if old_idx < len(submesh_vertices):
                    dist = np.linalg.norm(simplified_vertices[new_idx] - submesh_vertices[old_idx])
                    if dist < min_dist:
                        min_dist = dist
                        best_old_idx = old_idx

            if best_old_idx is not None and min_dist < 1e-5:
                old_to_new_map[best_old_idx] = new_idx

        # Now create the reverse maps using vertex lineage
        for new_idx in range(len(simplified_vertices)):
            simplified_reverse_map[new_idx] = None
            simplified_lineage_map[new_idx] = set()

            # Find the old index this new index came from
            for old_idx, mapped_new_idx in old_to_new_map.items():
                if mapped_new_idx == new_idx and old_idx in vertex_lineage:
                    # Get all original vertices this represents
                    original_local_indices = vertex_lineage[old_idx]
                    # Convert to global indices
                    for orig_local in original_local_indices:
                        if orig_local in reverse_map:
                            simplified_lineage_map[new_idx].add(reverse_map[orig_local])

                    # For reverse_map, pick the first one (for backward compatibility)
                    if len(simplified_lineage_map[new_idx]) > 0:
                        simplified_reverse_map[new_idx] = min(simplified_lineage_map[new_idx])
                    break

        # Store simplified submesh with mappings (use core_faces instead of all faces)
        simplified_submeshes.append({
            'vertices': simplified_vertices,
            'faces': core_faces,  # Only include core faces
            'vertex_map': vertex_map,
            'reverse_map': simplified_reverse_map,
            'lineage_map': simplified_lineage_map  # Set of all original vertices represented
        })

    # Step 2.5: Align boundaries between partitions
    print("\n[Step 2.5] Aligning boundaries...")
    simplified_submeshes = BoundaryAligner.align_boundaries(
        simplified_submeshes, vertices, partitioner.border_vertices
    )

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
    target_edges_per_partition = 2000  # Target edges per partition

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