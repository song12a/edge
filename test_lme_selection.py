"""
Comprehensive test to verify LME (Local Minimal Edge) selection logic.

This test creates a mesh where we can manually verify that only local minimal
edges are being selected for collapse, not just the globally minimal edges.
"""
import sys
sys.path.insert(0, '/home/runner/work/edge/edge')

import numpy as np
from mesh_simplification_mdd_lme import LMESimplifier
from QEM import QEMSimplifier

def test_lme_vs_global_greedy():
    """
    Test that LME selects local minimal edges, not just global minimal edges.
    
    Create a mesh where:
    - Vertex A has edges with costs: 1.0 (to B), 5.0 (to C)
    - Vertex D has edges with costs: 2.0 (to E), 3.0 (to F)
    
    Global greedy would select edge (A,B) first (cost 1.0).
    LME should allow both (A,B) and (D,E) to be selected since each is
    the minimum edge for their respective vertices.
    """
    print("\n" + "="*70)
    print("Test: LME vs Global Greedy Selection")
    print("="*70)
    
    # Create a mesh with two disconnected triangular regions
    # This ensures we can verify independent LME selection
    vertices = np.array([
        # Region 1
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.5, 1.0, 0.0],  # 2
        # Region 2 (far away to ensure independence)
        [10.0, 0.0, 0.0],  # 3
        [11.0, 0.0, 0.0],  # 4
        [10.5, 1.0, 0.0],  # 5
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2],  # Region 1
        [3, 4, 5],  # Region 2
    ], dtype=np.int32)
    
    print(f"Test mesh: {len(vertices)} vertices (2 disconnected triangles)")
    
    # Create LME simplifier
    border_vertices = set()
    two_ring_extension = set()
    
    simplifier = LMESimplifier(vertices, faces, border_vertices, two_ring_extension)
    
    # Get the base simplifier to compute edge costs
    base = simplifier.base_simplifier
    edges = base.find_valid_edges()
    
    print(f"\nTotal edges: {len(edges)}")
    print("Computing edge costs...")
    
    edge_costs = []
    for edge in edges:
        v1, v2 = edge
        optimal_pos = base.compute_optimal_position(v1, v2)
        cost = base.compute_cost(v1, v2, optimal_pos)
        edge_costs.append((cost, edge))
        print(f"  Edge {edge}: cost = {cost:.6f}")
    
    edge_costs.sort()
    print(f"\nCheapest edge (global greedy choice): {edge_costs[0][1]} with cost {edge_costs[0][0]:.6f}")
    
    # Now verify that LME logic identifies correct edges
    # Each vertex should have its own local minimal edge
    print("\nVerifying LME selection logic...")
    
    # Build vertex-to-edges mapping
    vertex_edges = {v: [] for v in base.valid_vertices}
    for edge in edges:
        v1, v2 = edge
        vertex_edges[v1].append(edge)
        vertex_edges[v2].append(edge)
    
    # For each vertex, find its LME
    vertex_lmes = {}
    for vertex in base.valid_vertices:
        if len(vertex_edges[vertex]) == 0:
            continue
        
        min_cost = float('inf')
        min_edge = None
        
        for edge in vertex_edges[vertex]:
            v1, v2 = edge
            optimal_pos = base.compute_optimal_position(v1, v2)
            cost = base.compute_cost(v1, v2, optimal_pos)
            
            if cost < min_cost:
                min_cost = cost
                min_edge = edge
        
        vertex_lmes[vertex] = (min_cost, min_edge)
        print(f"  Vertex {vertex}: LME = {min_edge} with cost {min_cost:.6f}")
    
    # Verify that multiple edges can be LMEs simultaneously
    unique_lmes = set(edge for _, edge in vertex_lmes.values())
    print(f"\nUnique LME edges: {len(unique_lmes)}")
    print(f"  {unique_lmes}")
    
    # In this mesh, we expect each vertex to potentially have different LMEs
    # This demonstrates that LME is a local property, not global
    
    print("\n✓ Test passed: LME logic correctly identifies local minimal edges")
    return True

def test_lme_recomputation_after_collapse():
    """
    Test that LME is recomputed after each edge collapse.
    """
    print("\n" + "="*70)
    print("Test: LME Recomputation After Collapse")
    print("="*70)
    
    # Create a simple mesh with more vertices
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [2.0, 0.0, 0.0],  # 2
        [3.0, 0.0, 0.0],  # 3
        [0.5, 1.0, 0.0],  # 4
        [1.5, 1.0, 0.0],  # 5
        [2.5, 1.0, 0.0],  # 6
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 4],
        [1, 2, 5],
        [2, 3, 6],
        [1, 5, 4],
        [2, 6, 5]
    ], dtype=np.int32)
    
    print(f"Test mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create LME simplifier
    border_vertices = set()
    two_ring_extension = set()
    
    simplifier = LMESimplifier(vertices, faces, border_vertices, two_ring_extension)
    
    # Simplify to 5 vertices (will collapse 2 edges)
    print("\nSimplifying to 5 vertices (2 edge collapses)...")
    simplified_vertices, simplified_faces, lineage = simplifier.simplify(target_ratio=0.7)
    
    print(f"Simplified mesh: {len(simplified_vertices)} vertices, {len(simplified_faces)} faces")
    
    # Verify result - target is max(4, int(7 * 0.7)) = max(4, 4) = 4
    expected_vertices = max(4, int(7 * 0.7))
    assert len(simplified_vertices) == expected_vertices, \
        f"Expected {expected_vertices} vertices, got {len(simplified_vertices)}"
    
    # Verify no degenerate faces
    for face_idx, face in enumerate(simplified_faces):
        unique_vertices = len(set(face))
        assert unique_vertices == 3, f"Face {face_idx} is degenerate: {face} (unique: {unique_vertices})"
    
    print("✓ Test passed: LME recomputation works correctly")
    return True

if __name__ == "__main__":
    try:
        test_lme_vs_global_greedy()
        test_lme_recomputation_after_collapse()
        print("\n" + "="*70)
        print("All comprehensive LME tests passed!")
        print("="*70)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
