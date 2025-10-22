"""
Test suite for 2-ring neighborhood implementation in MeshPartitioner.

This module tests the topology-based neighborhood calculation and validates
that the 2-ring neighborhoods are correctly computed and integrated with the
mesh simplification pipeline.
"""

import numpy as np
from mesh_simplification_mdd_lme import MeshPartitioner, simplify_mesh_with_partitioning
from QEM import PLYReader


def test_vertex_adjacency_simple_triangle():
    """Test vertex adjacency calculation for a simple triangle."""
    print("\n" + "="*70)
    print("Test 1: Vertex Adjacency - Simple Triangle")
    print("="*70)
    
    # Create a simple triangle
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float32)
    
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    
    # Create partitioner and build adjacency
    partitioner = MeshPartitioner(vertices, faces)
    adjacency = partitioner.build_vertex_adjacency()
    
    # Verify adjacency
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    print(f"Adjacency:")
    for v, neighbors in adjacency.items():
        print(f"  Vertex {v}: neighbors {sorted(neighbors)}")
    
    # Each vertex should be adjacent to the other two
    assert len(adjacency[0]) == 2, "Vertex 0 should have 2 neighbors"
    assert len(adjacency[1]) == 2, "Vertex 1 should have 2 neighbors"
    assert len(adjacency[2]) == 2, "Vertex 2 should have 2 neighbors"
    assert adjacency[0] == {1, 2}, "Vertex 0 should be adjacent to 1 and 2"
    assert adjacency[1] == {0, 2}, "Vertex 1 should be adjacent to 0 and 2"
    assert adjacency[2] == {0, 1}, "Vertex 2 should be adjacent to 0 and 1"
    
    print("✓ Test passed: Adjacency correctly computed")


def test_1ring_neighborhood():
    """Test 1-ring neighborhood calculation."""
    print("\n" + "="*70)
    print("Test 2: 1-Ring Neighborhood")
    print("="*70)
    
    # Create a simple mesh with 6 vertices forming two triangles
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.5, 1.0, 0.0],  # 2
        [1.5, 1.0, 0.0],  # 3
        [2.0, 0.0, 0.0],  # 4
        [1.5, 2.0, 0.0],  # 5
    ], dtype=np.float32)
    
    # Two triangles sharing an edge (1, 2)
    faces = np.array([
        [0, 1, 2],
        [1, 4, 3],
        [2, 3, 5]
    ], dtype=np.int32)
    
    partitioner = MeshPartitioner(vertices, faces)
    
    # Test 1-ring of vertex 1
    one_ring = partitioner.compute_n_ring_neighborhood({1}, n=1)
    print(f"1-ring of vertex 1: {sorted(one_ring)}")
    
    # Vertex 1 is connected to: 0, 2, 3, 4
    expected_1ring = {0, 1, 2, 3, 4}
    assert one_ring == expected_1ring, f"Expected {expected_1ring}, got {one_ring}"
    
    print("✓ Test passed: 1-ring correctly computed")


def test_2ring_neighborhood():
    """Test 2-ring neighborhood calculation."""
    print("\n" + "="*70)
    print("Test 3: 2-Ring Neighborhood")
    print("="*70)
    
    # Create a simple mesh with 6 vertices forming triangles
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.5, 1.0, 0.0],  # 2
        [1.5, 1.0, 0.0],  # 3
        [2.0, 0.0, 0.0],  # 4
        [1.5, 2.0, 0.0],  # 5
    ], dtype=np.float32)
    
    # Three triangles
    faces = np.array([
        [0, 1, 2],
        [1, 4, 3],
        [2, 3, 5]
    ], dtype=np.int32)
    
    partitioner = MeshPartitioner(vertices, faces)
    
    # Test 2-ring of vertex 0
    two_ring = partitioner.compute_n_ring_neighborhood({0}, n=2)
    print(f"2-ring of vertex 0: {sorted(two_ring)}")
    
    # Vertex 0's 1-ring: {1, 2}
    # From 1: adds {3, 4}
    # From 2: adds {3, 5}
    # So 2-ring should be: {0, 1, 2, 3, 4, 5}
    expected_2ring = {0, 1, 2, 3, 4, 5}
    assert two_ring == expected_2ring, f"Expected {expected_2ring}, got {two_ring}"
    
    print("✓ Test passed: 2-ring correctly computed")


def test_partition_2ring_expansion():
    """Test that partitions are correctly expanded with 2-ring neighborhoods."""
    print("\n" + "="*70)
    print("Test 4: Partition 2-Ring Expansion")
    print("="*70)
    
    # Load a test mesh
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    print(f"Test mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create partitioner (using BFS with target edges per partition)
    partitioner = MeshPartitioner(vertices, faces, target_edges_per_partition=200)
    partitions = partitioner.partition_bfs()
    
    print(f"Created {len(partitions)} partitions")
    
    # Verify that each partition has more vertices after 2-ring expansion
    for idx, partition in enumerate(partitions):
        core_count = len(partition['core_vertices'])
        total_count = len(partition['vertices'])
        expansion_ratio = total_count / core_count if core_count > 0 else 0
        
        print(f"Partition {idx}: {core_count} core vertices -> {total_count} total vertices "
              f"(expansion: {expansion_ratio:.2f}x)")
        
        # Each partition should have more vertices after 2-ring expansion
        assert total_count >= core_count, "Total vertices should be >= core vertices"
        # For a subdivided cube with octree partitioning, expect significant expansion
        assert expansion_ratio > 1.5, f"Expected expansion ratio > 1.5, got {expansion_ratio}"
    
    print("✓ Test passed: Partitions correctly expanded with 2-ring neighborhoods")


def test_border_vertex_classification():
    """Test that border vertices are correctly classified."""
    print("\n" + "="*70)
    print("Test 5: Border Vertex Classification")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    
    partitioner = MeshPartitioner(vertices, faces, target_edges_per_partition=200)
    partitions = partitioner.partition_bfs()
    
    print(f"Global border vertices: {len(partitioner.border_vertices)}")
    
    # Check each partition's border vertices
    for idx, partition in enumerate(partitions):
        border_count = len(partition['is_border'])
        core_count = len(partition['core_vertices'])
        total_count = len(partition['vertices'])
        
        print(f"Partition {idx}: {border_count} border vertices out of {total_count} total "
              f"({core_count} core)")
        
        # Border vertices should include:
        # 1. Vertices not in core (2-ring extension)
        # 2. Vertices on global borders
        assert border_count > 0, "Each partition should have some border vertices"
        
        # Verify that all non-core vertices are marked as border
        for v in partition['vertices']:
            if v not in partition['core_vertices']:
                assert v in partition['is_border'], \
                    f"Non-core vertex {v} should be marked as border"
    
    print("✓ Test passed: Border vertices correctly classified")


def test_mesh_coherence_with_2ring():
    """Test that mesh simplification with 2-ring produces coherent output."""
    print("\n" + "="*70)
    print("Test 6: Mesh Coherence with 2-Ring Support")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Simplify with 2-ring neighborhood support
    simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
        vertices, faces, target_ratio=0.5, target_edges_per_partition=200
    )
    
    print(f"\nOutput mesh: {len(simplified_vertices)} vertices, {len(simplified_faces)} faces")
    
    # Verify mesh coherence
    # 1. All face indices should be valid
    max_vertex_idx = len(simplified_vertices) - 1
    for face_idx, face in enumerate(simplified_faces):
        for v in face:
            assert 0 <= v <= max_vertex_idx, \
                f"Face {face_idx} has invalid vertex index {v} (max: {max_vertex_idx})"
    
    # 2. No degenerate faces (all three vertices should be different)
    for face_idx, face in enumerate(simplified_faces):
        assert len(set(face)) == 3, f"Face {face_idx} is degenerate: {face}"
    
    # 3. Mesh should be simplified (fewer vertices and faces than input)
    assert len(simplified_vertices) < len(vertices), "Mesh should be simplified"
    assert len(simplified_faces) < len(faces), "Faces should be reduced"
    
    # 4. Simplification should be reasonable (not too aggressive or too conservative)
    vertex_retention = len(simplified_vertices) / len(vertices)
    face_retention = len(simplified_faces) / len(faces)
    
    print(f"Vertex retention: {vertex_retention:.2%}")
    print(f"Face retention: {face_retention:.2%}")
    
    assert 0.3 < vertex_retention < 1.0, \
        f"Vertex retention {vertex_retention:.2%} should be between 30% and 100%"
    assert 0.3 < face_retention < 1.0, \
        f"Face retention {face_retention:.2%} should be between 30% and 100%"
    
    print("✓ Test passed: Simplified mesh is coherent and valid")


def test_2ring_vs_no_2ring_comparison():
    """Compare simplification with and without 2-ring neighborhoods."""
    print("\n" + "="*70)
    print("Test 7: Comparison - With vs Without 2-Ring")
    print("="*70)
    
    # Note: This is more of a demonstration than a strict test
    # The 2-ring neighborhood provides better context for QEM calculations
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Simplify with 2-ring (current implementation)
    print("\nSimplifying with 2-ring neighborhood support...")
    simplified_2ring_v, simplified_2ring_f = simplify_mesh_with_partitioning(
        vertices, faces, target_ratio=0.5, target_edges_per_partition=200
    )
    
    print(f"\nWith 2-ring: {len(simplified_2ring_v)} vertices, {len(simplified_2ring_f)} faces")
    print(f"  Vertex reduction: {100 * (1 - len(simplified_2ring_v)/len(vertices)):.1f}%")
    print(f"  Face reduction: {100 * (1 - len(simplified_2ring_f)/len(faces)):.1f}%")
    
    # Verify the output is valid
    assert len(simplified_2ring_v) > 0, "Should have vertices"
    assert len(simplified_2ring_f) > 0, "Should have faces"
    
    print("✓ Test passed: 2-ring neighborhood produces valid simplification")


def run_all_tests():
    """Run all tests for 2-ring neighborhood implementation."""
    print("\n" + "="*80)
    print("2-Ring Neighborhood Implementation Test Suite")
    print("="*80)
    
    tests = [
        test_vertex_adjacency_simple_triangle,
        test_1ring_neighborhood,
        test_2ring_neighborhood,
        test_partition_2ring_expansion,
        test_border_vertex_classification,
        test_mesh_coherence_with_2ring,
        test_2ring_vs_no_2ring_comparison,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*80)
    
    if failed == 0:
        print("✓ All tests passed!")
        return True
    else:
        print(f"✗ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
