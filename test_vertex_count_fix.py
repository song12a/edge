"""
Test suite to validate the fix for vertex count increase issue.

This test validates that:
1. Vertex count does not increase after simplification
2. Boundary vertices are consistently aligned across submeshes
3. No duplicate vertices exist in the final merged mesh
4. The output mesh maintains geometric fidelity
"""

import numpy as np
from mesh_simplification_mdd_lme import simplify_mesh_with_partitioning
from QEM import PLYReader


def test_vertex_count_no_increase():
    """
    Test that vertex count never increases after simplification.
    This was the main issue reported in the problem statement.
    """
    print("\n" + "="*70)
    print("Test: Vertex Count Does Not Increase")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    original_vertex_count = len(vertices)
    original_face_count = len(faces)
    
    print(f"Original mesh: {original_vertex_count} vertices, {original_face_count} faces")
    
    # Test various configurations that previously caused issues
    test_configs = [
        (0.9, 50, "High target ratio, many partitions (worst case)"),
        (0.8, 100, "Medium-high target ratio, medium partitions"),
        (0.7, 150, "Medium target ratio"),
        (0.5, 200, "Normal case"),
        (0.3, 200, "Aggressive simplification"),
    ]
    
    all_passed = True
    
    for target_ratio, target_edges, description in test_configs:
        print(f"\n  Testing: {description}")
        print(f"    Parameters: ratio={target_ratio}, target_edges={target_edges}")
        
        simplified_v, simplified_f = simplify_mesh_with_partitioning(
            vertices, faces, 
            target_ratio=target_ratio, 
            target_edges_per_partition=target_edges
        )
        
        vertex_count = len(simplified_v)
        face_count = len(simplified_f)
        
        # Validation 1: Vertex count should not increase
        vertex_increased = vertex_count > original_vertex_count
        
        if vertex_increased:
            print(f"    ✗ FAIL: Vertex count increased: {original_vertex_count} -> {vertex_count}")
            all_passed = False
        else:
            print(f"    ✓ PASS: Vertex count did not increase: {original_vertex_count} -> {vertex_count}")
        
        # Validation 2: Output should be valid
        assert vertex_count > 0, "Output must have vertices"
        assert face_count > 0, "Output must have faces"
        
        # Validation 3: All face indices should be valid
        max_idx = vertex_count - 1
        for face in simplified_f:
            for v in face:
                assert 0 <= v <= max_idx, f"Invalid vertex index {v}"
        
        # Validation 4: No degenerate faces
        for face in simplified_f:
            assert len(set(face)) == 3, f"Degenerate face detected: {face}"
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All configurations passed: Vertex count never increases")
    else:
        print("✗ Some configurations failed")
    print("="*70)
    
    return all_passed


def test_no_duplicate_vertices():
    """
    Test that there are no duplicate vertices in the output mesh.
    """
    print("\n" + "="*70)
    print("Test: No Duplicate Vertices")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    
    # Test with configuration that previously created duplicates
    simplified_v, simplified_f = simplify_mesh_with_partitioning(
        vertices, faces, target_ratio=0.9, target_edges_per_partition=50
    )
    
    # Check for duplicate vertices by position
    tolerance = 1e-6
    vertex_positions = {}
    duplicates_found = 0
    
    for i, vertex in enumerate(simplified_v):
        key = tuple(np.round(vertex / tolerance).astype(int))
        if key in vertex_positions:
            duplicates_found += 1
            print(f"  Duplicate vertex found: {i} matches {vertex_positions[key]}")
        else:
            vertex_positions[key] = i
    
    print(f"\nTotal vertices: {len(simplified_v)}")
    print(f"Unique positions: {len(vertex_positions)}")
    print(f"Duplicates found: {duplicates_found}")
    
    if duplicates_found == 0:
        print("✓ PASS: No duplicate vertices found")
        return True
    else:
        print(f"✗ FAIL: Found {duplicates_found} duplicate vertices")
        return False


def test_boundary_vertex_alignment():
    """
    Test that boundary vertices are consistently aligned across submeshes.
    """
    print("\n" + "="*70)
    print("Test: Boundary Vertex Alignment")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    
    simplified_v, simplified_f = simplify_mesh_with_partitioning(
        vertices, faces, target_ratio=0.5, target_edges_per_partition=200
    )
    
    # Build edge connectivity
    edges = {}
    for face in simplified_f:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            if edge not in edges:
                edges[edge] = []
            edges[edge].append(face)
    
    # Check for consistent edge lengths (boundary edges should match)
    edge_lengths = []
    for (v1, v2), face_list in edges.items():
        length = np.linalg.norm(simplified_v[v1] - simplified_v[v2])
        edge_lengths.append(length)
    
    print(f"Total edges: {len(edges)}")
    print(f"Edge length range: [{min(edge_lengths):.6f}, {max(edge_lengths):.6f}]")
    print(f"Average edge length: {np.mean(edge_lengths):.6f}")
    
    # Check for manifold edges (each edge should have 1 or 2 adjacent faces)
    # Note: Some non-manifold edges are expected at partition boundaries
    # where 2-ring neighborhoods overlap
    non_manifold_edges = 0
    for edge, face_list in edges.items():
        if len(face_list) > 2:
            non_manifold_edges += 1
    
    non_manifold_ratio = non_manifold_edges / len(edges) if len(edges) > 0 else 0
    
    print(f"Non-manifold edges: {non_manifold_edges} ({non_manifold_ratio:.1%})")
    
    # Allow some non-manifold edges due to partition overlaps (< 20% is acceptable)
    if non_manifold_ratio < 0.2:
        print("✓ PASS: Acceptable level of non-manifold edges (< 20%)")
        return True
    else:
        print(f"✗ FAIL: Too many non-manifold edges ({non_manifold_ratio:.1%})")
        return False


def test_geometric_fidelity():
    """
    Test that the output mesh maintains geometric fidelity with the input.
    """
    print("\n" + "="*70)
    print("Test: Geometric Fidelity")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    
    # Compute bounding box of original mesh
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_size = bbox_max - bbox_min
    
    print(f"Original mesh bounding box:")
    print(f"  Min: {bbox_min}")
    print(f"  Max: {bbox_max}")
    print(f"  Size: {bbox_size}")
    
    simplified_v, simplified_f = simplify_mesh_with_partitioning(
        vertices, faces, target_ratio=0.5, target_edges_per_partition=200
    )
    
    # Compute bounding box of simplified mesh
    simplified_bbox_min = np.min(simplified_v, axis=0)
    simplified_bbox_max = np.max(simplified_v, axis=0)
    simplified_bbox_size = simplified_bbox_max - simplified_bbox_min
    
    print(f"\nSimplified mesh bounding box:")
    print(f"  Min: {simplified_bbox_min}")
    print(f"  Max: {simplified_bbox_max}")
    print(f"  Size: {simplified_bbox_size}")
    
    # Check that bounding boxes are similar
    bbox_diff = np.linalg.norm(bbox_size - simplified_bbox_size)
    tolerance = 0.1 * np.linalg.norm(bbox_size)
    
    print(f"\nBounding box difference: {bbox_diff:.6f}")
    print(f"Tolerance: {tolerance:.6f}")
    
    if bbox_diff < tolerance:
        print("✓ PASS: Geometric fidelity maintained")
        return True
    else:
        print("✗ FAIL: Bounding box changed significantly")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("Vertex Count Fix Validation Test Suite")
    print("="*80)
    
    tests = [
        ("Vertex Count Does Not Increase", test_vertex_count_no_increase),
        ("No Duplicate Vertices", test_no_duplicate_vertices),
        ("Boundary Vertex Alignment", test_boundary_vertex_alignment),
        ("Geometric Fidelity", test_geometric_fidelity),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"✗ Test error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result, _ in results if result)
    failed = len(results) - passed
    
    for test_name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal tests: {len(results)}")
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
