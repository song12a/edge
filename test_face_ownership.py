"""
Test suite for face ownership and degenerate face fixes.

This module tests the fixes for:
1. Face count should DECREASE after simplification
2. No degenerate/distorted faces in output
3. No duplicate faces from overlapping 2-ring neighborhoods
4. Proper vertex mapping after simplification
"""

import numpy as np
from mesh_simplification_mdd_lme import (
    simplify_mesh_with_partitioning,
    is_valid_triangle,
    MeshPartitioner
)
from QEM import PLYReader


def test_face_count_decreases():
    """Test that face count decreases after simplification."""
    print("\n" + "="*70)
    print("Test 1: Face Count Decreases After Simplification")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_large.ply')
    print(f"Input: {len(vertices)} vertices, {len(faces)} faces")
    
    # Test with different ratios
    for ratio in [0.3, 0.5, 0.7]:
        simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
            vertices, faces, target_ratio=ratio, num_partitions=8
        )
        
        face_reduction = 1 - len(simplified_faces) / len(faces)
        print(f"  Ratio {ratio}: {len(faces)} -> {len(simplified_faces)} faces "
              f"({100 * face_reduction:.1f}% reduction)")
        
        # Face count MUST decrease
        assert len(simplified_faces) < len(faces), \
            f"Face count should decrease! Got {len(simplified_faces)} (was {len(faces)})"
    
    print("✓ Test passed: Face count decreases for all ratios")


def test_no_degenerate_faces():
    """Test that there are no degenerate faces in the output."""
    print("\n" + "="*70)
    print("Test 2: No Degenerate Faces in Output")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    print(f"Input: {len(vertices)} vertices, {len(faces)} faces")
    
    simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
        vertices, faces, target_ratio=0.5, num_partitions=8
    )
    
    print(f"Output: {len(simplified_vertices)} vertices, {len(simplified_faces)} faces")
    
    # Check each face
    degenerate_count = 0
    duplicate_vertex_count = 0
    invalid_index_count = 0
    zero_area_count = 0
    
    for face_idx, face in enumerate(simplified_faces):
        v1, v2, v3 = face
        
        # Check for duplicate vertices
        if v1 == v2 or v2 == v3 or v1 == v3:
            duplicate_vertex_count += 1
            degenerate_count += 1
            continue
        
        # Check for invalid indices
        if v1 >= len(simplified_vertices) or v2 >= len(simplified_vertices) or v3 >= len(simplified_vertices):
            invalid_index_count += 1
            degenerate_count += 1
            continue
        
        # Check for zero area
        if not is_valid_triangle(v1, v2, v3, simplified_vertices):
            zero_area_count += 1
            degenerate_count += 1
    
    print(f"  Degenerate faces: {degenerate_count}")
    print(f"    - Duplicate vertices: {duplicate_vertex_count}")
    print(f"    - Invalid indices: {invalid_index_count}")
    print(f"    - Zero area: {zero_area_count}")
    
    assert degenerate_count == 0, f"Found {degenerate_count} degenerate faces!"
    print("✓ Test passed: No degenerate faces in output")


def test_no_duplicate_faces():
    """Test that there are no duplicate faces in the output."""
    print("\n" + "="*70)
    print("Test 3: No Duplicate Faces in Output")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_large.ply')
    print(f"Input: {len(vertices)} vertices, {len(faces)} faces")
    
    simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
        vertices, faces, target_ratio=0.5, num_partitions=8
    )
    
    print(f"Output: {len(simplified_vertices)} vertices, {len(simplified_faces)} faces")
    
    # Check for duplicate faces (same vertices, any order)
    face_set = set()
    duplicate_count = 0
    
    for face in simplified_faces:
        # Sort vertices to create a canonical representation
        face_tuple = tuple(sorted(face))
        if face_tuple in face_set:
            duplicate_count += 1
        else:
            face_set.add(face_tuple)
    
    print(f"  Unique faces: {len(face_set)}")
    print(f"  Duplicate faces: {duplicate_count}")
    
    assert duplicate_count == 0, f"Found {duplicate_count} duplicate faces!"
    print("✓ Test passed: No duplicate faces in output")


def test_face_ownership():
    """Test that face ownership is properly assigned."""
    print("\n" + "="*70)
    print("Test 4: Face Ownership Assignment")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_subdivided.ply')
    print(f"Input: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create partitioner and partition mesh
    partitioner = MeshPartitioner(vertices, faces, num_partitions=8)
    partitions = partitioner.partition_octree()
    
    # Check that each face has exactly one owner
    face_ownership_count = {}
    total_owned_faces = 0
    
    for p_idx, partition in enumerate(partitions):
        owned_faces = partition['owned_faces']
        total_owned_faces += len(owned_faces)
        
        for face_idx in owned_faces:
            if face_idx not in face_ownership_count:
                face_ownership_count[face_idx] = []
            face_ownership_count[face_idx].append(p_idx)
    
    # Check for faces with multiple owners
    multiple_owners = 0
    for face_idx, owners in face_ownership_count.items():
        if len(owners) > 1:
            multiple_owners += 1
    
    print(f"  Total faces: {len(faces)}")
    print(f"  Owned faces across all partitions: {total_owned_faces}")
    print(f"  Unique owned faces: {len(face_ownership_count)}")
    print(f"  Faces with multiple owners: {multiple_owners}")
    
    # Each face in owned_faces should have exactly one owner
    assert multiple_owners == 0, f"Found {multiple_owners} faces with multiple owners!"
    
    # All faces should be owned by someone
    assert len(face_ownership_count) == len(faces), \
        f"Not all faces are owned! {len(face_ownership_count)} out of {len(faces)}"
    
    print("✓ Test passed: Each face has exactly one owner")


def test_vertex_merge_tracking():
    """Test that vertex merging is properly tracked."""
    print("\n" + "="*70)
    print("Test 5: Vertex Merge Tracking")
    print("="*70)
    
    from mesh_simplification_mdd_lme import LMESimplifier
    
    # Create a simple triangle mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2],
        [1, 4, 3],
    ], dtype=np.int32)
    
    print(f"Input: {len(vertices)} vertices, {len(faces)} faces")
    
    # No border vertices for this test
    border_vertices = set()
    
    simplifier = LMESimplifier(vertices, faces, border_vertices)
    simplified_vertices, simplified_faces = simplifier.simplify(target_ratio=0.5)
    
    print(f"Output: {len(simplified_vertices)} vertices, {len(simplified_faces)} faces")
    
    # Check that vertex merge map exists and is reasonable
    assert hasattr(simplifier, 'vertex_merge_map'), "vertex_merge_map should exist"
    assert len(simplifier.vertex_merge_map) > 0, "vertex_merge_map should not be empty"
    
    print(f"  Vertex merge map entries: {len(simplifier.vertex_merge_map)}")
    print(f"  Example merge map entries:")
    for v_idx, orig_verts in list(simplifier.vertex_merge_map.items())[:3]:
        print(f"    Vertex {v_idx} came from: {orig_verts}")
    
    print("✓ Test passed: Vertex merging is tracked")


def test_valid_triangle_function():
    """Test the is_valid_triangle function."""
    print("\n" + "="*70)
    print("Test 6: Valid Triangle Detection")
    print("="*70)
    
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 0.0],  # Collinear with 0 and 2
    ], dtype=np.float32)
    
    # Valid triangle
    assert is_valid_triangle(0, 1, 2, vertices), "Triangle 0-1-2 should be valid"
    print("  ✓ Valid triangle (0, 1, 2) detected correctly")
    
    # Degenerate: duplicate vertices
    assert not is_valid_triangle(0, 0, 1, vertices), "Triangle 0-0-1 should be invalid"
    print("  ✓ Duplicate vertices (0, 0, 1) detected correctly")
    
    # Degenerate: zero area (collinear)
    # Create truly collinear points
    collinear_vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],  # On the line between 0 and 1
    ], dtype=np.float32)
    assert not is_valid_triangle(0, 1, 2, collinear_vertices), \
        "Collinear triangle should be invalid"
    print("  ✓ Zero area triangle detected correctly")
    
    print("✓ Test passed: Valid triangle function works correctly")


def test_larger_mesh_reduction():
    """Test simplification on a larger mesh."""
    print("\n" + "="*70)
    print("Test 7: Larger Mesh Simplification")
    print("="*70)
    
    vertices, faces = PLYReader.read_ply('demo/data/cube_large.ply')
    print(f"Input: {len(vertices)} vertices, {len(faces)} faces")
    
    simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
        vertices, faces, target_ratio=0.5, num_partitions=8
    )
    
    print(f"Output: {len(simplified_vertices)} vertices, {len(simplified_faces)} faces")
    
    # Calculate reduction percentages
    vertex_reduction = 100 * (1 - len(simplified_vertices) / len(vertices))
    face_reduction = 100 * (1 - len(simplified_faces) / len(faces))
    
    print(f"  Vertex reduction: {vertex_reduction:.1f}%")
    print(f"  Face reduction: {face_reduction:.1f}%")
    
    # With a larger mesh, we should see significant reduction
    assert len(simplified_vertices) < len(vertices), "Vertices should be reduced"
    assert len(simplified_faces) < len(faces), "Faces should be reduced"
    assert face_reduction > 30, f"Face reduction should be > 30%, got {face_reduction:.1f}%"
    
    # Validate all faces
    degenerate_count = 0
    for face in simplified_faces:
        if not is_valid_triangle(face[0], face[1], face[2], simplified_vertices):
            degenerate_count += 1
    
    assert degenerate_count == 0, f"Found {degenerate_count} degenerate faces"
    
    print("✓ Test passed: Larger mesh simplifies correctly")


def run_all_tests():
    """Run all tests for face ownership and degenerate face fixes."""
    print("\n" + "="*80)
    print("Face Ownership and Degenerate Face Fixes Test Suite")
    print("="*80)
    
    tests = [
        test_face_count_decreases,
        test_no_degenerate_faces,
        test_no_duplicate_faces,
        test_face_ownership,
        test_vertex_merge_tracking,
        test_valid_triangle_function,
        test_larger_mesh_reduction,
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
