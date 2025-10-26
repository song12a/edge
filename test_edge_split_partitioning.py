"""
Test script to verify edge split with octree partitioning functionality
"""
import os
from edge_split import EdgeSplitter, PLYReader, PLYWriter, MeshPartitioner


def create_test_mesh():
    """Create a simple test mesh for partitioning tests"""
    vertices = []
    faces = []
    
    # Create a 4x4x4 grid of vertices
    for i in range(4):
        for j in range(4):
            for k in range(4):
                vertices.append([float(i), float(j), float(k)])
    
    # Create faces - simple cube subdivision
    def vertex_index(i, j, k):
        return i * 16 + j * 4 + k
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Create cube faces
                v000 = vertex_index(i, j, k)
                v001 = vertex_index(i, j, k+1)
                v010 = vertex_index(i, j+1, k)
                v011 = vertex_index(i, j+1, k+1)
                v100 = vertex_index(i+1, j, k)
                v101 = vertex_index(i+1, j, k+1)
                v110 = vertex_index(i+1, j+1, k)
                v111 = vertex_index(i+1, j+1, k+1)
                
                # Front face
                faces.extend([[v000, v100, v110], [v000, v110, v010]])
                # Back face
                faces.extend([[v001, v011, v111], [v001, v111, v101]])
                # Left face
                faces.extend([[v000, v010, v011], [v000, v011, v001]])
                # Right face
                faces.extend([[v100, v101, v111], [v100, v111, v110]])
                # Bottom face
                faces.extend([[v000, v001, v101], [v000, v101, v100]])
                # Top face
                faces.extend([[v010, v110, v111], [v010, v111, v011]])
    
    return vertices, faces


def test_partitioner():
    """Test 1: Basic partitioning functionality"""
    print("=" * 60)
    print("Test 1: MeshPartitioner Functionality")
    print("=" * 60)
    
    vertices, faces = create_test_mesh()
    print(f"Test mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create partitioner
    partitioner = MeshPartitioner(vertices, faces, num_partitions=8)
    partitions = partitioner.partition_octree()
    
    print(f"\nPartitioning results:")
    print(f"  Number of partitions: {len(partitions)}")
    print(f"  Border vertices: {len(partitioner.border_vertices)}")
    
    # Check each partition
    all_passed = True
    for i, partition in enumerate(partitions):
        core_v = len(partition['core_vertices'])
        total_v = len(partition['vertices'])
        faces_count = len(partition['faces'])
        border_v = len(partition['is_border'])
        
        print(f"\n  Partition {i}:")
        print(f"    Core vertices: {core_v}")
        print(f"    Total vertices (core + 2-ring): {total_v}")
        print(f"    Faces: {faces_count}")
        print(f"    Border vertices: {border_v}")
        
        # Verify that core vertices are a subset of total vertices
        if not partition['core_vertices'].issubset(partition['vertices']):
            print(f"    ✗ FAIL: Core vertices not subset of total vertices")
            all_passed = False
        
        # Verify that border vertices are a subset of total vertices
        if not partition['is_border'].issubset(partition['vertices']):
            print(f"    ✗ FAIL: Border vertices not subset of total vertices")
            all_passed = False
    
    if all_passed:
        print("\n✓ PASS: All partitions are valid")
        return True
    else:
        print("\n✗ FAIL: Some partitions are invalid")
        return False


def test_partitioned_splitting_subremeshing():
    """Test 2: Edge splitting with partitioning (subremeshing mode)"""
    print("\n" + "=" * 60)
    print("Test 2: Partitioned Edge Splitting (Subremeshing Mode)")
    print("=" * 60)
    
    vertices, faces = create_test_mesh()
    print(f"Original mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Test with partitioning
    splitter = EdgeSplitter(use_partitioning=True, num_partitions=8)
    splitter.initialize(vertices, faces)
    
    new_vertices, new_faces = splitter.split_edges(mode="subremeshing", max_iter=1)
    
    print(f"\nResults:")
    print(f"  New vertices: {len(new_vertices)}")
    print(f"  New faces: {len(new_faces)}")
    
    # Verify mesh validity
    vertex_set = set(range(len(new_vertices)))
    all_valid = True
    
    for i, face in enumerate(new_faces):
        if len(face) != 3:
            print(f"  ✗ FAIL: Face {i} does not have 3 vertices")
            all_valid = False
            break
        
        for v in face:
            if v not in vertex_set:
                print(f"  ✗ FAIL: Face {i} references invalid vertex {v}")
                all_valid = False
                break
        
        if not all_valid:
            break
    
    if all_valid:
        print("✓ PASS: Output mesh is valid")
        return True
    else:
        print("✗ FAIL: Output mesh has errors")
        return False


def test_partitioned_splitting_histogram():
    """Test 3: Edge splitting with partitioning (histogram mode)"""
    print("\n" + "=" * 60)
    print("Test 3: Partitioned Edge Splitting (Histogram Mode)")
    print("=" * 60)
    
    vertices, faces = create_test_mesh()
    print(f"Original mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Test with partitioning
    splitter = EdgeSplitter(use_partitioning=True, num_partitions=8)
    splitter.initialize(vertices, faces)
    
    new_vertices, new_faces = splitter.split_edges(mode="histogram", max_iter=3)
    
    print(f"\nResults:")
    print(f"  New vertices: {len(new_vertices)}")
    print(f"  New faces: {len(new_faces)}")
    
    # Verify mesh validity
    vertex_set = set(range(len(new_vertices)))
    all_valid = True
    
    for i, face in enumerate(new_faces):
        if len(face) != 3:
            print(f"  ✗ FAIL: Face {i} does not have 3 vertices")
            all_valid = False
            break
        
        for v in face:
            if v not in vertex_set:
                print(f"  ✗ FAIL: Face {i} references invalid vertex {v}")
                all_valid = False
                break
        
        if not all_valid:
            break
    
    if all_valid:
        print("✓ PASS: Output mesh is valid")
        return True
    else:
        print("✗ FAIL: Output mesh has errors")
        return False


def test_backward_compatibility():
    """Test 4: Ensure backward compatibility (no partitioning)"""
    print("\n" + "=" * 60)
    print("Test 4: Backward Compatibility (No Partitioning)")
    print("=" * 60)
    
    vertices, faces = create_test_mesh()
    print(f"Original mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Test without partitioning (original behavior)
    splitter = EdgeSplitter(use_partitioning=False)
    splitter.initialize(vertices, faces)
    
    # Test subremeshing mode
    new_vertices1, new_faces1 = splitter.split_edges(mode="subremeshing", max_iter=1)
    print(f"\nSubremeshing (no partition):")
    print(f"  Vertices: {len(new_vertices1)}, Faces: {len(new_faces1)}")
    
    # Test histogram mode
    splitter2 = EdgeSplitter(use_partitioning=False)
    splitter2.initialize(vertices, faces)
    new_vertices2, new_faces2 = splitter2.split_edges(mode="histogram", max_iter=3)
    print(f"Histogram (no partition):")
    print(f"  Vertices: {len(new_vertices2)}, Faces: {len(new_faces2)}")
    
    # Check that both modes produce valid output
    all_valid = True
    
    # Validate subremeshing output
    for face in new_faces1:
        if len(face) != 3 or any(v >= len(new_vertices1) or v < 0 for v in face):
            all_valid = False
            break
    
    # Validate histogram output
    for face in new_faces2:
        if len(face) != 3 or any(v >= len(new_vertices2) or v < 0 for v in face):
            all_valid = False
            break
    
    if all_valid:
        print("✓ PASS: Backward compatibility maintained")
        return True
    else:
        print("✗ FAIL: Backward compatibility broken")
        return False


def test_boundary_preservation():
    """Test 5: Verify that boundary vertices are preserved"""
    print("\n" + "=" * 60)
    print("Test 5: Boundary Vertex Preservation")
    print("=" * 60)
    
    vertices, faces = create_test_mesh()
    print(f"Original mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Get partitioning info
    partitioner = MeshPartitioner(vertices, faces, num_partitions=8)
    partitions = partitioner.partition_octree()
    
    border_vertices = partitioner.border_vertices
    print(f"Border vertices: {len(border_vertices)}")
    
    # Split with partitioning
    splitter = EdgeSplitter(use_partitioning=True, num_partitions=8)
    splitter.initialize(vertices, faces)
    new_vertices, new_faces = splitter.split_edges(mode="subremeshing", max_iter=1)
    
    # Check if original border vertices still exist in output
    # (Note: In merging, vertices might be deduplicated, so we check by position)
    preserved_count = 0
    tolerance = 1e-6
    
    for border_v_idx in border_vertices:
        orig_pos = vertices[border_v_idx]
        
        # Check if this position exists in the output
        found = False
        for new_pos in new_vertices:
            dist = sum((orig_pos[i] - new_pos[i]) ** 2 for i in range(3))
            if dist < tolerance:
                found = True
                preserved_count += 1
                break
    
    print(f"\nResults:")
    print(f"  Original border vertices: {len(border_vertices)}")
    print(f"  Preserved in output: {preserved_count}")
    
    # We expect most border vertices to be preserved
    preservation_ratio = preserved_count / len(border_vertices) if border_vertices else 1.0
    
    if preservation_ratio > 0.9:  # Allow some tolerance due to merging
        print(f"✓ PASS: {preservation_ratio*100:.1f}% of border vertices preserved")
        return True
    else:
        print(f"✗ FAIL: Only {preservation_ratio*100:.1f}% of border vertices preserved")
        return False


def test_output_consistency():
    """Test 6: Verify output files can be written and read"""
    print("\n" + "=" * 60)
    print("Test 6: Output File Consistency")
    print("=" * 60)
    
    os.makedirs("demo/output", exist_ok=True)
    
    vertices, faces = create_test_mesh()
    
    # Split with partitioning
    splitter = EdgeSplitter(use_partitioning=True, num_partitions=8)
    splitter.initialize(vertices, faces)
    new_vertices, new_faces = splitter.split_edges(mode="subremeshing", max_iter=1)
    
    # Write to file
    output_path = "demo/output/test_partitioned_output.ply"
    writer = PLYWriter()
    writer.write_ply(output_path, new_vertices, new_faces)
    
    print(f"Written to: {output_path}")
    
    # Read back
    reader = PLYReader()
    read_vertices, read_faces = reader.read_ply(output_path)
    
    print(f"Read back: {len(read_vertices)} vertices, {len(read_faces)} faces")
    
    # Verify consistency
    if len(read_vertices) == len(new_vertices) and len(read_faces) == len(new_faces):
        print("✓ PASS: File I/O is consistent")
        return True
    else:
        print("✗ FAIL: File I/O mismatch")
        return False


if __name__ == "__main__":
    print("Edge Split Partitioning Tests")
    print("Testing octree partitioning and boundary preservation\n")
    
    results = []
    
    # Run tests
    results.append(("Partitioner Functionality", test_partitioner()))
    results.append(("Partitioned Splitting (Subremeshing)", test_partitioned_splitting_subremeshing()))
    results.append(("Partitioned Splitting (Histogram)", test_partitioned_splitting_histogram()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Boundary Preservation", test_boundary_preservation()))
    results.append(("Output Consistency", test_output_consistency()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        exit(1)
