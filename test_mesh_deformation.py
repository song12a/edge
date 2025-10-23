"""
Test to verify that mesh simplification doesn't produce abnormal vertices,
edges, or faces that cause severe deformation.
"""
import sys
sys.path.insert(0, '/home/runner/work/edge/edge')

import numpy as np
from mesh_simplification_mdd_lme import simplify_mesh_with_partitioning
from QEM import PLYReader

def test_no_abnormal_deformation():
    """Test that simplified mesh doesn't have abnormal vertices far from original bounds."""
    print("\n" + "="*70)
    print("Test: No Abnormal Mesh Deformation")
    print("="*70)
    
    # Load test mesh
    vertices, faces = PLYReader.read_ply('/home/runner/work/edge/edge/demo/data/cube_subdivided.ply')
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Get original bounds
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    bbox_size = v_max - v_min
    
    print(f"Original bounds: [{v_min[0]:.3f}, {v_min[1]:.3f}, {v_min[2]:.3f}] to "
          f"[{v_max[0]:.3f}, {v_max[1]:.3f}, {v_max[2]:.3f}]")
    
    # Simplify
    simplified_v, simplified_f = simplify_mesh_with_partitioning(
        vertices, faces, target_ratio=0.5, target_edges_per_partition=200
    )
    
    print(f"\nSimplified mesh: {len(simplified_v)} vertices, {len(simplified_f)} faces")
    
    # Get simplified bounds
    sv_min = simplified_v.min(axis=0)
    sv_max = simplified_v.max(axis=0)
    
    print(f"Simplified bounds: [{sv_min[0]:.3f}, {sv_min[1]:.3f}, {sv_min[2]:.3f}] to "
          f"[{sv_max[0]:.3f}, {sv_max[1]:.3f}, {sv_max[2]:.3f}]")
    
    # Check 1: Vertices should stay within reasonable bounds (allow small tolerance)
    tolerance = 0.2  # 20% expansion beyond original bounds
    expanded_min = v_min - tolerance * bbox_size
    expanded_max = v_max + tolerance * bbox_size
    
    abnormal_verts = []
    for i, v in enumerate(simplified_v):
        if np.any(v < expanded_min) or np.any(v > expanded_max):
            abnormal_verts.append((i, v))
    
    if len(abnormal_verts) > 0:
        print(f"\n❌ FAILED: Found {len(abnormal_verts)} vertices outside reasonable bounds!")
        for i, v in abnormal_verts[:5]:
            print(f"  Vertex {i}: {v}")
        return False
    
    # Check 2: Edge lengths should be reasonable
    max_original_edge_length = 0
    for face in faces:
        for i in range(3):
            v1, v2 = vertices[face[i]], vertices[face[(i+1)%3]]
            edge_len = np.linalg.norm(v2 - v1)
            max_original_edge_length = max(max_original_edge_length, edge_len)
    
    max_simplified_edge_length = 0
    abnormal_edges = []
    for face_idx, face in enumerate(simplified_f):
        for i in range(3):
            v1, v2 = simplified_v[face[i]], simplified_v[face[(i+1)%3]]
            edge_len = np.linalg.norm(v2 - v1)
            max_simplified_edge_length = max(max_simplified_edge_length, edge_len)
            
            # Check for extremely long edges (> 5x original max)
            if edge_len > 5 * max_original_edge_length:
                abnormal_edges.append((face_idx, i, edge_len))
    
    print(f"\nOriginal max edge length: {max_original_edge_length:.3f}")
    print(f"Simplified max edge length: {max_simplified_edge_length:.3f}")
    
    if len(abnormal_edges) > 0:
        print(f"\n❌ FAILED: Found {len(abnormal_edges)} abnormally long edges!")
        for face_idx, edge_idx, length in abnormal_edges[:5]:
            print(f"  Face {face_idx}, edge {edge_idx}: length={length:.3f}")
        return False
    
    # Check 3: Face areas should be reasonable
    def compute_face_area(v0, v1, v2):
        """Compute triangle area using cross product."""
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        return 0.5 * np.linalg.norm(cross)
    
    max_original_area = 0
    for face in faces:
        area = compute_face_area(vertices[face[0]], vertices[face[1]], vertices[face[2]])
        max_original_area = max(max_original_area, area)
    
    abnormal_faces = []
    for face_idx, face in enumerate(simplified_f):
        area = compute_face_area(simplified_v[face[0]], simplified_v[face[1]], simplified_v[face[2]])
        
        # Check for extremely large faces (> 10x original max)
        if area > 10 * max_original_area:
            abnormal_faces.append((face_idx, area))
    
    print(f"Original max face area: {max_original_area:.3f}")
    print(f"Simplified max face area: {max([compute_face_area(simplified_v[f[0]], simplified_v[f[1]], simplified_v[f[2]]) for f in simplified_f]):.3f}")
    
    if len(abnormal_faces) > 0:
        print(f"\n❌ FAILED: Found {len(abnormal_faces)} abnormally large faces!")
        for face_idx, area in abnormal_faces[:5]:
            print(f"  Face {face_idx}: area={area:.3f}")
        return False
    
    print("\n✓ PASSED: No abnormal vertices, edges, or faces detected")
    print("✓ Mesh geometry is well-preserved")
    return True

if __name__ == "__main__":
    try:
        success = test_no_abnormal_deformation()
        if success:
            print("\n" + "="*70)
            print("All deformation tests passed!")
            print("="*70)
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
