#!/usr/bin/env python3
"""
Comprehensive validation of output meshes in demo/output directory.
Checks for:
- Abnormal vertices (outside expected bounds)
- Abnormally long edges
- Abnormally large faces
- Degenerate faces
- NaN/Inf values
- Collapsed dimensions
"""

import os
import numpy as np
from QEM import PLYReader

def compute_face_area(v0, v1, v2):
    """Compute area of a triangle face."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    return 0.5 * np.linalg.norm(cross)

def validate_mesh(input_path, output_path):
    """Validate an output mesh against its input."""
    print(f"\n{'='*70}")
    print(f"Validating: {os.path.basename(output_path)}")
    print(f"{'='*70}")
    
    # Read meshes
    try:
        orig_vertices, orig_faces = PLYReader.read_ply(input_path)
        simp_vertices, simp_faces = PLYReader.read_ply(output_path)
    except Exception as e:
        print(f"❌ ERROR reading files: {e}")
        return False
    
    print(f"Original: {len(orig_vertices)} vertices, {len(orig_faces)} faces")
    print(f"Simplified: {len(simp_vertices)} vertices, {len(simp_faces)} faces")
    
    issues = []
    
    # 1. Check bounds
    orig_min = orig_vertices.min(axis=0)
    orig_max = orig_vertices.max(axis=0)
    orig_bbox = orig_max - orig_min
    
    simp_min = simp_vertices.min(axis=0)
    simp_max = simp_vertices.max(axis=0)
    simp_bbox = simp_max - simp_min
    
    print(f"\nOriginal bounds: [{orig_min[0]:.4f}, {orig_min[1]:.4f}, {orig_min[2]:.4f}] to [{orig_max[0]:.4f}, {orig_max[1]:.4f}, {orig_max[2]:.4f}]")
    print(f"Simplified bounds: [{simp_min[0]:.4f}, {simp_min[1]:.4f}, {simp_min[2]:.4f}] to [{simp_max[0]:.4f}, {simp_max[1]:.4f}, {simp_max[2]:.4f}]")
    
    # Check for abnormal vertices (outside 20% tolerance)
    tolerance = 0.2
    expanded_min = orig_min - tolerance * orig_bbox
    expanded_max = orig_max + tolerance * orig_bbox
    
    abnormal_verts = []
    for i, v in enumerate(simp_vertices):
        if np.any(v < expanded_min) or np.any(v > expanded_max):
            abnormal_verts.append((i, v))
    
    if abnormal_verts:
        issues.append(f"Found {len(abnormal_verts)} abnormal vertices (outside 20% tolerance)")
        for i, v in abnormal_verts[:3]:
            print(f"  Abnormal vertex {i}: {v}")
        if len(abnormal_verts) > 3:
            print(f"  ... and {len(abnormal_verts) - 3} more")
    
    # 2. Check for collapsed dimensions
    collapsed_dims = simp_bbox < 0.01 * orig_bbox
    if np.any(collapsed_dims):
        dims = ['X' if collapsed_dims[0] else '', 'Y' if collapsed_dims[1] else '', 'Z' if collapsed_dims[2] else '']
        issues.append(f"Mesh collapsed in dimension(s): {' '.join([d for d in dims if d])}")
    
    # 3. Check for NaN/Inf
    if np.any(np.isnan(simp_vertices)):
        issues.append("NaN values detected in vertices")
    if np.any(np.isinf(simp_vertices)):
        issues.append("Inf values detected in vertices")
    
    # 4. Check for degenerate faces
    degenerate = sum(1 for face in simp_faces if len(set(face)) != 3)
    if degenerate > 0:
        issues.append(f"Found {degenerate} degenerate faces")
    
    # 5. Check edge lengths
    edge_lengths = []
    for face in simp_faces:
        for j in range(3):
            v1, v2 = simp_vertices[face[j]], simp_vertices[face[(j+1)%3]]
            length = np.linalg.norm(v2 - v1)
            edge_lengths.append(length)
    
    if edge_lengths:
        edge_lengths = np.array(edge_lengths)
        mean_len = edge_lengths.mean()
        std_len = edge_lengths.std()
        max_len = edge_lengths.max()
        
        print(f"\nEdge statistics:")
        print(f"  Mean: {mean_len:.4f}, Std: {std_len:.4f}, Max: {max_len:.4f}")
        
        # Check for abnormally long edges (> 5x original diagonal)
        orig_diag = np.linalg.norm(orig_bbox)
        if max_len > 5 * orig_diag:
            issues.append(f"Abnormally long edge: {max_len:.4f} (> 5x diagonal {orig_diag:.4f})")
    
    # 6. Check face areas
    face_areas = []
    for face in simp_faces:
        area = compute_face_area(simp_vertices[face[0]], simp_vertices[face[1]], simp_vertices[face[2]])
        face_areas.append(area)
    
    if face_areas:
        face_areas = np.array(face_areas)
        mean_area = face_areas.mean()
        max_area = face_areas.max()
        
        print(f"Face areas:")
        print(f"  Mean: {mean_area:.6f}, Max: {max_area:.6f}")
        
        # Original face areas
        orig_areas = []
        for face in orig_faces:
            area = compute_face_area(orig_vertices[face[0]], orig_vertices[face[1]], orig_vertices[face[2]])
            orig_areas.append(area)
        orig_areas = np.array(orig_areas)
        
        # Check for abnormally large faces (> 50x original max)
        if max_area > 50 * orig_areas.max():
            issues.append(f"Abnormally large face: {max_area:.6f} (> 50x original max {orig_areas.max():.6f})")
    
    # Print results
    print(f"\n{'='*70}")
    if issues:
        print("❌ VALIDATION FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ VALIDATION PASSED: No issues detected")
        return True

def main():
    """Validate all meshes in demo/output."""
    input_dir = "demo/data"
    output_dir = "demo/output"
    
    print("="*70)
    print("COMPREHENSIVE MESH VALIDATION")
    print("="*70)
    
    # Get all output files
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.ply')]
    
    if not output_files:
        print("\n❌ No output files found in demo/output/")
        return
    
    print(f"\nFound {len(output_files)} output file(s) to validate")
    
    results = {}
    
    for output_file in sorted(output_files):
        output_path = os.path.join(output_dir, output_file)
        
        # Find corresponding input file
        if output_file.startswith('simplified_'):
            input_file = output_file[len('simplified_'):]
        else:
            input_file = output_file
        
        input_path = os.path.join(input_dir, input_file)
        
        if not os.path.exists(input_path):
            print(f"\n⚠ Warning: Input file not found for {output_file}")
            continue
        
        results[output_file] = validate_mesh(input_path, output_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print(f"Total files: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓✓✓ ALL MESHES VALIDATED SUCCESSFULLY ✓✓✓")
    else:
        print(f"\n✗✗✗ {failed} MESH(ES) FAILED VALIDATION ✗✗✗")
        print("\nFailed files:")
        for fname, passed in results.items():
            if not passed:
                print(f"  - {fname}")

if __name__ == "__main__":
    main()
