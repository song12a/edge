# Bug Fix: Mesh Deformation in LME Selection

## Issue Report

User reported (@song12a comment #3432615792):
> "现在的问题是会出现异常点和异常边以及异常面，严重偏离原网格，导致网格出现严重形变"
> 
> Translation: "The current problem is that abnormal points, abnormal edges, and abnormal faces appear, seriously deviating from the original mesh, causing severe mesh deformation."

## Root Cause Analysis

The initial LME implementation had a critical bug in how it handled edge costs after edge collapses:

### The Problem

1. **Edge costs were cached once** at initialization in the `edge_costs` dictionary
2. **After collapsing edge (v1, v2)**:
   - Vertex v2 is deleted
   - Vertex v1 moves to the optimal collapse position
   - v1's position changes, but **edge costs for v1's remaining edges are NOT updated**
3. **When recomputing LME for affected vertices**:
   - The code used the **stale costs** from `edge_costs` that were computed with v1's **old position**
   - This caused incorrect LME selection using costs that didn't reflect current geometry
4. **Result**: 
   - Wrong edges were selected for collapse
   - Vertices were moved to positions that made no geometric sense
   - Severe mesh deformation with abnormal vertices, edges, and faces

### Example of the Bug

```python
# Initial state
v1 at position [0, 0, 0]
v2 at position [1, 0, 0]
edge(v1, v3) has cost 1.0 (computed with v1 at [0, 0, 0])

# After collapsing edge(v1, v2)
v1 moves to [0.5, 0, 0]  # New optimal position
v2 is deleted

# BUG: edge(v1, v3) still has cached cost 1.0
# But v1 is now at [0.5, 0, 0], not [0, 0, 0]
# The actual cost should be recomputed with v1's new position!
```

## The Fix

### Key Changes

1. **Removed persistent edge cost caching**: The `edge_costs` dictionary that stored costs across collapses was removed

2. **Recompute costs when finding LME**: The `find_vertex_lme()` function now recomputes edge costs fresh every time:
   ```python
   def find_vertex_lme(vertex):
       for edge in vertex_edges[vertex]:
           # CRITICAL: Recompute cost using CURRENT vertex positions
           cost, optimal_pos = compute_edge_cost(v1, v2)
           # ... find minimum
   ```

3. **Proper edge structure updates**: After collapse, edges involving the deleted vertex are properly removed from all data structures

4. **LME verification before collapse**: Double-check that a heap entry is still a valid LME before performing the collapse:
   ```python
   # Verify this edge is still the LME for at least one endpoint
   lme_v1 = find_vertex_lme(v1)  # Recomputes costs fresh
   if lme_v1 and lme_v1[1] == edge:
       is_lme = True
   ```

### Performance Implications

- **Trade-off**: More computation (costs recomputed each time) vs. correctness
- **Time complexity**: Still O(E log E) - same asymptotic complexity
- **Practical impact**: Negligible for typical meshes, critical for correctness
- **The cost is worth it**: Recomputing ensures geometric validity

## Verification

Created comprehensive test suite (`test_mesh_deformation.py`) that checks:

### Test 1: Vertex Bounds
- ✅ All vertices stay within 20% of original bounding box
- ✅ No abnormal vertices far from mesh

### Test 2: Edge Lengths
- ✅ No edges longer than 5x original maximum edge length
- ✅ Original max: 0.283, Simplified max: 0.639 (within bounds)

### Test 3: Face Areas
- ✅ No faces larger than 10x original maximum face area
- ✅ Original max: 0.020, Simplified max: 0.111 (within bounds)

### Test Results

```
Input mesh: 152 vertices, 300 faces
Original bounds: [0.000, 0.000, 0.000] to [1.000, 1.000, 1.000]

Simplified mesh: 88 vertices, 153 faces
Simplified bounds: [0.000, 0.000, 0.000] to [1.000, 1.000, 1.000]

✓ PASSED: No abnormal vertices, edges, or faces detected
✓ Mesh geometry is well-preserved
```

## Files Changed

1. **mesh_simplification_mdd_lme.py**
   - Removed `edge_costs` persistent dictionary
   - Added `compute_edge_cost()` helper function
   - Modified `find_vertex_lme()` to recompute costs fresh
   - Changed `vertex_edges` from list to set for efficient updates
   - Improved edge cleanup after collapse

2. **test_mesh_deformation.py** (new)
   - Comprehensive tests for abnormal vertices, edges, and faces
   - Validates mesh bounds preservation
   - Checks for geometric validity

## Commit

Commit hash: `b042a86`
Message: "Fix mesh deformation issue in LME selection - recompute edge costs after collapse"

## Conclusion

The bug was caused by using stale geometric information (cached edge costs) after vertex positions changed. The fix ensures that edge costs are always computed using current vertex positions, preventing mesh deformation and ensuring geometrically valid simplification.
