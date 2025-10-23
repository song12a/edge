# LME Selection Logic Update

## Overview

This update implements true Local Minimal Edge (LME) selection in the `LMESimplifier.simplify()` method. Previously, the algorithm was selecting edges based on global cost ordering (greedy approach). Now it correctly selects only edges that are locally minimal for at least one of their endpoints.

## What Changed

### Previous Behavior (Global Greedy)

The old implementation:
1. Computed costs for all edges
2. Added all edges to a single priority heap
3. Collapsed edges in order of global minimum cost

This meant that if vertex A had edges with costs [1.0, 5.0] and vertex B had edges with costs [2.0, 3.0], only the edge with cost 1.0 would be considered initially, even though the edge with cost 2.0 is the minimum for vertex B.

### New Behavior (Local Minimal Edges)

The new implementation:
1. For each vertex v_i, tracks all connected edges
2. For each vertex v_i, identifies the edge e(i,j) with minimal QEM cost among all edges connected to v_i
3. Only adds edges to the heap if they are the LME for at least one of their endpoints
4. After each collapse, recomputes the LME for all affected vertices
5. Adds newly identified LME edges to the heap

This ensures that local structure is preserved and multiple edges can be considered for collapse simultaneously if they are locally minimal.

## Benefits

1. **Better Preservation of Local Features**: By considering local minimal edges rather than just globally minimal edges, the algorithm better preserves local geometric features.

2. **More Balanced Simplification**: Instead of aggressively collapsing the cheapest edges globally, the algorithm distributes simplification more evenly across the mesh.

3. **Follows MDD/LME Paper**: This implementation aligns with the "Local Minimal Edges" concept from the paper on out-of-core mesh simplification.

## Implementation Details

### Key Data Structures

- `vertex_edges`: Maps each vertex to its list of connected edges
- `edge_costs`: Maps each edge to its (cost, optimal_position) tuple
- `vertex_lme`: Maps each vertex to its current LME (min_cost, edge) tuple
- `heap_edges`: Tracks which edges are currently in the priority heap

### Algorithm Flow

1. **Initialization**:
   - Build vertex-to-edges mapping
   - Compute costs for all edges
   - For each vertex, find its LME
   - Add all LME edges to the heap

2. **Collapse Loop**:
   - Pop the cheapest edge from heap
   - Verify it's still an LME for at least one endpoint
   - Collapse the edge
   - Identify affected vertices (neighbors of collapsed vertices)
   - Recompute LME for all affected vertices
   - Add new LME edges to the heap

3. **Termination**:
   - Continue until target vertex count is reached
   - Rebuild the simplified mesh

### Edge Selection Criteria

An edge is eligible for collapse if:
1. Both vertices are still valid (not already collapsed)
2. The edge is the LME for at least one of its endpoints
3. The edge is in the priority heap

## Testing

Three comprehensive tests verify the implementation:

1. **Basic LME Test**: Verifies LME selection with simple diamond mesh
2. **LME vs Global Greedy**: Demonstrates that multiple LME edges can be identified simultaneously
3. **LME Recomputation**: Verifies that LME is correctly recomputed after collapses

All tests pass successfully.

## Performance Implications

- **Time Complexity**: Remains O(E log E) per partition, where E is the number of edges
- **Space Complexity**: Additional O(V + E) for tracking vertex edges and LME data
- **Practical Impact**: Minimal overhead; the algorithm still processes edges efficiently

## Backward Compatibility

This change maintains full backward compatibility:
- Same function signature for `simplify()`
- Same return values
- No breaking changes to the API
- Existing code continues to work without modification

## Files Modified

- `mesh_simplification_mdd_lme.py`: Updated `LMESimplifier.simplify()` method
- `test_lme_selection.py`: New comprehensive test file for LME logic

## Security

CodeQL analysis: **0 vulnerabilities** detected.
