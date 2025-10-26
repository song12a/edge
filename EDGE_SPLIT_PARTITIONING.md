# Edge Split with Octree Partitioning - Implementation Guide

## Overview

This document describes the modifications made to `edge_split.py` to support octree-based dynamic partitioning with boundary vertex preservation, following the architecture from `mesh_simplification_mdd_lme.py`.

## Key Features

### 1. Octree Dynamic Partitioning
The mesh is divided into 8 spatial partitions (octree) with 2-ring neighborhood support:

- **Core vertices**: Vertices within the spatial bounds of each partition
- **Extended vertices**: Core vertices plus their 2-ring neighborhood
- **Border vertices**: Vertices on partition boundaries (preserved during splitting)

### 2. Boundary Vertex Preservation
Interior vertices only are modified during edge splitting:

- **Border vertices** (vertices on partition boundaries) are **not** split
- Only **interior vertices** (non-border vertices within each partition) undergo edge splitting
- This ensures mesh coherence when partitions are merged

### 3. Two Splitting Modes Preserved
Both original splitting modes are supported with and without partitioning:

- **Subremeshing mode**: Splits long edges based on average edge length
- **Histogram mode**: Splits edges based on curvature-weighted thresholds

## Architecture

### New Classes

#### `MeshPartitioner`
Handles octree-based mesh partitioning:

```python
partitioner = MeshPartitioner(vertices, faces, num_partitions=8)
partitions = partitioner.partition_octree()
```

**Key methods:**
- `partition_octree()`: Creates 8 spatial partitions with 2-ring neighborhoods
- `build_vertex_adjacency()`: Builds vertex-to-vertex connectivity
- `compute_n_ring_neighborhood()`: Computes n-ring neighborhood of vertices
- `extract_submesh()`: Extracts a submesh from a partition

### Modified Class

#### `EdgeSplitter`
Now supports both partitioned and non-partitioned modes:

```python
# Without partitioning (original behavior)
splitter = EdgeSplitter(use_partitioning=False)
splitter.initialize(vertices, faces)
result_vertices, result_faces = splitter.split_edges(mode="subremeshing", max_iter=10)

# With octree partitioning (new behavior)
splitter = EdgeSplitter(use_partitioning=True, num_partitions=8)
splitter.initialize(vertices, faces)
result_vertices, result_faces = splitter.split_edges(mode="subremeshing", max_iter=10)
```

**New methods:**
- `split_edges_with_partitioning()`: Main entry point for partitioned splitting
- `_split_partition_subremeshing()`: Splits edges in a partition (subremeshing mode)
- `_split_partition_histogram()`: Splits edges in a partition (histogram mode)
- `_merge_split_submeshes()`: Merges split partitions back into a single mesh

## Algorithm Flow

### Without Partitioning (Original Behavior)
```
1. Initialize mesh
2. Split edges (entire mesh at once)
   - Subremeshing: split based on E_ave
   - Histogram: split based on curvature
3. Return result
```

### With Partitioning (New Behavior)
```
1. Initialize mesh
2. Partition mesh using octree
   - Calculate bounding box center
   - Assign vertices to 8 octants
   - Expand each partition with 2-ring neighborhood
   - Identify border vertices
3. For each partition:
   - Extract submesh
   - Identify border vertices in local indices
   - Split edges (excluding border vertices)
     * Subremeshing: split long edges
     * Histogram: split based on curvature
4. Merge split partitions
   - Deduplicate vertices by position and original index
   - Deduplicate faces
5. Return result
```

## Usage Examples

### Basic Usage

```python
from edge_split import EdgeSplitter, PLYReader, PLYWriter

# Read mesh
reader = PLYReader()
vertices, faces = reader.read_ply("input.ply")

# Method 1: Without partitioning (original)
splitter = EdgeSplitter(use_partitioning=False)
splitter.initialize(vertices, faces)
new_vertices, new_faces = splitter.split_edges(mode="subremeshing", max_iter=1)

# Method 2: With partitioning (new)
splitter = EdgeSplitter(use_partitioning=True, num_partitions=8)
splitter.initialize(vertices, faces)
new_vertices, new_faces = splitter.split_edges(mode="histogram", max_iter=3)

# Write result
writer = PLYWriter()
writer.write_ply("output.ply", new_vertices, new_faces)
```

### Advanced Usage

```python
# Custom number of partitions (must be 8 for octree)
splitter = EdgeSplitter(use_partitioning=True, num_partitions=8)
splitter.initialize(vertices, faces)

# Subremeshing with partitioning
vertices_sub, faces_sub = splitter.split_edges(mode="subremeshing", max_iter=5)

# Histogram with partitioning
splitter2 = EdgeSplitter(use_partitioning=True, num_partitions=8)
splitter2.initialize(vertices, faces)
vertices_hist, faces_hist = splitter2.split_edges(mode="histogram", max_iter=10)
```

## Testing

### Run All Tests

```bash
# Original tests (backward compatibility)
python test_edge_split.py

# New partitioning tests
python test_edge_split_partitioning.py
```

### Test Coverage

The test suite covers:
1. **Partitioner functionality**: Octree partitioning, 2-ring neighborhoods
2. **Partitioned splitting (subremeshing)**: Edge splitting with partitions
3. **Partitioned splitting (histogram)**: Curvature-based splitting with partitions
4. **Backward compatibility**: Original behavior without partitioning
5. **Boundary preservation**: Border vertices are preserved
6. **Output consistency**: File I/O works correctly

## Implementation Details

### Boundary Vertex Detection

Border vertices are identified in two ways:

1. **Inter-partition boundaries**: Vertices whose incident faces span multiple core partitions
2. **2-ring extension vertices**: Vertices not in the core of a partition but in its 2-ring neighborhood

```python
# During partitioning
for v in partition['vertices']:
    if v not in partition['core_vertices']:
        partition['is_border'].add(v)  # 2-ring extension
    elif v in self.border_vertices:
        partition['is_border'].add(v)  # Inter-partition boundary
```

### Edge Splitting with Boundary Exclusion

When splitting edges, border vertices are skipped:

```python
for i in range(len(neighbor_out)):
    b1 = i
    if b1 in border_vertices:  # Skip border vertices
        continue
    
    for j in range(0, len(neighbor_out[i]), 2):
        b2 = neighbor_out[i][j]
        if b2 in border_vertices:  # Skip edges to border vertices
            continue
        
        # Process edge (b1, b2) for splitting
        ...
```

### Partition Merging

After splitting each partition, submeshes are merged:

1. **Vertex deduplication**: 
   - Match vertices by original global index
   - Match vertices by position (with tolerance)
   
2. **Face deduplication**:
   - Remap face indices to merged vertex indices
   - Remove duplicate faces (same vertices, any order)

```python
# Position-based matching
vertex_key = tuple(int(round(vertex[i] / tolerance)) for i in range(3))
if vertex_key in vertex_position_map:
    merged_idx = vertex_position_map[vertex_key]
```

## Performance Considerations

### When to Use Partitioning

**Use partitioning (`use_partitioning=True`) when:**
- Processing large meshes (> 100K vertices)
- Want to preserve mesh boundaries
- Need finer control over which vertices are modified

**Use original mode (`use_partitioning=False`) when:**
- Processing small meshes (< 10K vertices)
- Want maximum splitting aggressiveness
- Don't need boundary preservation

### Partition Size Impact

With octree partitioning (8 partitions):
- Small meshes may have most vertices as border vertices → less splitting
- Large meshes have better core/border ratio → more effective splitting

## Backward Compatibility

The implementation maintains **full backward compatibility**:

```python
# Old code (still works)
splitter = EdgeSplitter()
splitter.initialize(vertices, faces)
result = splitter.split_edges(mode="subremeshing")

# New code (explicit)
splitter = EdgeSplitter(use_partitioning=False)
splitter.initialize(vertices, faces)
result = splitter.split_edges(mode="subremeshing")
```

Both produce identical results.

## Dependencies

- Python 3.7+
- NumPy (for efficient array operations in partitioning)

## References

- Original `edge_split.py`: Edge splitting with subremeshing and histogram modes
- `mesh_simplification_mdd_lme.py`: Octree partitioning and 2-ring neighborhood implementation
- "Out-of-Core Framework for QEM-based Mesh Simplification": MDD concept
