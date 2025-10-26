# Mesh Simplification with MDD and LME

This repository implements a mesh simplification algorithm based on the concepts of **MDD (Minimal Simplification Domain)** and **LME (Local Minimal Edges)** from the paper on out-of-core mesh simplification, with support for **2-ring neighborhoods** for accurate QEM-based simplification.

It also includes **Edge Splitting with Octree Partitioning** for mesh refinement with boundary preservation.

## Overview

The implementation provides a modular approach to mesh processing:

### Mesh Simplification
1. **Partitions** large meshes into smaller sub-meshes (MDD - Minimal Simplification Domain) with 2-ring neighborhood support
2. **Simplifies** each sub-mesh independently using QEM with border preservation (LME - Local Minimal Edges)
3. **Merges** the simplified sub-meshes back into a single coherent output mesh

### Edge Splitting (New)
1. **Partitions** meshes using octree spatial subdivision (optional)
2. **Splits** edges in each partition while preserving boundary vertices
3. **Merges** split partitions back into a single refined mesh

This approach is particularly useful for:
- Processing very large meshes that don't fit in memory
- Parallel processing of mesh partitions
- Preserving geometric details at partition boundaries
- Ensuring accurate QEM calculations with topological context

## Key Features

### 2-Ring Neighborhood Support

The implementation now includes topology-based 2-ring neighborhood calculation for each partition:

- **Core Vertices**: Vertices within the spatial bounds of a partition
- **2-Ring Extension**: Vertices within 2 edges of the core vertices
- **Accurate QEM**: The 2-ring provides sufficient topological context for accurate quadric error metric calculations
- **Border Preservation**: Vertices in the 2-ring extension are preserved during simplification to maintain mesh coherence

This satisfies the requirements of the Minimal Simplification Domain (MDD) as described in the paper "Out-of-Core Framework for QEM-based Mesh Simplification."

### Edge Splitting with Octree Partitioning (New)

The edge splitting implementation supports two operational modes:

- **Without Partitioning**: Original behavior, splits edges across the entire mesh
- **With Octree Partitioning**: Divides mesh into 8 spatial partitions, processes each independently

**Key features:**
- Octree-based dynamic partitioning
- All edges eligible for splitting (including those with boundary vertices)
- Two splitting modes: Subremeshing and Histogram
- Full backward compatibility with original behavior

See [EDGE_SPLIT_PARTITIONING.md](EDGE_SPLIT_PARTITIONING.md) for detailed documentation.

## Files

- **`QEM.py`**: Base implementation of Quadric Error Metric (QEM) mesh simplification
  - `PLYReader`: Read PLY mesh files (ASCII and binary formats)
  - `PLYWriter`: Write PLY mesh files (ASCII format)
  - `QEMSimplifier`: Core QEM simplification algorithm

- **`mesh_simplification_mdd_lme.py`**: Main implementation of MDD/LME simplification with 2-ring neighborhoods
  - `MeshPartitioner`: Partitions meshes using octree spatial subdivision with 2-ring neighborhood support
  - `LMESimplifier`: Extends QEM to preserve border vertices
  - `MeshMerger`: Merges simplified sub-meshes with deduplication
  - Command-line interface for processing PLY files

- **`edge_split.py`**: Edge splitting implementation with octree partitioning support (New)
  - `MeshPartitioner`: Octree-based mesh partitioning (adapted from mesh_simplification_mdd_lme.py)
  - `EdgeSplitter`: Edge splitting with two modes (Subremeshing and Histogram)
  - Supports both partitioned and non-partitioned operation
  - See [EDGE_SPLIT_PARTITIONING.md](EDGE_SPLIT_PARTITIONING.md) for details

- **`create_test_mesh.py`**: Utility to generate test meshes (simple and subdivided cubes)

- **`examples.py`**: Comprehensive examples demonstrating various usage patterns

- **`test_2ring_neighborhood.py`**: Test suite for validating 2-ring neighborhood implementation

- **`test_edge_split.py`**: Test suite for edge splitting (backward compatibility)

- **`test_edge_split_partitioning.py`**: Test suite for edge splitting with partitioning (New)

## Installation

### Requirements

- Python 3.7+
- NumPy

```bash
pip install numpy
```

## Usage

### Basic Usage

Process all PLY files in a directory:

```bash
python mesh_simplification_mdd_lme.py
```

By default, the script:
- Reads PLY files from `./demo/data`
- Writes simplified meshes to `./demo/output`
- Uses simplification ratio of 0.5 (keeps 50% of vertices)
- Creates 8 partitions using octree subdivision

### Customizing Paths

Edit the `main()` function in `mesh_simplification_mdd_lme.py` to change:

```python
# Input/output directories
input_folder = "./demo/data"
output_folder = "./demo/output"

# Simplification parameters
simplification_ratio = 0.5  # Keep 50% of vertices
num_partitions = 8          # Octree partitioning (2x2x2)
```

### Using as a Library

**Mesh Simplification:**
```python
from mesh_simplification_mdd_lme import simplify_mesh_with_partitioning
from QEM import PLYReader, PLYWriter

# Read input mesh
vertices, faces = PLYReader.read_ply("input.ply")

# Simplify with partitioning
simplified_vertices, simplified_faces = simplify_mesh_with_partitioning(
    vertices, 
    faces,
    target_ratio=0.5,      # Keep 50% of vertices
    num_partitions=8       # Use 8 partitions
)

# Write output mesh
PLYWriter.write_ply("output.ply", simplified_vertices, simplified_faces)
```

**Edge Splitting (New):**
```python
from edge_split import EdgeSplitter, PLYReader, PLYWriter

# Read input mesh
vertices, faces = PLYReader.read_ply("input.ply")

# Method 1: Without partitioning (original behavior)
splitter = EdgeSplitter(use_partitioning=False)
splitter.initialize(vertices, faces)
new_vertices, new_faces = splitter.split_edges(mode="subremeshing", max_iter=1)

# Method 2: With octree partitioning (preserves boundaries)
splitter = EdgeSplitter(use_partitioning=True, num_partitions=8)
splitter.initialize(vertices, faces)
new_vertices, new_faces = splitter.split_edges(mode="histogram", max_iter=3)

# Write output mesh
PLYWriter.write_ply("output.ply", new_vertices, new_faces)
```

## Algorithm Details

### 1. Mesh Partitioning (MDD) with 2-Ring Neighborhoods

The mesh is partitioned using **octree spatial subdivision** with 2-ring neighborhood expansion:

#### Phase 1: Core Partitioning
- The bounding box of the mesh is divided into 8 octants
- Each vertex is assigned to an octant based on its spatial position
- These vertices form the "core" of each partition

#### Phase 2: Topology-Based Expansion
- For each partition, compute the 2-ring neighborhood of core vertices
- **1-ring**: All vertices directly connected by an edge to core vertices
- **2-ring**: All vertices connected to the 1-ring vertices
- The extended vertex set includes both core and 2-ring vertices

#### Phase 3: Face Assignment
- Faces are assigned to partitions if all their vertices are in the extended vertex set
- Vertices on partition boundaries are marked as **border vertices**

This creates smaller, independent sub-meshes with sufficient topological context for accurate QEM simplification.

### Why 2-Ring Neighborhoods?

The 2-ring neighborhood is essential for accurate QEM-based mesh simplification:

1. **Quadric Error Calculation**: The quadric error matrix for a vertex depends on adjacent faces. The 2-ring ensures all necessary face information is available.

2. **Edge Collapse Context**: When collapsing an edge, the optimal position calculation requires information about neighboring vertices and faces.

3. **Coherent Simplification**: The 2-ring provides enough context to make simplification decisions that are consistent with the global mesh structure.

4. **MDD Requirement**: The paper "Out-of-Core Framework for QEM-based Mesh Simplification" specifies 2-ring neighborhoods as a requirement for minimal simplification domains.

### 2. Local Simplification (LME)

Each sub-mesh is simplified using the **QEM (Quadric Error Metric)** method with border preservation:

- **Border vertices** (including 2-ring extension vertices) are preserved
- Only **core interior vertices** are simplified through edge collapse
- This ensures that partition boundaries remain compatible for merging
- The 2-ring context ensures accurate quadric error calculations

The QEM method:
- Computes a quadric error matrix for each vertex based on adjacent faces
- Iteratively collapses edges with minimal error
- Finds optimal vertex positions that minimize geometric distortion
- Benefits from 2-ring context for more accurate error estimation

### 3. Sub-mesh Merging

The simplified sub-meshes are merged back together:

- Vertices are deduplicated based on:
  - Original vertex indices (for border vertices)
  - Position-based matching (with tolerance)
- Faces are deduplicated to remove duplicates from overlapping partitions
- The result is a single, coherent simplified mesh

## Testing

### Running Tests

The repository includes a comprehensive test suite for validating the 2-ring neighborhood implementation:

```bash
python test_2ring_neighborhood.py
```

The test suite validates:
1. **Vertex Adjacency**: Topology-based adjacency graph construction
2. **1-Ring Neighborhoods**: Direct neighbor calculation
3. **2-Ring Neighborhoods**: Two-hop neighbor calculation
4. **Partition Expansion**: Core vertices correctly expanded with 2-ring
5. **Border Classification**: Border vertices correctly identified
6. **Mesh Coherence**: Simplified mesh is valid and coherent
7. **Simplification Quality**: Output meets quality expectations

### Test Meshes

Generate test meshes:

```bash
python create_test_mesh.py
```

This creates:
- `cube_simple.ply`: A basic cube (8 vertices, 12 faces)
- `cube_subdivided.ply`: A subdivided cube (152 vertices, 300 faces)

Run simplification on test meshes:

```bash
python mesh_simplification_mdd_lme.py
```

Check the output in `./demo/output/`

### Running Examples

See various usage examples:

```bash
python examples.py
```

This demonstrates:
1. Basic usage - single file simplification
2. Programmatic usage - load, simplify, save manually
3. Comparison with standard QEM
4. Different partition counts
5. Batch processing multiple files
6. Step-by-step component usage

## Configuration for Windows Paths

The script is designed to work with the paths specified in the requirements:

- Input: `D:\sxl08\rand1\neural-mesh-simplification\neural-mesh-simplification\demo\data`
- Output: `D:\sxl08\rand1\neural-mesh-simplification\neural-mesh-simplification\demo\output`

If these paths exist, the script will use them automatically. Otherwise, it falls back to relative paths (`./demo/data` and `./demo/output`).

To force specific paths, modify the `main()` function:

```python
input_folder = r"D:\sxl08\rand1\neural-mesh-simplification\neural-mesh-simplification\demo\data"
output_folder = r"D:\sxl08\rand1\neural-mesh-simplification\neural-mesh-simplification\demo\output"
```

## Performance Considerations

- **Partitioning overhead**: For small meshes, partitioning may add overhead. Consider using fewer partitions or the original QEM method.
- **Border vertices**: A high ratio of border vertices to interior vertices limits simplification effectiveness.
- **Memory usage**: Each partition is processed independently, reducing peak memory usage for large meshes.

## References

This implementation is based on concepts from:
- "Out-of-Core Framework for QEM-based Mesh Simplification" (included PDF)
- Quadric Error Metrics for surface simplification (Garland & Heckbert, 1997)

## License

This code is provided for educational and research purposes.
