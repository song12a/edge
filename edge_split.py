import math
import struct
import numpy as np
import os
from typing import List, Tuple, Any, Dict, Set

Point = List[float]
Triangle = List[int]
Neighbor = List[int]


class PLYReader:
    @staticmethod
    def read_ply(file_path):
        """读取PLY文件，返回顶点和面数据"""
        vertices = []
        faces = []

        with open(file_path, 'rb') as f:
            header_lines = []
            line = b''
            while b'end_header' not in line:
                line = f.readline()
                header_lines.append(line.decode('ascii', errors='ignore').strip())

            vertex_count = 0
            face_count = 0
            for line in header_lines:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('element face'):
                    face_count = int(line.split()[-1])

            format_line = [line for line in header_lines if line.startswith('format')][0]
            is_ascii = 'ascii' in format_line.lower()

            # 读取顶点数据
            if is_ascii:
                for _ in range(vertex_count):
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    vertex_data = list(map(float, line.split()[:3]))
                    vertices.append(vertex_data)
            else:
                for _ in range(vertex_count):
                    data = f.read(12)
                    x, y, z = struct.unpack('fff', data)
                    vertices.append([x, y, z])

            # 读取面数据
            if is_ascii:
                for _ in range(face_count):
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    face_data = list(map(int, line.split()))
                    if face_data[0] == 3:
                        faces.append(face_data[1:4])
                    elif face_data[0] > 3:
                        for j in range(1, face_data[0] - 1):
                            faces.append([face_data[1], face_data[1 + j], face_data[2 + j]])
            else:
                for _ in range(face_count):
                    count_data = f.read(1)
                    vertex_count_in_face = struct.unpack('B', count_data)[0]
                    if vertex_count_in_face == 3:
                        data = f.read(12)
                        v1, v2, v3 = struct.unpack('iii', data)
                        faces.append([v1, v2, v3])
                    elif vertex_count_in_face > 3:
                        data = f.read(4 * vertex_count_in_face)
                        indices = struct.unpack('i' * vertex_count_in_face, data)
                        for j in range(1, vertex_count_in_face - 1):
                            faces.append([indices[0], indices[j], indices[j + 1]])

        return vertices, faces


class PLYWriter:
    @staticmethod
    def write_ply(file_path, vertices, faces):
        """将顶点和面数据写入PLY文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_index\n")
            f.write("end_header\n")

            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


class MeshPartitioner:
    """
    Partitions a mesh into smaller sub-meshes based on octree spatial subdivision.
    Adapted from mesh_simplification_mdd_lme.py for edge splitting.
    """

    def __init__(self, vertices: List[Point], faces: List[Triangle], num_partitions: int = 8):
        """
        Initialize the mesh partitioner.

        Args:
            vertices: List of vertex coordinates
            faces: List of face indices
            num_partitions: Number of spatial partitions (default: 8 for octree)
        """
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        self.num_partitions = num_partitions
        self.partitions = []
        self.border_vertices = set()  # Vertices on partition boundaries
        self.vertex_adjacency = None
        self.boundary_edges = {}  # Maps (v1, v2) -> list of partition indices
        self.boundary_edge_split_plan = {}  # Maps (v1, v2) -> split plan

    def build_vertex_adjacency(self) -> Dict[int, Set[int]]:
        """Build vertex-to-vertex adjacency information from faces."""
        adjacency = {i: set() for i in range(len(self.vertices))}

        for face in self.faces:
            v0, v1, v2 = face
            adjacency[v0].add(v1)
            adjacency[v0].add(v2)
            adjacency[v1].add(v0)
            adjacency[v1].add(v2)
            adjacency[v2].add(v0)
            adjacency[v2].add(v1)

        return adjacency

    def compute_n_ring_neighborhood(self, vertex_set: Set[int], n: int = 1) -> Set[int]:
        """Compute the n-ring neighborhood of a set of vertices."""
        if self.vertex_adjacency is None:
            self.vertex_adjacency = self.build_vertex_adjacency()

        current_ring = vertex_set.copy()
        all_vertices = vertex_set.copy()

        for _ in range(n):
            next_ring = set()
            for vertex in current_ring:
                next_ring.update(self.vertex_adjacency[vertex])

            all_vertices.update(next_ring)
            current_ring = next_ring - vertex_set
            vertex_set = all_vertices.copy()

        return all_vertices

    def partition_octree(self) -> List[Dict]:
        """
        Partition the mesh using octree spatial subdivision with 2-ring neighborhood support.

        Returns:
            List of partition dictionaries, each containing:
                - 'vertices': all vertex indices in this partition (core + 2-ring)
                - 'core_vertices': vertex indices in the spatial bounds only
                - 'faces': face indices in this partition
                - 'is_border': set of border vertices (shared with other partitions)
        """
        # Calculate bounding box
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        center = (min_coords + max_coords) / 2

        # Determine which octant each vertex belongs to
        vertex_partitions = np.zeros(len(self.vertices), dtype=np.int32)

        for i, vertex in enumerate(self.vertices):
            octant = 0
            if vertex[0] > center[0]:
                octant += 1
            if vertex[1] > center[1]:
                octant += 2
            if vertex[2] > center[2]:
                octant += 4
            vertex_partitions[i] = octant

        # Initialize partition data structures
        partition_data = [{'core_vertices': set(), 'vertices': set(), 'faces': [], 'is_border': set()}
                         for _ in range(8)]

        # First pass: assign vertices to their core partitions
        for i in range(len(self.vertices)):
            partition_idx = vertex_partitions[i]
            partition_data[partition_idx]['core_vertices'].add(i)

        # Second pass: expand each partition with 2-ring neighborhoods
        for p_idx, p_data in enumerate(partition_data):
            if len(p_data['core_vertices']) > 0:
                extended_vertices = self.compute_n_ring_neighborhood(p_data['core_vertices'], n=2)
                p_data['vertices'] = extended_vertices

        # Third pass: assign faces to partitions
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = face

            core_partitions = {vertex_partitions[v0], vertex_partitions[v1], vertex_partitions[v2]}

            for p_idx, p_data in enumerate(partition_data):
                if v0 in p_data['vertices'] and v1 in p_data['vertices'] and v2 in p_data['vertices']:
                    p_data['faces'].append(face_idx)

            # Mark border vertices
            if len(core_partitions) > 1:
                for v in face:
                    self.border_vertices.add(v)

        # Fourth pass: identify border vertices for each partition
        for p_idx, p_data in enumerate(partition_data):
            for v in p_data['vertices']:
                if v not in p_data['core_vertices']:
                    p_data['is_border'].add(v)
                elif v in self.border_vertices:
                    p_data['is_border'].add(v)

        # Filter out empty partitions
        self.partitions = [p for p in partition_data if len(p['faces']) > 0]
        
        # Fifth pass: identify all boundary edges
        self.identify_boundary_edges(vertex_partitions)

        return self.partitions

    def identify_boundary_edges(self, vertex_partitions: np.ndarray) -> None:
        """
        Identify all boundary edges (edges shared by multiple partitions).
        
        Args:
            vertex_partitions: Array mapping vertex index to partition index
        """
        # Build adjacency if not already built
        if self.vertex_adjacency is None:
            self.vertex_adjacency = self.build_vertex_adjacency()
        
        # For each edge in the mesh, check if its vertices belong to different partitions
        edge_partitions = {}  # Maps (v1, v2) -> set of partition indices
        
        for face in self.faces:
            v0, v1, v2 = face
            edges = [(v0, v1), (v1, v2), (v2, v0)]
            
            for v_a, v_b in edges:
                # Normalize edge representation (smaller index first)
                edge = (min(v_a, v_b), max(v_a, v_b))
                
                # Get partitions for both vertices
                part_a = vertex_partitions[v_a]
                part_b = vertex_partitions[v_b]
                
                if edge not in edge_partitions:
                    edge_partitions[edge] = set()
                
                # Add the partition(s) this edge belongs to
                edge_partitions[edge].add(part_a)
                edge_partitions[edge].add(part_b)
        
        # Identify boundary edges (belong to multiple partitions)
        for edge, partitions in edge_partitions.items():
            if len(partitions) > 1:
                self.boundary_edges[edge] = sorted(list(partitions))

    def extract_submesh(self, partition: Dict) -> Tuple[List[Point], List[Triangle], Dict[int, int], Dict[int, int]]:
        """
        Extract a sub-mesh from a partition.

        Returns:
            Tuple of (vertices, faces, vertex_map, reverse_map) where:
                - vertices: list of vertex coordinates in the sub-mesh
                - faces: list of face indices in the sub-mesh (local indexing)
                - vertex_map: mapping from global vertex indices to local indices
                - reverse_map: mapping from local vertex indices to global indices
        """
        vertex_list = sorted(partition['vertices'])
        vertex_map = {global_idx: local_idx for local_idx, global_idx in enumerate(vertex_list)}
        reverse_map = {local_idx: global_idx for global_idx, local_idx in vertex_map.items()}

        # Extract vertices as list
        submesh_vertices = [self.vertices[i].tolist() for i in vertex_list]

        # Extract and reindex faces
        submesh_faces = []
        for face_idx in partition['faces']:
            face = self.faces[face_idx]
            local_face = [vertex_map[v] for v in face]
            submesh_faces.append(local_face)

        return submesh_vertices, submesh_faces, vertex_map, reverse_map


class EdgeSplitter:
    def __init__(self, use_partitioning: bool = False, num_partitions: int = 8):
        """
        Initialize EdgeSplitter.
        
        Args:
            use_partitioning: If True, use octree partitioning for edge splitting
            num_partitions: Number of spatial partitions (default: 8 for octree)
        """
        self.points = []
        self.faces = []
        self.point_neighbor = []
        self.cu_ave = None
        self.L_ave = 0.0
        self.original_vertex_count = 0
        self.use_partitioning = use_partitioning
        self.num_partitions = num_partitions

    def initialize(self, points: List[Point], faces: List[Triangle]) -> None:
        self.points = points
        self.faces = faces
        self.original_vertex_count = len(points)
        self._build_neighbor_structure()
        self.L_ave = self._compute_average_edge_length()

    def _build_neighbor_structure(self) -> None:
        self.point_neighbor = [[] for _ in range(len(self.points))]
        for face in self.faces:
            self._add_neighbor_triple(self.point_neighbor, face[0], face[1], face[2])

    def _compute_average_edge_length(self) -> float:
        edge_lengths = []
        visited = set()
        for face in self.faces:
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                if a < b:
                    pair = (a, b)
                else:
                    pair = (b, a)
                if pair not in visited:
                    visited.add(pair)
                    edge_lengths.append(self._edge_length(self.points[a], self.points[b]))
        self.L_ave = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 0.0
        return self.L_ave

    def _edge_length(self, a: Point, b: Point) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _vertex_normal(self, idx: int) -> List[float]:
        n = [0.0, 0.0, 0.0]
        pairs = self.point_neighbor[idx]
        for j in range(len(pairs) // 2):
            b2 = pairs[2 * j]
            b3 = pairs[2 * j + 1]
            if not self._is_valid_index(b2) or not self._is_valid_index(b3):
                continue
            nf = self._face_normal(idx, b2, b3)
            area = self._triangle_area(idx, b2, b3)
            n[0] += nf[0] * area
            n[1] += nf[1] * area
            n[2] += nf[2] * area
        return self._unit(n)

    def _face_normal(self, b1: int, b2: int, b3: int) -> List[float]:
        if not all(self._is_valid_index(i) for i in [b1, b2, b3]):
            return [0.0, 0.0, 0.0]
        p1, p2, p3 = self.points[b1], self.points[b2], self.points[b3]
        v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]]
        n = [
            v2[2] * v1[1] - v2[1] * v1[2],
            -v2[2] * v1[0] + v2[0] * v1[2],
            v2[1] * v1[0] - v2[0] * v1[1],
        ]
        return self._unit(n)

    def _triangle_area(self, b1: int, b2: int, b3: int) -> float:
        if not all(self._is_valid_index(i) for i in [b1, b2, b3]):
            return 0.0
        p1, p2, p3 = self.points[b1], self.points[b2], self.points[b3]
        v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
        v2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]]
        cross = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ]
        return 0.5 * math.sqrt(sum(x ** 2 for x in cross))

    def _unit(self, n: List[float]) -> List[float]:
        l = math.sqrt(sum(x ** 2 for x in n))
        return [x / l for x in n] if l != 0 else [0.0, 0.0, 0.0]

    def _angle_between(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        dot = max(min(dot, 1.0), -1.0)
        return math.acos(dot)

    def _unique_neighbors(self, idx: int) -> List[int]:
        if not self._is_valid_index(idx):
            return []
        unique: List[int] = []
        for b in self.point_neighbor[idx]:
            if b not in unique and self._is_valid_index(b):
                unique.append(b)
        return unique

    def compute_harmonic_like_measure(self) -> List[float]:
        """
        实现与C++MeshGeometric_Harmonic_N_Value一致的曲率计算
        使用余切权重和距离权重进行平滑处理
        """
        N = [0.0] * len(self.points)
        
        # 步骤1: 初始化N值和余切权重
        pointsN_withoutzRepeat = []
        ctan_weight_v = []
        distance_v = []
        
        for i in range(len(self.points)):
            ni = self._vertex_normal(i)
            points_i = self.points[i]
            pointsN_withoutzRepeat_i = []
            distance_i = []
            angle_sum = 0.0
            
            # 计算邻居点和距离
            for j in range(0, len(self.point_neighbor[i]), 2):
                if j + 1 >= len(self.point_neighbor[i]):
                    continue
                p2 = self.point_neighbor[i][j]
                p3 = self.point_neighbor[i][j + 1]
                
                if not self._is_valid_index(p2) or not self._is_valid_index(p3):
                    continue
                
                # 计算距离
                dpi2 = self._edge_length(points_i, self.points[p2])
                dpi3 = self._edge_length(points_i, self.points[p3])
                
                # 计算法向量夹角
                p2n = self._vertex_normal(p2)
                p3n = self._vertex_normal(p3)
                
                # 处理零法向量
                if all(abs(x) < 1e-10 for x in p2n):
                    p2n = p3n
                if all(abs(x) < 1e-10 for x in p3n):
                    p3n = p2n
                
                a2 = self._angle_between(ni, p2n)
                a3 = self._angle_between(ni, p3n)
                angle_sum += a2 + a3
                
                # 添加唯一邻居
                if p2 not in pointsN_withoutzRepeat_i:
                    pointsN_withoutzRepeat_i.append(p2)
                    distance_i.append(dpi2)
                if p3 not in pointsN_withoutzRepeat_i:
                    pointsN_withoutzRepeat_i.append(p3)
                    distance_i.append(dpi3)
            
            # 计算余切权重
            p1_Neighbor = self.point_neighbor[i]
            ctan_weight = [0.0] * len(pointsN_withoutzRepeat_i)
            
            for j, p2 in enumerate(pointsN_withoutzRepeat_i):
                # 找到包含p2的相邻三角形
                for k in range(0, len(p1_Neighbor), 2):
                    if k + 1 >= len(p1_Neighbor):
                        continue
                    p21 = p1_Neighbor[k]
                    p22 = p1_Neighbor[k + 1]
                    
                    if p21 == p2:
                        p2_real = p22
                    elif p22 == p2:
                        p2_real = p21
                    else:
                        continue
                    
                    if not self._is_valid_index(p2_real):
                        continue
                    
                    # 计算角度 p2_real-i-p2 的内角
                    angle = self._compute_inner_angle(p2_real, i, p2)
                    tan_value = abs(math.tan(angle))
                    
                    # 限制tan值范围
                    tan_value = max(0.1, min(tan_value, 10.0))
                    
                    # 余切权重 = 1/tan
                    ctan_weight[j] += 1.0 / tan_value
            
            ctan_weight_v.append(ctan_weight)
            pointsN_withoutzRepeat.append(pointsN_withoutzRepeat_i)
            distance_v.append(distance_i)
            
            # 初始N值
            if len(self.point_neighbor[i]) > 0:
                N[i] = angle_sum / len(self.point_neighbor[i])
            else:
                N[i] = 0.0
        
        # 步骤2: 使用距离加权归一化余切权重
        ctan_weight_v_f = []
        for i in range(len(self.points)):
            ctan_weight_i = ctan_weight_v[i]
            pointsN_withoutzRepeat_i = pointsN_withoutzRepeat[i]
            distance_i = distance_v[i]
            
            if len(pointsN_withoutzRepeat_i) == 0:
                ctan_weight_v_f.append([])
                continue
            
            # 计算权重和
            sum_weight = sum(ctan_weight_i[j] / distance_i[j] 
                           for j in range(len(pointsN_withoutzRepeat_i)) 
                           if distance_i[j] > 0)
            
            # 归一化
            if sum_weight > 0:
                ctan_weight_i_f = [(ctan_weight_i[j] / distance_i[j]) / sum_weight 
                                  for j in range(len(pointsN_withoutzRepeat_i))]
            else:
                ctan_weight_i_f = [0.0] * len(pointsN_withoutzRepeat_i)
            
            ctan_weight_v_f.append(ctan_weight_i_f)
        
        # 步骤3: 平滑处理（3次迭代，lambda=0.5）
        HN_Value = N.copy()
        landa = 0.5
        
        for _ in range(3):
            new_HN = HN_Value.copy()
            for i in range(len(self.points)):
                if len(pointsN_withoutzRepeat[i]) == 0:
                    continue
                
                ctan_weight_i = ctan_weight_v_f[i]
                pointsN_withoutzRepeat_i = pointsN_withoutzRepeat[i]
                
                HN_i = sum(ctan_weight_i[j] * HN_Value[pointsN_withoutzRepeat_i[j]]
                          for j in range(len(pointsN_withoutzRepeat_i)))
                
                new_HN[i] = HN_Value[i] * landa + HN_i * (1 - landa)
            
            HN_Value = new_HN
        
        return HN_Value
    
    def _compute_inner_angle(self, p1: int, p2: int, p3: int) -> float:
        """计算角度p1-p2-p3（在p2处的内角）"""
        if not all(self._is_valid_index(i) for i in [p1, p2, p3]):
            return 0.0
        
        p1p = self.points[p1]
        p2p = self.points[p2]
        p3p = self.points[p3]
        
        # 向量 p2->p1 和 p2->p3
        pn1 = [p1p[0] - p2p[0], p1p[1] - p2p[1], p1p[2] - p2p[2]]
        pn2 = [p3p[0] - p2p[0], p3p[1] - p2p[1], p3p[2] - p2p[2]]
        
        # 归一化
        pn1 = self._unit(pn1)
        pn2 = self._unit(pn2)
        
        # 计算夹角
        cosa1 = sum(pn1[k] * pn2[k] for k in range(3))
        cosa1 = max(min(cosa1, 1.0), -1.0)
        
        return math.acos(cosa1)

    def _init_histogram_factors(self) -> None:
        if not self.cu_ave:
            self.cu_ave = self.compute_harmonic_like_measure()

        sorted_cu = sorted(self.cu_ave)
        min_cu, max_cu = sorted_cu[0], sorted_cu[-1]
        unit_step = (max_cu - min_cu) / 40.0 if max_cu > min_cu else 0.1
        histogram = [0] * 40

        for val in self.cu_ave:
            bin_idx = min(int((val - min_cu) / unit_step), 39) if unit_step != 0 else 0
            histogram[bin_idx] += 1

        max_bin = histogram.index(max(histogram))
        c_base = min_cu + unit_step * (max_bin + 0.5)

        index_c_base = next((i for i, val in enumerate(sorted_cu) if val >= c_base), len(sorted_cu) - 1)
        max_idx = len(sorted_cu)
        unit_max = (max_idx - index_c_base) // 5 if (max_idx - index_c_base) > 0 else 1

        li1 = sorted_cu[index_c_base // 5 * 2] if index_c_base // 5 * 2 < max_idx else sorted_cu[-1]
        li2 = sorted_cu[index_c_base // 5 * 4] if index_c_base // 5 * 4 < max_idx else sorted_cu[-1]
        li3 = sorted_cu[index_c_base + unit_max] if index_c_base + unit_max < max_idx else sorted_cu[-1]
        li4 = sorted_cu[index_c_base + unit_max * 3] if index_c_base + unit_max * 3 < max_idx else sorted_cu[-1]

        # 修正：使用与C++一致的系数（AdpIsotropic.cpp lines 332-345）
        for i in range(len(self.cu_ave)):
            if self.cu_ave[i] < li1:
                self.cu_ave[i] = 1.8 * self.L_ave  # C++值: 1.8
            elif self.cu_ave[i] < li2:
                self.cu_ave[i] = 1.4 * self.L_ave  # C++值: 1.4
            elif self.cu_ave[i] < li3:
                self.cu_ave[i] = 1.0 * self.L_ave  # C++值: 1.0
            elif self.cu_ave[i] < li4:
                self.cu_ave[i] = 0.8 * self.L_ave  # C++值: 0.8
            else:
                self.cu_ave[i] = 0.6 * self.L_ave  # C++值: 0.6

    def split_edges(self, mode: str = "subremeshing", max_iter: int = 10) -> Tuple[List[Point], List[Triangle]]:
        if self.use_partitioning:
            return self.split_edges_with_partitioning(mode, max_iter)
        else:
            if mode == "subremeshing":
                return self.split_edges_subremeshing(max_iter)
            elif mode == "histogram":
                return self.split_edges_histogram(max_iter)
            else:
                raise ValueError("模式必须是 'subremeshing' 或 'histogram'")

    def split_edges_with_partitioning(self, mode: str = "subremeshing", max_iter: int = 10) -> Tuple[List[Point], List[Triangle]]:
        """
        Split edges using octree partitioning with global coordination for boundary edges.
        
        Args:
            mode: Either "subremeshing" or "histogram"
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple of (vertices, faces) after splitting
        """
        print(f"\n=== Edge Splitting with Octree Partitioning ===")
        print(f"Mode: {mode}")
        print(f"Input mesh: {len(self.points)} vertices, {len(self.faces)} faces")
        
        # Step 1: Partition the mesh
        print("\n[Step 1] Partitioning mesh...")
        partitioner = MeshPartitioner(self.points, self.faces, self.num_partitions)
        partitions = partitioner.partition_octree()
        print(f"Created {len(partitions)} non-empty partitions")
        print(f"Global border vertices: {len(partitioner.border_vertices)}")
        print(f"Global boundary edges: {len(partitioner.boundary_edges)}")
        
        # Step 1.5: Create global splitting plan for boundary edges
        print("\n[Step 1.5] Creating global splitting plan for boundary edges...")
        self._create_global_boundary_split_plan(partitioner, mode, max_iter)
        print(f"  Planned splits for {len(partitioner.boundary_edge_split_plan)} boundary edges")
        
        # Step 2: Split edges in each partition
        print("\n[Step 2] Splitting edges in partitions...")
        split_submeshes = []
        
        for idx, partition in enumerate(partitions):
            print(f"\nPartition {idx + 1}/{len(partitions)}:")
            print(f"  Core vertices: {len(partition['core_vertices'])}, Total vertices: {len(partition['vertices'])}")
            print(f"  Faces: {len(partition['faces'])}, Border vertices: {len(partition['is_border'])}")
            
            # Extract submesh
            submesh_vertices, submesh_faces, vertex_map, reverse_map = partitioner.extract_submesh(partition)
            
            # Identify border vertices in local indices
            local_border_vertices = {vertex_map[v] for v in partition['is_border'] if v in vertex_map}
            
            # Create a splitter for this submesh
            sub_splitter = EdgeSplitter(use_partitioning=False)
            sub_splitter.initialize(submesh_vertices, submesh_faces)
            
            # Perform splitting on this partition with global coordination
            if mode == "subremeshing":
                split_vertices, split_faces, split_metadata = self._split_partition_subremeshing(
                    sub_splitter, local_border_vertices, max_iter, partitioner, vertex_map, reverse_map
                )
            elif mode == "histogram":
                split_vertices, split_faces, split_metadata = self._split_partition_histogram(
                    sub_splitter, local_border_vertices, max_iter, partitioner, vertex_map, reverse_map
                )
            else:
                raise ValueError("模式必须是 'subremeshing' 或 'histogram'")
            
            print(f"  After splitting: {len(split_vertices)} vertices, {len(split_faces)} faces")
            
            # Store the split submesh with mappings
            split_submeshes.append({
                'vertices': split_vertices,
                'faces': split_faces,
                'vertex_map': vertex_map,
                'reverse_map': reverse_map,
                'core_vertices': partition['core_vertices'],
                'original_submesh_vertices': submesh_vertices,
                'split_metadata': split_metadata  # Metadata for topological merging
            })
        
        # Step 3: Merge split submeshes with topological matching
        print("\n[Step 3] Merging split submeshes...")
        final_vertices, final_faces = self._merge_split_submeshes_topological(split_submeshes, partitioner)
        
        print(f"\n=== Edge Splitting Complete ===")
        print(f"Output mesh: {len(final_vertices)} vertices, {len(final_faces)} faces")
        print(f"Vertex increase: {len(self.points)} -> {len(final_vertices)} "
              f"(+{len(final_vertices) - len(self.points)} vertices)")
        
        return final_vertices, final_faces

    def _create_global_boundary_split_plan(self, partitioner: MeshPartitioner, mode: str, max_iter: int) -> None:
        """
        Create a unified splitting plan for all boundary edges.
        
        Args:
            partitioner: The mesh partitioner with boundary edge information
            mode: Splitting mode ("subremeshing" or "histogram")
            max_iter: Maximum iterations
        """
        # For each boundary edge, determine how it should be split
        for edge, _ in partitioner.boundary_edges.items():
            v1, v2 = edge
            p1 = self.points[v1]
            p2 = self.points[v2]
            edge_length = self._edge_length(p1, p2)
            
            if mode == "subremeshing":
                # Use average edge length criterion
                E_ave = self.L_ave
                if edge_length > 2 * E_ave:
                    insert_num = int(edge_length / E_ave)
                    n = insert_num
                    # Calculate split positions
                    split_positions = []
                    segment = [(p2[i] - p1[i]) / (n + 1) for i in range(3)]
                    for k in range(1, n + 1):
                        pos = [p1[i] + segment[i] * k for i in range(3)]
                        split_positions.append(pos)
                    
                    partitioner.boundary_edge_split_plan[edge] = {
                        'num_splits': n,
                        'positions': split_positions
                    }
            
            elif mode == "histogram":
                # Use curvature-based criterion if available
                if self.cu_ave and v1 < len(self.cu_ave) and v2 < len(self.cu_ave):
                    m_a = self.cu_ave[v1]
                    m_b = self.cu_ave[v2]
                    threshold = 1.25 * min(m_a, m_b)
                    
                    if edge_length >= threshold:
                        # Split at midpoint for histogram mode
                        mid_point = [(p1[i] + p2[i]) / 2 for i in range(3)]
                        partitioner.boundary_edge_split_plan[edge] = {
                            'num_splits': 1,
                            'positions': [mid_point]
                        }

    def _split_partition_subremeshing(self, sub_splitter: 'EdgeSplitter', 
                                     border_vertices: Set[int], 
                                     max_iterations: int,
                                     partitioner: MeshPartitioner,
                                     vertex_map: Dict[int, int],
                                     reverse_map: Dict[int, int]) -> Tuple[List[Point], List[Triangle], Dict]:
        """
        Split edges in a partition using subremeshing mode with global coordination.
        
        Returns:
            Tuple of (vertices, faces, metadata) where metadata tracks split point origins
        """
        points_out = [p[:] for p in sub_splitter.points]
        neighbor_out = [n[:] for n in sub_splitter.point_neighbor]
        faces_out = [f[:] for f in sub_splitter.faces]
        original_number = len(points_out)
        
        # Metadata to track which vertices came from which global edge
        split_metadata = {}  # Maps local vertex index -> (global_edge, split_index)

        for it in range(max_iterations):
            E_ave = sub_splitter._compute_average_edge_length()
            if E_ave <= 0:
                break

            split_list: List[Tuple[int, int, int, List[Point]]] = []
            visited_edges = set()
            
            for i in range(len(neighbor_out)):
                b1 = i
                    
                for j in range(0, len(neighbor_out[i]), 2):
                    if j + 1 >= len(neighbor_out[i]):
                        continue
                    b2 = neighbor_out[i][j]
                    
                    # Skip if edge already visited
                    if b1 >= b2 or (b1, b2) in visited_edges:
                        continue
                        
                    visited_edges.add((b1, b2))
                    
                    # Check if this is a boundary edge (in global coordinates)
                    if b1 < original_number and b2 < original_number:
                        global_v1 = reverse_map.get(b1)
                        global_v2 = reverse_map.get(b2)
                        
                        if global_v1 is not None and global_v2 is not None:
                            global_edge = (min(global_v1, global_v2), max(global_v1, global_v2))
                            
                            # Use global plan if this is a boundary edge
                            if global_edge in partitioner.boundary_edge_split_plan:
                                plan = partitioner.boundary_edge_split_plan[global_edge]
                                n = plan['num_splits']
                                new_points = plan['positions']
                                split_list.append((b1, b2, n, new_points, global_edge))
                                continue
                    
                    # For non-boundary edges, use local criterion
                    edge_length = sub_splitter._edge_length(points_out[b1], points_out[b2])
                    if edge_length > 2 * E_ave:
                        insert_num = int(edge_length / E_ave)
                        n = insert_num
                        p1, p2 = points_out[b1], points_out[b2]
                        segment = [(p2[0] - p1[0]) / (n + 1), (p2[1] - p1[1]) / (n + 1), (p2[2] - p1[2]) / (n + 1)]
                        new_points = [
                            [p1[0] + segment[0] * k, p1[1] + segment[1] * k, p1[2] + segment[2] * k]
                            for k in range(1, n + 1)
                        ]
                        split_list.append((b1, b2, n, new_points, None))

            if not split_list:
                break

            points_end_idx = len(points_out)
            new_faces = []
            processed_triangles = set()
            
            for split_info in split_list:
                if len(split_info) == 5:
                    bs1, bs2, n, new_points, global_edge = split_info
                else:
                    bs1, bs2, n, new_points = split_info
                    global_edge = None
                    
                if not all(sub_splitter._is_valid_index(i, len(points_out)) for i in [bs1, bs2]):
                    continue

                # Add new points and track metadata
                new_point_indices = []
                for k, point in enumerate(new_points):
                    new_idx = len(points_out)
                    points_out.append(point)
                    new_point_indices.append(new_idx)
                    
                    # Track metadata for boundary edge splits
                    if global_edge is not None:
                        split_metadata[new_idx] = (global_edge, k)

                bs3_list = sub_splitter._collect_bs3_list(neighbor_out[bs1], bs1, bs2)
                triangle_vertices = [v for v in bs3_list if
                                     v != bs1 and v != bs2 and sub_splitter._is_valid_index(v, len(points_out))]

                for tv in triangle_vertices:
                    tri = tuple(sorted([bs1, bs2, tv]))
                    processed_triangles.add(tri)
                    
                    sub_splitter._local_reconnection(
                        points_out, neighbor_out, faces_out, new_faces,
                        bs1, bs2, tv, new_point_indices, n, E_ave
                    )

                sub_splitter._update_submesh_neighbors(
                    neighbor_out, bs1, bs2, new_point_indices, n
                )

            # Remove old faces that were subdivided
            faces_to_keep = []
            for face in faces_out:
                tri = tuple(sorted(face))
                if tri not in processed_triangles:
                    faces_to_keep.append(face)
            
            faces_out = faces_to_keep
            faces_out.extend(new_faces)

            for i in range(original_number, len(neighbor_out)):
                neighbor_out[i] = sub_splitter._structure_remove_repeat(neighbor_out[i])

        return points_out, faces_out, split_metadata

    def _split_partition_histogram(self, sub_splitter: 'EdgeSplitter',
                                   border_vertices: Set[int],
                                   max_iterations: int,
                                   partitioner: MeshPartitioner,
                                   vertex_map: Dict[int, int],
                                   reverse_map: Dict[int, int]) -> Tuple[List[Point], List[Triangle], Dict]:
        """
        Split edges in a partition using histogram mode with global coordination.
        
        Returns:
            Tuple of (vertices, faces, metadata) where metadata tracks split point origins
        """
        # Initialize histogram factors for the submesh
        sub_splitter._init_histogram_factors()
        
        points_out = [p[:] for p in sub_splitter.points]
        neighbor_out = [n[:] for n in sub_splitter.point_neighbor]
        faces_out = [f[:] for f in sub_splitter.faces]
        original_number = len(points_out)

        if not sub_splitter.cu_ave or len(sub_splitter.cu_ave) != original_number:
            raise ValueError("cu_ave初始化失败或长度不匹配顶点数量")

        # Metadata to track which vertices came from which global edge
        split_metadata = {}  # Maps local vertex index -> (global_edge, split_index)

        for it in range(max_iterations):
            split_list = []
            point_judge = [True] * len(points_out)
            visited_edges = set()

            for i in range(len(neighbor_out)):
                b1 = i
                    
                if not point_judge[b1]:
                    continue
                if b1 >= original_number:
                    continue
                    
                for j in range(0, len(neighbor_out[i]), 2):
                    if j + 1 >= len(neighbor_out[i]):
                        continue
                    b2 = neighbor_out[i][j]
                        
                    if (b1 >= b2 or not point_judge[b2] or
                            (b1, b2) in visited_edges or
                            not sub_splitter._is_valid_index(b2, original_number)):
                        continue
                    visited_edges.add((b1, b2))
                    
                    # Check if this is a boundary edge (in global coordinates)
                    global_edge = None
                    mid_point = None
                    
                    global_v1 = reverse_map.get(b1)
                    global_v2 = reverse_map.get(b2)
                    
                    if global_v1 is not None and global_v2 is not None:
                        global_edge = (min(global_v1, global_v2), max(global_v1, global_v2))
                        
                        # Use global plan if this is a boundary edge
                        if global_edge in partitioner.boundary_edge_split_plan:
                            plan = partitioner.boundary_edge_split_plan[global_edge]
                            if plan['num_splits'] > 0:
                                mid_point = plan['positions'][0]  # Histogram uses midpoint
                                split_list.append((b1, b2, mid_point, global_edge))
                                for nb in sub_splitter._unique_neighbors(b1):
                                    if sub_splitter._is_valid_index(nb, len(point_judge)):
                                        point_judge[nb] = False
                                for nb in sub_splitter._unique_neighbors(b2):
                                    if sub_splitter._is_valid_index(nb, len(point_judge)):
                                        point_judge[nb] = False
                            continue
                    
                    # For non-boundary edges, use local criterion
                    length12 = sub_splitter._edge_length(points_out[b1], points_out[b2])
                    if b1 >= len(sub_splitter.cu_ave) or b2 >= len(sub_splitter.cu_ave):
                        continue
                    m_a = sub_splitter.cu_ave[b1]
                    m_b = sub_splitter.cu_ave[b2]
                    threshold = 1.25 * min(m_a, m_b)

                    if length12 >= threshold:
                        mid_point = [
                            (points_out[b1][0] + points_out[b2][0]) / 2,
                            (points_out[b1][1] + points_out[b2][1]) / 2,
                            (points_out[b1][2] + points_out[b2][2]) / 2
                        ]
                        split_list.append((b1, b2, mid_point, None))
                        for nb in sub_splitter._unique_neighbors(b1):
                            if sub_splitter._is_valid_index(nb, len(point_judge)):
                                point_judge[nb] = False
                        for nb in sub_splitter._unique_neighbors(b2):
                            if sub_splitter._is_valid_index(nb, len(point_judge)):
                                point_judge[nb] = False

            if not split_list:
                break

            new_faces = []
            for split_info in split_list:
                bs1, bs2, mid_point, global_edge = split_info
                
                if not all(sub_splitter._is_valid_index(i, len(points_out)) for i in [bs1, bs2]):
                    continue

                mid_idx = len(points_out)
                points_out.append(mid_point)
                
                # Track metadata for boundary edge splits
                if global_edge is not None:
                    split_metadata[mid_idx] = (global_edge, 0)

                bs3_list = sub_splitter._collect_bs3_list(neighbor_out[bs1], bs1, bs2)
                triangle_vertices = [v for v in bs3_list if
                                     v != bs1 and v != bs2 and sub_splitter._is_valid_index(v, len(points_out))]

                sub_splitter._update_histogram_neighbors(
                    neighbor_out, bs1, bs2, mid_idx, triangle_vertices
                )

                for tv in triangle_vertices:
                    if [bs1, bs2, tv] in faces_out:
                        faces_out.remove([bs1, bs2, tv])
                    new_faces.append([bs1, mid_idx, tv])
                    new_faces.append([mid_idx, bs2, tv])

            for i in range(original_number, len(neighbor_out)):
                neighbor_out[i] = sub_splitter._structure_remove_repeat(neighbor_out[i])

            faces_out.extend(new_faces)

        return points_out, faces_out, split_metadata

    def _merge_split_submeshes(self, submeshes: List[Dict]) -> Tuple[List[Point], List[Triangle]]:
        """
        Merge split submeshes back into a single mesh.
        
        Args:
            submeshes: List of dictionaries containing split submesh data
            
        Returns:
            Tuple of (merged_vertices, merged_faces)
        """
        merged_vertices = []
        merged_faces = []
        vertex_global_to_merged = {}
        vertex_position_map = {}
        tolerance = 1e-6

        # First pass: collect all faces and determine which vertices are used
        temp_faces = []
        used_vertices = set()

        for submesh_idx, submesh in enumerate(submeshes):
            faces = submesh['faces']
            for face in faces:
                temp_faces.append((submesh_idx, face))
                for v in face:
                    used_vertices.add((submesh_idx, v))

        # Create mapping for submesh vertices
        global_vertex_map = {}

        # Second pass: process only used vertices
        for submesh_idx, submesh in enumerate(submeshes):
            vertices = submesh['vertices']
            reverse_map = submesh['reverse_map']
            original_submesh_vertices = submesh['original_submesh_vertices']

            for local_idx in range(len(vertices)):
                if (submesh_idx, local_idx) not in used_vertices:
                    continue

                vertex = vertices[local_idx]
                
                # Try to find the original vertex this came from
                original_global_idx = None
                if local_idx < len(original_submesh_vertices):
                    # Check if this vertex is close to an original vertex
                    min_dist = float('inf')
                    for orig_local_idx, orig_global_idx in reverse_map.items():
                        if orig_local_idx < len(original_submesh_vertices):
                            orig_vert = original_submesh_vertices[orig_local_idx]
                            dist = math.sqrt(sum((vertex[i] - orig_vert[i]) ** 2 for i in range(3)))
                            if dist < min_dist:
                                min_dist = dist
                                if dist < 1e-5:
                                    original_global_idx = orig_global_idx
                                    break

                # Check if this vertex was already added
                if original_global_idx is not None and original_global_idx in vertex_global_to_merged:
                    merged_idx = vertex_global_to_merged[original_global_idx]
                else:
                    # Check for duplicate by position
                    vertex_key = tuple(int(round(vertex[i] / tolerance)) for i in range(3))

                    if vertex_key in vertex_position_map:
                        merged_idx = vertex_position_map[vertex_key]
                    else:
                        # New vertex, add it
                        merged_idx = len(merged_vertices)
                        merged_vertices.append(vertex)
                        vertex_position_map[vertex_key] = merged_idx

                        if original_global_idx is not None:
                            vertex_global_to_merged[original_global_idx] = merged_idx

                global_vertex_map[(submesh_idx, local_idx)] = merged_idx

        # Third pass: process faces
        for submesh_idx, face in temp_faces:
            merged_face = [global_vertex_map[(submesh_idx, v)] for v in face]
            # Check for degenerate faces
            if len(set(merged_face)) == 3:
                merged_faces.append(merged_face)

        # Remove duplicate faces
        unique_faces = []
        seen_faces = set()
        for face in merged_faces:
            face_tuple = tuple(sorted(face))
            if face_tuple not in seen_faces:
                seen_faces.add(face_tuple)
                unique_faces.append(face)

        print(f"Merged {len(submeshes)} submeshes:")
        print(f"  Unique vertices: {len(merged_vertices)}")
        print(f"  Unique faces: {len(unique_faces)}")

        return merged_vertices, unique_faces

    def _merge_split_submeshes_topological(self, submeshes: List[Dict], partitioner: MeshPartitioner) -> Tuple[List[Point], List[Triangle]]:
        """
        Merge split submeshes using topological matching for boundary vertices.
        
        Args:
            submeshes: List of dictionaries containing split submesh data with metadata
            partitioner: The mesh partitioner with boundary edge information
            
        Returns:
            Tuple of (merged_vertices, merged_faces)
        """
        merged_vertices = []
        merged_faces = []
        global_vertex_map = {}  # Maps (submesh_idx, local_idx) -> merged_idx
        vertex_global_to_merged = {}  # Maps original global vertex idx to merged idx
        
        # Maps (global_edge, split_index) -> merged_idx for boundary split points
        boundary_split_map = {}
        
        tolerance = 1e-6

        # First pass: collect all faces and determine which vertices are used
        temp_faces = []
        used_vertices = set()

        for submesh_idx, submesh in enumerate(submeshes):
            faces = submesh['faces']
            for face in faces:
                temp_faces.append((submesh_idx, face))
                for v in face:
                    used_vertices.add((submesh_idx, v))

        # Second pass: process only used vertices
        for submesh_idx, submesh in enumerate(submeshes):
            vertices = submesh['vertices']
            reverse_map = submesh['reverse_map']
            original_submesh_vertices = submesh['original_submesh_vertices']
            split_metadata = submesh.get('split_metadata', {})

            for local_idx in range(len(vertices)):
                if (submesh_idx, local_idx) not in used_vertices:
                    continue

                vertex = vertices[local_idx]
                
                # Check if this is a boundary split point with metadata
                if local_idx in split_metadata:
                    global_edge, split_index = split_metadata[local_idx]
                    boundary_key = (global_edge, split_index)
                    
                    if boundary_key in boundary_split_map:
                        # This boundary split point already exists
                        merged_idx = boundary_split_map[boundary_key]
                    else:
                        # New boundary split point
                        merged_idx = len(merged_vertices)
                        merged_vertices.append(vertex)
                        boundary_split_map[boundary_key] = merged_idx
                    
                    global_vertex_map[(submesh_idx, local_idx)] = merged_idx
                    continue
                
                # Check if this is an original vertex
                original_global_idx = None
                if local_idx < len(original_submesh_vertices):
                    # Try to match to original vertex by position
                    min_dist = float('inf')
                    for orig_local_idx, orig_global_idx in reverse_map.items():
                        if orig_local_idx < len(original_submesh_vertices):
                            orig_vert = original_submesh_vertices[orig_local_idx]
                            dist = math.sqrt(sum((vertex[i] - orig_vert[i]) ** 2 for i in range(3)))
                            if dist < min_dist:
                                min_dist = dist
                                if dist < 1e-5:
                                    original_global_idx = orig_global_idx
                                    break

                # Check if this vertex was already added from another partition
                if original_global_idx is not None and original_global_idx in vertex_global_to_merged:
                    merged_idx = vertex_global_to_merged[original_global_idx]
                else:
                    # New vertex (either original or non-boundary split)
                    merged_idx = len(merged_vertices)
                    merged_vertices.append(vertex)

                    if original_global_idx is not None:
                        vertex_global_to_merged[original_global_idx] = merged_idx

                global_vertex_map[(submesh_idx, local_idx)] = merged_idx

        # Third pass: process faces
        for submesh_idx, face in temp_faces:
            merged_face = [global_vertex_map[(submesh_idx, v)] for v in face]
            # Check for degenerate faces
            if len(set(merged_face)) == 3:
                merged_faces.append(merged_face)

        # Remove duplicate faces
        unique_faces = []
        seen_faces = set()
        for face in merged_faces:
            face_tuple = tuple(sorted(face))
            if face_tuple not in seen_faces:
                seen_faces.add(face_tuple)
                unique_faces.append(face)

        print(f"Merged {len(submeshes)} submeshes with topological matching:")
        print(f"  Boundary split points matched: {len(boundary_split_map)}")
        print(f"  Unique vertices: {len(merged_vertices)}")
        print(f"  Unique faces: {len(unique_faces)}")

        return merged_vertices, unique_faces


    def split_edges_subremeshing(self, max_iterations: int = 10) -> Tuple[List[Point], List[Triangle]]:
        points_out = [p[:] for p in self.points]
        neighbor_out = [n[:] for n in self.point_neighbor]
        faces_out = [f[:] for f in self.faces]
        original_number = len(points_out)

        for it in range(max_iterations):
            E_ave = self._compute_average_edge_length()
            if E_ave <= 0:
                break

            split_list: List[Tuple[int, int, int]] = []
            visited_edges = set()
            for i in range(len(neighbor_out)):
                b1 = i
                for j in range(0, len(neighbor_out[i]), 2):
                    if j + 1 >= len(neighbor_out[i]):
                        continue
                    b2 = neighbor_out[i][j]
                    if b1 >= b2 or (b1, b2) in visited_edges:
                        continue
                    visited_edges.add((b1, b2))
                    edge_length = self._edge_length(points_out[b1], points_out[b2])
                    if edge_length > 2 * E_ave:
                        # 修正1：正确计算插入点数量 - 与C++一致
                        insert_num = int(edge_length / E_ave)  # 段数
                        n = insert_num  # 插入点数 = 段数（与C++一致）
                        split_list.append((b1, b2, n))

            if not split_list:
                break

            # Track which edges have been split
            split_edges = set()
            for (bs1, bs2, n) in split_list:
                split_edges.add((min(bs1, bs2), max(bs1, bs2)))

            points_end_idx = len(points_out)
            new_faces = []
            processed_triangles = set()  # Track which triangles have been subdivided
            
            for (bs1, bs2, n) in split_list:
                if not all(self._is_valid_index(i, len(points_out)) for i in [bs1, bs2]):
                    continue

                p1, p2 = points_out[bs1], points_out[bs2]
                segment = [(p2[0] - p1[0]) / (n + 1), (p2[1] - p1[1]) / (n + 1), (p2[2] - p1[2]) / (n + 1)]
                new_points = [
                    [p1[0] + segment[0] * k, p1[1] + segment[1] * k, p1[2] + segment[2] * k]
                    for k in range(1, n + 1)
                ]
                points_out.extend(new_points)
                new_point_indices = list(range(points_end_idx, points_end_idx + n))
                points_end_idx += n

                bs3_list = self._collect_bs3_list(neighbor_out[bs1], bs1, bs2)
                triangle_vertices = [v for v in bs3_list if
                                     v != bs1 and v != bs2 and self._is_valid_index(v, len(points_out))]

                # Mark triangles containing this split edge as processed
                for tv in triangle_vertices:
                    # Create a canonical representation of the triangle
                    tri = tuple(sorted([bs1, bs2, tv]))
                    processed_triangles.add(tri)
                    
                    self._local_reconnection(
                        points_out, neighbor_out, faces_out, new_faces,
                        bs1, bs2, tv, new_point_indices, n, E_ave
                    )

                self._update_submesh_neighbors(
                    neighbor_out, bs1, bs2, new_point_indices, n
                )

            # Remove old faces that were subdivided
            faces_to_keep = []
            for face in faces_out:
                tri = tuple(sorted(face))
                if tri not in processed_triangles:
                    faces_to_keep.append(face)
            
            faces_out = faces_to_keep
            faces_out.extend(new_faces)

            for i in range(original_number, len(neighbor_out)):
                neighbor_out[i] = self._structure_remove_repeat(neighbor_out[i])

        return points_out, faces_out

    def _compute_angle_cos(self, p1_idx: int, p2_idx: int, p3_idx: int, points: List[Point]) -> float:
        """计算角p1-p2-p3的余弦值"""
        p1, p2, p3 = points[p1_idx], points[p2_idx], points[p3_idx]

        d12 = self._edge_length(p1, p2)
        d23 = self._edge_length(p2, p3)
        d13 = self._edge_length(p1, p3)

        if d12 == 0 or d23 == 0:
            return 1.0

        # 余弦定理: cos(angle) = (d12^2 + d23^2 - d13^2) / (2 * d12 * d23)
        cos_val = (d12 * d12 + d23 * d23 - d13 * d13) / (2 * d12 * d23)
        return max(min(cos_val, 1.0), -1.0)

    def _create_point_pairs(self, b1_num: int, b2_num: int, b3_num: int,
                            bs1: int, bs2: int, tv: int, points: List[Point]) -> List[Tuple[int, int]]:
        """创建点配对列表，用于四边形分割"""
        pair_point = []
        b2b3_num = b2_num + b3_num + 1

        # 计算中间角的余弦值
        cos_middle = self._compute_angle_cos(bs1, tv, bs2, points)
        ang_middle = math.acos(cos_middle)

        # 修正2：添加钝角判断
        if ang_middle > 1.5:  # 钝角三角形
            if b2b3_num > b1_num:
                for i in range(1, b2b3_num + 1):
                    b1_num_index = int(b1_num * i / b2b3_num)
                    if b1_num_index >= b1_num:
                        b1_num_index = b1_num - 1
                    pair_point.append((b1_num_index, i - 1))
            else:
                for i in range(1, b1_num + 1):
                    b23_num_index = int(b2b3_num * i / b1_num)
                    if b23_num_index >= b2b3_num:
                        b23_num_index = b2b3_num - 1
                    pair_point.append((i - 1, b23_num_index))
        else:  # 非钝角
            for i in range(1, b1_num + 1):
                b2_num_index = int(b2_num * i / b1_num)
                if b2_num_index >= b2_num:
                    b2_num_index = b2_num - 1
                pair_point.append((i - 1, b2_num_index))

        # 去重
        if len(pair_point) >= 2:
            pair_point_unique = [pair_point[0]]
            for i in range(1, len(pair_point)):
                if pair_point[i] != pair_point_unique[-1]:
                    pair_point_unique.append(pair_point[i])
            pair_point = pair_point_unique

        return pair_point

    def _local_reconnection(self, points: List[Point], neighbors: List[Neighbor],
                            faces: List[Triangle], new_faces: List[Triangle],
                            bs1: int, bs2: int, tv: int, new_points: List[int],
                            n: int, E_ave: float) -> None:
        """
        Subdivide triangle (bs1, bs2, tv) where edge bs1-bs2 has been split with new points.
        Creates new triangular faces connecting the split edge points to tv.
        """
        if n == 0 or len(new_points) == 0:
            return
        
        # Create triangles connecting consecutive points on the split edge to tv
        # Triangle: bs1 - new_points[0] - tv
        new_faces.append([bs1, new_points[0], tv])
        
        # Intermediate triangles: new_points[i-1] - new_points[i] - tv
        for i in range(1, len(new_points)):
            new_faces.append([new_points[i-1], new_points[i], tv])
        
        # Final triangle: new_points[-1] - bs2 - tv
        new_faces.append([new_points[-1], bs2, tv])

    def _insert_points_on_edge(self, points: List[Point], p1: int, p2: int, n: int) -> List[Point]:
        if not all(self._is_valid_index(i, len(points)) for i in [p1, p2]) or n <= 0:
            return []
        start, end = points[p1], points[p2]
        segment = [(end[0] - start[0]) / (n + 1), (end[1] - start[1]) / (n + 1), (end[2] - start[2]) / (n + 1)]
        return [
            [start[0] + segment[0] * k, start[1] + segment[1] * k, start[2] + segment[2] * k]
            for k in range(1, n + 1)
        ]

    def _split_quadrilateral(self, neighbors: List[Neighbor], new_faces: List[Triangle],
                             v1: int, v2: int, v3: int, v4: int) -> None:
        valid_indices = [v1, v2, v3, v4]
        if not all(self._is_valid_index(i, len(self.points)) for i in valid_indices):
            return

        angle1 = self._max_angle_in_triangles(v1, v2, v3, v1, v3, v4)
        angle2 = self._max_angle_in_triangles(v2, v3, v4, v1, v2, v4)

        if angle1 <= angle2:
            new_faces.append([v1, v2, v3])
            new_faces.append([v1, v3, v4])
            self._add_neighbor_triple(neighbors, v1, v2, v3)
            self._add_neighbor_triple(neighbors, v1, v3, v4)
        else:
            new_faces.append([v2, v3, v4])
            new_faces.append([v1, v2, v4])
            self._add_neighbor_triple(neighbors, v2, v3, v4)
            self._add_neighbor_triple(neighbors, v1, v2, v4)

    def _max_angle_in_triangles(self, a1: int, a2: int, a3: int, b1: int, b2: int, b3: int) -> float:
        angles = []
        triangles = [(a1, a2, a3), (b1, b2, b3)]

        for tri in triangles:
            if not all(self._is_valid_index(i, len(self.points)) for i in tri):
                continue
            try:
                p1, p2, p3 = [self.points[i] for i in tri]
            except IndexError:
                continue

            v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
            v2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]]
            v3 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
            v4 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]]

            angles.append(self._angle_between(self._unit(v1), self._unit(v2)))
            angles.append(self._angle_between(self._unit(v3), self._unit(v4)))

        return max(angles) if angles else 0.0

    def _add_neighbor_triple(self, neighbors: List[Neighbor], p1: int, p2: int, p3: int) -> None:
        self._add_neighbor_pair_inplace(neighbors, p1, p2, p3)
        self._add_neighbor_pair_inplace(neighbors, p2, p3, p1)
        self._add_neighbor_pair_inplace(neighbors, p3, p1, p2)

    def _add_neighbor_pair_inplace(self, neighbors: List[Neighbor], p1: int, p2: int, p3: int) -> None:
        while p1 >= len(neighbors):
            neighbors.append([])
        neighbors[p1].append(p2)
        neighbors[p1].append(p3)

    def _update_submesh_neighbors(self, neighbors: List[Neighbor], bs1: int, bs2: int,
                                  new_points: List[int], n: int) -> None:
        for k in range(n - 1):
            if k + 1 >= len(new_points):
                break
            self._add_neighbor_pair_inplace(neighbors, new_points[k], new_points[k + 1], bs1)

        if new_points:
            self._add_neighbor_pair_inplace(neighbors, bs1, new_points[0], new_points[1] if n > 1 else bs2)
            self._add_neighbor_pair_inplace(neighbors, bs2, new_points[-1], new_points[-2] if n > 1 else bs1)

    def _structure_remove_repeat(self, lst: List[int]) -> List[int]:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _collect_bs3_list(self, neighbors: Neighbor, bs1: int, bs2: int) -> List[int]:
        bs3_list = []
        for i in range(0, len(neighbors), 2):
            if i + 1 >= len(neighbors):
                continue
            a, b = neighbors[i], neighbors[i + 1]
            if (a == bs2 or b == bs2) and a != bs1 and b != bs1:
                bs3 = a if b == bs2 else b
                bs3_list.append(bs3)
        return bs3_list

    def _is_valid_index(self, idx: int, length: int = None) -> bool:
        """检查索引是否有效"""
        if length is None:
            length = len(self.points)
        return 0 <= idx < length

    def split_edges_histogram(self, max_iterations: int = 10) -> Tuple[List[Point], List[Triangle]]:
        self._init_histogram_factors()
        points_out = [p[:] for p in self.points]
        neighbor_out = [n[:] for n in self.point_neighbor]
        faces_out = [f[:] for f in self.faces]
        original_number = len(points_out)

        if not self.cu_ave or len(self.cu_ave) != original_number:
            raise ValueError("cu_ave初始化失败或长度不匹配顶点数量")

        for it in range(max_iterations):
            split_list = []
            point_judge = [True] * len(points_out)
            visited_edges = set()

            for i in range(len(neighbor_out)):
                b1 = i
                if not point_judge[b1]:
                    continue
                if b1 >= original_number:
                    continue
                for j in range(0, len(neighbor_out[i]), 2):
                    if j + 1 >= len(neighbor_out[i]):
                        continue
                    b2 = neighbor_out[i][j]
                    if (b1 >= b2 or not point_judge[b2] or
                            (b1, b2) in visited_edges or
                            not self._is_valid_index(b2, original_number)):
                        continue
                    visited_edges.add((b1, b2))
                    length12 = self._edge_length(points_out[b1], points_out[b2])
                    if b1 >= len(self.cu_ave) or b2 >= len(self.cu_ave):
                        continue
                    m_a = self.cu_ave[b1]
                    m_b = self.cu_ave[b2]
                    # 修正：使用正确的阈值 5/4 = 1.25
                    threshold = 1.25 * min(m_a, m_b)

                    if length12 >= threshold:
                        split_list.append((b1, b2))
                        for nb in self._unique_neighbors(b1):
                            if self._is_valid_index(nb, len(point_judge)):
                                point_judge[nb] = False
                        for nb in self._unique_neighbors(b2):
                            if self._is_valid_index(nb, len(point_judge)):
                                point_judge[nb] = False

            if not split_list:
                break

            new_faces = []
            for (bs1, bs2) in split_list:
                if not all(self._is_valid_index(i, len(points_out)) for i in [bs1, bs2]):
                    continue

                mid_point = [
                    (points_out[bs1][0] + points_out[bs2][0]) / 2,
                    (points_out[bs1][1] + points_out[bs2][1]) / 2,
                    (points_out[bs1][2] + points_out[bs2][2]) / 2
                ]
                mid_idx = len(points_out)
                points_out.append(mid_point)

                bs3_list = self._collect_bs3_list(neighbor_out[bs1], bs1, bs2)
                triangle_vertices = [v for v in bs3_list if
                                     v != bs1 and v != bs2 and self._is_valid_index(v, len(points_out))]

                self._update_histogram_neighbors(
                    neighbor_out, bs1, bs2, mid_idx, triangle_vertices
                )

                for tv in triangle_vertices:
                    if [bs1, bs2, tv] in faces_out:
                        faces_out.remove([bs1, bs2, tv])
                    new_faces.append([bs1, mid_idx, tv])
                    new_faces.append([mid_idx, bs2, tv])

            for i in range(original_number, len(neighbor_out)):
                neighbor_out[i] = self._structure_remove_repeat(neighbor_out[i])

            faces_out.extend(new_faces)

        return points_out, faces_out

    def _update_histogram_neighbors(self, neighbors: List[Neighbor], bs1: int, bs2: int,
                                    mid_idx: int, triangle_vertices: List[int]) -> None:
        mid_neighbors = []

        for b in [bs1, bs2]:
            if not self._is_valid_index(b, len(neighbors)):
                continue
            for i in range(len(neighbors[b]) // 2):
                idx1 = 2 * i
                idx2 = 2 * i + 1
                if idx2 >= len(neighbors[b]):
                    continue
                if neighbors[b][idx1] == bs2 or neighbors[b][idx2] == bs2:
                    if neighbors[b][idx1] == bs2:
                        neighbors[b][idx1] = mid_idx
                    else:
                        neighbors[b][idx2] = mid_idx
                    mid_neighbors.extend([neighbors[b][idx2], b])

        for tv in triangle_vertices:
            if not self._is_valid_index(tv, len(neighbors)):
                continue
            for i in range(len(neighbors[tv]) // 2):
                idx1 = 2 * i
                idx2 = 2 * i + 1
                if idx2 >= len(neighbors[tv]):
                    continue
                if (neighbors[tv][idx1] == bs1 and neighbors[tv][idx2] == bs2) or \
                        (neighbors[tv][idx1] == bs2 and neighbors[tv][idx2] == bs1):
                    neighbors[tv][idx1] = mid_idx
                    neighbors[tv].append(bs1)
                    neighbors[tv].append(mid_idx)
                    mid_neighbors.extend([bs2, tv])
                    mid_neighbors.extend([tv, bs1])

        while mid_idx >= len(neighbors):
            neighbors.append([])
        neighbors[mid_idx] = mid_neighbors


# 使用示例
if __name__ == "__main__":
    reader = PLYReader()
    
    # Test file path - use a simple mesh for testing
    test_file = "demo/output/simplified_00011000_8a21002f126e4425a811e70a_trimesh_004.ply"
    
    # If test file doesn't exist, create a simple test mesh
    if not os.path.exists(test_file):
        print("Test file not found, creating simple test mesh...")
        os.makedirs("demo/output", exist_ok=True)
        
        # Create a simple subdivided cube
        vertices = []
        faces = []
        
        # Create a 3x3x3 grid of vertices
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    vertices.append([float(i), float(j), float(k)])
        
        # Create faces - this is a simple cube subdivision
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
        
        writer = PLYWriter()
        writer.write_ply(test_file, vertices, faces)
        print(f"Created test mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    vertices, faces = reader.read_ply(test_file)
    print(f"原始模型: 顶点数={len(vertices)}, 面数={len(faces)}")

    # Test 1: Original behavior (without partitioning)
    print("\n" + "="*60)
    print("Test 1: Original Behavior (No Partitioning)")
    print("="*60)
    
    # Subremeshing mode without partitioning
    print("\n=== Subremeshing模式 (无分区) ===")
    splitter = EdgeSplitter(use_partitioning=False)
    splitter.initialize(vertices, faces)
    new_vertices, new_faces = splitter.split_edges(mode="subremeshing", max_iter=1)
    print(f"处理后: 顶点数={len(new_vertices)}, 面数={len(new_faces)}")
    writer = PLYWriter()
    writer.write_ply("demo/output/output_subremeshing_no_partition.ply", new_vertices, new_faces)

    # Histogram mode without partitioning
    print("\n=== Histogram模式 (无分区) ===")
    splitter2 = EdgeSplitter(use_partitioning=False)
    splitter2.initialize(vertices, faces)
    new_vertices2, new_faces2 = splitter2.split_edges(mode="histogram", max_iter=3)
    print(f"处理后: 顶点数={len(new_vertices2)}, 面数={len(new_faces2)}")
    writer.write_ply("demo/output/output_histogram_no_partition.ply", new_vertices2, new_faces2)

    # Test 2: New behavior (with octree partitioning)
    print("\n" + "="*60)
    print("Test 2: New Behavior (With Octree Partitioning)")
    print("="*60)
    
    # Subremeshing mode with partitioning
    print("\n=== Subremeshing模式 (八叉树分区) ===")
    splitter3 = EdgeSplitter(use_partitioning=True, num_partitions=8)
    splitter3.initialize(vertices, faces)
    new_vertices3, new_faces3 = splitter3.split_edges(mode="subremeshing", max_iter=1)
    print(f"处理后: 顶点数={len(new_vertices3)}, 面数={len(new_faces3)}")
    writer.write_ply("demo/output/output_subremeshing_with_partition.ply", new_vertices3, new_faces3)

    # Histogram mode with partitioning
    print("\n=== Histogram模式 (八叉树分区) ===")
    splitter4 = EdgeSplitter(use_partitioning=True, num_partitions=8)
    splitter4.initialize(vertices, faces)
    new_vertices4, new_faces4 = splitter4.split_edges(mode="histogram", max_iter=3)
    print(f"处理后: 顶点数={len(new_vertices4)}, 面数={len(new_faces4)}")
    writer.write_ply("demo/output/output_histogram_with_partition.ply", new_vertices4, new_faces4)
    
    print("\n" + "="*60)
    print("测试完成！输出文件已保存到 demo/output/")
    print("="*60)