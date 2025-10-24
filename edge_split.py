import math
import struct
from typing import List, Tuple, Any

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


class EdgeSplitter:
    def __init__(self):
        self.points = []
        self.faces = []
        self.point_neighbor = []
        self.cu_ave = None
        self.L_ave = 0.0
        self.original_vertex_count = 0

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
        if mode == "subremeshing":
            return self.split_edges_subremeshing(max_iter)
        elif mode == "histogram":
            return self.split_edges_histogram(max_iter)
        else:
            raise ValueError("模式必须是 'subremeshing' 或 'histogram'")

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

            points_end_idx = len(points_out)
            new_faces = []
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

                for tv in triangle_vertices:
                    self._local_reconnection(
                        points_out, neighbor_out, faces_out, new_faces,
                        bs1, bs2, tv, new_point_indices, n, E_ave
                    )

                self._update_submesh_neighbors(
                    neighbor_out, bs1, bs2, new_point_indices, n
                )

            for i in range(original_number, len(neighbor_out)):
                neighbor_out[i] = self._structure_remove_repeat(neighbor_out[i])

            faces_out.extend(new_faces)

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
        len12 = self._edge_length(points[bs1], points[bs2])
        len13 = self._edge_length(points[bs1], points[tv])
        len23 = self._edge_length(points[bs2], points[tv])

        # 修正：完整实现三种情况
        if len13 <= 2 * E_ave and len23 <= 2 * E_ave:
            # 情况1：三条边都相对较短，在所有边上插入点
            p13_points = self._insert_points_on_edge(points, bs1, tv, n)
            p23_points = self._insert_points_on_edge(points, bs2, tv, n)
            p13_indices = list(range(len(points), len(points) + n))
            p23_indices = list(range(p13_indices[-1] + 1, p13_indices[-1] + 1 + n)) if n > 0 else []
            points.extend(p13_points + p23_points)

            # 创建均匀的四边形网格
            for k in range(n):
                if k >= len(new_points) or k >= len(p13_indices) or k >= len(p23_indices):
                    continue
                v1 = new_points[k]
                v2 = p13_indices[k]
                v3 = tv
                v4 = p23_indices[k]
                if all(self._is_valid_index(i, len(points)) for i in [v1, v2, v3, v4]):
                    self._split_quadrilateral(neighbors, new_faces, v1, v2, v3, v4)
        else:
            # 情况2和3：根据边长和角度决定分割策略
            edges = [(len12, bs1, bs2), (len13, bs1, tv), (len23, bs2, tv)]
            edges.sort(reverse=True, key=lambda x: x[0])
            (_, l1, l2), (_, s1, s2) = edges[:2]

            long_points = self._insert_points_on_edge(points, l1, l2, n)
            second_points = self._insert_points_on_edge(points, s1, s2, n)
            long_indices = list(range(len(points), len(points) + n))
            second_indices = list(range(long_indices[-1] + 1, long_indices[-1] + 1 + n)) if n > 0 else []
            points.extend(long_points + second_points)

            # 使用点配对算法
            for k in range(n):
                if k >= len(long_indices) or k >= len(second_indices):
                    continue
                v1 = long_indices[k]
                v2 = second_indices[k]
                if all(self._is_valid_index(i, len(points)) for i in [v1, v2, l1, s1]):
                    self._split_quadrilateral(neighbors, new_faces, v1, v2, l1, s1)

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
    vertices, faces = reader.read_ply("demo/output/simplified_00011000_8a21002f126e4425a811e70a_trimesh_004.ply")
    print(f"原始模型: 顶点数={len(vertices)}, 面数={len(faces)}")

    splitter = EdgeSplitter()
    splitter.initialize(vertices, faces)

    # 测试subremeshing模式
    print("\n=== Subremeshing模式 ===")
    new_vertices, new_faces = splitter.split_edges(mode="subremeshing", max_iter=1)
    print(f"处理后: 顶点数={len(new_vertices)}, 面数={len(new_faces)}")
    writer = PLYWriter()
    writer.write_ply("demo/output/output_subremeshing.ply", new_vertices, new_faces)

    # 测试histogram模式
    print("\n=== Histogram模式 ===")
    splitter2 = EdgeSplitter()
    splitter2.initialize(vertices, faces)
    new_vertices2, new_faces2 = splitter2.split_edges(mode="histogram", max_iter=10)
    print(f"处理后: 顶点数={len(new_vertices2)}, 面数={len(new_faces2)}")
    writer.write_ply("demo/output/output_histogram.ply", new_vertices2, new_faces2)