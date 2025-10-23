import numpy as np
import heapq
import os
import struct


class PLYReader:
    @staticmethod
    def read_ply(file_path):
        """读取PLY文件，返回顶点和面数据"""
        vertices = []
        faces = []

        # 检测文件格式并相应处理
        with open(file_path, 'rb') as f:  # 使用二进制模式读取
            # 读取文件头
            header_lines = []
            line = b''
            while b'end_header' not in line:
                line = f.readline()
                header_lines.append(line.decode('ascii', errors='ignore').strip())

            # 解析文件头
            vertex_count = 0
            face_count = 0
            properties = []

            for line in header_lines:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('element face'):
                    face_count = int(line.split()[-1])
                elif line.startswith('property'):
                    properties.append(line)

            # 检查文件格式
            format_line = [line for line in header_lines if line.startswith('format')][0]
            is_ascii = 'ascii' in format_line.lower()

            # 读取顶点数据
            if is_ascii:
                # ASCII格式
                for i in range(vertex_count):
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    vertex_data = list(map(float, line.split()[:3]))  # 只取前3个值(x,y,z)
                    vertices.append(vertex_data)
            else:
                # 二进制格式 - 简化处理，假设是float32和int32
                for i in range(vertex_count):
                    # 读取3个float32值 (x, y, z)
                    data = f.read(12)  # 3 * 4 bytes
                    x, y, z = struct.unpack('fff', data)
                    vertices.append([x, y, z])
                    # 跳过其他属性
                    # 这里需要根据实际属性调整

            # 读取面数据
            if is_ascii:
                # ASCII格式
                for i in range(face_count):
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    face_data = list(map(int, line.split()))
                    if face_data[0] == 3:  # 三角形面
                        faces.append(face_data[1:4])
                    elif face_data[0] > 3:  # 多边形面，分解为三角形
                        # 简单的三角化：从第一个顶点连接到其他顶点
                        for j in range(1, face_data[0] - 1):
                            faces.append([face_data[1], face_data[1 + j], face_data[2 + j]])
            else:
                # 二进制格式
                for i in range(face_count):
                    # 读取顶点数
                    count_data = f.read(1)  # 假设是uchar
                    vertex_count_in_face = struct.unpack('B', count_data)[0]

                    # 读取顶点索引
                    if vertex_count_in_face == 3:
                        data = f.read(12)  # 3 * 4 bytes
                        v1, v2, v3 = struct.unpack('iii', data)
                        faces.append([v1, v2, v3])
                    elif vertex_count_in_face > 3:
                        # 读取所有顶点索引
                        data = f.read(4 * vertex_count_in_face)
                        indices = struct.unpack('i' * vertex_count_in_face, data)
                        # 三角化
                        for j in range(1, vertex_count_in_face - 1):
                            faces.append([indices[0], indices[j], indices[j + 1]])

        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


class PLYWriter:
    @staticmethod
    def write_ply(file_path, vertices, faces):
        """将顶点和面数据写入PLY文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入PLY文件头
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_index\n")
            f.write("end_header\n")

            # 写入顶点数据
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

            # 写入面数据
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


class QEMSimplifier:
    def __init__(self, vertices, faces):
        self.vertices = vertices.copy()
        self.faces = faces.copy()
        self.vertex_count = len(vertices)
        self.face_count = len(faces)
        self.quadrics = {}
        self.vertex_faces = {}  # 顶点到面的映射
        self.valid_vertices = set(range(len(vertices)))
        self.compute_vertex_faces()
        self.compute_quadrics()

    def compute_vertex_faces(self):
        """计算每个顶点所属的面"""
        self.vertex_faces = {i: set() for i in range(len(self.vertices))}
        for idx, face in enumerate(self.faces):
            for vertex_id in face:
                self.vertex_faces[vertex_id].add(idx)

    def compute_quadrics(self):
        """为每个顶点计算二次误差矩阵"""
        # 初始化Q矩阵
        for i in range(len(self.vertices)):
            self.quadrics[i] = np.zeros((4, 4))

        # 为每个面计算平面方程并累加到对应顶点的Q矩阵
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]

            # 计算面的法向量
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)

            # 归一化法向量
            length = np.linalg.norm(normal)
            if length > 1e-10:
                normal /= length
            else:
                # 如果面退化，使用默认法向量
                normal = np.array([0, 0, 1])

            # 计算平面方程 ax + by + cz + d = 0
            a, b, c = normal
            d = -np.dot(normal, v0)

            # 平面方程的4D向量表示
            plane_eq = np.array([a, b, c, d])

            # 计算Kp矩阵
            Kp = np.outer(plane_eq, plane_eq)

            # 将Kp累加到面的每个顶点
            for vertex_id in face:
                self.quadrics[vertex_id] += Kp

    def compute_optimal_position(self, v1, v2):
        """计算两个顶点合并后的最优位置"""
        Q = self.quadrics[v1] + self.quadrics[v2]

        # 构造线性系统 Ax = b
        A = Q[:3, :3]
        b = -Q[:3, 3]

        # 检查矩阵是否可逆
        try:
            # 尝试直接求解
            optimal_pos = np.linalg.solve(A, b)

            # 验证解是否在边的范围内
            if self.is_valid_position(optimal_pos, v1, v2):
                return optimal_pos
        except np.linalg.LinAlgError:
            pass

        # 如果直接求解失败或位置无效，尝试边上的中点
        mid_point = (self.vertices[v1] + self.vertices[v2]) / 2
        if self.is_valid_position(mid_point, v1, v2):

            return mid_point

        # 最后返回v1或v2的位置
        return self.vertices[v1].copy()

    def is_valid_position(self, pos, v1, v2):
        """检查位置是否有效（在网格边界内）"""
        # 检查：位置不是NaN或无穷大
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            return False
        
        # CRITICAL: Check that position is within reasonable bounds
        # Position should not deviate too far from the edge endpoints
        # This prevents extreme positions from QEM solver numerical issues
        
        # Get the two endpoint positions
        p1 = self.vertices[v1]
        p2 = self.vertices[v2]
        
        # Compute bounding box of the edge with some tolerance
        edge_min = np.minimum(p1, p2)
        edge_max = np.maximum(p1, p2)
        edge_size = edge_max - edge_min
        
        # Allow position to be outside edge bbox by at most 2x the edge length in each dimension
        # This handles cases where optimal position is slightly outside the edge
        tolerance = 2.0
        expanded_min = edge_min - tolerance * (edge_size + 1e-6)  # Add small epsilon to handle zero-length edges
        expanded_max = edge_max + tolerance * (edge_size + 1e-6)
        
        # Check if position is within expanded bounds
        if np.any(pos < expanded_min) or np.any(pos > expanded_max):
            return False
        
        return True

    def compute_cost(self, v1, v2, optimal_pos):
        """计算边收缩的代价"""
        Q = self.quadrics[v1] + self.quadrics[v2]
        homo_pos = np.append(optimal_pos, 1)
        cost = homo_pos @ Q @ homo_pos.T
        return cost

    def find_valid_edges(self):
        """找到所有合法的边（共享边的顶点对）"""
        edges = set()

        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                if v1 != v2:  # 确保不是自环
                    edge = (min(v1, v2), max(v1, v2))
                    edges.add(edge)

        return list(edges)

    def simplify(self, target_ratio=0.5):
        """执行网格简化"""
        target_vertex_count = max(4, int(len(self.vertices) * target_ratio))

        print(f"开始简化: {len(self.vertices)} 顶点 -> {target_vertex_count} 顶点")

        # 找到所有合法的边
        edges = self.find_valid_edges()

        # 创建优先队列
        heap = []
        edge_info = {}  # 存储边的信息

        # 为每条边计算收缩代价
        for edge in edges:
            v1, v2 = edge
            if v1 not in self.valid_vertices or v2 not in self.valid_vertices:
                continue

            optimal_pos = self.compute_optimal_position(v1, v2)
            cost = self.compute_cost(v1, v2, optimal_pos)

            heapq.heappush(heap, (cost, v1, v2, optimal_pos))
            edge_info[(v1, v2)] = (cost, optimal_pos)

        # 逐步收缩代价最小的边
        contraction_count = 0
        while len(self.valid_vertices) > target_vertex_count and heap:
            cost, v1, v2, optimal_pos = heapq.heappop(heap)

            # 检查顶点是否仍然有效
            if v1 not in self.valid_vertices or v2 not in self.valid_vertices:
                continue

            # 执行边收缩
            self.contract_edge(v1, v2, optimal_pos)
            contraction_count += 1

            # 更新进度
            if contraction_count % 100 == 0:
                print(f"已收缩 {contraction_count} 条边, 剩余顶点: {len(self.valid_vertices)}")

        # 重建简化后的网格
        self.rebuild_mesh()

        print(f"简化完成: 剩余 {len(self.vertices)} 个顶点, {len(self.faces)} 个面")

    def contract_edge(self, v1, v2, optimal_pos):
        """收缩边 v1-v2，将v2合并到v1"""
        # 更新v1的位置
        self.vertices[v1] = optimal_pos

        # 合并quadrics
        self.quadrics[v1] += self.quadrics[v2]

        # 更新面的引用：将所有对v2的引用改为v1
        faces_to_remove = set()

        for face_idx in list(self.vertex_faces[v2]):
            face = self.faces[face_idx]
            new_face = []

            # 替换顶点引用
            for vertex_id in face:
                if vertex_id == v2:
                    new_face.append(v1)
                else:
                    new_face.append(vertex_id)

            # 检查面是否退化（包含重复顶点）
            if len(set(new_face)) < 3:
                faces_to_remove.add(face_idx)
            else:
                self.faces[face_idx] = new_face
                # 更新顶点-面映射
                for vertex_id in new_face:
                    self.vertex_faces[vertex_id].add(face_idx)

        # 移除退化面
        for face_idx in faces_to_remove:
            face = self.faces[face_idx]
            for vertex_id in face:
                if vertex_id in self.vertex_faces:
                    self.vertex_faces[vertex_id].discard(face_idx)

        # 标记v2为已删除
        self.valid_vertices.discard(v2)
        if v2 in self.quadrics:
            del self.quadrics[v2]
        if v2 in self.vertex_faces:
            del self.vertex_faces[v2]

    def rebuild_mesh(self):
        """重建简化后的网格"""
        # 创建顶点映射
        valid_vertex_list = sorted(self.valid_vertices)
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_vertex_list)}

        # 创建新的顶点数组
        new_vertices = [self.vertices[i] for i in valid_vertex_list]

        # 创建新的面数组
        new_faces = []
        seen_faces = set()

        for face in self.faces:
            # 检查面是否有效且不重复
            if all(v in self.valid_vertices for v in face):
                new_face = tuple(sorted([vertex_map[v] for v in face]))
                if new_face not in seen_faces and len(set(new_face)) == 3:
                    seen_faces.add(new_face)
                    new_faces.append(list(new_face))

        self.vertices = np.array(new_vertices, dtype=np.float32)
        self.faces = np.array(new_faces, dtype=np.int32)


def simplify_ply_file(input_file, output_file, simplification_ratio=0.5):
    """简化PLY文件的完整流程"""
    try:
        # 读取PLY文件
        print(f"读取文件: {input_file}")
        vertices, faces = PLYReader.read_ply(input_file)
        print(f"原始网格: {len(vertices)} 个顶点, {len(faces)} 个面")

        # 执行QEM简化
        simplifier = QEMSimplifier(vertices, faces)
        simplifier.simplify(target_ratio=simplification_ratio)

        # 保存简化后的网格
        PLYWriter.write_ply(output_file, simplifier.vertices, simplifier.faces)
        print(f"简化后的网格已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    # 设置输入输出文件路径
    input_folder = "../demo/data"  # 输入PLY文件所在的文件夹
    output_folder = "../demo/output"  # 输出PLY文件所在的文件夹
    simplification_ratio = 0.3  # 简化比例 (0.3 = 保留30%的顶点)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 处理文件夹中的所有PLY文件
    if os.path.exists(input_folder):
        successful = 0
        total = 0

        for filename in os.listdir(input_folder):
            if filename.endswith(".ply"):
                input_file = os.path.join(input_folder, filename)
                output_file = os.path.join(output_folder, f"simplified_{filename}")

                total += 1
                if simplify_ply_file(input_file, output_file, simplification_ratio):
                    successful += 1

        print(f"\n处理完成: {successful}/{total} 个文件成功处理")
    else:
        print(f"输入文件夹 '{input_folder}' 不存在")
        print("请创建该文件夹并放入PLY文件，或修改input_folder变量")