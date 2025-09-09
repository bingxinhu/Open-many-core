import yaml
import logging
from typing import List, Dict, Tuple, Any, Optional
import os

class ManyCoreYAMLConfig:
    """众核架构YAML配置解析器"""
    def __init__(self, config_path: str):
        # 加载并解析YAML配置
        self.config = self._load_config(config_path)
        
        # 验证配置完整性
        self._validate_config()
        
        logging.info(f"配置加载完成: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"配置文件解析错误: {str(e)}")
    
    def _validate_config(self) -> None:
        """验证配置的完整性和有效性"""
        required_sections = ["hardware", "core_roles", "model"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置缺少必要部分: {section}")
        
        # 验证硬件配置
        hardware = self.config["hardware"]
        required_hardware = ["total_cores", "array_shape", "core_spec", "noc"]
        for item in required_hardware:
            if item not in hardware:
                raise ValueError(f"硬件配置缺少必要项: {item}")
        
        # 验证核心角色配置
        core_roles = self.config["core_roles"]
        required_roles = ["input", "output", "compute"]
        for role in required_roles:
            if role not in core_roles:
                raise ValueError(f"核心角色配置缺少必要角色: {role}")
        
        # 验证模型配置
        model = self.config["model"]
        required_model = ["onnx_path", "input_shape", "input_dtype"]
        for item in required_model:
            if item not in model:
                raise ValueError(f"模型配置缺少必要项: {item}")
        
        # 验证核心ID有效性
        total_cores = hardware["total_cores"]
        for role, cores in core_roles.items():
            if role in ["input", "output"]:
                # 输入输出角色应为单个ID
                if not isinstance(cores, int) or cores < 0 or cores >= total_cores:
                    raise ValueError(f"无效的{role}核心ID: {cores}")
            else:
                # 其他角色应为ID列表
                if not isinstance(cores, list):
                    raise ValueError(f"{role}核心配置必须是列表: {cores}")
                for cid in cores:
                    if not isinstance(cid, int) or cid < 0 or cid >= total_cores:
                        raise ValueError(f"无效的{role}核心ID: {cid}")
    
    # ------------------------------
    # 硬件配置获取方法
    # ------------------------------
    
    def get_total_cores(self) -> int:
        """获取总核心数"""
        return self.config["hardware"]["total_cores"]
    
    def get_array_shape(self) -> Tuple[int, int]:
        """获取核心阵列形状"""
        return tuple(self.config["hardware"]["array_shape"])
    
    def get_core_spec(self) -> Dict[str, Any]:
        """获取单核心规格"""
        return self.config["hardware"]["core_spec"]
    
    def get_noc_spec(self) -> Dict[str, Any]:
        """获取片上网络规格"""
        return self.config["hardware"]["noc"]
    
    # ------------------------------
    # 核心角色配置获取方法
    # ------------------------------
    
    def get_core_ids_by_role(self, role: str) -> List[int]:
        """根据角色获取核心ID列表"""
        if role not in self.config["core_roles"]:
            return []
            
        cores = self.config["core_roles"][role]
        if role in ["input", "output"]:
            return [cores] if cores != -1 else []
        return cores if isinstance(cores, list) else []
    
    def get_core_neighbors(self, core_id: int) -> List[int]:
        """获取核心的直接邻居（用于缓存同步）"""
        rows, cols = self.get_array_shape()
        x, y = core_id // cols, core_id % cols
        neighbors = []
        
        # 检查上下左右四个方向的邻居
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                neighbors.append(nx * cols + ny)
        
        return neighbors
    
    def estimate_transfer_time(self, src_id: int, dst_id: int, data_size: int) -> float:
        """估算数据传输时间(ns)"""
        rows, cols = self.get_array_shape()
        x1, y1 = src_id // cols, src_id % cols
        x2, y2 = dst_id // cols, dst_id % cols
        hops = abs(x1 - x2) + abs(y1 - y2)
        
        noc_spec = self.get_noc_spec()
        transfer_time_ns = (data_size * 8) / (noc_spec["bandwidth"] * 1e9) * 1e9  # 传输时间
        return transfer_time_ns + hops * noc_spec["latency_per_hop"]  # 总时间=传输时间+路由延迟
    
    # ------------------------------
    # 模型配置获取方法
    # ------------------------------
    
    def get_onnx_path(self) -> str:
        """获取ONNX模型路径"""
        return self.config["model"]["onnx_path"]
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """获取输入形状"""
        return tuple(self.config["model"]["input_shape"])
    
    def get_input_dtype(self) -> str:
        """获取输入数据类型"""
        return self.config["model"]["input_dtype"]
    
    # ------------------------------
    # 层映射配置获取方法
    # ------------------------------
    
    def get_layer_mapping(self) -> Dict[str, List[int]]:
        """获取层到计算核心的映射"""
        return self.config.get("layer_mapping", {})
