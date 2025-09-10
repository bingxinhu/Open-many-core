import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

from manycore_config import ManyCoreYAMLConfig

class RoleBasedScheduler:
    """基于核心角色的任务调度器，支持所有原语的任务分配"""
    def __init__(self, config: ManyCoreYAMLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.layer_mapping = self._init_layer_mapping()  # 网络层到计算核心的映射
        self.layer_order = []  # 层执行顺序
        self.cache_map = self.map_cache_cores()  # 计算核心到缓存核心的映射
        
    def _init_layer_mapping(self) -> Dict[str, List[int]]:
        """从配置初始化层映射"""
        config_mapping = self.config.get_layer_mapping()
        compute_cores = self.config.get_core_ids_by_role("compute")
        
        # 如果配置中没有层映射，为所有层分配所有计算核心
        if not config_mapping:
            return {"default": compute_cores}
        
        return config_mapping
    
    def assign_layer_to_cores(self, layer_name: str, core_ids: Optional[List[int]] = None) -> None:
        """将网络层分配给指定计算核心"""
        if core_ids is None:
            # 如果未指定核心，使用默认映射
            core_ids = self.layer_mapping.get("default", self.config.get_core_ids_by_role("compute"))
        
        # 检查核心是否已配置为计算角色
        compute_cores = self.config.get_core_ids_by_role("compute")
        for cid in core_ids:
            assert cid in compute_cores, f"核心 {cid} 未配置为计算角色"
            
        self.layer_mapping[layer_name] = core_ids
        if layer_name not in self.layer_order:
            self.layer_order.append(layer_name)
    
    def split_workload(self, layer_name: str, data_size: int) -> Dict[int, Tuple[int, int]]:
        """将层工作负载分配到指定计算核心"""
        core_ids = self.get_layer_compute_cores(layer_name)
        core_count = len(core_ids)
        
        if core_count == 0:
            raise ValueError(f"未为层 {layer_name} 分配计算核心")
            
        # 根据层类型调整分配策略
        layer_type = self._infer_layer_type(layer_name)
        if layer_type in ["conv2d", "conv3d"]:
            # 卷积层按输出通道分配
            return self._split_by_output_channels(layer_name, data_size)
        elif layer_type in ["fully_connected", "dense"]:
            # 全连接层按输出神经元分配
            return self._split_by_output_dim(layer_name, data_size)
        else:
            # 默认均匀分配
            return self._split_uniformly(core_ids, data_size)
    
    def _split_uniformly(self, core_ids: List[int], data_size: int) -> Dict[int, Tuple[int, int]]:
        """均匀分配工作负载"""
        core_count = len(core_ids)
    
        if data_size <= 0:
            # 当data_size为0或负数时，返回一个默认的工作负载分配
            self.logger.warning(f"data_size ({data_size}) 小于等于0，使用默认分配")
            workload = {cid: (0, 0) for cid in core_ids}
            return workload
        
        if (data_size > 0):
            print(f"data_size: {data_size}")
            print(f"core_count: {core_count}")
        
        base_size = data_size // core_count 
        remainder = data_size % core_count
        
        workload = {}
        current = 0
        for i, cid in enumerate(core_ids):
            end = current + base_size + (1 if i < remainder else 0)
            workload[cid] = (current, end)
            current = end
        return workload
    
    def _split_by_output_channels(self, layer_name: str, data_size: int) -> Dict[int, Tuple[int, int]]:
        """按输出通道分配卷积层工作负载"""
        core_ids = self.get_layer_compute_cores(layer_name)
        # 假设通道数是核心数的倍数
        channels_per_core = data_size // len(core_ids)
        
        workload = {}
        current = 0
        for cid in core_ids:
            end = current + channels_per_core
            workload[cid] = (current, end)
            current = end
        return workload
    
    def _split_by_output_dim(self, layer_name: str, data_size: int) -> Dict[int, Tuple[int, int]]:
        """按输出维度分配全连接层工作负载"""
        core_ids = self.get_layer_compute_cores(layer_name)
        # 假设输出维度是核心数的倍数
        dims_per_core = data_size // len(core_ids)
        
        workload = {}
        current = 0
        for cid in core_ids:
            end = current + dims_per_core
            workload[cid] = (current, end)
            current = end
        return workload
    
    def map_cache_cores(self) -> Dict[int, int]:
        """为每个计算核心分配一个缓存核心（优先邻居）"""
        cache_map = {}
        compute_cores = self.config.get_core_ids_by_role("compute")
        cache_cores = self.config.get_core_ids_by_role("cache")
        
        if not cache_cores:
            logging.warning("未配置缓存核心，将跳过缓存同步")
            return {}
        
        for cid in compute_cores:
            # 优先选择邻居作为缓存核心
            neighbors = self.config.get_core_neighbors(cid)
            for neighbor in neighbors:
                if neighbor in cache_cores:
                    cache_map[cid] = neighbor
                    break
            # 如果没有邻居缓存核心，从缓存核心列表中选第一个
            if cid not in cache_map:
                cache_map[cid] = cache_cores[0]
        
        return cache_map
    
    def get_layer_compute_cores(self, layer_name: str) -> List[int]:
        """获取指定层的计算核心"""
        return self.layer_mapping.get(layer_name, self.layer_mapping.get("default", []))
    
    def _infer_layer_type(self, layer_name: str) -> str:
        """从层名称推断层类型"""
        layer_name = layer_name.lower()
        if "conv2d" in layer_name:
            return "conv2d"
        elif "conv3d" in layer_name:
            return "conv3d"
        elif "fc" in layer_name or "dense" in layer_name:
            return "fully_connected"
        elif "relu" in layer_name or "activation" in layer_name:
            return "activation"
        elif "pool" in layer_name:
            return "pooling"
        elif "concat" in layer_name:
            return "concat"
        elif "split" in layer_name:
            return "split"
        else:
            return "unknown"
    
    def get_layer_index(self, layer_name: str) -> int:
        """获取层在执行顺序中的索引"""
        try:
            return self.layer_order.index(layer_name)
        except ValueError:
            return -1
    
    def get_layer_name(self, index: int) -> Optional[str]:
        """根据索引获取层名称"""
        if 0 <= index < len(self.layer_order):
            return self.layer_order[index]
        return None
    
    def estimate_layer_performance(self, layer_name: str, input_size: int) -> Dict[str, float]:
        """估算层性能指标"""
        core_ids = self.get_layer_compute_cores(layer_name)
        layer_type = self._infer_layer_type(layer_name)
        
        # 基础性能参数
        mac_units = self.config.get_core_spec()["compute_capability"]["mac_units"]
        freq = 1.0  # 1GHz假设频率
        
        # 根据层类型估算计算量
        if layer_type in ["conv2d", "conv3d"]:
            # 卷积层计算量估算: 2 * N * C * H * W * K^2 / S
            compute_ops = 2 * input_size * 9  # 简化估算
        elif layer_type in ["fully_connected", "dense"]:
            # 全连接层计算量估算: 2 * input_size * output_size
            compute_ops = 2 * input_size * (input_size // 2)  # 假设输出是输入的一半
        else:
            # 其他层简化估算
            compute_ops = input_size
        
        # 计算时间 (秒)
        compute_time = compute_ops / (len(core_ids) * mac_units * freq * 1e9)
        
        # 通信时间估算
        comm_time = 0.0
        if len(core_ids) > 1:
            # 假设需要在核心间交换一半的数据
            data_size = input_size * 4  # 4字节/元素
            max_hops = self.config.get_noc_spec()["max_hops"]
            hop_latency = self.config.get_noc_spec()["latency_per_hop"]
            bandwidth = self.config.get_noc_spec()["bandwidth"]
            
            # 通信时间 (秒) = 数据传输时间 + 路由延迟
            transfer_time = (data_size * 0.5) / (bandwidth * 1e9)
            routing_delay = max_hops * hop_latency * 1e-9
            comm_time = transfer_time + routing_delay
        
        return {
            "compute_cores": len(core_ids),
            "compute_ops": compute_ops,
            "compute_time_s": compute_time,
            "comm_time_s": comm_time,
            "total_time_s": compute_time + comm_time
        }