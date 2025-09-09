import logging
import numpy as np
from typing import List, Dict, Any, Optional

from manycore_config import ManyCoreYAMLConfig

class RoleBasedRuntime:
    """基于核心角色的运行时系统"""
    def __init__(self, config: ManyCoreYAMLConfig):
        self.config = config
        self.total_cores = config.get_total_cores()
        self.active_roles = set()  # 激活的角色集合
        self.active_cores = set()  # 激活的核心集合
        
        # 模拟每个核心的SRAM数据
        sram_size = config.get_core_spec()["sram_size"]
        self.sram_data = [bytearray(sram_size) for _ in range(self.total_cores)]
        
        # 核心状态: idle, running, completed, error
        self.core_status = ["idle" for _ in range(self.total_cores)]
        
        logging.info(f"众核运行时初始化完成，总核心数: {self.total_cores}")
    
    def activate_roles(self, roles: List[str]) -> None:
        """激活指定角色的核心"""
        self.active_roles = set(roles)
        self.active_cores = set()
        
        # 激活各角色的核心
        for role in roles:
            core_ids = self.config.get_core_ids_by_role(role)
            self.active_cores.update(core_ids)
            # 更新核心状态
            for cid in core_ids:
                self.core_status[cid] = "ready"
        
        logging.info(f"激活角色: {roles}, 激活核心数: {len(self.active_cores)}")
    
    def load_input_data(self, data: np.ndarray) -> None:
        """将输入数据加载到输入核心的SRAM"""
        input_cores = self.config.get_core_ids_by_role("input")
        if not input_cores:
            raise ValueError("未配置输入核心")
            
        input_core = input_cores[0]
        data_bytes = data.tobytes()
        data_region_size = self.config.get_core_spec()["sram_layout"]["data"]
        
        # 检查数据大小
        if len(data_bytes) > data_region_size:
            raise MemoryError(f"输入数据过大({len(data_bytes)}字节)，超出输入核心SRAM数据区({data_region_size}字节)")
        
        # 写入输入核心的SRAM数据区域（从0x1000开始）
        data_addr = 0x1000
        self.sram_data[input_core][data_addr:data_addr+len(data_bytes)] = data_bytes
        self.core_status[input_core] = "loaded"
        
        logging.info(f"输入数据加载到核心{input_core}，大小: {len(data_bytes)}字节")
    
    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """将权重加载到输入核心，准备分发"""
        input_cores = self.config.get_core_ids_by_role("input")
        if not input_cores:
            raise ValueError("未配置输入核心")
            
        input_core = input_cores[0]
        sram_size = self.config.get_core_spec()["sram_size"]
        data_region_size = self.config.get_core_spec()["sram_layout"]["data"]
        
        # 从数据区末尾开始分配权重地址
        current_addr = 0x1000 + data_region_size
        
        for name, weight in weights.items():
            w_bytes = weight.tobytes()
            w_size = len(w_bytes)
            
            # 检查是否有足够空间
            if current_addr + w_size > sram_size:
                raise MemoryError(f"权重{name}过大，无法放入输入核心SRAM")
            
            # 写入输入核心的SRAM
            self.sram_data[input_core][current_addr:current_addr+w_size] = w_bytes
            logging.info(f"权重{name}加载到核心{input_core}，地址: 0x{current_addr:x}，大小: {w_size}字节")
            
            current_addr += w_size
        
        logging.info("所有权重加载完成")
    
    def run(self, binary: bytearray) -> np.ndarray:
        """执行二进制并返回结果"""
        if not self.active_cores:
            raise RuntimeError("未激活任何核心，请先调用activate_roles()")
        
        # 加载二进制到各核心（简化实现）
        self._load_binary(binary)
        
        # 启动所有激活的核心
        for cid in self.active_cores:
            self.core_status[cid] = "running"
        
        # 模拟执行（实际实现中由硬件执行）
        self._simulate_execution()
        
        # 从输出核心读取结果
        output_cores = self.config.get_core_ids_by_role("output")
        if not output_cores:
            raise ValueError("未配置输出核心")
            
        output_core = output_cores[0]
        if self.core_status[output_core] != "completed":
            raise RuntimeError(f"输出核心{output_core}执行失败，状态: {self.core_status[output_core]}")
        
        # 从输出核心的0x1000地址读取结果
        result_addr = 0x1000
        result_size = 4096  # 假设最大结果大小为4096字节
        result_bytes = self.sram_data[output_core][result_addr:result_addr+result_size]
        
        # 转换为numpy数组（假设float32类型）
        return np.frombuffer(result_bytes, dtype=np.float32)
    
    def _load_binary(self, binary: bytearray) -> None:
        """加载二进制到各核心（简化实现）"""
        ptr = 0
        while ptr < len(binary):
            # 读取核心ID（2字节）
            core_id = int.from_bytes(binary[ptr:ptr+2], byteorder='little')
            ptr += 2
            
            # 读取指令数量（2字节）
            instr_count = int.from_bytes(binary[ptr:ptr+2], byteorder='little')
            ptr += 2
            
            # 跳过指令数据（实际实现中应加载到核心指令存储器）
            for _ in range(instr_count):
                # 操作码（1字节）+ 操作数数量（1字节）+ 操作数（每个4字节）
                op_count = binary[ptr+1]
                ptr += 2 + op_count * 4
        
        logging.info(f"二进制加载完成，大小: {len(binary)}字节")
    
    def _simulate_execution(self) -> None:
        """模拟核心执行（实际实现中由硬件完成）"""
        # 简单模拟执行延迟
        import time
        time.sleep(0.1)  # 模拟100ms执行时间
        
        # 更新所有激活核心的状态为完成
        for cid in self.active_cores:
            self.core_status[cid] = "completed"
        
        logging.info(f"所有{len(self.active_cores)}个核心执行完成")
    
    def get_core_status(self, core_id: Optional[int] = None) -> Any:
        """获取核心状态，可选指定核心ID"""
        if core_id is not None:
            return self.core_status[core_id] if 0 <= core_id < self.total_cores else None
        return {cid: self.core_status[cid] for cid in range(self.total_cores)}
