import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

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
        
        # SRAM内存区域定义 - 合并MEM0和MEM1为一个整体区域
        self.MEM0_ADDR = 0x1000
        self.MEM0_SIZE = 65536  # 64KB
        self.MEM1_ADDR = 0x11000  # 0x1000 + 65536 = 0x11000
        self.MEM1_SIZE = 65536  # 64KB
        self.MEM2_ADDR = 0x21000  # 0x11000 + 65536 = 0x21000
        self.MEM2_SIZE = 16384  # 16KB
        
        # 将MEM0和MEM1合并视为一个128KB的整体空间
        self.COMBINED_MEM_ADDR = self.MEM0_ADDR
        self.COMBINED_MEM_SIZE = self.MEM0_SIZE + self.MEM1_SIZE  # 128KB
        
        # 用于记录权重和参数的存储位置
        self.weight_locations = {}
        self.param_locations = {}
        
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
        """将输入数据加载到输入核心的SRAM（使用合并后的128KB空间）"""
        input_cores = self.config.get_core_ids_by_role("input")
        if not input_cores:
            raise ValueError("未配置输入核心")
            
        input_core = input_cores[0]
        data_bytes = data.tobytes()
        data_size = len(data_bytes)
        
        # 检查数据大小是否超出合并后的空间
        if data_size > self.COMBINED_MEM_SIZE:
            raise MemoryError(f"输入数据过大({data_size}字节)，超出合并后的MEM0+MEM1空间({self.COMBINED_MEM_SIZE}字节)")
        
        # 尝试在合并后的空间中存储完整的输入数据
        # 从COMBINED_MEM_ADDR开始写入
        self.sram_data[input_core][self.COMBINED_MEM_ADDR:self.COMBINED_MEM_ADDR+data_size] = data_bytes
        
        # 记录输入数据的位置
        self.input_location = (input_core, self.COMBINED_MEM_ADDR, data_size)
        
        self.core_status[input_core] = "loaded"
        logging.info(f"输入数据加载到核心{input_core}的合并内存区域，地址: 0x{self.COMBINED_MEM_ADDR:x}，大小: {data_size}字节")
    
    def load_weights(self, weights: Dict[str, np.ndarray], model_params: Optional[Dict[str, np.ndarray]] = None) -> None:
        """将权重加载到存储核心，支持大权重分割存储
           合并的MEM0+MEM1：存储输入、权重和计算临时变量
           MEM2：仅存储模型参数
        """
        # 获取可用于存储的核心
        input_cores = self.config.get_core_ids_by_role("input")
        cache_cores = self.config.get_core_ids_by_role("cache")
        compute_cores = self.config.get_core_ids_by_role("compute")
        
        # 合并所有可用存储核心
        storage_cores = input_cores.copy()
        storage_cores.extend(cache_cores)
        storage_cores.extend(compute_cores)
        
        if not storage_cores:
            raise ValueError("未配置可用的存储核心")
            
        # 初始化权重位置映射表
        self.weight_locations = {}
        self.param_locations = {}
        
        # 为每个存储核心初始化地址指针
        # 将MEM0和MEM1视为一个连续的整体空间
        core_addresses = {
            core_id: {
                'combined': self.COMBINED_MEM_ADDR,  # 合并后的空间当前地址
                'mem2': self.MEM2_ADDR               # MEM2当前地址
            } for core_id in storage_cores
        }
        
        # 第一步：存储模型参数到MEM2区域
        if model_params:
            for name, param in model_params.items():
                param_bytes = param.tobytes()
                param_size = len(param_bytes)
                
                if param_size > self.MEM2_SIZE:
                    raise ValueError(f"模型参数{name}过大({param_size}字节)，无法放入MEM2区域({self.MEM2_SIZE}字节)")
                
                stored = False
                for core_id in storage_cores:
                    current_addr = core_addresses[core_id]['mem2']
                    
                    # 检查该核心的MEM2是否有足够空间
                    if current_addr + param_size <= self.MEM2_ADDR + self.MEM2_SIZE:
                        # 写入该核心的SRAM MEM2区域
                        self.sram_data[core_id][current_addr:current_addr+param_size] = param_bytes
                        # 记录参数位置
                        self.param_locations[name] = (core_id, current_addr, param_size)
                        # 更新该核心的MEM2当前地址
                        core_addresses[core_id]['mem2'] = current_addr + param_size
                        
                        logging.info(f"模型参数{name}加载到核心{core_id}的MEM2区域，地址: 0x{current_addr:x}，大小: {param_size}字节")
                        stored = True
                        break
                
                if not stored:
                    raise MemoryError(f"所有核心的MEM2区域空间不足，无法存储模型参数{name}")
            
            logging.info("所有模型参数加载完成")
        
        # 第二步：存储权重到合并后的MEM0+MEM1区域
        for name, weight in weights.items():
            weight_bytes = weight.tobytes()
            weight_size = len(weight_bytes)
            
            # 标记权重是否已存储
            stored = False
            blocks = []
            
            # 尝试在单个核心的合并空间中存储完整的权重
            for core_id in storage_cores:
                current_addr = core_addresses[core_id]['combined']
                
                # 检查该核心的合并空间是否有足够空间
                if current_addr + weight_size <= self.COMBINED_MEM_ADDR + self.COMBINED_MEM_SIZE:
                    # 写入该核心的SRAM合并区域
                    self.sram_data[core_id][current_addr:current_addr+weight_size] = weight_bytes
                    # 记录权重位置
                    self.weight_locations[name] = (core_id, current_addr, weight_size)
                    # 更新该核心的合并空间当前地址
                    core_addresses[core_id]['combined'] = current_addr + weight_size
                    
                    logging.info(f"权重{name}加载到核心{core_id}的合并内存区域，地址: 0x{current_addr:x}，大小: {weight_size}字节")
                    stored = True
                    break
            
            # 如果无法作为整体存储，尝试分割存储到多个核心的合并空间
            if not stored:
                logging.info(f"权重{name}过大，尝试分割存储")
                remaining_bytes = weight_bytes
                remaining_size = weight_size
                
                while remaining_size > 0:
                    block_stored = False
                    
                    # 遍历所有核心寻找空间
                    for core_id in storage_cores:
                        current_addr = core_addresses[core_id]['combined']
                        available_space = self.COMBINED_MEM_ADDR + self.COMBINED_MEM_SIZE - current_addr
                        
                        if available_space > 0:
                            # 计算本次存储的块大小
                            block_size = min(remaining_size, available_space)
                            # 提取块数据
                            block_data = remaining_bytes[:block_size]
                            # 写入核心SRAM合并区域
                            self.sram_data[core_id][current_addr:current_addr+block_size] = block_data
                            # 记录块位置
                            blocks.append((core_id, current_addr, block_size))
                            # 更新核心地址和剩余数据
                            core_addresses[core_id]['combined'] = current_addr + block_size
                            remaining_bytes = remaining_bytes[block_size:]
                            remaining_size -= block_size
                            
                            logging.info(f"权重{name}的块加载到核心{core_id}的合并内存区域，地址: 0x{current_addr:x}，大小: {block_size}字节")
                            block_stored = True
                            break
                    
                    # 如果所有核心都没有空间，抛出异常
                    if not block_stored:
                        raise MemoryError(f"权重{name}过大，即使分割后也无法放入任何可用核心的SRAM")
                
                # 记录权重的所有块位置
                self.weight_locations[name] = blocks
        
        logging.info("所有权重加载完成")
    
    def allocate_scratch_space(self, core_id: int, size: int) -> int:
        """在指定核心的合并内存区域中分配计算临时空间"""
        # 确保核心有效且已激活
        if core_id not in self.active_cores:
            raise ValueError(f"核心{core_id}未激活")
            
        # 检查是否有足够的空间
        # 注意：这里假设我们有一个机制来跟踪每个核心的当前分配状态
        # 为简化实现，我们从合并空间的末尾向前分配临时空间
        scratch_addr = self.COMBINED_MEM_ADDR + self.COMBINED_MEM_SIZE - size
        
        # 实际实现中，应该有更复杂的内存管理机制
        # 这里仅提供基本的占位实现
        logging.info(f"在核心{core_id}分配临时空间，地址: 0x{scratch_addr:x}，大小: {size}字节")
        
        return scratch_addr
    
    def get_memory_layout(self, core_id: int) -> Dict[str, Any]:
        """获取指定核心的内存布局信息"""
        return {
            'combined_memory': {
                'addr': self.COMBINED_MEM_ADDR,
                'size': self.COMBINED_MEM_SIZE,
                'description': '合并的MEM0和MEM1区域，用于存储输入、权重和计算临时变量'
            },
            'mem2': {
                'addr': self.MEM2_ADDR,
                'size': self.MEM2_SIZE,
                'description': 'MEM2区域，仅用于存储模型参数'
            }
        }
    
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
        
        # 从输出核心的合并内存区域读取结果
        result_addr = self.COMBINED_MEM_ADDR  # 假设结果存放在合并区域的起始位置
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
        """获取核心状态 ,可选指定核心ID"""
        if core_id is not None:
            return self.core_status[core_id] if 0 <= core_id < self.total_cores else None
        return {cid: self.core_status[cid] for cid in range(self.total_cores)}