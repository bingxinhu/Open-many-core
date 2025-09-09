import logging
from typing import List, Dict, Tuple, Set
import tvm
from tvm import tir

from manycore_config import ManyCoreYAMLConfig
from manycore_scheduler import RoleBasedScheduler
from manycore_primitives import RoleBasedPrimitives

class RoleAwareCodeGenerator:
    """感知核心角色的代码生成器，支持所有14种原语"""
    def __init__(self, config: ManyCoreYAMLConfig, scheduler: RoleBasedScheduler):
        self.config = config
        self.scheduler = scheduler
        self.core_programs = [[] for _ in range(config.get_total_cores())]
        self.opcode_map = RoleBasedPrimitives.OPCODE_MAP
        self.cache_map = scheduler.map_cache_cores()
        self.sram_allocation = {}  # 跟踪每个核心的SRAM分配
        
        # 初始化SRAM分配
        for core_id in range(config.get_total_cores()):
            self.sram_allocation[core_id] = {
                "data": 0,
                "metadata": 0,
                "scratchpad": 0
            }
    
    def generate_input_core_code(self, layer_name: str, input_size: int) -> None:
        """生成输入核心代码：接收数据并分散到计算核心"""
        input_core = self.config.get_core_ids_by_role("input")[0]
        compute_cores = self.scheduler.get_layer_compute_cores(layer_name)
        core_count = len(compute_cores)
        
        # 计算每个计算核心应接收的数据大小
        sizes = [input_size // core_count + (1 if i < input_size % core_count else 0)
                for i in range(core_count)]
        
        # 生成scatter指令
        self.core_programs[input_core].append({
            "opcode": self.opcode_map["scatter"],
            "operands": [
                input_core,
                len(compute_cores), *compute_cores,
                0x1000,  # 输入数据在输入核心的地址
                0x2000,  # 计算核心的接收地址
                len(sizes), *sizes
            ],
            "comment": f"将{layer_name}输入数据分散到计算核心"
        })
        logging.info(f"输入核心{input_core}生成{layer_name}数据分散指令，目标计算核心: {compute_cores}")
    
    def generate_compute_core_code(self, layer_name: str, layer_type: str, 
                                 input_size: int, weight_size: int) -> None:
        """生成计算核心代码：根据层类型选择对应原语执行计算"""
        compute_cores = self.scheduler.get_layer_compute_cores(layer_name)
        workload = self.scheduler.split_workload(layer_name, input_size)
        
        # 层类型到原语的映射
        layer_primitive_map = {
            "conv2d": "conv2d",
            "conv3d": "conv3d",
            "pool2d": "vector_accumulate",
            "relu": "ann_activation",
            "leaky_relu": "ann_activation",
            "tanh": "ann_activation",
            "fully_connected": "fully_connected",
            "dense": "fully_connected",
            "dot_product": "vector_dot",
            "elementwise_multiply": "vector_multiply",
            "batch_norm": "vector_scale",
            "concat": "vector_merge",
            "split": "vector_split",
            "division": "lookup_table",
            "lif": "snn_activation"
        }
        
        # 获取对应的原语
        primitive = layer_primitive_map.get(layer_type, "conv2d")
        sram_layout = self.config.get_core_spec()["sram_layout"]
        
        for core_id in compute_cores:
            # 为每个计算核心分配SRAM地址
            input_addr = self._alloc_sram_addr(core_id, "data", workload[core_id][1] - workload[core_id][0])
            output_addr = self._alloc_sram_addr(core_id, "data", workload[core_id][1] - workload[core_id][0])
            
            # 根据原语类型生成计算指令
            if primitive == "conv2d":
                weight_addr = self._alloc_sram_addr(core_id, "data", weight_size // len(compute_cores))
                cache_core = self.cache_map.get(core_id, -1)
                self.core_programs[core_id].append({
                    "opcode": self.opcode_map["conv2d"],
                    "operands": [
                        input_addr, weight_addr, output_addr,
                        3, 3,  # kernel size
                        1, 1,  # stride
                        1, 1,  # padding
                        core_id,
                        sram_layout["data"]
                    ],
                    "comment": f"{layer_name}卷积计算"
                })
                logging.info(f"计算核心{core_id}生成{layer_name}卷积指令")
                
            elif primitive == "fully_connected":
                weight_addr = self._alloc_sram_addr(core_id, "data", weight_size // len(compute_cores))
                bias_addr = self._alloc_sram_addr(core_id, "data", 4 * len(compute_cores))  # 假设每个核心4字节偏置
                self.core_programs[core_id].append({
                    "opcode": self.opcode_map["fully_connected"],
                    "operands": [
                        input_addr, weight_addr, bias_addr, output_addr,
                        core_id,
                        sram_layout["data"]
                    ],
                    "comment": f"{layer_name}全连接计算"
                })
                
            elif primitive == "ann_activation":
                self.core_programs[core_id].append({
                    "opcode": self.opcode_map["ann_activation"],
                    "operands": [
                        input_addr, output_addr,
                        0,  # ReLU激活
                        core_id,
                        sram_layout["data"]
                    ],
                    "comment": f"{layer_name}ReLU激活"
                })
                
            elif primitive == "vector_dot":
                input2_addr = self._alloc_sram_addr(core_id, "data", workload[core_id][1] - workload[core_id][0])
                self.core_programs[core_id].append({
                    "opcode": self.opcode_map["vector_dot"],
                    "operands": [
                        input_addr, input2_addr, 0.0,  # 输入1, 输入2, 偏置
                        output_addr,
                        core_id,
                        sram_layout["data"]
                    ],
                    "comment": f"{layer_name}向量点积"
                })
        
        # 生成层内同步指令
        self._generate_barrier(compute_cores, f"{layer_name}_complete")
    
    def generate_cache_sync_code(self, layer_name: str, data_size: int) -> None:
        """生成缓存核心同步代码"""
        cache_cores = self.config.get_core_ids_by_role("cache")
        if not cache_cores:
            return
            
        # 获取当前层和下一层的计算核心
        current_cores = self.scheduler.get_layer_compute_cores(layer_name)
        next_layer_idx = self.scheduler.get_layer_index(layer_name) + 1
        next_layer_name = self.scheduler.get_layer_name(next_layer_idx)
        next_cores = self.scheduler.get_layer_compute_cores(next_layer_name) if next_layer_name else []
        
        for cache_core in cache_cores:
            # 生成缓存同步指令
            self.core_programs[cache_core].append({
                "opcode": self.opcode_map["cache_sync"],
                "operands": [
                    cache_core,
                    len(current_cores), *current_cores,
                    len(next_cores), *next_cores,
                    0x3000,  # 缓存地址
                    data_size
                ],
                "comment": f"{layer_name}到{next_layer_name}的缓存同步"
            })
        
        # 生成缓存同步后的栅栏
        self._generate_barrier(current_cores + next_cores + cache_cores, f"{layer_name}_cache_sync")
    
    def generate_output_core_code(self, layer_name: str, output_size: int) -> None:
        """生成输出核心代码：从计算核心收集结果"""
        output_core = self.config.get_core_ids_by_role("output")[0]
        compute_cores = self.scheduler.get_layer_compute_cores(layer_name)
        core_count = len(compute_cores)
        
        # 计算每个计算核心应发送的数据大小
        sizes = [output_size // core_count + (1 if i < output_size % core_count else 0)
                for i in range(core_count)]
        
        # 生成gather指令
        self.core_programs[output_core].append({
            "opcode": self.opcode_map["gather"],
            "operands": [
                output_core,
                len(compute_cores), *compute_cores,
                0x2000,  # 计算核心的输出地址
                0x1000,  # 输出核心的接收地址
                len(sizes), *sizes
            ],
            "comment": f"收集{layer_name}计算结果"
        })
        logging.info(f"输出核心{output_core}生成{layer_name}数据收集指令，来源计算核心: {compute_cores}")
    
    def _generate_barrier(self, core_ids: List[int], barrier_id: str) -> None:
        """生成栅栏同步指令"""
        barrier_code = hash(barrier_id) % 0xFFFF  # 简单哈希生成barrier ID
        for cid in core_ids:
            self.core_programs[cid].append({
                "opcode": self.opcode_map["barrier"],
                "operands": [barrier_code, len(core_ids), *core_ids],
                "comment": f"{barrier_id}同步栅栏"
            })
    
    def _alloc_sram_addr(self, core_id: int, region: str, size: int) -> int:
        """为核心分配SRAM地址"""
        regions = {
            "data": 0x1000,
            "metadata": 0x10000,
            "scratchpad": 0x18000
        }
        
        # 计算所需字节数（假设32位数据）
        byte_size = size * 4
        
        # 获取当前区域已分配大小
        alloc_offset = self.sram_allocation[core_id][region]
        
        # 计算新地址
        addr = regions[region] + alloc_offset
        
        # 更新分配偏移
        self.sram_allocation[core_id][region] += byte_size
        
        # 检查是否超出区域大小
        if addr + byte_size > regions[region] + self.config.get_core_spec()["sram_layout"][region]:
            raise MemoryError(f"核心{core_id}的{region}区域内存不足")
            
        return addr
    
    def generate_binary(self) -> bytearray:
        """生成二进制镜像（简化实现）"""
        # 实际实现中应将指令转换为二进制格式
        binary = bytearray()
        
        # 为每个核心添加指令
        for core_id, program in enumerate(self.core_programs):
            if program:  # 只包含有指令的核心
                # 添加核心ID标记
                binary.extend(core_id.to_bytes(2, byteorder='little'))
                # 添加指令数量
                binary.extend(len(program).to_bytes(2, byteorder='little'))
                
                # 添加每条指令
                for instr in program:
                    # 操作码（1字节）
                    binary.extend(instr["opcode"].to_bytes(1, byteorder='little'))
                    # 操作数数量（1字节）
                    binary.extend(len(instr["operands"]).to_bytes(1, byteorder='little'))
                    # 操作数（每个4字节）
                    for op in instr["operands"]:
                        if isinstance(op, float):
                            binary.extend(bytearray(op.to_bytes(4, byteorder='little', signed=True)))
                        else:
                            # 将numpy.int64转换为Python内置整数类型
                            python_int = int(op)
                            binary.extend(python_int.to_bytes(4, byteorder='little', signed=True))

        return binary