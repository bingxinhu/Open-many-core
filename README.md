# 众核架构神经网络部署框架

## 项目概述

本框架专为160核众核架构设计，实现了从ONNX模型到众核硬件执行的端到端部署能力。通过**角色化核心分配**（输入/计算/输出/缓存）和**分布式存储**（无全局内存）设计，支持灵活的核心资源配置，仅激活部分核心完成神经网络推理，最大化硬件利用率。

框架内置14种计算原语，涵盖CNN、MLP、SNN等多种神经网络类型，支持通过YAML配置文件实现全参数定制，无需修改代码即可适配不同模型和硬件需求。


## 核心特性

- **丰富的计算原语**：内置14种原语（2D/3D卷积、向量运算、全连接等），支持多种神经网络类型
- **灵活的核心配置**：通过YAML定义核心角色（输入/计算/输出/缓存），支持动态资源分配
- **分布式存储优化**：利用每个核心的144KB本地SRAM，结合缓存核心减少跨核通信
- **完整的工作流**：覆盖ONNX模型导入、IR转换、任务调度、代码生成到执行的全流程
- **配置驱动开发**：所有参数通过YAML集中管理，模型切换和硬件适配无需修改代码


## 架构设计

### 模块结构

| 架构 | 代码 | 功能描述 |
|------|----------|----------|
│**应用层**| pipeline_controller.py  | 主程序     │
│**模型处理层**| model_processor.py   | 模型加载与转换 │
│**原语层**|  manycore_primitives.py | 14种计算/通信原语 │
│**调度层**|  manycore_scheduler.py  | 任务分配/负载均衡 │
│**代生层**|  manycore_codegen.py    | 指令生成/二进制打包 │
│**运行层**|  manycore_runtime.py    | 核心管理/数据加载/执行 │
│**配置层**| manycore_config.py      | 配置解析/硬件抽象 |   
│**配置文件层**│ manycore_config.yaml | 硬件/角色/模型配置 │
    

### 核心数据流
ONNX模型 → [model_processor.py] → TVM IR + 权重提取
                                  ↓
YAML配置 → [manycore_config.py] → 配置对象（硬件/角色/模型参数）
                                  ↓
[manycore_scheduler.py] → 层-核心映射 + 工作负载分配
                                  ↓
[manycore_codegen.py] → 调用原语生成指令 → 二进制镜像
                                  ↓
[manycore_runtime.py] → 加载数据/执行 → 推理结果

## 原语支持

框架实现了14种核心原语，覆盖各类神经网络计算需求：

| 类型 | 原语名称 | 功能描述 | 适用场景 |
|------|----------|----------|----------|
| **卷积计算** | `conv2d` | 2D卷积运算 | CNN特征提取 |
| | `conv3d` | 3D卷积运算 | 视频/3D数据处理 |
| **向量运算** | `vector_accumulate` | 向量累加 | 求和计算 |
| | `vector_dot` | 向量点积 | 相似度计算 |
| | `vector_multiply` | 向量乘法 | 元素级运算 |
| | `vector_scale` | 向量缩放/扩张 | 特征缩放 |
| | `vector_merge` | 向量合并 | 特征拼接 |
| | `vector_split` | 向量裂解 | 特征分解 |
| **神经网络层** | `fully_connected` | 全连接计算 | MLP层 |
| | `ann_activation` | ANN激活函数 | ReLU/Sigmoid等 |
| | `snn_activation` | SNN激活函数 | LIF神经元模型 |
| **特殊运算** | `lookup_table` | 查找表运算 | 除法/LUT加速 |
| **通信原语** | `scatter` | 数据分散 | 输入核心分发数据 |
| | `gather` | 数据聚合 | 输出核心收集结果 |


## 快速开始

### 环境依赖

- Python 3.8+
- TVM 0.15.0+（需支持Relax IR）
- ONNX 1.13.0+
- PyYAML 6.0+
- NumPy 1.24.0+

安装依赖：pip install onnx tvm pyyaml numpy
### 配置说明

修改`manycore_config.yaml`配置核心分配和模型参数：
# 硬件配置
hardware:
  total_cores: 160                 # 总核心数（固定160核）
  array_shape: [16, 10]            # 核心阵列布局
  # ...（其他硬件参数）

# 核心角色分配（1输入+6计算+1输出+2缓存）
core_roles:
  input: 0                         # 输入核心ID
  output: 7                        # 输出核心ID
  compute: [1, 2, 3, 4, 5, 6]      # 计算核心ID列表
  cache: [8, 9]                    # 缓存核心ID列表

# 模型配置
model:
  onnx_path: "../../onnx_model/lenet.onnx"  # 模型路径
  input_shape: [1, 1, 28, 28]              # 输入形状
  input_dtype: "float32"                   # 输入数据类型

# 层-核心映射
layer_mapping:
  conv2d_0: [1, 2, 3]              # 第一层卷积使用核心1-3
  conv2d_1: [4, 5, 6]              # 第二层卷积使用核心4-6
  # ...（其他层映射）
### 运行部署

执行端到端部署流程：python pipeline_controller.py
运行成功后，将输出：
- 核心配置信息
- 模型层分析结果
- 代码生成进度
- 推理结果大小和示例值


## 使用指南

### 更换模型

1. 修改`manycore_config.yaml`中的`model.onnx_path`为新模型路径
2. 调整`model.input_shape`和`model.input_dtype`匹配新模型输入要求
3. （可选）修改`layer_mapping`调整各层使用的计算核心
4. 重新运行`pipeline_controller.py`

### 调整核心分配

1. 修改`core_roles`部分调整各角色核心：
   ```yaml
   core_roles:
     input: 10                       # 更换输入核心为ID=10
     output: 11                      # 更换输出核心为ID=11
     compute: [12,13,14,15,16,17,18,19]  # 增加计算核心到8个
   ```
2. （可选）相应调整`layer_mapping`中的核心分配

### 扩展新原语

1. 在`manycore_primitives.py`中实现新原语的静态方法
2. 在`register_primitives()`中注册新原语
3. 在`manycore_codegen.py`的`primitive_map`中添加层类型与新原语的映射
4. （可选）在YAML配置中添加对应层的映射关系


## 常见问题

### Q1: 如何处理模型输入过大的问题？
A1: 可通过以下方式解决：
- 调整`core_spec.sram_layout.data`增大数据区大小
- 在`manycore_scheduler.py`中实现数据分块传输逻辑
- 增加计算核心数量，分散数据处理压力

### Q2: 如何优化推理性能？
A2: 性能优化建议：
- 确保计算核心与缓存核心为邻居关系（通过`get_core_neighbors`验证）
- 根据层计算复杂度调整`layer_mapping`，平衡负载
- 对于大型卷积层，增加分配的计算核心数量

### Q3: 支持哪些ONNX算子？
A3: 目前支持：
- 卷积（Conv）、池化（Pool）、全连接（Gemm）
- 激活函数（Relu、Sigmoid、Tanh）
- 张量操作（Reshape、Concat、Split）
- 可通过扩展原语层支持更多算子
    
