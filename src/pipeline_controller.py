import logging
import numpy as np

# 导入组件
from model_processor import ModelProcessor
from manycore_config import ManyCoreYAMLConfig
from manycore_primitives import RoleBasedPrimitives
from manycore_scheduler import RoleBasedScheduler
from manycore_codegen import RoleAwareCodeGenerator
from manycore_runtime import RoleBasedRuntime

class PipelineController:
    """众核架构部署全流程控制器"""
    def __init__(self, config_path: str):
        # 初始化配置
        self.config = ManyCoreYAMLConfig(config_path)
        
        # 初始化核心组件
        self._init_components()
        
        # 初始化模型处理器
        self.model_processor = ModelProcessor(self.config)
        
        # 初始化流程状态
        self.binary = None
        self.weights = None
        
    def _init_components(self) -> None:
        """初始化众核架构核心组件"""
        # 注册原语
        RoleBasedPrimitives.register_primitives()
        logging.info("1. 原语注册完成")
        # 创建调度器
        self.scheduler = RoleBasedScheduler(self.config)
        logging.info("2. 调度器初始化完成")
        # 创建代码生成器
        self.codegen = RoleAwareCodeGenerator(self.config, self.scheduler)
        logging.info("3. 代码生成器初始化完成")
        # 创建运行时
        self.runtime = RoleBasedRuntime(self.config)
    
        logging.info("4. 众核架构核心组件初始化完成")
        logging.info(f"核心配置: 输入={self.config.get_core_ids_by_role('input')}, "
                    f"输出={self.config.get_core_ids_by_role('output')}, "
                    f"计算核心数={len(self.config.get_core_ids_by_role('compute'))}")
    
    def prepare_model(self) -> None:
        """准备模型：加载、转换和提取权重"""
        # 加载并转换模型
        self.mod, self.input_name, self.onnx_model = self.model_processor.load_and_convert()
        
        # 提取权重
        self.weights = self.model_processor.extract_weights(self.onnx_model)
        
        # 分析模型层
        self.layers = self.model_processor.analyze_layers(self.mod)
        
        # 为每个层分配计算核心
        for layer_name, layer_type in self.layers:
            self.scheduler.assign_layer_to_cores(layer_name)
            
            # 估算层性能
            input_size = np.prod(self.config.get_input_shape())
            perf = self.scheduler.estimate_layer_performance(layer_name, input_size)
            logging.debug(f"{layer_name} 性能估算: 计算时间={perf['compute_time_s']:.6f}s, "
                         f"通信时间={perf['comm_time_s']:.6f}s")
        
        logging.info(f"模型层信息: {[name for name, _ in self.layers]}")
    
    def generate_executable(self) -> None:
        """生成众核可执行代码"""
        if not hasattr(self, 'layers'):
            raise RuntimeError("请先调用prepare_model()准备模型")
            
        # 计算输入数据字节大小
        input_size = np.prod(self.config.get_input_shape()) * np.dtype(self.config.get_input_dtype()).itemsize
        
        # 为每个层生成代码
        logging.info(f"开始为 {len(self.layers)} 层生成代码...")
        for i, (layer_name, layer_type) in enumerate(self.layers):
            logging.info(f"为层 {layer_name} (类型: {layer_type}) 生成代码")
            
            # 1. 生成输入核心代码（第一层需要）
            if i == 0:
                self.codegen.generate_input_core_code(layer_name, input_size)
            
            # 2. 生成计算核心代码
            weight_size = 0
            if layer_type in ["conv2d", "conv3d", "fully_connected"]:
                # 估算权重大小
                weight_size = input_size // 2
                
            self.codegen.generate_compute_core_code(layer_name, layer_type, input_size, weight_size)
            
            # 3. 生成缓存同步代码（除了最后一层）
            if i < len(self.layers) - 1 and self.config.get_core_ids_by_role("cache"):
                self.codegen.generate_cache_sync_code(layer_name, input_size // 2)
            
            # 4. 更新输入大小（假设每层输出是输入的一半）
            input_size = input_size // 2
        
        # 为最后一层生成输出代码
        last_layer_name, _ = self.layers[-1]
        self.codegen.generate_output_core_code(last_layer_name, input_size)
        
        # 生成最终二进制
        self.binary = self.codegen.generate_binary()
        logging.info(f"二进制代码生成完成，大小: {len(self.binary)}字节")
    
    def run_inference(self) -> np.ndarray:
        """执行推理并返回结果"""
        if not self.binary or not self.weights:
            raise RuntimeError("请先调用prepare_model()和generate_executable()")
            
        # 生成符合配置的输入数据
        input_data = np.random.rand(*self.config.get_input_shape()).astype(self.config.get_input_dtype())
        
        # 激活所需角色
        self.runtime.activate_roles(["input", "output", "compute", "cache"])
        
        # 加载输入数据和权重
        self.runtime.load_input_data(input_data)
        self.runtime.load_weights(self.weights)
        
        # 执行并返回结果
        result = self.runtime.run(self.binary)
        return result

def main():
    # 配置全局日志
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    # 仅需指定配置文件路径
    config_path = "manycore_config.yaml"
    
    try:
        # 创建流程控制器
        controller = PipelineController(config_path)
        logging.info("流程控制器初始化完成")
        # 准备模型
        controller.prepare_model()
        logging.info("模型准备完成")
        # 生成可执行文件
        controller.generate_executable()
        logging.info("可执行文件生成成功")
        # 执行推理
        result = controller.run_inference()
        logging.info("推理执行完成")
        # 输出结果信息
        valid_result = result[result != 0]
        logging.info(f"推理完成，有效输出大小: {valid_result.shape}")
        logging.info(f"输出示例: {valid_result[:5]}")
        
    except Exception as e:
        logging.error(f"执行出错: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
