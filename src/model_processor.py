import tvm
import tvm.relax as relax  # 替换原来的relay导入
import numpy as np
import logging
import onnx  # 添加缺失的onnx导入
import onnx.checker
from typing import Tuple, Dict, List
from tvm.relax.frontend.onnx import from_onnx

class ModelProcessor:
    """模型处理类,负责ONNX模型的加载、转换和分析(兼容TVM Relay)"""
    
    def __init__(self, config):
        """初始化模型处理器
        
        Args:
            config: 众核架构配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_and_convert(self) -> Tuple[tvm.IRModule, str, onnx.ModelProto]:
        """加载ONNX模型并转换为TVM Relay IR
        
        Returns:
            tvm.IRModule: TVM Relay中间表示
            str: 输入名称
            onnx.ModelProto: ONNX模型对象
        """
        # 加载ONNX模型
        onnx_path = self.config.get_onnx_path()
        self.logger.info(f"加载ONNX模型: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        self.logger.info("ONNX模型格式验证通过")
        
        # 获取输入名称和形状
        input_name = onnx_model.graph.input[0].name
        input_shape = self.config.get_input_shape()
        self.logger.info(f"模型输入名称: {input_name}, 形状: {input_shape}")
        
        # 转换为Relay IR（兼容大多数TVM版本）
        shape_dict = {input_name: input_shape}
        try:
            mod = from_onnx(onnx_model, shape_dict={input_name: input_shape}, keep_params_in_input=False, opset=13)
            if not isinstance(mod, tvm.IRModule):
                mod = tvm.IRModule.from_expr(mod)
        except Exception as e:
            self.logger.warning(f"opset=13转换失败,尝试opset=11: {str(e)}")
            mod = from_onnx(onnx_model, shape_dict={input_name: input_shape}, keep_params_in_input=False, opset=11)
            if not isinstance(mod, tvm.IRModule):
                mod = tvm.IRModule.from_expr(mod)
        
        # 应用优化转换
        self.logger.info("优化Relay IR...")
        with tvm.transform.PassContext(opt_level=3):
           mod = relax.transform.ToNonDataflow()(mod)
           mod = relax.transform.FoldConstant()(mod)
           mod = relax.transform.FuseOps()(mod)
           mod = relax.transform.RemoveUnusedOutputs()(mod)
           mod = relax.transform.RemoveUnusedParameters()(mod)
           mod = relax.transform.CallTIRRewrite()(mod)
           mod = relax.transform.LegalizeOps()(mod)

        return mod, input_name, onnx_model
    
    def extract_weights(self, onnx_model: onnx.ModelProto) -> Dict[str, np.ndarray]:
        """从ONNX模型中提取权重参数"""
        self.logger.info("从ONNX模型提取权重...")
        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = onnx.numpy_helper.to_array(init)
        self.logger.info(f"提取完成，共 {len(weights)} 个权重参数")
        return weights
    
    def analyze_layers(self, mod: tvm.IRModule) -> List[Tuple[str, str]]:
        """分析模型层类型（基于Relay IR）"""
        self.logger.info("分析模型层结构...")
        layers = []
        
        # 遍历Relay函数
        for var in mod.get_global_vars():
            func = mod[var]
            layer_name = var.name_hint
            
            # 分析算子类型
            if "conv2d" in layer_name.lower():
                layers.append((layer_name, "conv2d"))
            elif "conv3d" in layer_name.lower():
                layers.append((layer_name, "conv3d"))
            elif "relu" in layer_name.lower() or "sigmoid" in layer_name.lower():
                layers.append((layer_name, "ann_activation"))
            elif "dense" in layer_name.lower() or "fc" in layer_name.lower():
                layers.append((layer_name, "fully_connected"))
            elif "pool" in layer_name.lower():
                layers.append((layer_name, "pool"))
            elif "dot" in layer_name.lower():
                layers.append((layer_name, "vector_dot"))
            elif "multiply" in layer_name.lower():
                layers.append((layer_name, "vector_multiply"))
            elif "split" in layer_name.lower():
                layers.append((layer_name, "vector_split"))
            elif "concat" in layer_name.lower():
                layers.append((layer_name, "vector_merge"))
            else:
                self.logger.warning(f"未识别的层类型: {layer_name}")
                layers.append((layer_name, "unknown"))
        
        self.logger.info(f"模型层分析完成，共 {len(layers)} 层")
        return layers
    
    def get_layer_input_output_shapes(self, onnx_model: onnx.ModelProto) -> Dict[str, Dict[str, List[int]]]:
        """获取各层的输入输出形状"""
        layer_shapes = {}
        
        # 处理输入层
        for input_tensor in onnx_model.graph.input:
            layer_name = input_tensor.name
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            layer_shapes[layer_name] = {
                "input": None,
                "output": shape
            }
        
        # 处理中间层
        for node in onnx_model.graph.node:
            layer_name = node.name or node.output[0]  # 处理无名称节点
                
            # 获取输入形状
            input_shapes = []
            for input_name in node.input:
                for tensor in onnx_model.graph.value_info + onnx_model.graph.input + onnx_model.graph.output:
                    if tensor.name == input_name and hasattr(tensor.type, 'tensor_type'):
                        shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]
                        input_shapes.append(shape)
                        break
            
            # 获取输出形状
            output_shapes = []
            for output_name in node.output:
                for tensor in onnx_model.graph.value_info + onnx_model.graph.output:
                    if tensor.name == output_name and hasattr(tensor.type, 'tensor_type'):
                        shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]
                        output_shapes.append(shape)
                        break
            
            layer_shapes[layer_name] = {
                "input": input_shapes[0] if input_shapes else None,
                "output": output_shapes[0] if output_shapes else None
            }
            
        return layer_shapes
