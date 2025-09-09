import tvm
import tvm.relax as relax
import numpy as np
import logging
import onnx
import onnx.checker
from typing import Tuple, Dict, List, Optional
from tvm.relax.frontend.onnx import from_onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

class ModelProcessor:
    """模型处理类,负责ONNX模型的加载、转换和分析(兼容TVM Relax)"""
    
    def __init__(self, config):
        """初始化模型处理器
        
        Args:
            config: 众核架构配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        # 层类型映射表，与原语保持一致
        self.layer_type_map = {
            "conv2d": "conv2d",
            "conv3d": "conv3d",
            "pool": "vector_accumulate",
            "ann_activation": "ann_activation",
            "fully_connected": "fully_connected",
            "vector_dot": "vector_dot",
            "vector_multiply": "vector_multiply",
            "vector_split": "vector_split",
            "vector_merge": "vector_merge",
            "tensor_reshape": "vector_scale",  # reshape可映射到缩放原语
            "vector_max": "vector_accumulate",  # max可复用累加原语
            "vector_min": "vector_accumulate",  # min可复用累加原语
            "vector_round": "lookup_table",     # round可通过LUT实现
            "batch_norm": "vector_scale",       # 批归一化映射到缩放
            "snn_activation": "snn_activation"  # SNN激活函数
        }
        
    def load_and_convert(self, convert_format=True, quantize_to_uint8=False) -> Tuple[tvm.IRModule, str, onnx.ModelProto]:
        """加载ONNX模型并转换为TVM Relax IR
        
        Args:
            convert_format: 是否将NCHW格式转换为NCWH格式
            quantize_to_uint8: 是否将模型量化为UINT8格式
        
        Returns:
            tvm.IRModule: TVM Relax中间表示
            str: 输入名称
            onnx.ModelProto: ONNX模型对象
        """
        # 加载ONNX模型
        onnx_path = self.config.get_onnx_path()
        self.logger.info(f"加载ONNX模型: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        self.logger.info("ONNX模型格式验证通过")
        
        # 如果需要量化为UINT8
        if quantize_to_uint8:
            onnx_model = self.quantize_to_uint8(onnx_model)
        
        # 获取输入名称和形状
        input_name = onnx_model.graph.input[0].name
        input_shape = self.config.get_input_shape()
        self.logger.info(f"模型输入名称: {input_name}, 形状: {input_shape}")
        
        # 转换为Relax IR（兼容大多数TVM版本）
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
        self.logger.info("优化Relax IR...")
        with tvm.transform.PassContext(opt_level=3):
           mod = relax.transform.ToNonDataflow()(mod)
           mod = relax.transform.FoldConstant()(mod)
           mod = relax.transform.FuseOps()(mod)
           mod = relax.transform.RemoveUnusedOutputs()(mod)
           mod = relax.transform.RemoveUnusedParameters()(mod)
           mod = relax.transform.CallTIRRewrite()(mod)
           mod = relax.transform.LegalizeOps()(mod)

        # 如果需要转换格式
        if convert_format:
            mod = self.transform_nchw_to_ncwh(mod)

        return mod, input_name, onnx_model
    
    def quantize_to_uint8(self, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """将ONNX模型量化为UINT8格式
        
        Args:
            onnx_model: 原始ONNX模型
        
        Returns:
            onnx.ModelProto: 量化后的ONNX模型
        """
        self.logger.info("开始量化ONNX模型为UINT8格式...")
        
        # 保存原始模型到临时文件
        temp_path = "temp_original.onnx"
        quantized_path = "temp_quantized_uint8.onnx"
        onnx.save(onnx_model, temp_path)
        
        # 使用onnxruntime进行动态量化
        quantize_dynamic(
            temp_path,
            quantized_path,
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=True
        )
        
        # 加载量化后的模型
        quantized_model = onnx.load(quantized_path)
        onnx.checker.check_model(quantized_model)
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(quantized_path):
            os.remove(quantized_path)
        
        self.logger.info("ONNX模型量化为UINT8格式完成")
        return quantized_model
    
    def transform_nchw_to_ncwh(self, mod: tvm.IRModule) -> tvm.IRModule:
        """将模型中的NCHW格式转换为NCWH格式
        
        Args:
            mod: TVM IRModule对象
        
        Returns:
            tvm.IRModule: 转换后的IRModule对象
        """
        self.logger.info("开始将NCHW格式转换为NCWH格式...")
        
        # 定义转置变换函数，将NCHW(0,1,2,3)转换为NCWH(0,1,3,2)
        def transform_layout(func):
            if isinstance(func, relax.Function):
                # 处理Relax函数中的所有表达式
                updated_body = self._recursive_transform(func.body)
                return func.with_body(updated_body)
            return func
        
        # 应用布局转换到模型中的所有函数
        for gv in mod.get_global_vars():
            mod[gv] = transform_layout(mod[gv])
        
        self.logger.info("NCHW格式转换为NCWH格式完成")
        return mod
    
    def _recursive_transform(self, expr):
        """递归处理表达式，执行布局转换"""
        # 处理Call节点中的算子
        if isinstance(expr, relax.Call) and hasattr(expr.op, "name"):
            op_name = expr.op.name.lower()
            # 扩展需要处理的算子类型
            target_ops = {"conv2d", "conv3d", "pool", "max_pool2d", "avg_pool2d", 
                          "batch_norm", "relu", "sigmoid", "tanh"}
            
            if any(op in op_name for op in target_ops):
                # 检查输入形状是否是4D(NCHW)
                if len(expr.args[0].struct_info.shape) == 4:
                    # 构建转置操作，将NCHW(0,1,2,3)转置为NCWH(0,1,3,2)
                    transposed = relax.op.transpose(expr.args[0], axes=[0, 1, 3, 2])
                    # 替换原始参数
                    new_args = list(expr.args)
                    new_args[0] = transposed
                    # 构建新的调用
                    new_call = relax.Call(expr.op, new_args, expr.attrs, expr.span)
                    # 对结果进行反向转置，保持输出格式一致
                    result_transposed = relax.op.transpose(new_call, axes=[0, 1, 3, 2])
                    return result_transposed
        
        # 递归处理子表达式
        if hasattr(expr, "body"):
            expr = expr.with_body(self._recursive_transform(expr.body))
        if hasattr(expr, "then_branch"):
            expr = expr.with_then_branch(self._recursive_transform(expr.then_branch))
        if hasattr(expr, "else_branch"):
            expr = expr.with_else_branch(self._recursive_transform(expr.else_branch))
        if hasattr(expr, "args") and isinstance(expr.args, list):
            new_args = []
            for arg in expr.args:
                new_args.append(self._recursive_transform(arg))
            expr = expr.replace(args=new_args)
        
        return expr
    
    def extract_weights(self, onnx_model: onnx.ModelProto) -> Dict[str, np.ndarray]:
        """从ONNX模型中提取权重参数"""
        self.logger.info("从ONNX模型提取权重...")
        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = onnx.numpy_helper.to_array(init)
        self.logger.info(f"提取完成，共 {len(weights)} 个权重参数")
        return weights
    
    def analyze_layers(self, mod: tvm.IRModule) -> List[Tuple[str, str]]:
        """分析模型层类型（基于Relax IR）"""
        self.logger.info("分析模型层结构...")
        layers = []
        
        # 遍历Relax函数
        for var in mod.get_global_vars():
            func = mod[var]
            layer_name = var.name_hint.lower()  # 转为小写统一处理
            
            # 扩展层类型识别逻辑
            if "conv2d" in layer_name:
                layers.append((layer_name, "conv2d"))
            elif "conv3d" in layer_name:
                layers.append((layer_name, "conv3d"))
            elif "relu" in layer_name or "sigmoid" in layer_name or "tanh" in layer_name:
                layers.append((layer_name, "ann_activation"))
            elif "dense" in layer_name or "fc" in layer_name:
                layers.append((layer_name, "fully_connected"))
            elif "pool" in layer_name:
                layers.append((layer_name, "pool"))
            elif "dot" in layer_name or "matmul" in layer_name:
                layers.append((layer_name, "vector_dot"))
            elif "multiply" in layer_name:
                layers.append((layer_name, "vector_multiply"))
            elif "split" in layer_name:
                layers.append((layer_name, "vector_split"))
            elif "concat" in layer_name:
                layers.append((layer_name, "vector_merge"))
            elif "reshape" in layer_name:
                layers.append((layer_name, "tensor_reshape"))
            elif "maximum" in layer_name:
                layers.append((layer_name, "vector_max"))
            elif "minimum" in layer_name:
                layers.append((layer_name, "vector_min"))
            elif "round" in layer_name:
                layers.append((layer_name, "vector_round"))
            elif "batch_norm" in layer_name:
                layers.append((layer_name, "batch_norm"))
            elif "lif" in layer_name or "spike" in layer_name:
                layers.append((layer_name, "snn_activation"))
            else:
                # 尝试从函数内部分析未识别的层
                inferred_type = self._infer_layer_type_from_func(func)
                if inferred_type:
                    layers.append((layer_name, inferred_type))
                else:
                    self.logger.warning(f"未识别的层类型: {layer_name}")
                    layers.append((layer_name, "unknown"))
        
        self.logger.info(f"模型层分析完成，共 {len(layers)} 层")
        return layers
    
    def _infer_layer_type_from_func(self, func: relax.Function) -> Optional[str]:
        """从函数内部推断层类型"""
        # 简单的算子类型推断逻辑
        if hasattr(func, "body"):
            expr = func.body
            if isinstance(expr, relax.Call) and hasattr(expr.op, "name"):
                op_name = expr.op.name.lower()
                if "matmul" in op_name:
                    return "vector_dot"
                elif "max" in op_name:
                    return "vector_max"
                elif "min" in op_name:
                    return "vector_min"
                elif "reshape" in op_name:
                    return "tensor_reshape"
        return None
    
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
        
        # 处理中间层和输出层
        all_tensors = list(onnx_model.graph.value_info) + list(onnx_model.graph.input) + list(onnx_model.graph.output)
        tensor_map = {t.name: t for t in all_tensors}
        
        for node in onnx_model.graph.node:
            layer_name = node.name or node.output[0]  # 处理无名称节点
                
            # 获取输入形状
            input_shapes = []
            for input_name in node.input:
                if input_name in tensor_map and hasattr(tensor_map[input_name].type, 'tensor_type'):
                    shape = [dim.dim_value for dim in tensor_map[input_name].type.tensor_type.shape.dim]
                    input_shapes.append(shape)
            
            # 获取输出形状
            output_shapes = []
            for output_name in node.output:
                if output_name in tensor_map and hasattr(tensor_map[output_name].type, 'tensor_type'):
                    shape = [dim.dim_value for dim in tensor_map[output_name].type.tensor_type.shape.dim]
                    output_shapes.append(shape)
            
            layer_shapes[layer_name] = {
                "input": input_shapes[0] if input_shapes else None,
                "output": output_shapes[0] if output_shapes else None
            }
        
        # 处理输出层
        for output_tensor in onnx_model.graph.output:
            layer_name = output_tensor.name
            if layer_name not in layer_shapes:
                shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                layer_shapes[layer_name] = {
                    "input": None,
                    "output": shape
                }
            
        return layer_shapes
    
    def get_primitive_for_layer(self, layer_type: str) -> str:
        """获取层类型对应的原语
        
        Args:
            layer_type: 层类型字符串
            
        Returns:
            对应的原语名称
        """
        return self.layer_type_map.get(layer_type, "unknown")

    def validate_layer_mapping(self, layer_mapping: Dict[str, List[int]]) -> bool:
        """验证层到核心的映射是否有效
        
        Args:
            layer_mapping: 层到核心的映射字典
            
        Returns:
            bool: 映射是否有效
        """
        compute_cores = set(self.config.get_core_ids_by_role("compute"))
        valid = True
        
        for layer_name, core_ids in layer_mapping.items():
            for cid in core_ids:
                if cid not in compute_cores:
                    self.logger.error(f"层 {layer_name} 映射到无效的计算核心 {cid}")
                    valid = False
        
        return valid
