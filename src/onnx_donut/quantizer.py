from onnxruntime.quantization import quantize_dynamic, quant_pre_process, registry
import os
import shutil


def quantize(src_folder, dst_folder=None):
    # Quantize a float ONNX model to an int8 ONNX model

    # Conv layer quantization is not supported with dynamic quantization
    all_op = registry.CommonOpsRegistry
    all_op.update(registry.IntegerOpsRegistry)
    all_op.update(registry.QDQRegistry)
    all_op.update(registry.QLinearOpsRegistry)
    all_op_but_conv = {k: v for k, v in all_op.items() if k != "Conv"}

    if dst_folder is None:
        dst_folder = src_folder + "_quant"

    shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

    for file in ["encoder", "decoder", "decoder_with_past"]:
        print("Quantizing "+file+"...")
        src = os.path.join(dst_folder, file + '.onnx')
        dst = os.path.join(dst_folder, file + '.onnx')
        quant_pre_process(src, src)
        quantize_dynamic(src, dst, op_types_to_quantize=list(all_op_but_conv.keys()))
        print("Done.")
