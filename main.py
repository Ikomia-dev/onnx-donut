from onnx_donut.exporter import export_onnx
from onnx_donut.predictor import OnnxPredictor
from onnx_donut.quantizer import quantize
import numpy as np
from PIL import Image

# Hugging Face model card or folder
model_path = "naver-clova-ix/donut-base-finetuned-docvqa"

# Image path to run on
img_path = "/path/to/your/image.png"

# Folder where the exported model will be stored
dst_folder = "converted_donut"

# Export from Pytorch to ONNX
export_onnx(model_path, dst_folder, opset_version=16)

# Quantize your model to int8
quantize(dst_folder, dst_folder)

# Read image
img = np.array(Image.open(img_path).convert('RGB'))

# Instantiate ONNX predictor
predictor = OnnxPredictor(model_folder=dst_folder)

# Write your prompt accordingly to the model you use
prompt = f"<s_docvqa><s_question>what is the title?</s_question><s_answer>"

# Run prediction
out = predictor.generate(img, prompt)

# Display prediction
print(out)
