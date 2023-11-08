from onnx_donut.exporter import export_onnx
from onnx_donut.predictor import OnnxPredictor
import numpy as np
from PIL import Image
import onnxruntime

# Hugging Face model card or folder
model_path = "naver-clova-ix/donut-base-finetuned-docvqa"

# Image path to run on
img_path = "/path/to/your/image.png"

# Folder where the exported model will be stored
dst_folder = "converted_donut"

# Export from Pytorch to ONNX
export_onnx(model_path, dst_folder, opset_version=16)

# Read image
img = np.array(Image.open(img_path))

# Avoid increase of memory usage between inferences
options = onnxruntime.SessionOptions()
options.enable_mem_pattern = False

# Instantiate ONNX predictor
predictor = OnnxPredictor(export_folder=dst_folder, sess_options=options)

# Write your prompt
# Adapt it based on the model you use
prompt = f"<s_docvqa><s_question>what is the title?</s_question><s_answer>"

# Run prediction
out = predictor.generate(img, prompt)

# Display prediction
print(out)
