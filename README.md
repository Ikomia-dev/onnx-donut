# Onnx Donut

Package to export a [Donut](https://github.com/clovaai/donut) model from pytorch to ONNX, then run it with onnxruntime.

## Installation

```shell
pip install onnx-donut
```

## Export to onnx

```python
from onnx_donut.exporter import export_onnx

# Hugging Face model card or folder
model_path = "naver-clova-ix/donut-base-finetuned-docvqa"

# Folder where the exported model will be stored
dst_folder = "converted_donut"

# Export from Pytorch to ONNX
export_onnx(model_path, dst_folder, opset_version=16)
```

## Model inference with onnxruntime

```python
from onnx_donut.predictor import OnnxPredictor
import numpy as np
from PIL import Image
import onnxruntime

# Image path to run on
img_path = "/path/to/your/image.png"

# Folder where the exported model will be stored
onnx_folder = "converted_donut"

# Read image
img = np.array(Image.open(img_path).convert('RGB'))

# Avoid increase of memory usage between inferences
options = onnxruntime.SessionOptions()
options.enable_mem_pattern = False

# Instantiate ONNX predictor
predictor = OnnxPredictor(model_folder=onnx_folder, sess_options=options)

# Write your prompt accordingly to the model you use
prompt = f"<s_docvqa><s_question>what is the title?</s_question><s_answer>"

# Run prediction
out = predictor.generate(img, prompt)

# Display prediction
print(out)
```