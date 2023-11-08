from setuptools import setup

setup(name="onnx_donut",
      version="0.1.0",
      dependency_links=["https://download.pytorch.org/whl/torch_stable.html"],
      install_requires=[
          "transformers >= 4.26.0, <=4.30.0",
          "torch==1.13.1+cu116; python_version >= '3.10'",
          "torchvision==0.14.1+cu116; python_version >= '3.10'",
          "torch==1.9.0+cu111; python_version < '3.10'",
          "torchvision==0.10.0+cu111; python_version < '3.10'",
          "timm==0.5.4",
          "donut-python==1.0.9",
          "onnx",
          "onnxruntime"
      ])
