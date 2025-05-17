import torch
from onnx2torch import convert
from transformers import AutoTokenizer,AutoModel
import os
import shutil


def copy_files_excluding_safetensors(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_root) if rel_root != "." else dst_dir
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            if file.endswith(".onnx") or file.endswith(".onnx_data"):
                continue
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_root, file)
            shutil.copy2(src_file, dst_file)

onnx_model_directory = "../models/onnx-models/bert-base-cased-qa-onnx/"

onnx_model_path = "../models/onnx-models/bert-base-cased-qa-onnx/model.onnx"
torch_model = convert(onnx_model_path)

torch.save(torch_model.state_dict(), "bert-base-cased-qa/pytorch_model.bin")

copy_files_excluding_safetensors(onnx_model_directory,"bert-base-cased-qa/")


