import os
import shutil
import torch
import argparse
from onnx2torch import convert

def copy_files_excluding_onnx(src_dir, dst_dir):
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


def main():
    parser = argparse.ArgumentParser(description="Load Onnx model and save as a pytorch_model.bin")
    parser.add_argument("--input_onnx_dir",type=str,required=True,help="Path to the input ONNX model directory")
    parser.add_argument("--output_dir",type=str,required=True,help=" Output Directory to save the pytorch_model.bin file")
    args = parser.parse_args()

    print(f"Loading model from {args.input_onnx_dir} ...")
    input_model_path = os.path.join(args.input_onnx_dir,"model.onnx")

    torch_model = convert(input_model_path)

    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, "pytorch_model.bin")
    print(f"Saving weights to {output_path} ...")

    torch.save(torch_model, output_path)  ##changed from state_dict only to full model
    copy_files_excluding_onnx(args.input_onnx_dir,args.output_dir)

    model = torch.load(output_path,weights_only=False)

    with open("model_out.txt","w") as file:
        file.write(str(model))
    
    print("Export Completed!")

if __name__ == "__main__":
    main()

