import argparse
import os
import shutil
import torch
from transformers import AutoModel

def copy_files_excluding_safetensors(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_root) if rel_root != "." else dst_dir
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            if file.endswith(".safetensors"):
                continue
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_root, file)
            shutil.copy2(src_file, dst_file)


def main():
    parser = argparse.ArgumentParser(description="Merge HF model shards and save as pytorch_model.bin")
    parser.add_argument("--input_dir",type=str,required=True,help="Path to the input Hugging Face model directory")
    parser.add_argument("--output_dir",type=str,required=True,help=" Output Directory to save the pytorch_model.bin file")
    args = parser.parse_args()

    print(f"Loading model from {args.input_dir} ...")
    model = AutoModel.from_pretrained(args.input_dir,torch_dtype= "auto",device_map="auto")

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "pytorch_model.bin")
    print(f"Saving merged weights to {output_path} ...")

    torch.save(model, output_path) #changed from state_dict to full model save
    copy_files_excluding_safetensors(args.input_dir, args.output_dir)

    model = torch.load(output_path,weights_only=False)

    with open("model.txt","w") as file:
        file.write(str(model))

    print("Export Completed!")


if __name__ == "__main__":
    main()
