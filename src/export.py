import argparse
import onnx.checker
import torch

from config import get_config
from models import build_model

import onnx
import onnxsim

# PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def export_norm_onnx(model, file, input):
    torch.onnx.export(
        model = model,
        args = (input,),
        f = file,
        input_names = ["input0"],
        output_names = ["output0"],
        opset_version = 12
    )
    print("Finished normal onnx export")
    model_onnx = onnx.load(file)
    #检查导入的onnx
    onnx.checker.check_model(model_onnx)
    #使用onnx-simplifier进行简化
    print(f"Simplifying with onnxsim {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)

def main(config):
    model = build_model(config)
    input = torch.rand(1, 3, 224, 224)
    model.eval()
    export_norm_onnx(model, "../models/swin-tiny-after-simplify-opset12.onnx", input)

if __name__ == '__main__':
    args, config = parse_option()
    main(config)

#run command:
# python export.py --eval --cfg configs/swin/swin_base_patch4_window7_224.yaml --resume ../weights/swin_tiny_patch4_window7_224.pth --data-path data/ --local_ran 0

#Error: RuntimeError: Exporting the operator roll to ONNX opset version 12 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub.
#Solution:
# opset12中并没有加入roll的算子
# cd ~/miniconda3/envs/swin/lib/python3.7/site-packages/torch/onnx
# code symbolic_opset12.py

# #==========add roll =================
# from sys import maxsize as maxsize

# @parse_args('v', 'is', 'is')
# def roll(g, self, shifts, dims):
#     assert len(shifts) == len(dims)

#     result = self
#     for i in range(len(shifts)):
#         shapes = []
#         shape = sym_help._slice_helper(
#             g,
#             result,
#             axes=[dims[i]],
#             starts=[-shifts[i]],
#             ends=[maxsize]
#         )
#         shapes.append(shape)
#         shape = sym_help._slice_helper(
#             g,
#             result,
#             axes=[dims[i]],
#             starts=[0],
#             ends=[-shifts[i]]
#         )
#         shapes.append(shape)
#         result = g.op("Concat", *shapes, axis_i=dims[i])
    
#     return result