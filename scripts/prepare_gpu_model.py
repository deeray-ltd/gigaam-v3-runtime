#!/usr/bin/env python3
"""Offline GigaAM export and TensorRT EPContext preparation."""
import argparse
from pathlib import Path
import subprocess


def profile_shape(value):
    try:
        name, minimum, optimum, maximum = value.split(":")
        shapes = tuple(tuple(int(dim) for dim in shape.split("x")) for shape in (minimum, optimum, maximum))
        if not name or not shapes[0] or any(dim <= 0 for shape in shapes for dim in shape):
            raise ValueError
        if len({len(shape) for shape in shapes}) != 1:
            raise ValueError
        if any(not low <= mid <= high for low, mid, high in zip(*shapes)):
            raise ValueError
        return name, shapes
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME:MIN:OPT:MAX with positive x-separated dimensions") from error


def build_python(args):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    if not args.tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    if args.shape:
        profile = builder.create_optimization_profile()
        names = set()
        for name, (minimum, optimum, maximum) in args.shape:
            if name in names:
                raise ValueError(f"duplicate profile input: {name}")
            names.add(name)
            profile.set_shape(name, minimum, optimum, maximum)
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError("TensorRT rejected the optimization profile")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    args.engine.write_bytes(bytes(engine))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="export freshly loaded GigaAM weights")
    export.add_argument("source", help="upstream model name or supported checkpoint")
    export.add_argument("output", type=Path)
    export.add_argument("--dtype", choices=("fp32", "fp16"), required=True)
    export.add_argument("--device", default="cuda")
    export.add_argument("--download-root")
    build = commands.add_parser("build", help="build a TensorRT engine with explicit trtexec options")
    build.add_argument("onnx", type=Path)
    build.add_argument("engine", type=Path)
    build.add_argument("trtexec_options", nargs=argparse.REMAINDER, help="trtexec shape and precision options")
    python_build = commands.add_parser("build-python", help="build a strongly typed TensorRT engine using Python")
    python_build.add_argument("onnx", type=Path)
    python_build.add_argument("engine", type=Path)
    python_build.add_argument("--shape", type=profile_shape, action="append", default=[], metavar="NAME:MIN:OPT:MAX", help="repeat for dynamic inputs; dimensions use x separators")
    python_build.add_argument("--tf32", action="store_true", help="permit TF32 for FP32 operations")
    wrap = commands.add_parser("wrap", help="wrap an existing engine using the ONNX graph interface")
    wrap.add_argument("onnx", type=Path)
    wrap.add_argument("engine", type=Path)
    wrap.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        import gigaam
        import torch
        args.output.mkdir(parents=True, exist_ok=True)
        model = gigaam.load_model(args.source, fp16_encoder=False, device=args.device, download_root=args.download_root)
        model.to_onnx(str(args.output), dtype={"fp32": torch.float32, "fp16": torch.float16}[args.dtype])
    elif args.command == "build":
        subprocess.run(["trtexec", f"--onnx={args.onnx}", f"--saveEngine={args.engine}", *args.trtexec_options], check=True)
    elif args.command == "build-python":
        build_python(args)
    else:
        import onnx
        from onnx import helper
        if args.engine.resolve().parent != args.output.resolve().parent:
            parser.error("engine and wrapper must share a directory")
        graph = onnx.load(str(args.onnx), load_external_data=False).graph
        node = helper.make_node("EPContext", [item.name for item in graph.input], [item.name for item in graph.output], domain="com.microsoft", embed_mode=0, ep_cache_context=args.engine.name, source="TensorrtExecutionProvider", main_context=1)
        wrapped = helper.make_model(helper.make_graph([node], "TensorRT", list(graph.input), list(graph.output)), opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)])
        onnx.save(wrapped, str(args.output))


if __name__ == "__main__":
    main()
