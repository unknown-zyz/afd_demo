"""
Main entry point for AFD Demo.

Runs either as attention node or FFN node based on configuration.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings

import torch
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils import logging as hfhub_logging
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from .distributed import init_distributed, get_distributed_context, DistributedConfig
from .model import DisaggregatedQwenModel
from .pipeline import SimplePipelineScheduler, AsyncPipelineScheduler
from .utils.profiler import get_profiler, print_memory_stats

# Suppress transformers warnings by default
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
hfhub_logging.set_verbosity_error()
disable_progress_bars()

# Use Chinese HuggingFace mirror if not set
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
DEFAULT_MODEL_PATH = "/data/Qwen/Qwen3-30B-A3B/"


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Suppress noisy loggers
    for name in [
        "transformers",
        "torch.distributed",
        "huggingface_hub",
        "httpx",
        "httpcore",
        "urllib3",
        "src.distributed",
        "src.model",
        "src.pipeline",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING if verbose else logging.ERROR)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s" if verbose 
               else "%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AFD Demo - Attention-FFN Disaggregation")
    
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_PATH,
                        help="HuggingFace model name or path")
    parser.add_argument("--role", type=str, choices=["attention", "ffn", "auto"], default="auto",
                        help="Node role (auto = determine from rank)")
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 2)),
                        help="Distributed world size (default: 2)")
    parser.add_argument("--rank", type=int, default=None,
                        help="Global rank (optional, overrides role-based default)")
    parser.add_argument("--local-rank", type=int, default=None,
                        help="Local CUDA rank (optional)")
    parser.add_argument("--attn-node-rank", type=int, default=0,
                        help="Global rank used as attention node")
    parser.add_argument("--ffn-node-rank", type=int, default=1,
                        help="Global rank used as ffn node")
    parser.add_argument("--master-addr", type=str, default="localhost",
                        help="Master node address")
    parser.add_argument("--master-port", type=str, default="29500",
                        help="Master node port")
    parser.add_argument("--local-test", action="store_true",
                        help="Run in local test mode with torchrun")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Total batch size (divided into micro-batches)")
    parser.add_argument("--max-seq-len", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--prefill-seq-len", type=int, default=None,
                        help="Prefill sequence length (if set, pad/truncate prompt to this length)")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"],
                        default="bfloat16", help="Data type")
    parser.add_argument("--prompt", type=str, default="Hello, how are you today?",
                        help="Test prompt")
    parser.add_argument("--timing", action="store_true",
                        help="Enable detailed per-MB timing (saves to results/)")
    parser.add_argument("--timing-mode", type=str, choices=["cuda_events", "sync"],
                        default="cuda_events",
                        help="Timing method: cuda_events (zero-overhead, default) or sync (legacy, breaks pipeline)")
    parser.add_argument("--timing-suffix", type=str, default="",
                        help="Suffix for timing file (e.g., 'local_b4_t5' -> timing_attention_local_b4_t5.json)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    # DBO (Dual Batch Overlap) options
    parser.add_argument("--no-dbo", action="store_true",
                        help="Disable DBO for both prefill and decode (AF separation only)")
    parser.add_argument("--num-micro-batches", type=int, default=2,
                        help="Number of micro-batches for DBO pipeline")
    
    # Generation options (enabled by default)
    parser.add_argument("--no-generate", action="store_true",
                        help="Disable autoregressive generation, run prefill only")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of sampling")
    
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]


def build_distributed_config(args) -> DistributedConfig:
    """Create distributed config from args/env."""
    if args.local_test:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    else:
        if args.role == "attention":
            rank = args.attn_node_rank
        elif args.role == "ffn":
            rank = args.ffn_node_rank
        else:
            rank = args.rank if args.rank is not None else int(os.environ.get("RANK", args.attn_node_rank))
        local_rank = args.local_rank if args.local_rank is not None else int(os.environ.get("LOCAL_RANK", 0))
        world_size = args.world_size

    if args.rank is not None:
        rank = args.rank
    if args.local_rank is not None:
        local_rank = args.local_rank

    return DistributedConfig(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://" if args.local_test else f"tcp://{args.master_addr}:{args.master_port}",
        master_addr=args.master_addr,
        master_port=args.master_port,
        attn_node_rank=args.attn_node_rank,
        ffn_node_rank=args.ffn_node_rank,
    )


def run_inference_demo(args):
    """Run the main inference demo."""
    logger = setup_logging(args.verbose)
    
    # Initialize distributed
    config = build_distributed_config(args)
    
    ctx = get_distributed_context()
    ctx.initialize(config)
    
    device = ctx.device
    dtype = get_dtype(args.dtype)
    
    # Log initialization (concise)
    logger.info(f"[{ctx.role.upper()}] rank={ctx.rank}, device={device}, dtype={dtype}")
    
    # Load model
    if args.verbose:
        logger.info(f"Loading model: {args.model_name}")
    model = DisaggregatedQwenModel.from_pretrained(
        args.model_name, device=device, dtype=dtype,
        max_seq_len=args.max_seq_len, max_batch_size=args.batch_size,
    )
    logger.info(
        f"[{ctx.role.upper()}] model_type={model.model_type}, moe={model.is_moe}, "
        f"router={model.has_router}, moe_timing={model.supports_moe_timing}"
    )
    if args.verbose:
        print_memory_stats()
    
    ctx.barrier()
    
    # Tokenizer (attention node only)
    tokenizer = None
    if ctx.is_attention_node:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare input
    if ctx.is_attention_node:
        prompts = [args.prompt] * args.batch_size
        
        # 如果指定了 prefill_seq_len，先 tokenize 然后 pad/truncate
        if args.prefill_seq_len:
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=args.prefill_seq_len)
            logger.info(f"Prefill seq_len set to {args.prefill_seq_len}")
        else:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=args.max_seq_len)
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        logger.info(f"Input shape: {input_ids.shape}")
    else:
        # FFN 节点：创建占位符
        seq_len = args.prefill_seq_len if args.prefill_seq_len else args.max_seq_len
        input_ids = torch.zeros(args.batch_size, seq_len, dtype=torch.long, device=device)
        attention_mask = None
    
    def run_with_scheduler(scheduler):
        """Run inference with timing."""
        ctx.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = scheduler.run(input_ids, attention_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return output, elapsed
    
    # Select scheduler based on --no-dbo flag
    use_dbo = not args.no_dbo
    if use_dbo:
        scheduler = AsyncPipelineScheduler(
            model=model, num_micro_batches=args.num_micro_batches,
            use_cuda_streams=True, enable_timing=args.timing,
            timing_mode=args.timing_mode,
        )
        scheduler_name = "DBO"
    else:
        scheduler = SimplePipelineScheduler(model=model, num_micro_batches=args.num_micro_batches)
        scheduler_name = "SYNC"
    
    output, elapsed = run_with_scheduler(scheduler)
    
    if ctx.is_attention_node:
        logger.info(f"[{scheduler_name}] prefill_time={elapsed*1000:.2f}ms")
        if use_dbo and hasattr(scheduler, 'get_stats'):
            stats = scheduler.get_stats()
            logger.info(f"[DBO] {stats}")
    
    # Save timing data if enabled
    if args.timing and use_dbo:
        timing_data = scheduler.get_timing_data()
        if timing_data:
            os.makedirs("results/prefill_dbo", exist_ok=True)
            # Build timing file name with configuration info
            if args.timing_suffix:
                timing_file = f"results/prefill_dbo/timing_{ctx.role}_{args.timing_suffix}.json"
            else:
                # Auto-generate suffix from config
                suffix = f"b{args.batch_size}_t{args.max_new_tokens}"
                timing_file = f"results/prefill_dbo/timing_{ctx.role}_{suffix}.json"
            timing_data.save(timing_file)
            logger.info(f"Timing saved: {timing_file}")
            logger.info(timing_data.summary())
    elif args.timing and not use_dbo:
        # Save minimal timing for serial baseline
        os.makedirs("results/prefill_dbo", exist_ok=True)
        import json
        serial_data = {
            "mode": "serial",
            "role": ctx.role,
            "total_time_ms": elapsed * 1000,
            "batch_size": args.batch_size,
            "prefill_seq_len": getattr(args, 'prefill_seq_len', 1),
            "max_new_tokens": args.max_new_tokens,
        }
        if args.timing_suffix:
            timing_file = f"results/prefill_dbo/timing_{ctx.role}_{args.timing_suffix}.json"
        else:
            suffix = f"b{args.batch_size}_t{args.max_new_tokens}"
            timing_file = f"results/prefill_dbo/timing_{ctx.role}_serial_{suffix}.json"
        with open(timing_file, 'w') as f:
            json.dump(serial_data, f, indent=2)
        logger.info(f"Timing saved: {timing_file}")
    
    # Output results (attention node only)
    if ctx.is_attention_node and output is not None:
        predicted_ids = output.argmax(dim=-1)
        for i in range(min(args.batch_size, 2)):
            next_token = tokenizer.decode(predicted_ids[i, -1])
            logger.info(f"Output[{i}]: '{args.prompt}' → '{next_token}'")
    
    ctx.barrier()
    ctx.cleanup()


def run_generation_demo(args):
    """Run autoregressive generation demo."""
    logger = setup_logging(args.verbose)
    
    # Initialize distributed
    config = build_distributed_config(args)
    
    ctx = get_distributed_context()
    ctx.initialize(config)
    
    device = ctx.device
    dtype = get_dtype(args.dtype)
    
    logger.info(f"[{ctx.role.upper()}] rank={ctx.rank}, device={device}, dtype={dtype}")
    logger.info(f"Generation mode: max_new_tokens={args.max_new_tokens}, temp={args.temperature}")
    
    # Load model with larger max_seq_len for generation
    max_total_len = args.max_seq_len + args.max_new_tokens
    model = DisaggregatedQwenModel.from_pretrained(
        args.model_name, device=device, dtype=dtype,
        max_seq_len=max_total_len, max_batch_size=args.batch_size,
    )
    logger.info(
        f"[{ctx.role.upper()}] model_type={model.model_type}, moe={model.is_moe}, "
        f"router={model.has_router}, moe_timing={model.supports_moe_timing}"
    )
    
    ctx.barrier()
    
    # Tokenizer
    tokenizer = None
    if ctx.is_attention_node:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare input - support batch prompts (comma-separated or repeat single prompt)
    if ctx.is_attention_node:
        # Parse prompts: if batch_size > 1 and single prompt, repeat it
        prompts = [args.prompt] * args.batch_size
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_seq_len)
        input_ids = inputs["input_ids"].to(device)
        batch_size, prompt_len = input_ids.shape
        
        # Broadcast batch_size and prompt length to FFN node
        meta_tensor = torch.tensor([batch_size, prompt_len], device=device)
        logger.info(f"Input: batch_size={batch_size}, prompt_len={prompt_len}")
    else:
        # Receive metadata from attention node
        meta_tensor = torch.zeros(2, dtype=torch.long, device=device)
    
    import torch.distributed as dist
    dist.broadcast(meta_tensor, src=0)
    batch_size = meta_tensor[0].item()
    prompt_len = meta_tensor[1].item()
    
    if not ctx.is_attention_node:
        # FFN node creates dummy input with correct shape
        input_ids = torch.zeros(batch_size, prompt_len, dtype=torch.long, device=device)
    
    ctx.barrier()
    
    # Generate with DBO setting
    use_dbo = not args.no_dbo
    logger.info(f"Starting generation... (DBO: {use_dbo})")
    start_time = time.perf_counter()
    
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        eos_token_id=tokenizer.eos_token_id if tokenizer else None,
        pad_token_id=tokenizer.pad_token_id if tokenizer else None,
        use_decode_dbo=use_dbo,
        num_decode_micro_batches=args.num_micro_batches,
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_time = time.perf_counter() - start_time
    
    # Output results
    if ctx.is_attention_node:
        num_generated = output_ids.shape[1] - input_ids.shape[1]
        tokens_per_sec = num_generated / gen_time
        
        logger.info(f"Generated {num_generated} tokens in {gen_time*1000:.2f}ms ({tokens_per_sec:.1f} tok/s)")
        
        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"\n=== Generated Text ===\n{output_text}\n{'=' * 22}")
    
    ctx.barrier()
    ctx.cleanup()


def main():
    args = parse_args()
    try:
        if args.no_generate:
            # Prefill only (no generation)
            run_inference_demo(args)
        else:
            # Default: autoregressive generation
            run_generation_demo(args)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
