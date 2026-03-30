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
    parser.add_argument("--num-micro-batches", type=int, default=2,
                        help="Number of micro-batches for pipeline")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"],
                        default="bfloat16", help="Data type")
    parser.add_argument("--prompt", type=str, default="Hello, how are you today?",
                        help="Test prompt")
    parser.add_argument("--scheduler", type=str, choices=["sync", "async", "compare"],
                        default="async", help="sync | async (DBO) | compare")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Warmup runs before timing")
    parser.add_argument("--benchmark-runs", type=int, default=3,
                        help="Benchmark runs for timing")
    parser.add_argument("--timing", action="store_true",
                        help="Enable detailed per-MB timing (saves to results/)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    # Generation options
    parser.add_argument("--generate", action="store_true",
                        help="Enable autoregressive generation (requires KV cache)")
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
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_seq_len)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
    else:
        input_ids = torch.zeros(args.batch_size, args.max_seq_len, dtype=torch.long, device=device)
        attention_mask = None
    
    # Create schedulers
    def create_sync_scheduler():
        return SimplePipelineScheduler(model=model, num_micro_batches=args.num_micro_batches)
    
    def create_async_scheduler():
        return AsyncPipelineScheduler(
            model=model, num_micro_batches=args.num_micro_batches,
            use_cuda_streams=True, enable_timing=args.timing,
        )
    
    def run_with_scheduler(scheduler, warmup=0, runs=1):
        """Run inference with timing."""
        for _ in range(warmup):
            ctx.barrier()
            scheduler.run(input_ids, attention_mask)
            ctx.barrier()
        
        times = []
        for _ in range(runs):
            ctx.barrier()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = scheduler.run(input_ids, attention_mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        return output, sum(times) / len(times), times
    
    # Run based on mode
    if args.scheduler == "sync":
        scheduler = create_sync_scheduler()
        output, avg_time, _ = run_with_scheduler(scheduler, args.warmup_runs, args.benchmark_runs)
        if ctx.is_attention_node:
            logger.info(f"[SYNC] avg_time={avg_time*1000:.2f}ms")
        
    elif args.scheduler == "async":
        scheduler = create_async_scheduler()
        output, avg_time, _ = run_with_scheduler(scheduler, args.warmup_runs, args.benchmark_runs)
        if ctx.is_attention_node:
            stats = scheduler.get_stats()
            logger.info(f"[DBO] {stats}")
        
        # Save timing data if enabled
        if args.timing:
            timing_data = scheduler.get_timing_data()
            if timing_data:
                os.makedirs("results", exist_ok=True)
                timing_file = f"results/timing_{ctx.role}.json"
                timing_data.save(timing_file)
                logger.info(f"Timing saved: {timing_file}")
                logger.info(timing_data.summary())
        
    else:  # compare
        sync_scheduler = create_sync_scheduler()
        sync_output, sync_time, _ = run_with_scheduler(sync_scheduler, args.warmup_runs, args.benchmark_runs)
        
        async_scheduler = create_async_scheduler()
        output, async_time, _ = run_with_scheduler(async_scheduler, args.warmup_runs, args.benchmark_runs)
        
        if ctx.is_attention_node:
            speedup = sync_time / async_time if async_time > 0 else 1.0
            logger.info("=" * 40)
            logger.info("DBO Comparison:")
            logger.info(f"  Sync:    {sync_time*1000:.2f}ms")
            logger.info(f"  Async:   {async_time*1000:.2f}ms")
            logger.info(f"  Speedup: {speedup:.2f}x")
            logger.info("=" * 40)
        
        # Save timing if enabled
        if args.timing:
            timing_data = async_scheduler.get_timing_data()
            if timing_data:
                os.makedirs("results", exist_ok=True)
                timing_data.save(f"results/timing_{ctx.role}.json")
    
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
    
    # Prepare input
    if ctx.is_attention_node:
        inputs = tokenizer(args.prompt, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_seq_len)
        input_ids = inputs["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        # Broadcast prompt length to FFN node
        prompt_len_tensor = torch.tensor([prompt_len], device=device)
    else:
        # Receive prompt length from attention node
        prompt_len_tensor = torch.zeros(1, dtype=torch.long, device=device)
    
    import torch.distributed as dist
    dist.broadcast(prompt_len_tensor, src=0)
    prompt_len = prompt_len_tensor.item()
    
    if not ctx.is_attention_node:
        # FFN node creates dummy input with correct shape
        input_ids = torch.zeros(1, prompt_len, dtype=torch.long, device=device)
    
    ctx.barrier()
    
    # Generate
    logger.info("Starting generation...")
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
        if args.generate:
            run_generation_demo(args)
        else:
            run_inference_demo(args)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
