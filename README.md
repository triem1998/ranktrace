# ranktrace

Per-rank profiling for distributed PyTorch training on NVIDIA and AMD GPUs. ranktrace only
captures one trace per rank; the analysis is done by existing tools:
[HTA](https://github.com/facebookresearch/HolisticTraceAnalysis),
[TraceLens](https://github.com/AMD-AGI/TraceLens) and nsys recipes.

## Install

```
pip install "ranktrace @ git+https://github.com/triem1998/ranktrace.git"
```

nsys itself comes from the system (or a cluster module), not from pip.

## Capture

```python
from ranktrace import make_profiler

with make_profiler("torch", trace_dir="traces", wait=5, warmup=1, active=2) as prof:
    for batch in loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()  # once per batch
```

- `mode`: `None` (off, `Profiler`), `"torch"` (`TorchProfiler`, writes `rank{r}.pt.trace.json`
  per rank), `"nvtx"` (`NvtxProfiler`, one `step` NVTX range per recorded batch, for nsys).
- `wait`, `warmup`, `active`: as in `torch.profiler.schedule`. With gradient accumulation over A
  batches, make `wait + warmup + active` a multiple of A so the last recorded batch has the
  gradient all-reduce and the optimizer step.
- `region(name)`: optional section (e.g. forward, backward). torch traces already label the step,
  the DataLoader and the optimizer. Inside a region autograd runs on the calling thread.
- `record_shapes=True`: torch mode only.
- `analyze=True` (default, torch mode only): at the end of the run, rank 0 runs HTA and TraceLens
  on all ranks' traces and prints the time of each. Meanwhile the job keeps its GPUs, and HTA needs
  about 5x the total trace size in RAM; for large runs set `analyze=False` and analyze later.
- `stats=True`: writes `stats/rank{r}.json` (step times, peak GPU / host memory, trace size).

`nvtx` mode runs under nsys. `--capture-range-end=stop` is required: the default
(`stop-shutdown`) kills the application when the capture ends.

```
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -o traces/rank0 python train.py
```

## Analyze

In torch mode HTA and TraceLens run automatically (`analyze=True`). By hand:

```
python -m ranktrace.analyze traces --tool hta        # torch traces
python -m ranktrace.analyze traces --tool tracelens  # torch traces
python -m ranktrace.analyze traces --tool nsys       # rank*.nsys-rep
python -m ranktrace.analyze traces --tool stats      # stats/rank*.json
```

Results go to `traces/analysis/<tool>/` as CSV tables and HTML figures (TraceLens figures in
`figures/`). HTA figures are HTA's own; the other figures are drawn by ranktrace from the tools'
tables, and their titles say so. Outputs answering the same question have the same number in every
tool.

| # | Question | HTA | TraceLens | nsys |
|---|---|---|---|---|
| 01 | Time per step, DataLoader wait | `01_sections` | | `01_nvtx_gpu_proj_sum` |
| 02 | GPU busy or idle | `02_temporal` | `02_gpu_timeline` (also exposed comm and copies) | `02_gpu_time_util` |
| 03 | Why idle / when idle | `03_idle` | | `03_gpu_gaps` |
| 04 | What GPU work, which kernels | `04_kernel_types` | `04_ops_by_category` | `04_cuda_gpu_kern_sum` |
| 05 | Communication hidden by compute * | `05_comm_overlap` | in `02_gpu_timeline` | `05_nccl_gpu_overlap_trace` |
| 06 | Which rank is slow * | `06_stragglers` | `06_stragglers` | |
| 07 | CPU-bound or GPU-bound | `07_queue_length`, `07b_full_queue` (tables) | | `07_cuda_api_sync` (table) |
| 08 | Copies | `08_memory_bw` (table) | in `02_gpu_timeline` | `08_cuda_gpu_mem_time_sum` |
| 09 | Collectives: transfer or wait * | | `09_collectives` | `09_nccl_sum` |

\* needs at least 2 ranks with NCCL; on 1 GPU these outputs are not written.

## Reading the results

Words used below:
- **kernel**: one program run on the GPU; every PyTorch op launches one or more.
- **rank**: one process, usually one GPU.
- **collective**: a communication between ranks (all-reduce, all-gather, ...), done by NCCL kernels.
- **exposed**: time when only this kind of work runs on the GPU, not hidden behind compute.
- **H2D / D2H / D2D**: copies host (CPU) to device (GPU), device to host, device to device.

**01: How long is a step, and is it waiting for data?**
- HTA `01_sections`: CPU time per batch of each label. `ProfilerStep#N` is one whole recorded step,
  `enumerate(DataLoader)#...` the wait for the next batch, `Optimizer.step#...` the optimizer, plus
  any `region()` you added. A DataLoader wait that is a large share of the step means the run is
  input-bound: more workers, faster storage or lighter preprocessing.
- nsys `01_nvtx_gpu_proj_sum`: per NVTX range (`step` and your regions), CPU time (how long the
  range is open) and GPU time (how long the GPU works on what was launched inside it).

**02: Is the GPU busy?**
- HTA `02_temporal`: share of time in compute kernels, non-compute kernels (communication, copies)
  and idle (nothing runs).
- TraceLens `02_gpu_timeline`: the same split, in ms. `computation_time` = a compute kernel runs;
  `exposed_comm_time` = communication runs with no compute next to it, so the GPU waits for the
  network; `exposed_memcpy_time` = a copy runs with no compute or communication next to it;
  `idle_time` = nothing runs. They add up to the total.
- nsys `02_gpu_time_util`: `In-Use` = share of time a kernel runs.
- Idle above ~10% is worth a look (03 says why). Exposed comm above a few % means communication is
  not hidden (see 05).

**03: Why, and when, is the GPU idle?**
- HTA `03_idle`: `host wait` = the CPU did not send work fast enough (CPU-bound); `kernel wait` =
  short gaps between kernels, the cost of launching them (many tiny kernels); `other` = the rest
  (e.g. waiting for a sync or a copy).
- nsys `03_gpu_gaps`: every idle gap longer than 1 ms, with its start time and length. Regular gaps
  at the same place in each step point to one cause (DataLoader, a `.item()`, logging).

**04: What does the GPU spend its time on?**
- HTA `04_kernel_types`: share of compute, communication and memory (copy, memset) kernels.
- TraceLens `04_ops_by_category`: GPU time per kind of PyTorch op (`CONV_fwd`, `CONV_bwd`, `GEMM`,
  `elementwise`, `reduce`, ...). Many small `elementwise` ops can often be fused (`torch.compile`).
- nsys `04_cuda_gpu_kern_sum`: the 10 kernels with the most total time, by kernel name (nsys does not
  know PyTorch ops). cuDNN convolution names: `fprop` = forward, `dgrad` / `wgrad` = backward (input /
  weight gradient). `nchwToNhwc` / `nhwcToNchw` = layout conversions around convolutions; if they
  are large, try `channels_last` memory format.

**05: Is communication hidden behind compute?** *
- HTA `05_comm_overlap`: share of communication time that overlaps with compute. Close to 100% is
  good; low means the GPU waits for communication.
- nsys `05_nccl_gpu_overlap_trace`: the same per NCCL kernel, one box per rank.
- TraceLens: `exposed_comm_time` in `02_gpu_timeline`.

**06: Which rank is slow?** *
- HTA `06_stragglers`: how often each rank started its collectives last. The slow rank has the
  highest count.
- TraceLens `06_stragglers`: per rank, `total_wait_time_us` (time spent waiting for the others) and
  `pct_arrived_last` (on the bar). The slow rank waits the least and arrives last most often; the
  others wait for it. Then compare that rank's 01 and 04 with the others: more data wait, more
  compute (bigger input) or a slower node.

**07: Is the CPU or the GPU the limit?**
- HTA `07_queue_length`: number of operations the CPU has queued for the GPU and not yet run (max
  1024). Near 0 = the GPU waits for the CPU (CPU-bound); high = the CPU is ahead (GPU-bound, normal).
  `07b_full_queue`: share of time the queue was full and the CPU had to wait (GPU-bound).
- nsys `07_cuda_api_sync`: every CPU call that blocks until the GPU finishes
  (`cudaStreamSynchronize`, ...), with its duration. Often caused by `.item()`, `.cpu()` or printing a
  tensor. In a GPU-bound run long waits are normal; many of them stop the CPU from queueing ahead.

**08: How much time goes to copies?**
- HTA `08_memory_bw`: per copy type (H2D, D2H, D2D, memset), count and bandwidth (GB/s).
- nsys `08_cuda_gpu_mem_time_sum`: total time per copy type, per rank.
- TraceLens: `exposed_memcpy_time` in `02_gpu_timeline`. Large H2D: use `pin_memory=True` and
  `.to(device, non_blocking=True)`. Large D2D: often extra copies (`.contiguous()`, `torch.cat`).

**09: Do collectives take long because of the transfer, or because ranks wait?** *
- TraceLens `09_collectives`: per collective, `comm_latency_mean` (the transfer itself) and
  `skew in start time_mean` (how long the first rank waits for the last). Large skew = load imbalance
  (see 06); large latency = network or message size.
- nsys `09_nccl_sum`: total time per collective type, per rank (transfer and wait together).

**stats** (`--tool stats`, from `stats=True`): per rank, median step time, peak GPU memory
(`gpu_alloc_peak_gb` = PyTorch tensors, `gpu_device_peak_gb` = the whole GPU), peak CPU memory
(`host_rss_peak_gb`) and trace size. A GPU peak close to the GPU size means a risk of
out-of-memory.
