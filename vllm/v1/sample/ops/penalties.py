# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch, nvtx, time, numpy, os
from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils.torch_utils import make_tensor_with_pad

times = numpy.array([])

DEBUG = int(os.getenv("DEBUG", 0))


@nvtx.annotate(message="apply_all_penalties", color="blue")
def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    global times
    start = time.perf_counter()
    _, vocab_size = logits.shape
    reqs, reqs_len = prompt_token_ids.shape
    out_reqs = len(output_token_ids)
    assert reqs == out_reqs
    for req in range(reqs):
        elems = torch.sum(prompt_token_ids[req] != 156940)
        if not torch.equal(prompt_token_ids[req, :elems], token_ids[req, :elems].to("cuda", torch.int64)):
            print("Warning: prompt and output mismatch!", prompt_token_ids[req, :elems], token_ids[req, :elems].to("cuda"))
    max_len = max(map(len, output_token_ids), default=0) + reqs_len
    pinned_output_tokens_ids = torch.empty((reqs, max_len), device="cpu", dtype=torch.int64, pin_memory=True)
    pinned_output_tokens_ids.copy_(token_ids[:reqs, :max_len])
    output_token_ids_t = pinned_output_tokens_ids.to(device=logits.device, dtype=torch.int64, non_blocking=True)
    masked = prompt_token_ids.masked_fill(prompt_token_ids.eq(156940), 0)
    output_token_ids_t[:, :reqs_len].sub_(masked).abs_()
    
    output_lens = torch.tensor([len(x) for x in output_token_ids]).cuda()
    prompt_lens = torch.sum(prompt_token_ids != 156940, 1).cuda()
    for req in range(reqs):
        output_token_ids_t[req , output_lens[req] + prompt_lens[req] : ] = 0

    output_token_ids_t.masked_fill_(output_token_ids_t.eq(0), vocab_size)
    if (output_token_ids_t == -1).any():
        print("Warning: tensor contains placeholder -1")
    if (output_token_ids_t < 0).any():
        print("Warning: tensor contains negative values")
    ret = apply_penalties(
        logits,
        prompt_token_ids,
        output_token_ids_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )
    if DEBUG:
        times = numpy.append(times, time.perf_counter() - start)
        if len(times) % 64 == 0:
            p90, p95, p99 = numpy.percentile(times, [90, 95, 99])
            print(
                f"penalties mean ({numpy.mean(times):.4}s) | "
                f"p90 ({p90:.4}) p95 ({p95:.4}) p99 ({p99:.4})"
            )
    return ret


@nvtx.annotate(message="_convert_to_tensors", color="blue")
def _convert_to_tensors(
    output_token_ids: list[list[int]], vocab_size: int, device: torch.device
) -> torch.Tensor:
    """
    Convert the different list data structures to tensors.
    """
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=False,
    )
    return output_tokens_tensor.to(device, non_blocking=True)
