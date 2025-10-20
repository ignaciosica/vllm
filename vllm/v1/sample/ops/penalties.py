# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch, nvtx, time, numpy
from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils import make_tensor_with_pad

times = numpy.array([])


@nvtx.annotate(message="apply_all_penalties", color="blue")
def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    global times
    start = time.perf_counter()
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)
    ret = apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )
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
