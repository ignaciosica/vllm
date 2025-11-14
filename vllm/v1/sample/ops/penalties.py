# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
    token_ids_cpu_tensor: torch.Tensor,
    num_tokens,
    num_reqs: int,
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape

    num_tokens_cpu_tensor = torch.tensor(num_tokens)
    max_num_computed = torch.max(num_tokens_cpu_tensor[:num_reqs])
    token_ids_slice = token_ids_cpu_tensor[:num_reqs, :max_num_computed]
    idx = torch.arange(max_num_computed).expand(num_reqs, max_num_computed)
    token_ids_slice[idx > num_tokens_cpu_tensor[:num_reqs].unsqueeze(1)] = vocab_size
    token_ids_dev_tensor = token_ids_slice.to(logits.device, dtype=torch.int64, non_blocking=True)

    logits_ref = logits.clone()
    logits_c = logits.clone()
    prompt_token_ids_c = prompt_token_ids.clone()
    output_tokens_t_c = token_ids_dev_tensor
    presence_penalties_c = presence_penalties.clone()
    frequency_penalties_c = frequency_penalties.clone()
    repetition_penalties_c = repetition_penalties.clone()

    res = apply_penalties(
        logits_c,
        torch.full_like(prompt_token_ids_c, vocab_size),
        output_tokens_t_c,
        presence_penalties_c,
        frequency_penalties_c,
        repetition_penalties_c,
    )

    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)

    # In the async scheduling case, rows that won't have penalties applied may contain
    # -1 placeholder token ids. We must replace these with valid token ids so that the
    # scatter done in apply_penalties is valid.
    # NOTE(nick): The penalties implementation is currently quite inefficient and
    # will be reworked anyhow.
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    ref = apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )

    return ref

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
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)
