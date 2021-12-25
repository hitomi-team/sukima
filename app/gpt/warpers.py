# @title Tail Free Sampling Warper
from typing import List

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsWarper
from math import exp


class TailFreeSamplingLogitsWarper(LogitsWarper):
    """
    :class:`transformers.LogitsWarper` that performs tail free sampling according to:
        https://trentbrick.github.io/Tail-Free-Sampling/#tail-free-sampling-algorithm
    Args:
        threshold (:obj:`float`):
            This sets the threshold z. A reasonable value is 0.95.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
    """

    def __init__(self, threshold: float, filter_value: float = -float("inf")):
        if not isinstance(threshold, float) or (threshold < 0 or threshold > 1.0):
            raise ValueError(f"`threshold` has to be a float > 0 and < 1, but is {threshold}")

        self.z = threshold
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)

        d = sorted_logits.softmax(dim=-1)
        d = d[:, 1:] - d[:, :-1]
        d = d[:, 1:] - d[:, :-1]
        d = d.abs()
        d = d / d.sum(dim=-1).view(1, -1).T

        cumulative_probs = d.cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = torch.zeros(sorted_indices.shape).bool().to(scores.device)
        sorted_indices_to_remove[:, :-2] = (cumulative_probs > self.z)[:, :]

        # Shift the indices to the right to keep also the first token above the threshold
        # sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()

        # Always keep the first token
        sorted_indices_to_remove[:, 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper(LogitsWarper):
    def __init__(self, threshold: float, filter_value: float = -float("inf")):
        if not isinstance(threshold, float) or (threshold < 0 or threshold > 1.0):
            raise ValueError(f"`threshold` has to be a float > 0 and < 1, but is {threshold}")

        self.z = threshold
        self.filter_value = filter_value
        print(threshold)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.nn.functional.softmax(scores, dim=-1)
        limit = torch.pow(torch.max(probs), 2.0) * self.z
        indices_to_remove = probs < limit
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


# @title Repetition Penalty Processor
class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.
    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float = 1.0, slope=3.33, penalize_last=250, alpha_frequency=None, alpha_presence=None, whitelist=None):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = 1.0 if penalty < 1.0 else penalty
        self.raw_penalty = penalty
        self.penalize_last = None

        if slope is not None and penalize_last is not None and penalize_last >= 1:
            self.penalty = (torch.arange(penalize_last) / (penalize_last - 1)) * 2. - 1
            self.penalty = (slope * self.penalty) / (1 + torch.abs(self.penalty) * (slope - 1))
            self.penalty = 1 + ((self.penalty + 1) / 2).unsqueeze(0) * (penalty - 1)

            self.penalize_last = penalize_last

        self.alpha_frequency = alpha_frequency if alpha_frequency is not None and alpha_frequency > 0.0 else None
        self.alpha_presence = alpha_presence if alpha_presence is not None and alpha_presence > 0.0 else None
        self.alpha_enable = self.alpha_frequency is not None or self.alpha_presence is not None

        self.whitelist = None
        self.whitelist_list = None

        if whitelist is not None:
            self.whitelist_list = whitelist

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.whitelist is None and self.whitelist_list is not None:
            self.whitelist_list = list(filter(lambda x: x >= 0 and x < scores.shape[1], self.whitelist_list))

            if len(self.whitelist_list) > 0:
                self.whitelist = torch.tensor(self.whitelist_list).long().sort()[0]
                self.whitelist = self.whitelist.to(input_ids.device)

        if self.whitelist is not None:
            unpenalized = scores.gather(1, self.whitelist.view(1, -1))

        if self.raw_penalty > 1.0:
            if self.penalize_last is not None:
                penality_len = min(input_ids.shape[1], self.penalize_last)
                input_ids = input_ids[:, -penality_len:]

            score = torch.gather(scores, 1, input_ids)

            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            if self.penalize_last is not None:
                penalty = self.penalty.type(score.dtype).to(score.device)
                score = torch.where(score < 0, score * penalty[:, -penality_len:], score / penalty[:, -penality_len:])

            else:
                score = torch.where(score < 0, score * self.penalty, score / self.penalty)

            scores.scatter_(1, input_ids, score)

        if self.alpha_enable:
            c = torch.zeros(scores.shape).long().to(input_ids.device)
            # unique only returns counts for first item in batch, so manually iterate
            for i in range(input_ids.shape[0]):
                if self.penalize_last is not None:
                    token_input_ids, counts = torch.unique(input_ids[i, -self.penalize_last:], sorted=True, return_counts=True, dim=-1)

                else:
                    token_input_ids, counts = torch.unique(input_ids[i], sorted=True, return_counts=True, dim=-1)

                c[i].scatter_(0, token_input_ids, counts)

            if self.alpha_frequency:
                scores -= c * self.alpha_frequency

            if self.alpha_presence:
                scores[c > 0] -= self.alpha_presence

        if self.whitelist is not None:
            scores.scatter_(1, self.whitelist.view(1, -1), unpenalized)

        return scores


class PhraseBiasProcessor(LogitsProcessor):
    def __init__(self, words_ids: List[List[int]], bias: float):
        if not isinstance(words_ids, list) or len(words_ids) == 0:
            return

        if any(not isinstance(word_ids, list) for word_ids in words_ids):
            raise ValueError("`words_ids` has to be a list of lists")

        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in word_ids)
            for word_ids in words_ids
        ):
            raise ValueError(
                "Each list in `words_ids` has to be a list of positive integers"
            )

        self.words_ids = words_ids
        self.bias = exp(bias)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for phrase_ids in self.words_ids:
            if phrase_ids[0] not in input_ids:
                scores[:, phrase_ids[0]] += self.bias

            else:
                for token_id in phrase_ids:
                    if token_id in input_ids:
                        continue

                    else:
                        scores[:, token_id] += self.bias
                        break

        return scores
