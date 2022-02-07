# @title Tail Free Sampling Warper
from typing import List, Tuple

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsWarper
from math import exp


class TailFreeSamplingLogitsWarper(LogitsWarper):
    r"""
    :class:`transformers.LogitsWarper` that performs tail free sampling, as described in
    https://www.trentonbricken.com/Tail-Free-Sampling/.
    Args:
        tfs (:obj:`float`):
            If set to < 1, only the most probable tokens where the second derivative of the probabilities of the tokens
            sorted in descending order of probability add up to at most :obj:`tfs` are kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float > 0 and < 1, but is {tfs}")

        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.filter_value >= 1.0:
            return scores

        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs
        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.nn.functional.softmax(scores, dim=-1)
        limit = torch.pow(torch.max(probs), 2.0) * self.z
        indices_to_remove = probs < limit
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class TypicalLogitsWarper(LogitsWarper):
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

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

class LogitBiasProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` adding bias to specific tokens
    Args:
        logit_biases (:obj:`List[Tuple[int, float]]`):
            Adds a float bias to the given token's logit.
    """

    def __init__(self, logit_bias: List[Tuple[int, float]]=[]):
        if not isinstance(logit_bias, list) and len(logit_bias) > 0:
            raise ValueError("`logit_bias` has to be a non-empty list")
        
        # apply exp to each bias
        self.logit_bias = [(token, exp(bias)) for token, bias in logit_bias]
        self.bias = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.bias is None:
            self.bias = torch.zeros(scores.shape[1]).float()
            logit_bias = torch.tensor(self.logit_bias)
            self.bias.scatter_(0, logit_bias[:,0].long(), logit_bias[:,1].float())
            self.bias = self.bias.to(scores.dtype).to(scores.device).unsqueeze(0)
        return scores + self.bias

class PhraseBiasProcessor(LogitsProcessor):
    def __init__(self, words_ids: List[List[int]], bias: float, ensure_sequence_finish: bool, generate_once: bool):
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
        self.ensure_sequence_finish = ensure_sequence_finish
        self.generate_once = generate_once
    
    def slice_in_list(self, l, s):
        a = 0
        for i in range(l.shape[1]):
            for j in range(len(s)):
                if l[:,i].item() == s[j]:
                    a += 1
        return a
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for phrase_ids in self.words_ids:
            if self.generate_once:
                if phrase_ids[0] not in input_ids:
                    scores[:, phrase_ids[0]] += self.bias
                    continue
            else:
                scores[:, phrase_ids[0]] += self.bias
            idx = self.slice_in_list(input_ids, phrase_ids)
            if idx == len(phrase_ids) or idx > len(phrase_ids):
                continue # sequence is finished
            else:
                if self.ensure_sequence_finish:
                    if self.generate_once:
                        scores[:, phrase_ids[idx]] -= self.bias
                    scores[:, phrase_ids[idx]] = 1000.0 # max bias
                    break
                else:
                    scores[:, phrase_ids[idx]] += self.bias
                continue

        return scores
