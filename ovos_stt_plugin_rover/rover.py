"""
ROVER (Recognizer Output Voting Error Reduction).

Implements dynamic-programming alignment of multiple token sequences to build a
Word Transition Network (WTN), then performs majority voting to obtain a final
consensus transcript.

Reference:
    J. G. Fiscus, "A post-processing system to yield reduced word error rates:
    Recognizer Output Voting Error Reduction (ROVER)," ASRU 1997.
"""
from copy import deepcopy
from enum import Enum, unique
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@unique
class AlignmentAction(Enum):
    """Edit operations used during DP alignment."""

    DELETION = "DELETION"
    SUBSTITUTION = "SUBSTITUTION"
    INSERTION = "INSERTION"
    CORRECT = "CORRECT"


class AlignmentEdge:
    """
    A single token node used in the Word Transition Network (WTN).

    Parameters
    ----------
    value:
        Token text.
    sources_count:
        Number of hypotheses in which this token occurred in this aligned
        position. Used for majority voting.
    """

    value: str
    sources_count: int

    def __init__(self, value: str, sources_count: int) -> None:
        self.value = value
        self.sources_count = sources_count


class ROVER:
    """
    Recognizer Output Voting Error Reduction (ROVER).

    This class aligns multiple hypotheses using dynamic programming, constructs
    a Word Transition Network, and returns the majority-voted output string.

    Parameters
    ----------
    tokenizer:
        Callable that converts raw text to a list of tokens.
    detokenizer:
        Callable that converts a list of tokens back to text.
    silent:
        If False, a progress bar could be shown (currently unused).

    Notes
    -----
    * Alignment cost model: insertion = deletion = substitution = 1.
    * Correct match cost = 0.
    """

    tokenizer: Callable[[str], List[str]]
    detokenizer: Callable[[List[str]], str]
    silent: bool

    def __init__(
            self,
            tokenizer: Callable[[str], List[str]] = str.split,
            detokenizer: Callable[[List[str]], str] = lambda x: " ".join(x),
            silent: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.silent = silent

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, hyps: Sequence[str]) -> str:
        """
        Compute a ROVER consensus from input hypotheses.

        Parameters
        ----------
        hyps:
            Iterable of raw hypothesis strings. Each is tokenized and aligned.

        Returns
        -------
        str
            Majority-voted output transcript.
        """
        tokenized = [self.tokenizer(h) for h in hyps]
        edges = self._build_word_transition_network(tokenized)
        consensus = self._get_result(edges)
        return self.detokenizer(consensus)

    # ------------------------------------------------------------------ #
    # Word Transition Network (WTN) construction
    # ------------------------------------------------------------------ #

    def _build_word_transition_network(
            self, hypotheses: List[List[str]]
    ) -> List[Dict[str, AlignmentEdge]]:
        """
        Build the WTN by iteratively aligning hypotheses to an accumulated graph.

        Parameters
        ----------
        hypotheses:
            List of token sequences.

        Returns
        -------
        list of dict
            Each element is a dict mapping token text to an AlignmentEdge.
        """
        edges: List[Dict[str, AlignmentEdge]] = [
            {e.value: e} for e in self._edges_from_tokens(hypotheses[0])
        ]

        for src_idx, hyp in enumerate(hypotheses[1:], start=1):
            edges = self._align(edges, self._edges_from_tokens(hyp), sources_count=src_idx)

        return edges

    @staticmethod
    def _edges_from_tokens(words: List[str]) -> List[AlignmentEdge]:
        """Wrap raw tokens into AlignmentEdge objects."""
        return [AlignmentEdge(w, 1) for w in words]

    # ------------------------------------------------------------------ #
    # Dynamic Programming Alignment
    # ------------------------------------------------------------------ #

    @staticmethod
    def _align(
            ref_edges_sets: List[Dict[str, AlignmentEdge]],
            hyp_edges: List[AlignmentEdge],
            sources_count: int,
    ) -> List[Dict[str, AlignmentEdge]]:
        """
        Align a hypothesis sequence against an existing WTN layer using DP.

        Parameters
        ----------
        ref_edges_sets:
            Previously built WTN columns (each a dict of token→edge).
        hyp_edges:
            Token edges for the current hypothesis.
        sources_count:
            Index of the hypothesis; used to update source counts when merging.

        Returns
        -------
        list of dict
            Updated WTN edge sets reflecting the alignment.
        """

        H, R = len(hyp_edges), len(ref_edges_sets)
        distance = np.zeros((H + 1, R + 1), dtype=float)

        # DP initialization
        distance[:, 0] = np.arange(H + 1)
        distance[0, :] = np.arange(R + 1)

        memo: List[List[
            Optional[Tuple[AlignmentAction, Dict[str, AlignmentEdge], AlignmentEdge]]
        ]] = [[None] * (R + 1) for _ in range(H + 1)]

        # First column: insertions
        for i in range(1, H + 1):
            memo[i][0] = (
                AlignmentAction.INSERTION,
                {"": AlignmentEdge("", sources_count)},
                hyp_edges[i - 1],
            )

        # First row: deletions
        for j in range(1, R + 1):
            memo[0][j] = (
                AlignmentAction.DELETION,
                ref_edges_sets[j - 1],
                AlignmentEdge("", 1),
            )

        # Fill DP table
        for i in range(1, H + 1):
            hyp_edge = hyp_edges[i - 1]
            for j in range(1, R + 1):
                ref_edges = ref_edges_sets[j - 1]
                hyp_word = hyp_edge.value
                in_ref = hyp_word in ref_edges

                candidates: List[
                    Tuple[
                        float,
                        Tuple[AlignmentAction, Dict[str, AlignmentEdge], AlignmentEdge],
                    ]
                ] = []

                # match / substitution
                if in_ref:
                    candidates.append(
                        (
                            distance[i - 1, j - 1],
                            (AlignmentAction.CORRECT, ref_edges, hyp_edge),
                        )
                    )
                else:
                    candidates.append(
                        (
                            distance[i - 1, j - 1] + 1,
                            (AlignmentAction.SUBSTITUTION, ref_edges, hyp_edge),
                        )
                    )

                # deletion
                del_cost = 1 if "" not in ref_edges else 0
                candidates.append(
                    (
                        distance[i, j - 1] + del_cost,
                        (AlignmentAction.DELETION, ref_edges, AlignmentEdge("", 1)),
                    )
                )

                # insertion
                candidates.append(
                    (
                        distance[i - 1, j] + 1,
                        (
                            AlignmentAction.INSERTION,
                            {"": AlignmentEdge("", sources_count)},
                            hyp_edge,
                        ),
                    )
                )

                distance[i, j], memo[i][j] = min(candidates, key=lambda t: t[0])

        # Backtrace
        aligned: List[Dict[str, AlignmentEdge]] = []
        i, j = H, R

        while i > 0 or j > 0:
            action, ref_edges, hyp_edge = memo[i][j]  # type: ignore
            merged = deepcopy(ref_edges)

            w = hyp_edge.value
            if w not in merged:
                merged[w] = hyp_edge
            else:
                merged[w].sources_count += 1

            aligned.append(merged)

            if action in (AlignmentAction.CORRECT, AlignmentAction.SUBSTITUTION):
                i -= 1
                j -= 1
            elif action == AlignmentAction.INSERTION:
                i -= 1
            else:  # DELETION
                j -= 1

        return aligned[::-1]

    # ------------------------------------------------------------------ #
    # Majority Voting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_result(edges: List[Dict[str, AlignmentEdge]]) -> List[str]:
        """
        Produce a consensus token sequence from a Word Transition Network by majority voting.
        
        For each WTN column, selects the token with the maximum tuple (sources_count, token length, token) and returns the sequence of selected tokens.
        
        Parameters:
            edges (List[Dict[str, AlignmentEdge]]): WTN represented as a list of dictionaries mapping token text to its AlignmentEdge.
        
        Returns:
            List[str]: The voted output token sequence.
        """
        result: List[str] = []

        for edge_set in edges:
            # Max by (sources_count, token_length, token)
            _, _, token = max(
                (e.sources_count, len(e.value), e.value)
                for e in edge_set.values()
            )
            result.append(token)

        return result


class WeightedROVER(ROVER):
    """
    wROVER: Confidence-Weighted ROVER.
    Uses a fixed list of weights corresponding to the order of hypotheses.
    """
    def __init__(self, weights: List[float], **kwargs):
        """
        Initialize a WeightedROVER with per-hypothesis weights and base ROVER configuration.
        
        Parameters:
            weights (List[float]): Per-hypothesis weights in the same order as input hypotheses; used to weight contributions during consensus voting.
            **kwargs: Passed through to the base ROVER initializer (tokenizer, detokenizer, silent, etc.).
        """
        super().__init__(**kwargs)
        self.weights = weights

    def fit(self, hyps: Sequence[str]) -> str:
        # If weights aren't provided for all hyps, default missing ones to 1.0
        """
        Generate a consensus transcript from multiple hypothesis strings using confidence weights for voting.
        
        Builds a word transition network by tokenizing each hypothesis, merging them with the per-hypothesis weights supplied at construction (any missing weights default to 1.0 in order), performs alignment and weighted voting to produce a consensus token sequence, and returns the detokenized consensus.
        
        Parameters:
            hyps (Sequence[str]): Ordered sequence of hypothesis strings to combine; weights correspond by index to these hypotheses.
        
        Returns:
            str: The detokenized consensus transcript produced by weighted ROVER voting.
        """
        active_weights = self.weights + [1.0] * (len(hyps) - len(self.weights))

        tokenized = [self.tokenizer(h) for h in hyps]

        # Initial WTN from first hypothesis using its weight
        edges: List[Dict[str, AlignmentEdge]] = [
            {e.value: e} for e in [AlignmentEdge(w, active_weights[0]) for w in tokenized[0]]
        ]

        # Iteratively align and add weighted counts
        for i, tokens in enumerate(tokenized[1:], start=1):
            hyp_token_edges = [AlignmentEdge(w, active_weights[i]) for w in tokens]
            edges = self._align(edges, hyp_token_edges, sources_count=i)

        consensus = self._get_result(edges)
        return self.detokenizer(consensus)


class IterativeROVER(ROVER):
    """
    itROVER: Multi-Pass / Iterative ROVER.
    Uses the first pass consensus as a high-confidence anchor for a second pass.
    """

    def fit(self, hyps: Sequence[str]) -> str:
        # Pass 1: Standard consensus
        """
        Refines a consensus transcript by running a two-pass ROVER alignment.
        
        Performs an initial consensus on the provided hypotheses, then re-aligns all hypotheses using that initial consensus as the primary reference (prepended) to produce a stabilized final consensus.
        
        Parameters:
            hyps (Sequence[str]): Candidate transcripts (hypotheses) to merge.
        
        Returns:
            str: Final consensus transcript produced after the second-pass alignment.
        """
        initial_consensus = super().fit(hyps)

        # Pass 2: Re-align all hypotheses using the consensus as the primary reference
        # We put the consensus first to stabilize the WTN structure
        augmented_hyps = [initial_consensus] + list(hyps)
        return super().fit(augmented_hyps)