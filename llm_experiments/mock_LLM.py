"""
Sequence Distribution via Subsequence Rewards
==============================================
Defines a probability distribution over all sequences of length T 
over a vocabulary of size V, where:

    log p(x) ∝ sum_s  reward(s) * count(s in x)

Key design choices:
- All V^T sequences enumerated once, scores stored as log-probs
- Aho-Corasick style prefix counting for O(T) score per sequence
- Conditionals computed by marginalising over pre-grouped score table
"""

import os
import numpy as np
from itertools import product
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# 1.  Aho-Corasick automaton for counting all subsequence hits in O(T)
# ---------------------------------------------------------------------------

class MultiPatternCounter:
    """
    Builds a DFA over the vocabulary so that scanning a sequence of length T
    costs exactly T transitions rather than T * sum(len(s)) naive matching.

    For small vocabularies (V <= 10) and short patterns this is the 
    cleanest approach and avoids repeated scans.
    """

    def __init__(self, patterns: list[tuple], rewards: list[float], V: int):
        self.patterns = patterns
        self.rewards  = rewards
        self.V        = V
        self._build()

    def _build(self):
        # --- build trie ---
        # Each node: children dict, output reward accumulated here
        goto   = [{}]          # goto[state][token] = next_state. List of dictionaries
        output = [0.0]         # output[state] = reward emitted on arriving here

        # Insert patterns into trie
        for pat, rew in zip(self.patterns, self.rewards):
            cur = 0
            for tok in pat:
                if tok not in goto[cur]:
                    goto[cur][tok] = len(goto)
                    goto.append({})
                    output.append(0.0)
                cur = goto[cur][tok]
            output[cur] += rew          # reward emitted when pattern completes

        n_states = len(goto)

        # --- compute failure links (BFS) ---
        failure = [0] * n_states
        queue   = []

        for tok, s in goto[0].items():
            failure[s] = 0
            queue.append(s)

        head = 0
        while head < len(queue):
            r = queue[head]; head += 1
            for tok, s in goto[r].items():
                queue.append(s)
                state = failure[r]
                while state != 0 and tok not in goto[state]:
                    state = failure[state]
                failure[s] = goto[state].get(tok, 0)
                if failure[s] == s:
                    failure[s] = 0
                output[s] += output[failure[s]]   # propagate suffix outputs

        # --- build complete DFA transition table ---
        # dfa[state, token] = next_state  (fill missing with failure links)
        dfa    = np.zeros((n_states, self.V), dtype=np.int32)
        reward_vec = np.array(output, dtype=np.float64)

        for s in range(n_states):
            for tok in range(self.V):
                cur = s
                while cur != 0 and tok not in goto[cur]:
                    cur = failure[cur]
                dfa[s, tok] = goto[cur].get(tok, 0)

        self.dfa        = dfa          # shape (n_states, V)
        self.reward_vec = reward_vec   # shape (n_states,)
        self.n_states   = n_states

    def score_sequence(self, seq: tuple | np.ndarray) -> float:
        """Score a single sequence: O(T) transitions."""
        state     = 0
        total_rew = 0.0
        for tok in seq:
            state      = self.dfa[state, tok]
            total_rew += self.reward_vec[state]
        return total_rew

    def score_all_sequences(self, V: int, T: int) -> np.ndarray:
        """
        Score every sequence in lexicographic order.

        Uses a BFS/DP over (position, automaton_state) so we visit each
        (position, state, token) triple exactly once.

        Returns: scores array of shape (V^T,) in lex order
                 i.e. index = x[0]*V^(T-1) + x[1]*V^(T-2) + ... + x[T-1]
        """
        total = V ** T

        # dp[state] = array of (index_offset, cumulative_reward) pairs
        # We track: for each automaton state after processing t tokens,
        # what are the sequence indices and accumulated rewards?

        # More memory-efficient: propagate a (V^T,) reward array in T sweeps,
        # each sweep doing one token step.

        # current_rewards[i] = reward accumulated for the i-th sequence prefix so far
        # At step t, prefix has t tokens.

        # Represent state jointly: shape (V^t, ) -> too large to store states explicitly
        # Instead, keep a parallel (V^T,) array of automaton states.

        # Initialise: all sequences start at automaton state 0 with reward 0
        auto_states = np.zeros(total, dtype=np.int32)
        rewards     = np.zeros(total, dtype=np.float64)

        stride = V ** (T - 1)   # how many sequences share the same first token

        for t in range(T):
            # Token at position t for sequence index i is:  (i // V^(T-1-t)) % V
            pos_stride  = V ** (T - 1 - t)
            token_at_t  = (np.arange(total) // pos_stride) % V   # shape (total,)

            # Vectorised DFA transition
            next_states  = self.dfa[auto_states, token_at_t]      # shape (total,)
            rewards     += self.reward_vec[next_states]
            auto_states  = next_states

        return rewards   # log unnormalised scores


# ---------------------------------------------------------------------------
# 2.  The distribution itself
# ---------------------------------------------------------------------------

@dataclass
class SubseqRewardDistribution:
    """
    Full distribution over V^T sequences.

    Attributes
    ----------
    V          : vocabulary size
    T          : sequence length
    patterns   : list of token tuples to reward
    rewards    : reward per occurrence (default 1.0 each)
    alpha      : exponent for power distribution p^alpha (default 1.0 = base)
    """
    V        : int
    T        : int
    patterns : list[tuple]
    rewards  : list[float] = field(default_factory=list)
    alpha    : float       = 1.0

    # Computed on first call to build()
    _counter      : MultiPatternCounter = field(init=False, repr=False, default=None)
    _log_scores   : np.ndarray          = field(init=False, repr=False, default=None)
    _log_probs    : np.ndarray          = field(init=False, repr=False, default=None)
    _probs        : np.ndarray          = field(init=False, repr=False, default=None)
    _all_seqs     : np.ndarray          = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if not self.rewards:
            self.rewards = [1.0] * len(self.patterns)
        assert len(self.rewards) == len(self.patterns)

    def build(self):
        """Enumerate all V^T sequences and compute their (log) probabilities."""
        total = self.V ** self.T

        # Memory estimate: 4 arrays of size total
        #   auto_states (int32=4B) + rewards (float64=8B) + token_at_t (int64=8B)
        #   + _probs (float64=8B) + _log_probs (float64=8B) + _all_seqs (int32, T*4B)
        bytes_per_seq = 4 + 8 + 8 + 8 + 8 + self.T * 4
        est_bytes = total * bytes_per_seq
        est_gb = est_bytes / (1024 ** 3)

        print(f"Building distribution: V={self.V}, T={self.T}, alpha={self.alpha}, "
              f"{total:,} sequences, {len(self.patterns)} patterns")
        print(f"Estimated memory: {est_gb:.2f} GB")

        if est_gb > 1.0:
            resp = input(f"This will use ~{est_gb:.1f} GB of RAM (alpha={self.alpha}). Continue? [y/N] ")
            if resp.strip().lower() != 'y':
                raise MemoryError(
                    f"Aborted: V={self.V}, T={self.T} requires ~{est_gb:.1f} GB"
                )

        # -- 1. build automaton & score all sequences --
        self._counter    = MultiPatternCounter(self.patterns, self.rewards, self.V)
        self._log_scores = self._counter.score_all_sequences(self.V, self.T)
        # log_scores[i] = sum of rewards = log unnorm prob under alpha=1

        # -- 2. apply power (multiply log scores by alpha) --
        log_unnorm = self.alpha * self._log_scores

        # -- 3. normalise in log space --
        log_unnorm -= log_unnorm.max()               # numerical stability
        self._log_probs = log_unnorm - np.log(np.sum(np.exp(log_unnorm)))
        self._probs     = np.exp(self._log_probs)

        # -- 4. build sequence index array (lex order) --
        # _all_seqs[i] = the i-th sequence as an array of T tokens
        indices = np.arange(total)
        self._all_seqs = np.zeros((total, self.T), dtype=np.int32)
        for t in range(self.T):
            pos_stride = self.V ** (self.T - 1 - t)
            self._all_seqs[:, t] = (indices // pos_stride) % self.V

        print(f"Done. Entropy = {self.entropy():.4f} bits  |  "
              f"Top-5 sequence scores: {np.sort(self._log_scores)[-5:][::-1]}")

        resp = input("Save distribution to disk? [y/N] ")
        if resp.strip().lower() == 'y':
            folder = input("Save folder: ").strip()
            fname = f"dist_V{self.V}_T{self.T}_alpha{self.alpha}"
            path = os.path.join(folder, fname)
            self.save(path)

        return self

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    def prob(self, seq: tuple) -> float:
        """Probability of a single sequence."""
        idx = self._seq_to_index(seq)
        return self._probs[idx]

    def log_prob(self, seq: tuple) -> float:
        idx = self._seq_to_index(seq)
        return self._log_probs[idx]

    def score(self, seq: tuple) -> float:
        """Raw reward score (sum of pattern matches)."""
        idx = self._seq_to_index(seq)
        return self._log_scores[idx]

    # ------------------------------------------------------------------
    # Conditional probabilities  p(x_t = v | x_{<t})
    # ------------------------------------------------------------------

    def conditional(self, prefix: tuple, v: int) -> float:
        """
        p(x_t = v | prefix) computed exactly by marginalising.

        This is just:  sum of probs[i] for all i whose sequence starts with prefix+(v,)
        divided by:    sum of probs[i] for all i whose sequence starts with prefix

        Both sums are O(V^(T - len(prefix))) but we use the index structure to
        slice contiguous blocks without looping over all sequences.
        """
        t = len(prefix)
        assert t < self.T, "prefix is already a full sequence"

        denom = self._marginal_prob(prefix)
        if denom < 1e-300:
            return 1.0 / self.V   # uniform fallback for zero-prob prefix

        numer = self._marginal_prob(prefix + (v,))
        return numer / denom

    def conditional_vector(self, prefix: tuple) -> np.ndarray:
        """Returns p(x_t = v | prefix) for all v in one call."""
        t     = len(prefix)
        probs = np.array([self._marginal_prob(prefix + (v,)) for v in range(self.V)])
        total = probs.sum()
        if total < 1e-300:
            return np.ones(self.V) / self.V
        return probs / total

    def _marginal_prob(self, prefix: tuple) -> float:
        """
        Sum of p(x) over all x starting with prefix.
        Uses the lex-order index structure: all such sequences form a
        contiguous block of size V^(T - len(prefix)).
        """
        t     = len(prefix)
        block = self.V ** (self.T - t)    # how many sequences share this prefix

        # Index of first sequence with this prefix
        start = 0
        for i, tok in enumerate(prefix):
            start += tok * (self.V ** (self.T - 1 - i))

        return self._probs[start : start + block].sum()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n sequences from the exact distribution."""
        indices = np.random.choice(len(self._probs), size=n, p=self._probs)
        return self._all_seqs[indices]   # shape (n, T)

    def ancestral_sample(self, temperature: float = 1.0) -> tuple:
        """
        Sample token-by-token using exact conditionals.
        temperature < 1 sharpens; temperature > 1 flattens.
        This is the 'low-temperature' baseline from the paper.
        """
        seq = []
        for t in range(self.T):
            probs = self.conditional_vector(tuple(seq))
            if temperature != 1.0:
                log_p  = np.log(probs + 1e-300)
                log_p  = log_p / temperature
                log_p -= log_p.max()
                probs  = np.exp(log_p)
                probs /= probs.sum()
            tok = np.random.choice(self.V, p=probs)
            seq.append(tok)
        return tuple(seq)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """Shannon entropy in bits."""
        p = self._probs[self._probs > 0]
        return float(-np.sum(p * np.log2(p)))

    def top_k_sequences(self, k: int = 10):
        """Return the k highest-probability sequences with their stats."""
        top_idx = np.argsort(self._probs)[-k:][::-1]
        results = []
        for idx in top_idx:
            seq   = tuple(int(x) for x in self._all_seqs[idx])
            results.append({
                "seq"      : seq,
                "prob"     : float(self._probs[idx]),
                "log_prob" : float(self._log_probs[idx]),
                "score"    : float(self._log_scores[idx]),
            })
        return results

    def score_histogram(self, bins: int = 20):
        """Histogram of scores weighted by probability mass."""
        return np.histogram(self._log_scores, bins=bins, weights=self._probs)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save the built distribution to a .npz file."""
        np.savez_compressed(
            path,
            V=self.V, T=self.T, alpha=self.alpha,
            patterns=np.array(self.patterns, dtype=object),
            rewards=np.array(self.rewards),
            log_scores=self._log_scores,
            log_probs=self._log_probs,
            probs=self._probs,
            all_seqs=self._all_seqs,
            dfa=self._counter.dfa,
            reward_vec=self._counter.reward_vec,
        )
        savepath = path if path.endswith('.npz') else path + '.npz'
        size_mb = os.path.getsize(savepath) / (1024**2)
        print(f"Saved to {savepath} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "SubseqRewardDistribution":
        """Load a previously saved distribution."""
        data = np.load(path, allow_pickle=True)
        dist = cls(
            V=int(data['V']),
            T=int(data['T']),
            patterns=[tuple(p) for p in data['patterns']],
            rewards=data['rewards'].tolist(),
            alpha=float(data['alpha']),
        )
        dist._log_scores = data['log_scores']
        dist._log_probs  = data['log_probs']
        dist._probs      = data['probs']
        dist._all_seqs   = data['all_seqs']

        # Rebuild the counter with the saved DFA
        dist._counter = MultiPatternCounter.__new__(MultiPatternCounter)
        dist._counter.dfa        = data['dfa']
        dist._counter.reward_vec = data['reward_vec']
        dist._counter.n_states   = len(data['reward_vec'])
        dist._counter.V          = dist.V
        dist._counter.patterns   = dist.patterns
        dist._counter.rewards    = dist.rewards

        print(f"Loaded distribution: V={dist.V}, T={dist.T}, "
              f"{len(dist._probs):,} sequences, entropy={dist.entropy():.4f} bits")
        return dist

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seq_to_index(self, seq: tuple) -> int:
        """Convert a sequence tuple to its lex-order index."""
        idx = 0
        for t, tok in enumerate(seq):
            idx += tok * (self.V ** (self.T - 1 - t))
        return idx


# ---------------------------------------------------------------------------
# 3.  Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    N_SAMPLES = 50
    V = 3
    T = 15
    patterns = [(0,1), (0,1,2,0,1,2)]
    rewards  = [0.5, 2]

    alpha = float(input("Enter alpha for comparison: "))
    temperature = 1.0 / alpha

    # -- Build base distribution (alpha=1) --
    print("\n=== Building base distribution (alpha=1) ===")
    dist_base = SubseqRewardDistribution(
        V=V, T=T, patterns=patterns, rewards=rewards, alpha=1.0,
    ).build()

    print("\n--- Top 10 sequences (base) ---")
    for entry in dist_base.top_k_sequences(10):
        print(f"  {entry['seq']}  score={entry['score']:.1f}  "
              f"prob={entry['prob']:.6f}")

    # -- Build power distribution (alpha=user) --
    print(f"\n=== Building power distribution (alpha={alpha}) ===")
    dist_power = SubseqRewardDistribution(
        V=V, T=T, patterns=patterns, rewards=rewards, alpha=alpha,
    ).build()

    print(f"\n--- Top 10 sequences (alpha={alpha}) ---")
    for entry in dist_power.top_k_sequences(10):
        print(f"  {entry['seq']}  score={entry['score']:.1f}  "
              f"prob={entry['prob']:.6f}")

    # -- Sample and compare --
    print(f"\n=== Sampling comparison ({N_SAMPLES} samples each) ===")
    print(f"  Low-temp:  alpha=1, temperature=1/{alpha}={temperature:.4f}")
    print(f"  Power:     alpha={alpha}, temperature=1.0\n")

    # Low-temp sampling: base distribution + temperature = 1/alpha
    low_temp_scores = []
    for _ in range(N_SAMPLES):
        s = dist_base.ancestral_sample(temperature=temperature)
        low_temp_scores.append(dist_base.score(s))
    low_temp_scores = np.array(low_temp_scores)

    # Power sampling: alpha distribution + temperature = 1
    power_scores = []
    for _ in range(N_SAMPLES):
        s = dist_power.ancestral_sample(temperature=1.0)
        power_scores.append(dist_power.score(s))
    power_scores = np.array(power_scores)

    print(f"  {'Method':<25} {'Mean Score':>12} {'Variance':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Low-temp (1/alpha)':<25} {low_temp_scores.mean():>12.4f} {low_temp_scores.var():>12.4f}")
    print(f"  {'Power (alpha)':<25} {power_scores.mean():>12.4f} {power_scores.var():>12.4f}")