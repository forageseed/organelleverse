# OrganelleVerse dependency patch

Source: https://github.com/luksgrin/rust-MAFFT
Crate: mafft-align 0.1.2, commit aff77991c0a07372ea6afa622bc7c841c4baecc5.
Upstream MIT and BSD license files are included unchanged.

The profile DP base case dropped nonempty columns when an FFT-anchored
segment was empty on only one side. The corrected base case emits Insert
or Delete operations for all columns on the nonempty side. No residues
are restored by post-processing, and scoring/nonempty DP are unchanged.
Regression: the three 600-bp repetitive sequences in
 tests/phylogeny/test_alignment_integrity.py.
