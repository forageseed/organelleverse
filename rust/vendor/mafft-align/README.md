# mafft-align

> Pairwise and profile sequence alignment for [rust-MAFFT](https://github.com/luksgrin/rust-MAFFT).

> **Acknowledgment.** rust-MAFFT is a port of [MAFFT](https://mafft.cbrc.jp/alignment/software/)
> by **Kazutaka Katoh** and colleagues (CBRC). The science is theirs.
> If you use this crate in published work, please cite
> [Katoh & Standley 2013](https://doi.org/10.1093/molbev/mst010);
> full guidance at [the project citation page](https://luksgrin.github.io/rust-MAFFT/citation/).

The DP layer: Needleman-Wunsch, Smith-Waterman, generalized affine
(`G__align11`, `L__align11`, `A__align`), profile-vs-profile DP
(`MSalign`, `Salignmm`, `partA__align`), FFT-anchored DP (`Falign`),
and constraint integration via `Falign_localhom`. All byte-identical
to the corresponding C MAFFT routines for the BAliBASE 3 fixture set.

## Install

```sh
cargo add mafft-align
```

Most callers want the top-level [`mafft`](https://crates.io/crates/mafft)
crate. Use `mafft-align` directly when you need the pairwise DP
primitives without the progressive-alignment / guide-tree machinery
on top.

## Documentation

API reference: [docs.rs/mafft-align](https://docs.rs/mafft-align)

Project documentation: [luksgrin.github.io/rust-MAFFT](https://luksgrin.github.io/rust-MAFFT/) ·
[Cross-validation harness](https://luksgrin.github.io/rust-MAFFT/architecture/cross-validation/)

## License

MIT for the Rust port; the DP algorithms are derived from upstream
MAFFT under BSD-3-Clause. See `LICENSE-MIT` and `LICENSE-BSD` in the
workspace root.
