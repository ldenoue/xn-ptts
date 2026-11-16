# tiny-voca browser support

`dizione.zip` contains the unmodified Python package used by tiny-voca's
official browser demo. It was retrieved from <https://tiny-voca.n4ze3m.com>
on November 15, 2026 and matches Dizione revision
`8cd889e4de7be6bd1dd2e6891344a0729fa14dd4` exactly.

The model and pronunciation data are not stored in this Git repository. The
browser downloads them from the official
[`n4ze3m/tiny-voca`](https://huggingface.co/n4ze3m/tiny-voca) repository,
pinned to revision `e5f9b11651a4db525fb207a56a91e4df2d1dd387`.

tiny-voca and Dizione are Apache-2.0 licensed. Their upstream licenses are
included here as `TINY_VOCA_LICENSE` and `DIZIONE_LICENSE`. The pronunciation
data's CMUdict and homograph notices remain available in the pinned Hugging
Face repository.

The integration wrapper is `../tiny_voca_worker.js`.
