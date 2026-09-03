# GigaAM v3 Runtime Rust dependency notices

This generated payload covers the exact registry crates statically linked into
`gigaam-service` for the Linux release target. Do not edit it by hand; run
`python3 scripts/generate_rust_notice_payload.py --check` to compare it with the
locked Cargo metadata and checksum-verified local registry crate archives.

## Scope

- Target: `x86_64-unknown-linux-gnu`.
- Service configurations: CPU (`--no-default-features`), CUDA
  (`--no-default-features --features gigaam-service/cuda`), and TensorRT
  (`--no-default-features --features gigaam-service/tensorrt`).
- Selection: normal non-dev dependency edges only. Packages whose only library target
  is a procedural macro are excluded because their code is executed during compilation,
  not statically linked into `asr-serve`.
- Registry packages: 101; the three selected closures are identical.
- Cargo.lock SHA-256: `6da8635180ddf06d3a72b02acfc56aadf3a701b03c9453ccddc078cffdf3647d`.
- Closure SHA-256: `1fab76abcfb7c7f03ed7344e9ee6cc984429adb8af21090dec99b8e8ab819108`.
- Project workspace code is covered by the parent `COPYING` and `COPYING.LESSER` files.

## Package records

### `arrayvec` 0.7.8
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `d3fb67a6e08acf24fdeccbac2cb6ac4305825bd1f117462e0e6f2f193345ad56`.
- Source archive: `https://static.crates.io/crates/arrayvec/arrayvec-0.7.8.crate`.
- Repository declared by package metadata: `https://github.com/bluss/arrayvec`.
- Authors declared by package metadata: `bluss`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/4da95ec4ecb65b738d470b7d762894ad9c97da93e6cbfb18b570fc2c96f4b871.txt` (SHA-256 `4da95ec4ecb65b738d470b7d762894ad9c97da93e6cbfb18b570fc2c96f4b871`).

### `atomic-waker` 1.1.2
- License expression: `Apache-2.0 OR MIT`.
- Locked crate archive SHA-256: `1505bd5d3d116872e7271a6d4e16d81d0c8570876c8de68093a09ac269d8aac0`.
- Source archive: `https://static.crates.io/crates/atomic-waker/atomic-waker-1.1.2.crate`.
- Repository declared by package metadata: `https://github.com/smol-rs/atomic-waker`.
- Authors declared by package metadata: `Stjepan Glavina <stjepang@gmail.com>`; `Contributors to futures-rs`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).
  - `LICENSE-THIRD-PARTY`: `texts/6226d0632e2e1a80c23597e964da9812ae193c535fe058154afb034e94167aa5.txt` (SHA-256 `6226d0632e2e1a80c23597e964da9812ae193c535fe058154afb034e94167aa5`).

### `axum` 0.8.9
- License expression: `MIT`.
- Locked crate archive SHA-256: `31b698c5f9a010f6573133b09e0de5408834d0c82f8d7475a89fc1867a71cd90`.
- Source archive: `https://static.crates.io/crates/axum/axum-0.8.9.crate`.
- Repository declared by package metadata: `https://github.com/tokio-rs/axum`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/6a13bc24a100a6812f053879ec51b126b103af7cda6dbf48c4188722da44da9f.txt` (SHA-256 `6a13bc24a100a6812f053879ec51b126b103af7cda6dbf48c4188722da44da9f`).

### `axum-core` 0.5.6
- License expression: `MIT`.
- Locked crate archive SHA-256: `08c78f31d7b1291f7ee735c1c6780ccde7785daae9a9206026862dab7d8792d1`.
- Source archive: `https://static.crates.io/crates/axum-core/axum-core-0.5.6.crate`.
- Repository declared by package metadata: `https://github.com/tokio-rs/axum`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/008c87afcd2e626eaf564093250bed06dd7efb5732113264bba3dda8f1c556a1.txt` (SHA-256 `008c87afcd2e626eaf564093250bed06dd7efb5732113264bba3dda8f1c556a1`).

### `base64` 0.22.1
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `72b3254f16251a8381aa12e40e3c4d2f0199f8c6508fbecb9d91f575e0fbb8c6`.
- Source archive: `https://static.crates.io/crates/base64/base64-0.22.1.crate`.
- Repository declared by package metadata: `https://github.com/marshallpierce/rust-base64`.
- Authors declared by package metadata: `Marshall Pierce <marshall@mpierce.org>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/0dd882e53de11566d50f8e8e2d5a651bcf3fabee4987d70f306233cf39094ba7.txt` (SHA-256 `0dd882e53de11566d50f8e8e2d5a651bcf3fabee4987d70f306233cf39094ba7`).

### `bitflags` 1.3.2
- License expression: `MIT/Apache-2.0`.
- Locked crate archive SHA-256: `bef38d45163c2f1dde094a7dfd33ccf595c92905c8f8f4fdc18d06fb1037718a`.
- Source archive: `https://static.crates.io/crates/bitflags/bitflags-1.3.2.crate`.
- Repository declared by package metadata: `https://github.com/bitflags/bitflags`.
- Authors declared by package metadata: `The Rust Project Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb.txt` (SHA-256 `6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb`).

### `block-buffer` 0.10.4
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `3078c7629b62d3f0439517fa394996acacc5cbc91c5a20d8c658e77abd503a71`.
- Source archive: `https://static.crates.io/crates/block-buffer/block-buffer-0.10.4.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/utils`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/d5c22aa3118d240e877ad41c5d9fa232f9c77d757d4aac0c2f943afc0a95e0ef.txt` (SHA-256 `d5c22aa3118d240e877ad41c5d9fa232f9c77d757d4aac0c2f943afc0a95e0ef`).

### `block-buffer` 0.12.1
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `d2f6c7dbe95a6ed67ad9f18e57daf93a2f034c524b99fd2b76d18fdfeb6660aa`.
- Source archive: `https://static.crates.io/crates/block-buffer/block-buffer-0.12.1.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/utils`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/98181e7249d0c01737645ec982499ce99a0f07eb8f7d625b8840d799d10dbc01.txt` (SHA-256 `98181e7249d0c01737645ec982499ce99a0f07eb8f7d625b8840d799d10dbc01`).

### `bytemuck` 1.25.2
- License expression: `Zlib OR Apache-2.0 OR MIT`.
- Locked crate archive SHA-256: `95832e849adfb21180ccb6826a99da14e5d266ae5c2e668e1602cf234f153797`.
- Source archive: `https://static.crates.io/crates/bytemuck/bytemuck-1.25.2.crate`.
- Repository declared by package metadata: `https://github.com/Lokathor/bytemuck`.
- Authors declared by package metadata: `Lokathor <zefria@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/e3ba223bb1423f0aad8c3dfce0fe3148db48926d41e6fbc3afbbf5ff9e1c89cb.txt` (SHA-256 `e3ba223bb1423f0aad8c3dfce0fe3148db48926d41e6fbc3afbbf5ff9e1c89cb`).
  - `LICENSE-MIT`: `texts/9df9ba60a11af705f2e451b53762686e615d86f76b169cf075c3237730dbd7e2.txt` (SHA-256 `9df9ba60a11af705f2e451b53762686e615d86f76b169cf075c3237730dbd7e2`).
  - `LICENSE-ZLIB`: `texts/84b34dd7608f7fb9b17bd588a6bf392bf7de504e2716f024a77d89f1b145a151.txt` (SHA-256 `84b34dd7608f7fb9b17bd588a6bf392bf7de504e2716f024a77d89f1b145a151`).

### `bytes` 1.12.1
- License expression: `MIT`.
- Locked crate archive SHA-256: `fc652a48c352aef3ea3aed32080501cf3ef6ed5da78602a020c991775b0aff04`.
- Source archive: `https://static.crates.io/crates/bytes/bytes-1.12.1.crate`.
- Repository declared by package metadata: `https://github.com/tokio-rs/bytes`.
- Authors declared by package metadata: `Carl Lerche <me@carllerche.com>`; `Sean McArthur <sean@seanmonstar.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/45f522cacecb1023856e46df79ca625dfc550c94910078bd8aec6e02880b3d42.txt` (SHA-256 `45f522cacecb1023856e46df79ca625dfc550c94910078bd8aec6e02880b3d42`).

### `cfg-if` 1.0.4
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `9330f8b2ff13f34540b44e946ef35111825727b38d33286ef986142615121801`.
- Source archive: `https://static.crates.io/crates/cfg-if/cfg-if-1.0.4.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang/cfg-if`.
- Authors declared by package metadata: `Alex Crichton <alex@alexcrichton.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/378f5840b258e2779c39418f3f2d7b2ba96f1c7917dd6be0713f88305dbda397.txt` (SHA-256 `378f5840b258e2779c39418f3f2d7b2ba96f1c7917dd6be0713f88305dbda397`).

### `const-oid` 0.10.2
- License expression: `Apache-2.0 OR MIT`.
- Locked crate archive SHA-256: `a6ef517f0926dd24a1582492c791b6a4818a4d94e789a334894aa15b0d12f55c`.
- Source archive: `https://static.crates.io/crates/const-oid/const-oid-0.10.2.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/formats`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/73b9dc2e79c7308998dd30296e073aefaefb944a68fb89aa412c23c0edcabcaa.txt` (SHA-256 `73b9dc2e79c7308998dd30296e073aefaefb944a68fb89aa412c23c0edcabcaa`).

### `cpufeatures` 0.2.17
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `59ed5838eebb26a2bb2e58f6d5b5316989ae9d08bab10e0e6d103e656d1b0280`.
- Source archive: `https://static.crates.io/crates/cpufeatures/cpufeatures-0.2.17.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/utils`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/ae9baa7beea910273c2f384c2a6b721fb7bd02bda3436074a1072e4ee689f985.txt` (SHA-256 `ae9baa7beea910273c2f384c2a6b721fb7bd02bda3436074a1072e4ee689f985`).

### `cpufeatures` 0.3.0
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `8b2a41393f66f16b0823bb79094d54ac5fbd34ab292ddafb9a0456ac9f87d201`.
- Source archive: `https://static.crates.io/crates/cpufeatures/cpufeatures-0.3.0.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/utils`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/ae9baa7beea910273c2f384c2a6b721fb7bd02bda3436074a1072e4ee689f985.txt` (SHA-256 `ae9baa7beea910273c2f384c2a6b721fb7bd02bda3436074a1072e4ee689f985`).

### `crypto-common` 0.1.7
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `78c8292055d1c1df0cce5d180393dc8cce0abec0a7102adb6c7b1eef6016d60a`.
- Source archive: `https://static.crates.io/crates/crypto-common/crypto-common-0.1.7.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/traits`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/3521672491a3479422d5fe1aca6645dd2984090f85da6e5205abfb18fb7a6897.txt` (SHA-256 `3521672491a3479422d5fe1aca6645dd2984090f85da6e5205abfb18fb7a6897`).

### `crypto-common` 0.2.2
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `ce6e4c961d6cd6c9a86db418387425e8bdeaf05b3c8bc1411e6dca4c252f1453`.
- Source archive: `https://static.crates.io/crates/crypto-common/crypto-common-0.2.2.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/traits`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/d2e7ec5355c96eeade56b09187ceb48a6a30299da3ce7531a66d3d11405ab963.txt` (SHA-256 `d2e7ec5355c96eeade56b09187ceb48a6a30299da3ce7531a66d3d11405ab963`).

### `data-encoding` 2.11.1
- License expression: `MIT`.
- Locked crate archive SHA-256: `4583a4551df46e2792f82ceeac45e850d2e2d5debba0b91f102385cda5b11f06`.
- Source archive: `https://static.crates.io/crates/data-encoding/data-encoding-2.11.1.crate`.
- Repository declared by package metadata: `https://github.com/ia0/data-encoding`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/b68ad1a3367b825447089e1f8d6829b97f47a89eb78d2f4ebaef4672f5606186.txt` (SHA-256 `b68ad1a3367b825447089e1f8d6829b97f47a89eb78d2f4ebaef4672f5606186`).

### `digest` 0.10.7
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `9ed9a281f7bc9b7576e61468ba615a66a5c8cfdff42420a70aa82701a3b1e292`.
- Source archive: `https://static.crates.io/crates/digest/digest-0.10.7.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/traits`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/9e0dfd2dd4173a530e238cb6adb37aa78c34c6bc7444e0e10c1ab5d8881f63ba.txt` (SHA-256 `9e0dfd2dd4173a530e238cb6adb37aa78c34c6bc7444e0e10c1ab5d8881f63ba`).

### `digest` 0.11.3
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `f1dd6dbb5841937940781866fa1281a1ff7bd3bf827091440879f9994983d5c2`.
- Source archive: `https://static.crates.io/crates/digest/digest-0.11.3.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/traits`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/af59cea35d7f5e2777a713b8d155d65efa2c339eb43f3c14e868c6ac8506edad.txt` (SHA-256 `af59cea35d7f5e2777a713b8d155d65efa2c339eb43f3c14e868c6ac8506edad`).

### `encoding_rs` 0.8.35
- License expression: `(Apache-2.0 OR MIT) AND BSD-3-Clause`.
- Locked crate archive SHA-256: `75030f3c4f45dafd7586dd6780965a8c7e8e285a5ecb86713e63a79c5b2766f3`.
- Source archive: `https://static.crates.io/crates/encoding_rs/encoding_rs-0.8.35.crate`.
- Repository declared by package metadata: `https://github.com/hsivonen/encoding_rs`.
- Authors declared by package metadata: `Henri Sivonen <hsivonen@hsivonen.fi>`.
- Packaged license, permission, and copyright notices:
  - `COPYRIGHT`: `texts/11789f45bb180841cd362a5eee6789c68ddb573a11105e30768c308a6add0190.txt` (SHA-256 `11789f45bb180841cd362a5eee6789c68ddb573a11105e30768c308a6add0190`).
  - `LICENSE-APACHE`: `texts/cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30.txt` (SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
  - `LICENSE-MIT`: `texts/3fa4ca83dcc9237839b1bdeb2e6d16bdfb5ec0c5ce42b24694d8bbf0dcbef72c.txt` (SHA-256 `3fa4ca83dcc9237839b1bdeb2e6d16bdfb5ec0c5ce42b24694d8bbf0dcbef72c`).
  - `LICENSE-WHATWG`: `texts/838118388fe5c2e7f1dbbaeed13e1c7f3ebf88be91319c7c1d77c18e987d1a50.txt` (SHA-256 `838118388fe5c2e7f1dbbaeed13e1c7f3ebf88be91319c7c1d77c18e987d1a50`).

### `errno` 0.3.14
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `39cab71617ae0d63f51a36d69f866391735b51691dbda63cf6f96d042b63efeb`.
- Source archive: `https://static.crates.io/crates/errno/errno-0.3.14.crate`.
- Repository declared by package metadata: `https://github.com/lambda-fairy/rust-errno`.
- Authors declared by package metadata: `Chris Wong <lambda.fairy@gmail.com>`; `Dan Gohman <dev@sunfishcode.online>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/8764a597675778ddfd4e25f81b08a05dbcf089ac05662df7613fe67f150e3aa2.txt` (SHA-256 `8764a597675778ddfd4e25f81b08a05dbcf089ac05662df7613fe67f150e3aa2`).

### `extended` 0.1.0
- License expression: `MIT`.
- Locked crate archive SHA-256: `af9673d8203fcb076b19dfd17e38b3d4ae9f44959416ea532ce72415a6020365`.
- Source archive: `https://static.crates.io/crates/extended/extended-0.1.0.crate`.
- Repository declared by package metadata: `https://github.com/depp/extended-rs`.
- Authors declared by package metadata: `Dietrich Epp <depp@zdome.net>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE.txt`: `texts/25a0874d15e7c834a47c3adc80901edb2219759254992023ef1010c3065413d5.txt` (SHA-256 `25a0874d15e7c834a47c3adc80901edb2219759254992023ef1010c3065413d5`).

### `form_urlencoded` 1.2.2
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `cb4cb245038516f5f85277875cdaa4f7d2c9a0fa0468de06ed190163b1581fcf`.
- Source archive: `https://static.crates.io/crates/form_urlencoded/form_urlencoded-1.2.2.crate`.
- Repository declared by package metadata: `https://github.com/servo/rust-url`.
- Authors declared by package metadata: `The rust-url developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/20c7855c364d57ea4c97889a5e8d98470a9952dade37bd9248b9a54431670e5e.txt` (SHA-256 `20c7855c364d57ea4c97889a5e8d98470a9952dade37bd9248b9a54431670e5e`).

### `futures-channel` 0.3.34
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `b1f9e3d69d39e4862ffed03ed071a76f9a13ba1d9109d355b0f0aa6b15e393c4`.
- Source archive: `https://static.crates.io/crates/futures-channel/futures-channel-0.3.34.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang/futures-rs`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427.txt` (SHA-256 `275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427`).
  - `LICENSE-MIT`: `texts/6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd.txt` (SHA-256 `6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd`).

### `futures-core` 0.3.34
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `92d699e522242e69e3003b94ecc1f960f3a5e015aa7c5d7486e65ad01dd94f5e`.
- Source archive: `https://static.crates.io/crates/futures-core/futures-core-0.3.34.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang/futures-rs`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427.txt` (SHA-256 `275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427`).
  - `LICENSE-MIT`: `texts/6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd.txt` (SHA-256 `6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd`).

### `futures-sink` 0.3.34
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `1944426bf7d03f1d14f708785e4b33efd750b36d48a157b836b3efc15ede8e1d`.
- Source archive: `https://static.crates.io/crates/futures-sink/futures-sink-0.3.34.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang/futures-rs`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427.txt` (SHA-256 `275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427`).
  - `LICENSE-MIT`: `texts/6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd.txt` (SHA-256 `6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd`).

### `futures-task` 0.3.34
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `cd417de3d1d015fc3bfd2b1ea46dfc7bab72ef86f1cc7cc9c78e728b34a6d1fd`.
- Source archive: `https://static.crates.io/crates/futures-task/futures-task-0.3.34.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang/futures-rs`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427.txt` (SHA-256 `275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427`).
  - `LICENSE-MIT`: `texts/6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd.txt` (SHA-256 `6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd`).

### `futures-util` 0.3.34
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `0d50a92467f8ba5dd6e3ee5d4bd04d73ab2e4e1c44474a0674821dfce14b79bc`.
- Source archive: `https://static.crates.io/crates/futures-util/futures-util-0.3.34.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang/futures-rs`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427.txt` (SHA-256 `275c491d6d1160553c32fd6127061d7f9606c3ea25abfad6ca3f6ed088785427`).
  - `LICENSE-MIT`: `texts/6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd.txt` (SHA-256 `6652c868f35dfe5e8ef636810a4e576b9d663f3a17fb0f5613ad73583e1b88fd`).

### `generic-array` 0.14.7
- License expression: `MIT`.
- Locked crate archive SHA-256: `85649ca51fd72272d7821adaf274ad91c288277713d9c18820d8499a7ff69e9a`.
- Source archive: `https://static.crates.io/crates/generic-array/generic-array-0.14.7.crate`.
- Repository declared by package metadata: `https://github.com/fizyk20/generic-array.git`.
- Authors declared by package metadata: `Bartłomiej Kamiński <fizyk20@gmail.com>`; `Aaron Trent <novacrazy@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/c09aae9d3c77b531f56351a9947bc7446511d6b025b3255312d3e3442a9a7583.txt` (SHA-256 `c09aae9d3c77b531f56351a9947bc7446511d6b025b3255312d3e3442a9a7583`).

### `getrandom` 0.3.4
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `899def5c37c4fd7b2664648c28120ecec138e4d395b459e5ca34f9cce2dd77fd`.
- Source archive: `https://static.crates.io/crates/getrandom/getrandom-0.3.4.crate`.
- Repository declared by package metadata: `https://github.com/rust-random/getrandom`.
- Authors declared by package metadata: `The Rand Project Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/aaff376532ea30a0cd5330b9502ad4a4c8bf769c539c87ffe78819d188a18ebf.txt` (SHA-256 `aaff376532ea30a0cd5330b9502ad4a4c8bf769c539c87ffe78819d188a18ebf`).
  - `LICENSE-MIT`: `texts/29e9fe5074bd27e0e5d5d110394fbbcd841baee2651a3c4b4560a632702cede4.txt` (SHA-256 `29e9fe5074bd27e0e5d5d110394fbbcd841baee2651a3c4b4560a632702cede4`).

### `http` 1.5.0
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `918d3568bebf352712bc2ef3d46a8bcf1a75b373be6539de198e9105cbbf9ce0`.
- Source archive: `https://static.crates.io/crates/http/http-1.5.0.crate`.
- Repository declared by package metadata: `https://github.com/hyperium/http`.
- Authors declared by package metadata: `Alex Crichton <alex@alexcrichton.com>`; `Carl Lerche <me@carllerche.com>`; `Sean McArthur <sean@seanmonstar.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/8bb1b50b0e5c9399ae33bd35fab2769010fa6c14e8860c729a52295d84896b7a.txt` (SHA-256 `8bb1b50b0e5c9399ae33bd35fab2769010fa6c14e8860c729a52295d84896b7a`).
  - `LICENSE-MIT`: `texts/dc91f8200e4b2a1f9261035d4c18c33c246911a6c0f7b543d75347e61b249cff.txt` (SHA-256 `dc91f8200e4b2a1f9261035d4c18c33c246911a6c0f7b543d75347e61b249cff`).

### `http-body` 1.1.0
- License expression: `MIT`.
- Locked crate archive SHA-256: `ca2a8f2913ee65f60facd6a5905613afaa448497a0230cc41ce022d93290bc2c`.
- Source archive: `https://static.crates.io/crates/http-body/http-body-1.1.0.crate`.
- Repository declared by package metadata: `https://github.com/hyperium/http-body`.
- Authors declared by package metadata: `Carl Lerche <me@carllerche.com>`; `Lucio Franco <luciofranco14@gmail.com>`; `Sean McArthur <sean@seanmonstar.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/248378d0a3383c173fb925f17141b88e71580b3ba17ddc6ac3d2a344683232ab.txt` (SHA-256 `248378d0a3383c173fb925f17141b88e71580b3ba17ddc6ac3d2a344683232ab`).

### `http-body-util` 0.1.5
- License expression: `MIT`.
- Locked crate archive SHA-256: `23169fe34a5fbcdd3f3862e78fb9b6fccd5f02a6dc6f732547005d45631ce71c`.
- Source archive: `https://static.crates.io/crates/http-body-util/http-body-util-0.1.5.crate`.
- Repository declared by package metadata: `https://github.com/hyperium/http-body`.
- Authors declared by package metadata: `Carl Lerche <me@carllerche.com>`; `Lucio Franco <luciofranco14@gmail.com>`; `Sean McArthur <sean@seanmonstar.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/248378d0a3383c173fb925f17141b88e71580b3ba17ddc6ac3d2a344683232ab.txt` (SHA-256 `248378d0a3383c173fb925f17141b88e71580b3ba17ddc6ac3d2a344683232ab`).

### `httparse` 1.10.1
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `6dbf3de79e51f3d586ab4cb9d5c3e2c14aa28ed23d180cf89b4df0454a69cc87`.
- Source archive: `https://static.crates.io/crates/httparse/httparse-1.10.1.crate`.
- Repository declared by package metadata: `https://github.com/seanmonstar/httparse`.
- Authors declared by package metadata: `Sean McArthur <sean@seanmonstar.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/391a5396cec6230bfabd4ef4eb2350eb895bc5efce377a2218f5702ed020d3e3.txt` (SHA-256 `391a5396cec6230bfabd4ef4eb2350eb895bc5efce377a2218f5702ed020d3e3`).

### `httpdate` 1.0.3
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `df3b46402a9d5adb4c86a0cf463f42e19994e3ee891101b1841f30a545cb49a9`.
- Source archive: `https://static.crates.io/crates/httpdate/httpdate-1.0.3.crate`.
- Repository declared by package metadata: `https://github.com/pyfisch/httpdate`.
- Authors declared by package metadata: `Pyfisch <pyfisch@posteo.org>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/4d10fe5f3aa176b05b229a248866bad70b834c173f1252a814ff4748d8a13837.txt` (SHA-256 `4d10fe5f3aa176b05b229a248866bad70b834c173f1252a814ff4748d8a13837`).
  - `LICENSE-MIT`: `texts/934887691e05d69d7c86ad3f2c360980fa30c15b035e351f3c9865e99da4debc.txt` (SHA-256 `934887691e05d69d7c86ad3f2c360980fa30c15b035e351f3c9865e99da4debc`).

### `hybrid-array` 0.4.14
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `707114b52a152fa7bdb290cd7cd5912d9467273b6d74e21b8d81aca1f8533f6b`.
- Source archive: `https://static.crates.io/crates/hybrid-array/hybrid-array-0.4.14.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/hybrid-array`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/70c9d40f1f9545c3f133b8a67206e89da850f6468eed072281bb3701514114a9.txt` (SHA-256 `70c9d40f1f9545c3f133b8a67206e89da850f6468eed072281bb3701514114a9`).

### `hyper` 1.11.0
- License expression: `MIT`.
- Locked crate archive SHA-256: `d22053281f852e11534f5198498373cbb59295120a20771d90f7ed1897490a72`.
- Source archive: `https://static.crates.io/crates/hyper/hyper-1.11.0.crate`.
- Repository declared by package metadata: `https://github.com/hyperium/hyper`.
- Authors declared by package metadata: `Sean McArthur <sean@seanmonstar.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/2d01890414494742ba4a509fcec8efa40f6d8be22cbd72be7cff08d6fda4ec89.txt` (SHA-256 `2d01890414494742ba4a509fcec8efa40f6d8be22cbd72be7cff08d6fda4ec89`).

### `hyper-util` 0.1.20
- License expression: `MIT`.
- Locked crate archive SHA-256: `96547c2556ec9d12fb1578c4eaf448b04993e7fb79cbaad930a656880a6bdfa0`.
- Source archive: `https://static.crates.io/crates/hyper-util/hyper-util-0.1.20.crate`.
- Repository declared by package metadata: `https://github.com/hyperium/hyper-util`.
- Authors declared by package metadata: `Sean McArthur <sean@seanmonstar.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/9e0a97848ea543aef745c98e84fde696a9a3e0735538f6daefdd3cb1942effc1.txt` (SHA-256 `9e0a97848ea543aef745c98e84fde696a9a3e0735538f6daefdd3cb1942effc1`).

### `itoa` 1.0.18
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `8f42a60cbdf9a97f5d2305f08a87dc4e09308d1276d28c869c684d7777685682`.
- Source archive: `https://static.crates.io/crates/itoa/itoa-1.0.18.crate`.
- Repository declared by package metadata: `https://github.com/dtolnay/itoa`.
- Authors declared by package metadata: `David Tolnay <dtolnay@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a.txt` (SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a`).
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).

### `lazy_static` 1.5.0
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `bbd2bcb4c963f2ddae06a2efc7e9f3591312473c50c6685e1f298068316e66fe`.
- Source archive: `https://static.crates.io/crates/lazy_static/lazy_static-1.5.0.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang-nursery/lazy-static.rs`.
- Authors declared by package metadata: `Marvin Löbel <loebel.marvin@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/0621878e61f0d0fda054bcbe02df75192c28bde1ecc8289cbd86aeba2dd72720.txt` (SHA-256 `0621878e61f0d0fda054bcbe02df75192c28bde1ecc8289cbd86aeba2dd72720`).

### `libc` 0.2.189
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `3eaf3ede3fee6db1a4c2ee091bf8a8b4dccdc6d17f656fb07896ee72867612f2`.
- Source archive: `https://static.crates.io/crates/libc/libc-0.2.189.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang/libc`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a.txt` (SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a`).
  - `LICENSE-MIT`: `texts/123a331b5dbf04c30097fa43b8f858bc85df671fe776de498d01f3d6b7c1f69e.txt` (SHA-256 `123a331b5dbf04c30097fa43b8f858bc85df671fe776de498d01f3d6b7c1f69e`).

### `libloading` 0.9.0
- License expression: `ISC`.
- Locked crate archive SHA-256: `754ca22de805bb5744484a5b151a9e1a8e837d5dc232c2d7d8c2e3492edc8b60`.
- Source archive: `https://static.crates.io/crates/libloading/libloading-0.9.0.crate`.
- Repository declared by package metadata: `https://github.com/nagisa/rust_libloading/`.
- Authors declared by package metadata: `Simonas Kazlauskas <libloading@kazlauskas.me>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/b29f8b01452350c20dd1af16ef83b598fea3053578ccc1c7a0ef40e57be2620f.txt` (SHA-256 `b29f8b01452350c20dd1af16ef83b598fea3053578ccc1c7a0ef40e57be2620f`).

### `log` 0.4.34
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `f9f8bd3e56ce4dfc153cf470fffbfa98c7620958b312ca5c3a4b8d5181fd13c6`.
- Source archive: `https://static.crates.io/crates/log/log-0.4.34.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang/log`.
- Authors declared by package metadata: `The Rust Project Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb.txt` (SHA-256 `6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb`).

### `matchit` 0.8.4
- License expression: `MIT AND BSD-3-Clause`.
- Locked crate archive SHA-256: `47e1ffaa40ddd1f3ed91f717a33c8c0ee23fff369e3aa8772b9605cc1d22f4c3`.
- Source archive: `https://static.crates.io/crates/matchit/matchit-0.8.4.crate`.
- Repository declared by package metadata: `https://github.com/ibraheemdev/matchit`.
- Authors declared by package metadata: `Ibraheem Ahmed <ibraheem@ibraheem.ca>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/de701d0618d694feb1af90f02181a1763d9b0bdeb70a3a592781e529077dba65.txt` (SHA-256 `de701d0618d694feb1af90f02181a1763d9b0bdeb70a3a592781e529077dba65`).
  - `LICENSE.httprouter`: `texts/162ce11ad71338d0a0c6ebaf5c48af72c6ae237b468859d3656fe2d9ed3f3a85.txt` (SHA-256 `162ce11ad71338d0a0c6ebaf5c48af72c6ae237b468859d3656fe2d9ed3f3a85`).

### `matrixmultiply` 0.3.11
- License expression: `MIT/Apache-2.0`.
- Locked crate archive SHA-256: `3f607c237553f086e7043417a51df26b2eb899d3caff94e6a67592ff992fedc7`.
- Source archive: `https://static.crates.io/crates/matrixmultiply/matrixmultiply-0.3.11.crate`.
- Repository declared by package metadata: `https://github.com/bluss/matrixmultiply/`.
- Authors declared by package metadata: `bluss`; `R. Janis Goldschmidt`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/792d075c7bad6dac258a44e799eb64cbf465e24d9932d27669be08c5ec957e27.txt` (SHA-256 `792d075c7bad6dac258a44e799eb64cbf465e24d9932d27669be08c5ec957e27`).

### `memchr` 2.8.3
- License expression: `Unlicense OR MIT`.
- Locked crate archive SHA-256: `cf8baf1c55e62ffcace7a9f06f4bd9cd3f0c4beb022d3b367256b91b87513d98`.
- Source archive: `https://static.crates.io/crates/memchr/memchr-2.8.3.crate`.
- Repository declared by package metadata: `https://github.com/BurntSushi/memchr`.
- Authors declared by package metadata: `Andrew Gallant <jamslam@gmail.com>`; `bluss`.
- Packaged license, permission, and copyright notices:
  - `COPYING`: `texts/01c266bced4a434da0051174d6bee16a4c82cf634e2679b6155d40d75012390f.txt` (SHA-256 `01c266bced4a434da0051174d6bee16a4c82cf634e2679b6155d40d75012390f`).
  - `LICENSE-MIT`: `texts/0f96a83840e146e43c0ec96a22ec1f392e0680e6c1226e6f3ba87e0740af850f.txt` (SHA-256 `0f96a83840e146e43c0ec96a22ec1f392e0680e6c1226e6f3ba87e0740af850f`).

### `mime` 0.3.17
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `6877bb514081ee2a7ff5ef9de3281f14a4dd4bceac4c09388074a6b5df8a139a`.
- Source archive: `https://static.crates.io/crates/mime/mime-0.3.17.crate`.
- Repository declared by package metadata: `https://github.com/hyperium/mime`.
- Authors declared by package metadata: `Sean McArthur <sean@seanmonstar.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/df9cfd06d8a44d9a671eadd39ffd97f166481da015a30f45dfd27886209c5922.txt` (SHA-256 `df9cfd06d8a44d9a671eadd39ffd97f166481da015a30f45dfd27886209c5922`).

### `mio` 1.2.2
- License expression: `MIT`.
- Locked crate archive SHA-256: `30d65c71f1ce40ab09135ce117d742b9f8a19ff91a41a8b57ed50bc2de59c427`.
- Source archive: `https://static.crates.io/crates/mio/mio-1.2.2.crate`.
- Repository declared by package metadata: `https://github.com/tokio-rs/mio`.
- Authors declared by package metadata: `Carl Lerche <me@carllerche.com>`; `Thomas de Zeeuw <thomasdezeeuw@gmail.com>`; `Tokio Contributors <team@tokio.rs>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/07919255c7e04793d8ea760d6c2ce32d19f9ff02bdbdde3ce90b1e1880929a9b.txt` (SHA-256 `07919255c7e04793d8ea760d6c2ce32d19f9ff02bdbdde3ce90b1e1880929a9b`).

### `ndarray` 0.17.2
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `520080814a7a6b4a6e9070823bb24b4531daac8c4627e08ba5de8c5ef2f2752d`.
- Source archive: `https://static.crates.io/crates/ndarray/ndarray-0.17.2.crate`.
- Repository declared by package metadata: `https://github.com/rust-ndarray/ndarray`.
- Authors declared by package metadata: `Ulrik Sverdrup "bluss"`; `Jim Turner`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/1fd6747d2c8e80f9fa766f57c5888864774621deb85cc2838ccaed727db32d45.txt` (SHA-256 `1fd6747d2c8e80f9fa766f57c5888864774621deb85cc2838ccaed727db32d45`).

### `num-complex` 0.4.6
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `73f88a1307638156682bada9d7604135552957b7818057dcef22705b4d509495`.
- Source archive: `https://static.crates.io/crates/num-complex/num-complex-0.4.6.crate`.
- Repository declared by package metadata: `https://github.com/rust-num/num-complex`.
- Authors declared by package metadata: `The Rust Project Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb.txt` (SHA-256 `6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb`).

### `num-integer` 0.1.47
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `7ce2d95d4b3734dc35aa2f45e1aa22cd416814592a4f9d9205e11affd5b8e10b`.
- Source archive: `https://static.crates.io/crates/num-integer/num-integer-0.1.47.crate`.
- Repository declared by package metadata: `https://github.com/rust-num/num-integer`.
- Authors declared by package metadata: `The Rust Project Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb.txt` (SHA-256 `6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb`).

### `num-traits` 0.2.19
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `071dfc062690e90b734c0b2273ce72ad0ffa95f0c74596bc250dcfd960262841`.
- Source archive: `https://static.crates.io/crates/num-traits/num-traits-0.2.19.crate`.
- Repository declared by package metadata: `https://github.com/rust-num/num-traits`.
- Authors declared by package metadata: `The Rust Project Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb.txt` (SHA-256 `6485b8ed310d3f0340bf1ad1f47645069ce4069dcc6bb46c7d5c6faf41de1fdb`).

### `once_cell` 1.21.4
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `9f7c3e4beb33f85d45ae3e3a1792185706c8e16d043238c593331cc7cd313b50`.
- Source archive: `https://static.crates.io/crates/once_cell/once_cell-1.21.4.crate`.
- Repository declared by package metadata: `https://github.com/matklad/once_cell`.
- Authors declared by package metadata: `Aleksey Kladov <aleksey.kladov@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).

### `opus-decoder` 0.1.1
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `15ae32c275dabb0cd2c863460bf349e9abc3568ba2abf0f9eb7b7d2edeeed07e`.
- Source archive: `https://static.crates.io/crates/opus-decoder/opus-decoder-0.1.1.crate`.
- Repository declared by package metadata: `https://github.com/TadeuszWolfGang/Rusopus`.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `ort` 2.0.0-rc.13
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `4336a1e2b38848325241c72889086886004e589b7c74f335e60a8e8db5138a0b`.
- Source archive: `https://static.crates.io/crates/ort/ort-2.0.0-rc.13.crate`.
- Repository declared by package metadata: `https://github.com/pykeio/ort`.
- Authors declared by package metadata: `pyke.io <contact@pyke.io>`; `Nicolas Bigaouette <nbigaouette@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30.txt` (SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
  - `LICENSE-MIT`: `texts/6bba08de46289c40986a2e6b310e2da61fcea33b3c112c8320f6093f8f9cb71b.txt` (SHA-256 `6bba08de46289c40986a2e6b310e2da61fcea33b3c112c8320f6093f8f9cb71b`).

### `ort-sys` 2.0.0-rc.13
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `cf211e3776eea6aec988552fa118dd746d70e1b1e5e244058d1c98015f3e5872`.
- Source archive: `https://static.crates.io/crates/ort-sys/ort-sys-2.0.0-rc.13.crate`.
- Repository declared by package metadata: `https://github.com/pykeio/ort`.
- Authors declared by package metadata: `pyke.io <contact@pyke.io>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30.txt` (SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
  - `LICENSE-MIT`: `texts/6bba08de46289c40986a2e6b310e2da61fcea33b3c112c8320f6093f8f9cb71b.txt` (SHA-256 `6bba08de46289c40986a2e6b310e2da61fcea33b3c112c8320f6093f8f9cb71b`).

### `percent-encoding` 2.3.2
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `9b4f627cb1b25917193a259e49bdad08f671f8d9708acfd5fe0a8c1455d87220`.
- Source archive: `https://static.crates.io/crates/percent-encoding/percent-encoding-2.3.2.crate`.
- Repository declared by package metadata: `https://github.com/servo/rust-url/`.
- Authors declared by package metadata: `The rust-url developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/b38f11f6096706e6de553dabe2a7ed142d59b6fa8c97e290c67496154745cdd5.txt` (SHA-256 `b38f11f6096706e6de553dabe2a7ed142d59b6fa8c97e290c67496154745cdd5`).

### `pin-project-lite` 0.2.17
- License expression: `Apache-2.0 OR MIT`.
- Locked crate archive SHA-256: `a89322df9ebe1c1578d689c92318e070967d1042b512afbe49518723f4e6d5cd`.
- Source archive: `https://static.crates.io/crates/pin-project-lite/pin-project-lite-0.2.17.crate`.
- Repository declared by package metadata: `https://github.com/taiki-e/pin-project-lite`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594.txt` (SHA-256 `0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594`).
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).

### `ppv-lite86` 0.2.21
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `85eae3c4ed2f50dcfe72643da4befc30deadb458a9b590d720cde2f2b1e97da9`.
- Source archive: `https://static.crates.io/crates/ppv-lite86/ppv-lite86-0.2.21.crate`.
- Repository declared by package metadata: `https://github.com/cryptocorrosion/cryptocorrosion`.
- Authors declared by package metadata: `The CryptoCorrosion Contributors`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/0218327e7a480793ffdd4eb792379a9709e5c135c7ba267f709d6f6d4d70af0a.txt` (SHA-256 `0218327e7a480793ffdd4eb792379a9709e5c135c7ba267f709d6f6d4d70af0a`).
  - `LICENSE-MIT`: `texts/4cada0bd02ea3692eee6f16400d86c6508bbd3bafb2b65fed0419f36d4f83e8f.txt` (SHA-256 `4cada0bd02ea3692eee6f16400d86c6508bbd3bafb2b65fed0419f36d4f83e8f`).

### `rand` 0.9.5
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `b9ef1d0d795eb7d84685bca4f72f3649f064e6641543d3a8c415898726a57b41`.
- Source archive: `https://static.crates.io/crates/rand/rand-0.9.5.crate`.
- Repository declared by package metadata: `https://github.com/rust-random/rand`.
- Authors declared by package metadata: `The Rand Project Developers`; `The Rust Project Developers`.
- Packaged license, permission, and copyright notices:
  - `COPYRIGHT`: `texts/90eb64f0279b0d9432accfa6023ff803bc4965212383697eee27a0f426d5f8d5.txt` (SHA-256 `90eb64f0279b0d9432accfa6023ff803bc4965212383697eee27a0f426d5f8d5`).
  - `LICENSE-APACHE`: `texts/35242e7a83f69875e6edeff02291e688c97caafe2f8902e4e19b49d3e78b4cab.txt` (SHA-256 `35242e7a83f69875e6edeff02291e688c97caafe2f8902e4e19b49d3e78b4cab`).
  - `LICENSE-MIT`: `texts/209fbbe0ad52d9235e37badf9cadfe4dbdc87203179c0899e738b39ade42177b.txt` (SHA-256 `209fbbe0ad52d9235e37badf9cadfe4dbdc87203179c0899e738b39ade42177b`).

### `rand_chacha` 0.9.0
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `d3022b5f1df60f26e1ffddd6c66e8aa15de382ae63b3a0c1bfc0e4d3e3f325cb`.
- Source archive: `https://static.crates.io/crates/rand_chacha/rand_chacha-0.9.0.crate`.
- Repository declared by package metadata: `https://github.com/rust-random/rand`.
- Authors declared by package metadata: `The Rand Project Developers`; `The Rust Project Developers`; `The CryptoCorrosion Contributors`.
- Packaged license, permission, and copyright notices:
  - `COPYRIGHT`: `texts/90eb64f0279b0d9432accfa6023ff803bc4965212383697eee27a0f426d5f8d5.txt` (SHA-256 `90eb64f0279b0d9432accfa6023ff803bc4965212383697eee27a0f426d5f8d5`).
  - `LICENSE-APACHE`: `texts/35242e7a83f69875e6edeff02291e688c97caafe2f8902e4e19b49d3e78b4cab.txt` (SHA-256 `35242e7a83f69875e6edeff02291e688c97caafe2f8902e4e19b49d3e78b4cab`).
  - `LICENSE-MIT`: `texts/209fbbe0ad52d9235e37badf9cadfe4dbdc87203179c0899e738b39ade42177b.txt` (SHA-256 `209fbbe0ad52d9235e37badf9cadfe4dbdc87203179c0899e738b39ade42177b`).

### `rand_core` 0.9.5
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `76afc826de14238e6e8c374ddcc1fa19e374fd8dd986b0d2af0d02377261d83c`.
- Source archive: `https://static.crates.io/crates/rand_core/rand_core-0.9.5.crate`.
- Repository declared by package metadata: `https://github.com/rust-random/rand`.
- Authors declared by package metadata: `The Rand Project Developers`; `The Rust Project Developers`.
- Packaged license, permission, and copyright notices:
  - `COPYRIGHT`: `texts/90eb64f0279b0d9432accfa6023ff803bc4965212383697eee27a0f426d5f8d5.txt` (SHA-256 `90eb64f0279b0d9432accfa6023ff803bc4965212383697eee27a0f426d5f8d5`).
  - `LICENSE-APACHE`: `texts/6df43f6f4b5d4587f3d8d71e45532c688fd168afa5fe89d571cb32fa09c4ef51.txt` (SHA-256 `6df43f6f4b5d4587f3d8d71e45532c688fd168afa5fe89d571cb32fa09c4ef51`).
  - `LICENSE-MIT`: `texts/209fbbe0ad52d9235e37badf9cadfe4dbdc87203179c0899e738b39ade42177b.txt` (SHA-256 `209fbbe0ad52d9235e37badf9cadfe4dbdc87203179c0899e738b39ade42177b`).

### `rawpointer` 0.2.1
- License expression: `MIT/Apache-2.0`.
- Locked crate archive SHA-256: `60a357793950651c4ed0f3f52338f53b2f809f32d83a07f72909fa13e4c6c1e3`.
- Source archive: `https://static.crates.io/crates/rawpointer/rawpointer-0.2.1.crate`.
- Repository declared by package metadata: `https://github.com/bluss/rawpointer/`.
- Authors declared by package metadata: `bluss`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/7576269ea71f767b99297934c0b2367532690f8c4badc695edf8e04ab6a1e545.txt` (SHA-256 `7576269ea71f767b99297934c0b2367532690f8c4badc695edf8e04ab6a1e545`).

### `ryu` 1.0.23
- License expression: `Apache-2.0 OR BSL-1.0`.
- Locked crate archive SHA-256: `9774ba4a74de5f7b1c1451ed6cd5285a32eddb5cccb8cc655a4e50009e06477f`.
- Source archive: `https://static.crates.io/crates/ryu/ryu-1.0.23.crate`.
- Repository declared by package metadata: `https://github.com/dtolnay/ryu`.
- Authors declared by package metadata: `David Tolnay <dtolnay@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a.txt` (SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a`).
  - `LICENSE-BOOST`: `texts/c9bff75738922193e67fa726fa225535870d2aa1059f91452c411736284ad566.txt` (SHA-256 `c9bff75738922193e67fa726fa225535870d2aa1059f91452c411736284ad566`).

### `serde` 1.0.229
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `4148590afebada386688f18773da617792bf2ef03ffc1e4cbd2b1d45b023e0ba`.
- Source archive: `https://static.crates.io/crates/serde/serde-1.0.229.crate`.
- Repository declared by package metadata: `https://github.com/serde-rs/serde`.
- Authors declared by package metadata: `Erick Tryzelaar <erick.tryzelaar@gmail.com>`; `David Tolnay <dtolnay@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a.txt` (SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a`).
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).

### `serde_core` 1.0.229
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `67dca2c9c51e58a4791a4b1ed58308b39c64224d349a935ab5039aa360942a48`.
- Source archive: `https://static.crates.io/crates/serde_core/serde_core-1.0.229.crate`.
- Repository declared by package metadata: `https://github.com/serde-rs/serde`.
- Authors declared by package metadata: `Erick Tryzelaar <erick.tryzelaar@gmail.com>`; `David Tolnay <dtolnay@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a.txt` (SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a`).
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).

### `serde_json` 1.0.151
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `c841b55ecdae098c80dcae9cf767f6f8a0c2cdb3416bbef72181df4d0fe73f14`.
- Source archive: `https://static.crates.io/crates/serde_json/serde_json-1.0.151.crate`.
- Repository declared by package metadata: `https://github.com/serde-rs/json`.
- Authors declared by package metadata: `Erick Tryzelaar <erick.tryzelaar@gmail.com>`; `David Tolnay <dtolnay@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a.txt` (SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a`).
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).

### `serde_path_to_error` 0.1.20
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `10a9ff822e371bb5403e391ecd83e182e0e77ba7f6fe0160b795797109d1b457`.
- Source archive: `https://static.crates.io/crates/serde_path_to_error/serde_path_to_error-0.1.20.crate`.
- Repository declared by package metadata: `https://github.com/dtolnay/path-to-error`.
- Authors declared by package metadata: `David Tolnay <dtolnay@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a.txt` (SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a`).
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).

### `serde_urlencoded` 0.7.1
- License expression: `MIT/Apache-2.0`.
- Locked crate archive SHA-256: `d3491c14715ca2294c4d6a88f15e84739788c1d030eed8c110436aafdaa2f3fd`.
- Source archive: `https://static.crates.io/crates/serde_urlencoded/serde_urlencoded-0.7.1.crate`.
- Repository declared by package metadata: `https://github.com/nox/serde_urlencoded`.
- Authors declared by package metadata: `Anthony Ramine <n.oxyde@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a.txt` (SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a`).
  - `LICENSE-MIT`: `texts/b9eb266294324f672cbe945fe8f2e32f85024f0d61a1a7d14382cdde0ac44769.txt` (SHA-256 `b9eb266294324f672cbe945fe8f2e32f85024f0d61a1a7d14382cdde0ac44769`).

### `sha1` 0.10.7
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `a978451301f4db1d02937a4ab3ccce137717b81826e79b7d49ffe3244a13c3b8`.
- Source archive: `https://static.crates.io/crates/sha1/sha1-0.10.7.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/hashes`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/b4eb00df6e2a4d22518fcaa6a2b4646f249b3a3c9814509b22bd2091f1392ff1.txt` (SHA-256 `b4eb00df6e2a4d22518fcaa6a2b4646f249b3a3c9814509b22bd2091f1392ff1`).

### `sha2` 0.11.0
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `446ba717509524cb3f22f17ecc096f10f4822d76ab5c0b9822c5f9c284e825f4`.
- Source archive: `https://static.crates.io/crates/sha2/sha2-0.11.0.crate`.
- Repository declared by package metadata: `https://github.com/RustCrypto/hashes`.
- Authors declared by package metadata: `RustCrypto Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5.txt` (SHA-256 `a9040321c3712d8fd0b09cf52b17445de04a23a10165049ae187cd39e5c86be5`).
  - `LICENSE-MIT`: `texts/831e0f43ad0bf014c1c4fec5767aae470434c1d66d6e671be2d823e729063e25.txt` (SHA-256 `831e0f43ad0bf014c1c4fec5767aae470434c1d66d6e671be2d823e729063e25`).

### `signal-hook-registry` 1.4.8
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `c4db69cba1110affc0e9f7bcd48bbf87b3f4fc7c61fc9155afd4c469eb3d6c1b`.
- Source archive: `https://static.crates.io/crates/signal-hook-registry/signal-hook-registry-1.4.8.crate`.
- Repository declared by package metadata: `https://github.com/vorner/signal-hook`.
- Authors declared by package metadata: `Michal 'vorner' Vaner <vorner@vorner.cz>`; `Masaki Hara <ackie.h.gmai@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/503558bfefe66ca15e4e3f7955b3cb0ec87fd52f29bf24b336af7bd00e946d5c.txt` (SHA-256 `503558bfefe66ca15e4e3f7955b3cb0ec87fd52f29bf24b336af7bd00e946d5c`).

### `slab` 0.4.12
- License expression: `MIT`.
- Locked crate archive SHA-256: `0c790de23124f9ab44544d7ac05d60440adc586479ce501c1d6d7da3cd8c9cf5`.
- Source archive: `https://static.crates.io/crates/slab/slab-0.4.12.crate`.
- Repository declared by package metadata: `https://github.com/tokio-rs/slab`.
- Authors declared by package metadata: `Carl Lerche <me@carllerche.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/8ce0830173fdac609dfb4ea603fdc002c2f4af0dc9b1a005653f5da9cf534b18.txt` (SHA-256 `8ce0830173fdac609dfb4ea603fdc002c2f4af0dc9b1a005653f5da9cf534b18`).

### `smallvec` 1.15.2
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `8ed6a63f02c8539c91a8685a86f4099661ba3da017932f6ebbea6de3f0fa7c90`.
- Source archive: `https://static.crates.io/crates/smallvec/smallvec-1.15.2.crate`.
- Repository declared by package metadata: `https://github.com/servo/rust-smallvec`.
- Authors declared by package metadata: `The Servo Project Developers`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/0b28172679e0009b655da42797c03fd163a3379d5cfa67ba1f1655e974a2a1a9.txt` (SHA-256 `0b28172679e0009b655da42797c03fd163a3379d5cfa67ba1f1655e974a2a1a9`).

### `socket2` 0.6.5
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `c3d1e2c7f27f8d4cb10542a02c49005dbd6e93095799d6f3be745fae9f8fedd4`.
- Source archive: `https://static.crates.io/crates/socket2/socket2-0.6.5.crate`.
- Repository declared by package metadata: `https://github.com/rust-lang/socket2`.
- Authors declared by package metadata: `Alex Crichton <alex@alexcrichton.com>`; `Thomas de Zeeuw <thomasdezeeuw@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/378f5840b258e2779c39418f3f2d7b2ba96f1c7917dd6be0713f88305dbda397.txt` (SHA-256 `378f5840b258e2779c39418f3f2d7b2ba96f1c7917dd6be0713f88305dbda397`).

### `symphonia` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `5773a4c030a19d9bfaa090f49746ff35c75dfddfa700df7a5939d5e076a57039`.
- Source archive: `https://static.crates.io/crates/symphonia/symphonia-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-bundle-flac` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `c91565e180aea25d9b80a910c546802526ffd0072d0b8974e3ebe59b686c9976`.
- Source archive: `https://static.crates.io/crates/symphonia-bundle-flac/symphonia-bundle-flac-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-bundle-mp3` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `4872dd6bb56bf5eac799e3e957aa1981086c3e613b27e0ac23b176054f7c57ed`.
- Source archive: `https://static.crates.io/crates/symphonia-bundle-mp3/symphonia-bundle-mp3-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-codec-aac` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `4c263845aa86881416849c1729a54c7f55164f8b96111dba59de46849e73a790`.
- Source archive: `https://static.crates.io/crates/symphonia-codec-aac/symphonia-codec-aac-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`; `Kostya Shishkov <kostya.shiskov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-codec-pcm` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `4e89d716c01541ad3ebe7c91ce4c8d38a7cf266a3f7b2f090b108fb0cb031d95`.
- Source archive: `https://static.crates.io/crates/symphonia-codec-pcm/symphonia-codec-pcm-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-codec-vorbis` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `f025837c309cd69ffef572750b4a2257b59552c5399a5e49707cc5b1b85d1c73`.
- Source archive: `https://static.crates.io/crates/symphonia-codec-vorbis/symphonia-codec-vorbis-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-core` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `ea00cc4f79b7f6bb7ff87eddc065a1066f3a43fe1875979056672c9ef948c2af`.
- Source archive: `https://static.crates.io/crates/symphonia-core/symphonia-core-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-format-isomp4` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `243739585d11f81daf8dac8d9f3d18cc7898f6c09a259675fc364b382c30e0a5`.
- Source archive: `https://static.crates.io/crates/symphonia-format-isomp4/symphonia-format-isomp4-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-format-mkv` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `122d786d2c43a49beb6f397551b4a050d8229eaa54c7ddf9ee4b98899b8742d0`.
- Source archive: `https://static.crates.io/crates/symphonia-format-mkv/symphonia-format-mkv-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Dariusz Niedoba <dariusz.niedoba@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-format-ogg` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `2b4955c67c1ed3aa8ae8428d04ca8397fbef6a19b2b051e73b5da8b1435639cb`.
- Source archive: `https://static.crates.io/crates/symphonia-format-ogg/symphonia-format-ogg-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-format-riff` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `c2d7c3df0e7d94efb68401d81906eae73c02b40d5ec1a141962c592d0f11a96f`.
- Source archive: `https://static.crates.io/crates/symphonia-format-riff/symphonia-format-riff-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`; `dedobbin <dedobbindedobbin@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-metadata` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `36306ff42b9ffe6e5afc99d49e121e0bd62fe79b9db7b9681d48e29fa19e6b16`.
- Source archive: `https://static.crates.io/crates/symphonia-metadata/symphonia-metadata-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `symphonia-utils-xiph` 0.5.5
- License expression: `MPL-2.0`.
- Locked crate archive SHA-256: `ee27c85ab799a338446b68eec77abf42e1a6f1bb490656e121c6e27bfbab9f16`.
- Source archive: `https://static.crates.io/crates/symphonia-utils-xiph/symphonia-utils-xiph-0.5.5.crate`.
- Repository declared by package metadata: `https://github.com/pdeljanov/Symphonia`.
- Authors declared by package metadata: `Philip Deljanov <philip.deljanov@gmail.com>`.
- MPL-2.0 terms: `https://www.mozilla.org/MPL/2.0/`; obtain the checksum-bound source archive above.
- The locked crate archive contains no standalone license or notice file; its metadata and source archive above are retained without inventing text.

### `sync_wrapper` 1.0.2
- License expression: `Apache-2.0`.
- Locked crate archive SHA-256: `0bf256ce5efdfa370213c1dabab5935a12e49f2c58d15e9eac2870d3b4f27263`.
- Source archive: `https://static.crates.io/crates/sync_wrapper/sync_wrapper-1.0.2.crate`.
- Repository declared by package metadata: `https://github.com/Actyx/sync_wrapper`.
- Authors declared by package metadata: `Actyx AG <developer@actyx.io>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594.txt` (SHA-256 `0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594`).

### `thiserror` 2.0.20
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `ec86235f5fcc2a73650310756d2ac5b138a5780bbbdfae3eeccec992c435ba4f`.
- Source archive: `https://static.crates.io/crates/thiserror/thiserror-2.0.20.crate`.
- Repository declared by package metadata: `https://github.com/dtolnay/thiserror`.
- Authors declared by package metadata: `David Tolnay <dtolnay@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a.txt` (SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a`).
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).

### `tokio` 1.53.1
- License expression: `MIT`.
- Locked crate archive SHA-256: `202caea871b69668250d242070849eb495be178ed697a3e98aebce5bc81a0bed`.
- Source archive: `https://static.crates.io/crates/tokio/tokio-1.53.1.crate`.
- Repository declared by package metadata: `https://github.com/tokio-rs/tokio`.
- Authors declared by package metadata: `Tokio Contributors <team@tokio.rs>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/253cd04c6714889df2d32f3f64d669179a1c95c76ac43c40882c52eb06bc3552.txt` (SHA-256 `253cd04c6714889df2d32f3f64d669179a1c95c76ac43c40882c52eb06bc3552`).

### `tokio-tungstenite` 0.29.0
- License expression: `MIT`.
- Locked crate archive SHA-256: `8f72a05e828585856dacd553fba484c242c46e391fb0e58917c942ee9202915c`.
- Source archive: `https://static.crates.io/crates/tokio-tungstenite/tokio-tungstenite-0.29.0.crate`.
- Repository declared by package metadata: `https://github.com/snapview/tokio-tungstenite`.
- Authors declared by package metadata: `Daniel Abramov <dabramov@snapview.de>`; `Alexey Galakhov <agalakhov@snapview.de>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/fdd55e2b2da854b0fbdc1e607df7c2ba1e1ebf91ecb77c515511ebeef972bc8f.txt` (SHA-256 `fdd55e2b2da854b0fbdc1e607df7c2ba1e1ebf91ecb77c515511ebeef972bc8f`).

### `tower` 0.5.3
- License expression: `MIT`.
- Locked crate archive SHA-256: `ebe5ef63511595f1344e2d5cfa636d973292adc0eec1f0ad45fae9f0851ab1d4`.
- Source archive: `https://static.crates.io/crates/tower/tower-0.5.3.crate`.
- Repository declared by package metadata: `https://github.com/tower-rs/tower`.
- Authors declared by package metadata: `Tower Maintainers <team@tower-rs.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/4249c8e6c5ebb85f97c77e6457c6fafc1066406eb8f1ef61e796fbdc5ff18482.txt` (SHA-256 `4249c8e6c5ebb85f97c77e6457c6fafc1066406eb8f1ef61e796fbdc5ff18482`).

### `tower-layer` 0.3.3
- License expression: `MIT`.
- Locked crate archive SHA-256: `121c2a6cda46980bb0fcd1647ffaf6cd3fc79a013de288782836f6df9c48780e`.
- Source archive: `https://static.crates.io/crates/tower-layer/tower-layer-0.3.3.crate`.
- Repository declared by package metadata: `https://github.com/tower-rs/tower`.
- Authors declared by package metadata: `Tower Maintainers <team@tower-rs.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/4249c8e6c5ebb85f97c77e6457c6fafc1066406eb8f1ef61e796fbdc5ff18482.txt` (SHA-256 `4249c8e6c5ebb85f97c77e6457c6fafc1066406eb8f1ef61e796fbdc5ff18482`).

### `tower-service` 0.3.3
- License expression: `MIT`.
- Locked crate archive SHA-256: `8df9b6e13f2d32c91b9bd719c00d1958837bc7dec474d94952798cc8e69eeec3`.
- Source archive: `https://static.crates.io/crates/tower-service/tower-service-0.3.3.crate`.
- Repository declared by package metadata: `https://github.com/tower-rs/tower`.
- Authors declared by package metadata: `Tower Maintainers <team@tower-rs.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/4249c8e6c5ebb85f97c77e6457c6fafc1066406eb8f1ef61e796fbdc5ff18482.txt` (SHA-256 `4249c8e6c5ebb85f97c77e6457c6fafc1066406eb8f1ef61e796fbdc5ff18482`).

### `tracing` 0.1.44
- License expression: `MIT`.
- Locked crate archive SHA-256: `63e71662fa4b2a2c3a26f570f037eb95bb1f85397f3cd8076caed2f026a6d100`.
- Source archive: `https://static.crates.io/crates/tracing/tracing-0.1.44.crate`.
- Repository declared by package metadata: `https://github.com/tokio-rs/tracing`.
- Authors declared by package metadata: `Eliza Weisman <eliza@buoyant.io>`; `Tokio Contributors <team@tokio.rs>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/898b1ae9821e98daf8964c8d6c7f61641f5f5aa78ad500020771c0939ee0dea1.txt` (SHA-256 `898b1ae9821e98daf8964c8d6c7f61641f5f5aa78ad500020771c0939ee0dea1`).

### `tracing-core` 0.1.36
- License expression: `MIT`.
- Locked crate archive SHA-256: `db97caf9d906fbde555dd62fa95ddba9eecfd14cb388e4f491a66d74cd5fb79a`.
- Source archive: `https://static.crates.io/crates/tracing-core/tracing-core-0.1.36.crate`.
- Repository declared by package metadata: `https://github.com/tokio-rs/tracing`.
- Authors declared by package metadata: `Tokio Contributors <team@tokio.rs>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/898b1ae9821e98daf8964c8d6c7f61641f5f5aa78ad500020771c0939ee0dea1.txt` (SHA-256 `898b1ae9821e98daf8964c8d6c7f61641f5f5aa78ad500020771c0939ee0dea1`).

### `tungstenite` 0.29.0
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `6c01152af293afb9c7c2a57e4b559c5620b421f6d133261c60dd2d0cdb38e6b8`.
- Source archive: `https://static.crates.io/crates/tungstenite/tungstenite-0.29.0.crate`.
- Repository declared by package metadata: `https://github.com/snapview/tungstenite-rs`.
- Authors declared by package metadata: `Alexey Galakhov`; `Daniel Abramov`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2.txt` (SHA-256 `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`).
  - `LICENSE-MIT`: `texts/7fea0ee51a4ca5d5cea7464135fd55e8b09caf3a61da3d451ac8a22af95c033f.txt` (SHA-256 `7fea0ee51a4ca5d5cea7464135fd55e8b09caf3a61da3d451ac8a22af95c033f`).

### `typenum` 1.20.1
- License expression: `MIT OR Apache-2.0`.
- Locked crate archive SHA-256: `b6f5e870be6c3b371b77fe0ee0bafb859fa4964b4404c27de1d380043c4dda20`.
- Source archive: `https://static.crates.io/crates/typenum/typenum-1.20.1.crate`.
- Repository declared by package metadata: `https://github.com/paholg/typenum`.
- Packaged license, permission, and copyright notices:
  - `LICENSE`: `texts/db11fec9946737df39ca3898d9cd8c10ec6f6c3a884a6802b0ad0b81b4e8f23a.txt` (SHA-256 `db11fec9946737df39ca3898d9cd8c10ec6f6c3a884a6802b0ad0b81b4e8f23a`).
  - `LICENSE-APACHE`: `texts/516b24e051bf5630880ebbd55c40a25ce9552ebaf8970a53e8976eb70e522406.txt` (SHA-256 `516b24e051bf5630880ebbd55c40a25ce9552ebaf8970a53e8976eb70e522406`).
  - `LICENSE-MIT`: `texts/a825bd853ab71619a4923d7b4311221427848070ff44d990da39b0b274c1683f.txt` (SHA-256 `a825bd853ab71619a4923d7b4311221427848070ff44d990da39b0b274c1683f`).

### `zerocopy` 0.8.56
- License expression: `BSD-2-Clause OR Apache-2.0 OR MIT`.
- Locked crate archive SHA-256: `556764e583adb45a9f8d413c2a147fa7e8d821e48e12b14fd560b607998b75eb`.
- Source archive: `https://static.crates.io/crates/zerocopy/zerocopy-0.8.56.crate`.
- Repository declared by package metadata: `https://github.com/google/zerocopy`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-APACHE`: `texts/9d185ac6703c4b0453974c0d85e9eee43e6941009296bb1f5eb0b54e2329e9f3.txt` (SHA-256 `9d185ac6703c4b0453974c0d85e9eee43e6941009296bb1f5eb0b54e2329e9f3`).
  - `LICENSE-BSD`: `texts/83c1763356e822adde0a2cae748d938a73fdc263849ccff6b27776dff213bd32.txt` (SHA-256 `83c1763356e822adde0a2cae748d938a73fdc263849ccff6b27776dff213bd32`).
  - `LICENSE-MIT`: `texts/1a2f5c12ddc934d58956aa5dbdd3255fe55fd957633ab7d0d39e4f0daa73f7df.txt` (SHA-256 `1a2f5c12ddc934d58956aa5dbdd3255fe55fd957633ab7d0d39e4f0daa73f7df`).

### `zmij` 1.0.23
- License expression: `MIT`.
- Locked crate archive SHA-256: `29666d0abbfad1e3dc4dcf6144730dd3a3ab225bbbdac83319345b1b44ccfc1b`.
- Source archive: `https://static.crates.io/crates/zmij/zmij-1.0.23.crate`.
- Repository declared by package metadata: `https://github.com/dtolnay/zmij`.
- Authors declared by package metadata: `David Tolnay <dtolnay@gmail.com>`.
- Packaged license, permission, and copyright notices:
  - `LICENSE-MIT`: `texts/23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3.txt` (SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`).
