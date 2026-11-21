# SentencePiece model types

The checked-in Rust types are generated from Google's SentencePiece
[`src/sentencepiece_model.proto` at release v0.2.3](https://github.com/google/sentencepiece/blob/v0.2.3/src/sentencepiece_model.proto), SHA-256
`75c6fd6676ec53eba5da38336692cea80364c7625177d5a680dbde569db12279`.
The schema is Copyright 2016 Google Inc. and licensed under Apache-2.0;
see the [license](../../crates/model-package/licenses/sentencepiece/LICENSE).
The Rust types are derived from that schema.

Regeneration uses prost-build 0.14.6, this directory's Cargo.lock, and
protoc 36.2 (Linux x86_64 release archive SHA-256
`121f6c7afe1d4d0e3ea6aab9432038599250134cbf4474cb1167d2c7decd4278`).
Check both SHA-256 values with `sha256sum` and check `protoc --version` before generation.
Place the verified schema in its own directory. Run the following from this directory
with the repository Rust toolchain and rustfmt.

```sh
PROTOC=/path/to/protoc cargo run --locked -- /absolute/path/to/schema-directory
```

The tool removes schema comments and emits rustfmt-formatted output; compare
`crates/model-package/src/generated/sentencepiece.rs` to the checked-in copy.
Ordinary server builds use the generated file and do not run this tool or protoc.

The small text fixture in `crates/model-package/tests/fixtures` was encoded
independently using protoc 36.2 and the same verified schema with message type
`sentencepiece.ModelProto`. Its encoded SHA-256 is
`7f5b5d0fb374668606334e37842cd926465c60d93ed7738441ad785aa33e9243`.
Both crates compare their prost serialization to that witness.
