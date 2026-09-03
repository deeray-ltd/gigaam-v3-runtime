# GigaAM v3 model packs

GigaAM v3 Runtime accepts only a GigaAM v3 model package. Weights, vocabularies, and
the optional VAD graph are external artifacts under their own terms; neither this
repository nor its container definitions include a model package.

The runtime reads one authoritative file, config.kv, from a package directory. A nonblank,
non-comment line must be one nonempty key=value pair. Duplicate keys, empty keys or
values, malformed lines, non-UTF-8 configuration, unsupported values, and unknown keys
are refusals. The package is validated before a frontend asset, an ONNX graph, or an ONNX
Runtime session is selected.

## Version 1 schema

format_version=1 is mandatory. It is the required schema discriminator and is validated
and consumed separately from the runtime definitions. There is no unversioned or legacy
configuration path.

Version 1 is a closed inventory:

| Class | Count | Meaning |
| --- | ---: | --- |
| Required schema discriminator | 1 | format_version |
| Required typed runtime keys | 35 | Frontend, CTC, RNN-T, and VAD definitions below |
| Required entries | 36 | The discriminator plus all typed runtime keys |
| Optional opaque retained keys | 8 | Named metadata/extensions below |
| Admitted version-1 keys | 44 | No other key is accepted |

### Required discriminator

- format_version

### Required typed runtime keys

| Capability | Keys |
| --- | --- |
| Frontend (8) | sample_rate, n_mels, hop_length, n_fft, center, log_clamp_min, log_clamp_max, frames_per_sec |
| CTC (8) | ctc.vocab, ctc.blank_id, ctc.encoder_fp32, ctc.encoder_fp16io32, ctc.input_names, ctc.output_names, ctc.out_dim, ctc.out_layout |
| RNN-T (18) | rnnt.vocab, rnnt.blank_id, rnnt.pred_hidden, rnnt.max_symbols_per_step, rnnt.encoder_fp16io32, rnnt.decoder_fp16, rnnt.joint_fp16, rnnt.encoder_fp32, rnnt.decoder_fp32, rnnt.joint_fp32, rnnt.out_dim, rnnt.out_layout, rnnt.input_names, rnnt.output_names, rnnt.decoder_inputs, rnnt.decoder_outputs, rnnt.joint_inputs, rnnt.joint_outputs |
| VAD (1) | vad.model |

### Optional opaque retained keys

The following eight keys may be absent. If present, their values must be nonempty, but
they are retained only as metadata or extensions: they do not select an artifact, do not
make an artifact required, and do not configure runtime behavior.

- win_length
- subsampling_factor
- pos_emb_max_len
- ctc.encoder_fp16
- rnnt.pred_layers
- rnnt.encoder_fp16
- source
- exported

A new key, a changed key meaning, or a new selectable artifact role requires an explicitly
supported schema version and a migration rule. Version 1 never expands silently. Package
hashes, provenance verification, runtime compatibility ranges, offline verification, and
schema retirement are not current package guarantees.

## Typed definitions and compatibility

The required frontend values define the model rate and feature contract. sample_rate,
n_mels, hop_length, and n_fft are nonzero; n_fft is at least two; center is exactly true
or false; log clamp values are finite, positive, and ordered; frames_per_sec is finite and
positive. Service and CLI bind their time-based transcription settings to the validated
model sample rate before runtime construction.

CTC and RNN-T blank/output dimensions, RNN-T prediction width, and the RNN-T
max_symbols_per_step are checked before use. ctc.out_layout and rnnt.out_layout are
spelled exactly t_d or d_t.

Tensor-name values are comma-separated, nonempty, and unique. Their V1 arities are fixed:

| Contract | Required names |
| --- | ---: |
| CTC encoder inputs / outputs | 2 / 2 |
| RNN-T encoder inputs / outputs | 2 / 2 |
| RNN-T decoder inputs / outputs | 3 / 3 |
| RNN-T joint inputs / outputs | 2 / 1 |

The array order gives the typed data/length and recurrent/joint roles; callers do not
choose these roles through positional command-line configuration.

## Artifacts

The frontend tables have fixed roles and names:

- stft_window.f32
- mel_fbank.f32

Each table is a little-endian binary sequence of a u32 dimension count, that many u32
dimensions, and exactly the resulting number of f32 values. The package selects CTC
fp32 or fp16io32 encoder artifacts through the corresponding required CTC keys. It
selects RNN-T encoder, decoder, joint, and vocabulary artifacts through the corresponding
required RNN-T keys when RNN-T is enabled. vad.model is selected only when VAD endpointing
is requested.

Every configured artifact path must be a nonempty relative path without a backslash,
absolute prefix, current-directory component, or parent traversal. Immediately before a
selected artifact is opened, its canonical target must remain within the package directory
and be a regular file. An unused RNN-T, precision, or VAD artifact is not opened merely
because its configuration key exists.

## Operating a pack

Pass the package directory with the Service or CLI model option. The package must contain
config.kv and the selected runtime artifacts. An image intentionally starts without a
bundled package; mount one at the configured model path before starting the service.

The schema establishes configuration and selected-asset safety, not redistribution rights
or model quality. Obtain authorization for the exact GigaAM, vocabulary, and VAD artifacts
you deploy, and validate a real package with the selected CPU, CUDA, or TensorRT runtime
before treating an image as a release artifact.
