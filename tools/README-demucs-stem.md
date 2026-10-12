
# Offline Demucs stems for Mixxx

`demucs_onnx_to_stem.py` separates a stereo audio file into four stems and
creates a `.stem.mp4` file that Mixxx can load with its existing stem volume,
mute, and effects controls. Separation runs offline; the fixed-size
`htdemucs` ONNX model processes 7.8-second segments and is not suitable for
low-latency live separation.

## Requirements

- Mixxx built with `STEM=ON` and `FFMPEG=ON`. Mixxx 2.5.6 cannot load these
  Stem files; installing FFmpeg separately does not change that.
- Python with NumPy, ONNX Runtime, and Mutagen installed.
- OpenVINO for the preferred CPU FP32 engine. It is included in the dependency
  list; if it is absent, the helper warns and uses ONNX Runtime CPU instead.
- The `htdemucs.onnx` model and matching `htdemucs.json` contract file.
- FFmpeg available on `PATH` (or pass its executable with `--ffmpeg`).

Keep the model outside the Mixxx source tree. The tool does not install,
download, or redistribute the model. Check the applicable licenses for the
model weights and dependencies before using or distributing generated files.

Install the helper dependencies into the Python environment Mixxx will use:

```powershell
python -m pip install -r tools\requirements-demucs-stem.txt
```

## Generate from the library

Right-click one or more original tracks and select **Analyze > Create Stems**.
Generation starts without executable or model selection dialogs. There is no
Stem generation settings menu.

- Tracks are processed one at a time in a non-modal background queue.
- Each `song.stem.mp4` is saved **next to its original**, not in a separate
  `stems` folder, and automatically added to the library.
- The source file and its metadata are not modified.
- Existing outputs, missing sources, Stem inputs, and conflicting output names
  are skipped and reported. The library action never overwrites existing files.
- The progress window allows cancellation of the active job and remaining queue.
  Inference cancellation waits for the current segment to finish. During FFmpeg
  processing, the helper terminates and waits for the child process.
- Temporary files are cleaned up after success, failure, or cancellation.
  Application shutdown requests cancellation, waits up to 30 seconds, and then
  terminates the helper if necessary.
- Failures, warnings, and skipped tracks are available in the completion dialog.
  The existing Analyze/Reanalyze actions keep their original behavior.

### Automatic discovery

Mixxx first honors valid previously saved Python, FFmpeg, and model paths.
Missing or stale paths fall back to automatic discovery:

| Component | Search locations, in order after any valid saved path |
| --- | --- |
| Python | `tools/python/` under the resource directory, then `python.exe` on Windows or `python3` elsewhere on `PATH`, then `python` |
| FFmpeg | `tools/` under the resource directory, the resource directory itself, then `PATH` |
| Model | `models/htdemucs.onnx` under the resource directory, under the current settings directory, then `Documents\mixxx\htdemucs.onnx` |

Each model location must also contain the matching `htdemucs.json` file.
The helper and dependency list are installed with STEM-enabled builds, but
Python, its packages, FFmpeg, and the model remain external dependencies unless
provided separately in the supported bundled locations. Nothing is downloaded
automatically. Missing components produce an explicit error.

### Inference engine

The library action uses four inference threads and automatically prefers
**OpenVINO CPU FP32**. ONNX Runtime is still required: it validates the model's
input/output contract and embedded metadata before OpenVINO compilation.
The ONNX Runtime session is released before OpenVINO compiles the model.

If OpenVINO is not installed, the helper warns and uses ONNX Runtime CPU with
its memory arena disabled. OpenVINO initialization and inference errors are
reported rather than silently switching engines. Mixxx logs the selected engine.
No DirectML, GPU, or reduced-precision execution is enabled.

## Output and metadata

The generated MP4 contains five stereo audio streams: the original mix, drums,
bass, other, and vocals. It also includes the `moov/udta/stem` JSON manifest
Mixxx uses to identify stem names and colors. Audio is lossless ALAC by default.
Load the output in a primary deck to use the existing Stem volume, mute, and
per-stem effects controls.

Generated files copy the original's FFmpeg-supported text tags, including title,
artist, album, date, genre, and track number. MP4 input tags are also preserved
directly; ID3 and Vorbis/FLAC BPM, key, and ReplayGain tags are translated into
Mixxx-compatible MP4 fields. Embedded JPEG/PNG artwork is preserved in the MP4
cover-art tag, without adding a video track (ffprobe may display this tag as an
attached-picture stream). Unsupported artwork formats produce a warning.
Stem stream names and the Stem manifest remain separate from the song title.
The generated song title is prefixed with `[STEM] ` so it is easy to distinguish
from the original in Mixxx. An existing prefix is not duplicated. If the source
has no non-blank title tag, its filename without the extension is used.
Already-generated or imported Stem files are not retagged automatically.

## Standalone helper

Run the same generator from PowerShell:

```powershell
python tools\demucs_onnx_to_stem.py "C:\Music\track.wav" `
  --model "C:\Models\htdemucs.onnx"
```

This creates `track.stem.mp4` next to the input, using lossless ALAC audio by
default. To create a smaller AAC file, pass `--codec aac`. `--threads N`
sets the inference thread count for either engine (default: 4).
`--backend auto` is the default; `--backend openvino` requires OpenVINO,
and `--backend onnxruntime` explicitly selects ONNX Runtime CPU.
Use `--output` to choose a different destination ending in `.stem.mp4`.
An existing output is preserved
unless `--force` is explicitly supplied.

## Tests

Run the focused helper tests with:

```powershell
python tools\test_demucs_onnx_to_stem.py -v
```

FFmpeg and ffprobe are needed for metadata integration tests. Without a model,
the real-engine comparison is skipped. To include the CPU FP32 numeric comparison:

```powershell
$env:MIXXX_STEM_TEST_MODEL = "C:\Models\htdemucs.onnx"
python tools\test_demucs_onnx_to_stem.py -v
```

The C++ queue tests cover discovery, output protection, failures, cancellation,
and shutdown. Run `mixxx-test.exe` from your build directory with
`--gtest_filter=*Stem*` to include these and the existing Stem tests.
The optional real-model queue test requires `MIXXX_STEM_TEST_MODEL`,
`MIXXX_STEM_TEST_HELPER` (the generator script path), and
`MIXXX_STEM_TEST_FFMPEG` (the executable path).

## Windows build

Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for the supported build tools and
dependency setup. Enable `STEM=ON` and `FFMPEG=ON` in the CMake configuration.
Enable `BUILD_TESTING=ON` if you want to build the C++ tests.
Choose build and installation directories separate from your stable installation.

Launch the resulting executable with `--settings-path` pointing to a separate
profile directory, for example `C:\MixxxProfiles\stems`, to keep its settings and
database separate from the installed stable version. Example paths in this
document should be replaced with your own audio, model, and profile locations.
This is a development build; verify audio routing and Stem controls on a real
DJ setup before live use.