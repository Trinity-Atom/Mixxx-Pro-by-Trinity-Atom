#!/usr/bin/env python
import argparse
import base64
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import warnings

import mutagen
from mutagen.flac import Picture
from mutagen.id3 import ID3
from mutagen.mp4 import MP4, MP4Cover
import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 44100
SOURCES = ("drums", "bass", "other", "vocals")
COLORS = ("#fd4a4a", "#ffff00", "#00e8e8", "#ad65ff")


def validate_contract(session, contract):
    expected_input = {
        "name": "audio",
        "dtype": "float32",
        "shape": [1, 2, 343980],
    }
    expected_output = {
        "name": "stems",
        "dtype": "float32",
        "shape": [1, 4, 2, 343980],
    }
    if contract.get("model") != "htdemucs":
        raise ValueError("Expected the htdemucs model.")
    if contract.get("sample_rate") != SAMPLE_RATE:
        raise ValueError(f"Expected a {SAMPLE_RATE} Hz model.")
    if contract.get("sources") != list(SOURCES):
        raise ValueError(f"Expected sources in this order: {', '.join(SOURCES)}.")
    if contract.get("input") != expected_input:
        raise ValueError(f"Unexpected model input contract: {contract.get('input')!r}")
    if contract.get("output") != expected_output:
        raise ValueError(f"Unexpected model output contract: {contract.get('output')!r}")

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
            len(inputs) != 1
            or inputs[0].name != "audio"
            or inputs[0].shape != expected_input["shape"]):
        raise ValueError("The ONNX input does not match the htdemucs contract.")
    if inputs[0].type != "tensor(float)":
        raise ValueError("The ONNX input must use float32.")
    if (
            len(outputs) != 1
            or outputs[0].name != "stems"
            or outputs[0].shape != expected_output["shape"]):
        raise ValueError("The ONNX output does not match the htdemucs contract.")
    if outputs[0].type != "tensor(float)":
        raise ValueError("The ONNX output must use float32.")
    embedded_contract = session.get_modelmeta().custom_metadata_map.get(
        "demucs_contract"
    )
    if embedded_contract is None or json.loads(embedded_contract) != contract:
        raise ValueError("The JSON contract does not match the ONNX model metadata.")


class OpenVinoSession:
    def __init__(self, model_path, threads):
        import openvino as ov

        core = ov.Core()
        self.compiled = core.compile_model(
            core.read_model(str(model_path)),
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": "f32",
                "INFERENCE_NUM_THREADS": threads,
            },
        )
        self.request = self.compiled.create_infer_request()
        if (
                list(self.compiled.input(0).shape) != [1, 2, 343980]
                or list(self.compiled.output(0).shape) != [1, 4, 2, 343980]
                or self.compiled.input(0).element_type != ov.Type.f32
                or self.compiled.output(0).element_type != ov.Type.f32):
            raise ValueError("OpenVINO model ports do not match the float32 Stem contract.")

    def run(self, output_names, inputs):
        if output_names != ["stems"] or set(inputs) != {"audio"}:
            raise ValueError("Unexpected OpenVINO input or output names.")
        self.request.infer({self.compiled.input(0): inputs["audio"]})
        return [self.request.get_output_tensor(0).data.copy()]


def create_session(model_path, contract, threads, backend="auto"):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    validate_contract(session, contract)
    if backend == "onnxruntime":
        print("STEM_BACKEND ONNX Runtime CPU", flush=True)
        return session
    if importlib.util.find_spec("openvino") is None:
        if backend == "openvino":
            raise RuntimeError("OpenVINO is not installed.")
        warnings.warn("OpenVINO is not installed; using ONNX Runtime CPU.")
        print("STEM_BACKEND ONNX Runtime CPU", flush=True)
        return session
    del session
    gc.collect()
    session = OpenVinoSession(model_path, threads)
    print("STEM_BACKEND OpenVINO CPU FP32", flush=True)
    return session


def separate_to_raw(
        input_path, output_path, session, segment_frames, overlap_frames,
        progress=None, cancel_file=None):
    audio = np.memmap(input_path, dtype=np.float32, mode="r")
    if audio.size == 0 or audio.size % 2:
        raise ValueError("Decoded input must contain non-empty stereo float32 audio.")
    audio = audio.reshape(-1, 2)
    frame_count = audio.shape[0]
    if segment_frames <= 0 or overlap_frames <= 0 or overlap_frames >= segment_frames // 2:
        raise ValueError("The overlap must be positive and less than half the segment.")
    hop = segment_frames - overlap_frames
    padded_frames = frame_count + 2 * overlap_frames
    chunk_count = max(1, math.ceil(max(0, padded_frames - segment_frames) / hop) + 1)

    separated = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(SOURCES), frame_count, 2),
    )
    separated[:] = 0
    weights = np.memmap(
        output_path.parent / "weights.f32",
        dtype=np.float32,
        mode="w+",
        shape=(frame_count,),
    )
    weights[:] = 0
    ramp = np.arange(1, overlap_frames + 1, dtype=np.float32) / (overlap_frames + 1)

    for chunk_index in range(chunk_count):
        _check_canceled(cancel_file)
        padded_start = chunk_index * hop
        source_start = padded_start - overlap_frames
        source_end = source_start + segment_frames
        chunk = np.zeros((2, segment_frames), dtype=np.float32)
        copy_start = max(source_start, 0)
        copy_end = min(source_end, frame_count)
        if copy_start < copy_end:
            chunk_offset = copy_start - source_start
            chunk[:, chunk_offset:chunk_offset + copy_end - copy_start] = (
                audio[copy_start:copy_end].T
            )

        window = np.ones(segment_frames, dtype=np.float32)
        if chunk_index:
            window[:overlap_frames] = ramp
        if chunk_index + 1 < chunk_count:
            window[-overlap_frames:] = ramp[::-1]

        result = session.run(["stems"], {"audio": chunk[np.newaxis, ...]})[0]
        if result.shape != (1, len(SOURCES), 2, segment_frames):
            raise ValueError(f"Unexpected model output shape: {result.shape!r}")
        if not np.isfinite(result).all():
            raise ValueError(f"The model returned non-finite samples in chunk {chunk_index}.")

        output_start = max(source_start, 0)
        output_end = min(source_end, frame_count)
        if output_start >= output_end:
            continue
        chunk_output_start = output_start - source_start
        chunk_output_end = chunk_output_start + output_end - output_start
        chunk_weights = window[chunk_output_start:chunk_output_end]
        separated[:, output_start:output_end, :] += (
            result[0, :, :, chunk_output_start:chunk_output_end]
            .transpose(0, 2, 1)
            * chunk_weights[np.newaxis, :, np.newaxis]
        )
        weights[output_start:output_end] += chunk_weights
        if progress is not None:
            progress((chunk_index + 1) * 100 // chunk_count)

    if not np.all(weights > 0):
        raise ValueError("Overlapping inference left input samples without output.")
    block_frames = 65536
    for start in range(0, frame_count, block_frames):
        end = min(start + block_frames, frame_count)
        separated[:, start:end, :] /= weights[np.newaxis, start:end, np.newaxis]
    separated.flush()
    weights.flush()
    del separated
    del weights
    del audio


def _iter_atoms(data, start, end):
    offset = start
    while offset < end:
        if end - offset < 8:
            raise ValueError("Invalid MP4 atom header.")
        size = struct.unpack_from(">I", data, offset)[0]
        atom_type = bytes(data[offset + 4:offset + 8])
        header_size = 8
        if size == 1:
            if end - offset < 16:
                raise ValueError("Invalid extended MP4 atom header.")
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header_size = 16
        elif size == 0:
            size = end - offset
        if size < header_size or offset + size > end:
            raise ValueError(f"Invalid size for MP4 atom {atom_type!r}.")
        yield offset, size, header_size, atom_type
        offset += size
    if offset != end:
        raise ValueError("MP4 atoms do not end at their container boundary.")


def _write_atom_size(data, offset, size, header_size):
    if header_size == 16:
        struct.pack_into(">Q", data, offset + 8, size)
    elif size < 2**32:
        struct.pack_into(">I", data, offset, size)
    else:
        raise ValueError("MP4 atom exceeds the supported 32-bit size.")


def _read_top_level_atoms(file_path):
    file_size = file_path.stat().st_size
    offset = 0
    with file_path.open("rb") as source:
        while offset < file_size:
            source.seek(offset)
            header = source.read(8)
            if len(header) != 8:
                raise ValueError("Invalid MP4 atom header.")
            size = struct.unpack_from(">I", header)[0]
            atom_type = header[4:8]
            header_size = 8
            if size == 1:
                extended_size = source.read(8)
                if len(extended_size) != 8:
                    raise ValueError("Invalid extended MP4 atom header.")
                size = struct.unpack(">Q", extended_size)[0]
                header_size = 16
            elif size == 0:
                size = file_size - offset
            if size < header_size or offset + size > file_size:
                raise ValueError(f"Invalid size for MP4 atom {atom_type!r}.")
            yield offset, size, header_size, atom_type
            offset += size


def embed_stem_manifest(mp4_path, manifest):
    top_atoms = list(_read_top_level_atoms(mp4_path))
    moov = next((atom for atom in top_atoms if atom[3] == b"moov"), None)
    if moov is None:
        raise ValueError("FFmpeg output has no MP4 moov atom.")
    moov_offset, moov_size, moov_header, _ = moov
    if any(
            atom_type == b"mdat" and offset > moov_offset
            for offset, _, _, atom_type in top_atoms):
        raise ValueError("The MP4 moov atom must follow media data to preserve offsets.")

    with mp4_path.open("rb") as source:
        source.seek(moov_offset)
        moov_data = bytearray(source.read(moov_size))
    if len(moov_data) != moov_size:
        raise ValueError("Unable to read the MP4 moov atom.")
    udta = next(
            (
                atom
                for atom in _iter_atoms(moov_data, moov_header, moov_size)
                if atom[3] == b"udta"
            ),
            None,
    )
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    stem_atom = struct.pack(">I", len(manifest_bytes) + 8) + b"stem" + manifest_bytes
    if udta is None:
        addition = struct.pack(">I", len(stem_atom) + 8) + b"udta" + stem_atom
        moov_data.extend(addition)
        _write_atom_size(moov_data, 0, moov_size + len(addition), moov_header)
    else:
        udta_offset, udta_size, udta_header, _ = udta
        if any(
                atom_type == b"stem"
                for _, _, _, atom_type in _iter_atoms(
                    moov_data,
                    udta_offset + udta_header,
                    udta_offset + udta_size)):
            raise ValueError("The output MP4 already contains a stem manifest.")
        udta_end = udta_offset + udta_size
        moov_data[udta_end:udta_end] = stem_atom
        _write_atom_size(
            moov_data, udta_offset, udta_size + len(stem_atom), udta_header
        )
        _write_atom_size(moov_data, 0, moov_size + len(stem_atom), moov_header)

    with tempfile.NamedTemporaryFile(
            prefix=f".{mp4_path.name}.",
            suffix=".tmp",
            dir=mp4_path.parent,
            delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with mp4_path.open("rb") as source, temporary_path.open("wb") as destination:
            bytes_to_copy = moov_offset
            while bytes_to_copy:
                chunk = source.read(min(bytes_to_copy, 1024 * 1024))
                if not chunk:
                    raise ValueError("Unable to copy MP4 data before the moov atom.")
                destination.write(chunk)
                bytes_to_copy -= len(chunk)
            destination.write(moov_data)
            source.seek(moov_offset + moov_size)
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        os.replace(temporary_path, mp4_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _check_canceled(cancel_file):
    if cancel_file is not None and cancel_file.exists():
        raise InterruptedError("Stem generation canceled.")


def _run_command(command, cancel_file=None):
    if cancel_file is None:
        subprocess.run(command, check=True)
        return
    _check_canceled(cancel_file)
    with subprocess.Popen(command) as process:
        while process.poll() is None:
            if cancel_file.exists():
                process.terminate()
                process.wait()
                raise InterruptedError("Stem generation canceled.")
            time.sleep(0.1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    _check_canceled(cancel_file)


def _decode_input(ffmpeg, input_path, raw_path, cancel_file=None):
    _run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "-c:a",
            "pcm_f32le",
            str(raw_path),
        ],
        cancel_file,
    )
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg produced no decoded audio.")


def _copy_extra_metadata(input_path, output):
    source = mutagen.File(input_path)
    if source is None:
        warnings.warn(
            f"Cannot read supplemental tags or artwork from {input_path}; "
            "only FFmpeg-supported metadata will be copied."
        )
        return
    if not source.tags and not getattr(source, "pictures", []):
        return
    if isinstance(source, MP4):
        output.tags.update({
            key: value for key, value in source.tags.items() if key != "\xa9too"
        })
    else:
        if isinstance(source.tags, ID3):
            text_tags = {}
            for name, frame in (("bpm", "TBPM"), ("initialkey", "TKEY")):
                values = source.tags.getall(frame)
                if values:
                    text_tags[name] = [str(value) for value in values[0].text]
            for frame in source.tags.getall("TXXX"):
                text_tags[frame.desc.lower()] = [str(value) for value in frame.text]
            pictures = source.tags.getall("APIC")
        else:
            text_tags = {
                key.lower(): value for key, value in (source.tags or {}).items()
                if isinstance(value, list) and all(isinstance(item, str) for item in value)
            }
            pictures = list(getattr(source, "pictures", []))
            pictures.extend(
                Picture(base64.b64decode(value, validate=True))
                for value in text_tags.get("metadata_block_picture", [])
            )
        for aliases, name in (
                (("bpm", "tempo"), "BPM"),
                (("initialkey", "key"), "initialkey"),
                (("replaygain_track_gain",), "replaygain_track_gain"),
                (("replaygain_track_peak",), "replaygain_track_peak"),
                (("replaygain_album_gain",), "replaygain_album_gain"),
                (("replaygain_album_peak",), "replaygain_album_peak")):
            values = next((text_tags[key] for key in aliases if text_tags.get(key)), None)
            if values:
                output.tags[f"----:com.apple.iTunes:{name}"] = [
                    value.encode("utf-8") for value in values
                ]
        covers = []
        for picture in sorted(pictures, key=lambda picture: picture.type != 3):
            if picture.mime in ("image/jpeg", "image/jpg"):
                image_format = MP4Cover.FORMAT_JPEG
            elif picture.mime == "image/png":
                image_format = MP4Cover.FORMAT_PNG
            else:
                warnings.warn(f"Unsupported artwork format {picture.mime!r} in {input_path}.")
                continue
            covers.append(MP4Cover(picture.data, imageformat=image_format))
        if covers:
            output.tags["covr"] = covers


def _mux_stems(ffmpeg, raw_paths, input_path, output_path, codec, cancel_file=None):
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for raw_path in raw_paths:
        command.extend(
            [
                "-f",
                "f32le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-i",
                str(raw_path),
            ]
        )
    metadata_index = len(raw_paths)
    command.extend(["-i", str(input_path)])
    for input_index in range(len(raw_paths)):
        command.extend(["-map", f"{input_index}:a:0"])
    command.extend(["-map_metadata", str(metadata_index), "-c:a", codec])
    if codec == "aac":
        command.extend(["-b:a", "256k"])
    for stream_index, title in enumerate(("Mix",) + SOURCES):
        command.extend([f"-metadata:s:a:{stream_index}", f"title={title.title()}"])
    command.extend(["-f", "mp4", str(output_path)])
    _run_command(command, cancel_file)
    output = MP4(output_path)
    _copy_extra_metadata(input_path, output)
    titles = output.tags.get("\xa9nam", [])
    title = titles[0] if titles and titles[0].strip() else input_path.stem
    if not title.startswith("[STEM] "):
        title = "[STEM] " + title
    output.tags["\xa9nam"] = [title]
    output.save()


def _manifest():
    return {
        "version": 1,
        "stems": [
            {"name": source.title(), "color": color}
            for source, color in zip(SOURCES, COLORS)
        ],
    }


def _publish_output(temp_output, output_path, force):
    if force:
        os.replace(temp_output, output_path)
    elif os.name == "nt":
        os.rename(temp_output, output_path)
    else:
        os.link(temp_output, output_path)
        temp_output.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Separate a stereo track offline into a Mixxx-compatible .stem.mp4."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend", choices=("auto", "openvino", "onnxruntime"), default="auto")
    parser.add_argument("--codec", choices=("alac", "aac"), default="alac")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()

    input_path = args.input.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    contract_path = (args.contract or model_path.with_suffix(".json")).resolve(strict=True)
    output_path = (args.output or input_path.with_name(
        input_path.stem + ".stem.mp4"
    )).resolve()
    if input_path == output_path:
        parser.error("Input and output paths must be different.")
    if output_path.suffix.lower() != ".mp4" or not output_path.name.lower().endswith(".stem.mp4"):
        parser.error("Output filename must end in .stem.mp4.")
    if output_path.exists() and not args.force:
        parser.error(f"Output already exists; use --force to replace it: {output_path}")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        parser.error(f"FFmpeg executable not found: {args.ffmpeg}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    session = create_session(model_path, contract, args.threads, args.backend)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="mixxx-stems-", dir=args.temp_dir or output_path.parent) as temp_dir:
        temp = Path(temp_dir)
        decoded_path = temp / "input.f32"
        _decode_input(ffmpeg, input_path, decoded_path, args.cancel_file)
        segment_frames = contract["input"]["shape"][2]
        overlap_frames = segment_frames // 4
        separated_path = temp / "separated.f32"
        separated_paths = [
            temp / f"{source}.f32"
            for source in SOURCES
        ]

        separate_to_raw(
            decoded_path,
            separated_path,
            session,
            segment_frames,
            overlap_frames,
            (lambda percent: print(f"STEM_PROGRESS {percent}", flush=True))
            if args.progress else None,
            args.cancel_file,
        )
        frame_count = separated_path.stat().st_size // (
            np.dtype(np.float32).itemsize * len(SOURCES) * 2
        )
        separated_data = np.memmap(
            separated_path,
            dtype=np.float32,
            mode="r",
            shape=(len(SOURCES), frame_count, 2),
        )
        for index, raw_path in enumerate(separated_paths):
            separated_data[index].tofile(raw_path)
        del separated_data

        temp_output = temp / output_path.name
        _mux_stems(
            ffmpeg, [decoded_path, *separated_paths], input_path, temp_output,
            args.codec, args.cancel_file,
        )
        embed_stem_manifest(temp_output, _manifest())
        _check_canceled(args.cancel_file)
        _publish_output(temp_output, output_path, args.force)
    print(f"Created {output_path}")


if __name__ == "__main__":
    try:
        main()
    except InterruptedError as error:
        print(error, file=sys.stderr)
        sys.exit(130)
