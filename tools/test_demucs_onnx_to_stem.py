import base64
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from threading import Timer
import unittest
from unittest.mock import patch

from mutagen.flac import FLAC, Picture
from mutagen.id3 import APIC, ID3, TBPM, TKEY
from mutagen.mp4 import MP4, MP4Cover
import numpy as np

from demucs_onnx_to_stem import (
    _manifest,
    _mux_stems,
    _publish_output,
    _run_command,
    create_session,
    embed_stem_manifest,
    separate_to_raw,
    validate_contract,
)


class FakeSession:
    def __init__(self, scales):
        self.scales = np.asarray(scales, dtype=np.float32)

    def run(self, output_names, inputs):
        audio = inputs["audio"]
        stems = audio[:, np.newaxis, :, :] * self.scales[None, :, None, None]
        return [stems]


class FakeNode:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.type = "tensor(float)"


class FakeModelMeta:
    def __init__(self, contract):
        self.custom_metadata_map = {"demucs_contract": json.dumps(contract)}


class ContractSession:
    def __init__(self, contract):
        self.contract = contract

    def get_inputs(self):
        return [FakeNode("audio", [1, 2, 343980])]

    def get_outputs(self):
        return [FakeNode("stems", [1, 4, 2, 343980])]

    def get_modelmeta(self):
        return FakeModelMeta(self.contract)


def atom(atom_type, payload=b""):
    return struct.pack(">I", len(payload) + 8) + atom_type + payload


def child_atoms(data, start, end):
    offset = start
    while offset < end:
        size = struct.unpack_from(">I", data, offset)[0]
        yield bytes(data[offset + 4:offset + 8]), offset, size
        offset += size


class DemucsOnnxToStemTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MIXXX_STEM_TEST_MODEL"),
                         "Set MIXXX_STEM_TEST_MODEL for real engine validation")
    def test_real_openvino_matches_cpu_contract_tolerance(self):
        model = Path(os.environ["MIXXX_STEM_TEST_MODEL"])
        contract = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
        audio = np.random.default_rng(42).normal(
            0, 0.1, (1, 2, 343980)).astype(np.float32)
        cpu = create_session(model, contract, 4, "onnxruntime")
        reference = cpu.run(["stems"], {"audio": audio})[0]
        del cpu
        accelerated = create_session(model, contract, 4, "openvino")
        actual = accelerated.run(["stems"], {"audio": audio})[0]
        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, reference, atol=0.0002, rtol=0.002)

    def test_auto_backend_selects_openvino_after_contract_validation(self):
        with patch("demucs_onnx_to_stem.ort.InferenceSession"), patch(
                "demucs_onnx_to_stem.validate_contract") as validate, patch(
                "demucs_onnx_to_stem.importlib.util.find_spec", return_value=object()), patch(
                "demucs_onnx_to_stem.OpenVinoSession") as accelerated:
            session = create_session(Path("model.onnx"), {}, 4)
        validate.assert_called_once()
        accelerated.assert_called_once_with(Path("model.onnx"), 4)
        self.assertIs(session, accelerated.return_value)

    def test_missing_openvino_warns_and_uses_cpu(self):
        with patch("demucs_onnx_to_stem.ort.InferenceSession") as cpu, patch(
                "demucs_onnx_to_stem.validate_contract"), patch(
                "demucs_onnx_to_stem.importlib.util.find_spec", return_value=None):
            with self.assertWarnsRegex(UserWarning, "using ONNX Runtime CPU"):
                session = create_session(Path("model.onnx"), {}, 4)
        self.assertIs(session, cpu.return_value)
        self.assertFalse(cpu.call_args.kwargs["sess_options"].enable_cpu_mem_arena)

    def test_openvino_initialization_errors_are_not_hidden(self):
        with patch("demucs_onnx_to_stem.ort.InferenceSession"), patch(
                "demucs_onnx_to_stem.validate_contract"), patch(
                "demucs_onnx_to_stem.importlib.util.find_spec", return_value=object()), patch(
                "demucs_onnx_to_stem.OpenVinoSession", side_effect=RuntimeError("compile failed")):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                create_session(Path("model.onnx"), {}, 4)

    def test_explicit_cpu_backend_does_not_initialize_openvino(self):
        with patch("demucs_onnx_to_stem.ort.InferenceSession") as cpu, patch(
                "demucs_onnx_to_stem.validate_contract"), patch(
                "demucs_onnx_to_stem.OpenVinoSession") as accelerated:
            self.assertIs(create_session(Path("model.onnx"), {}, 4, "onnxruntime"),
                          cpu.return_value)
        accelerated.assert_not_called()

    def test_cancellation_terminates_running_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cancel = root / "cancel"
            output = root / "unwanted"
            timer = Timer(0.3, cancel.touch)
            timer.start()
            try:
                with self.assertRaises(InterruptedError):
                    _run_command([
                        sys.executable, "-c",
                        "import time; from pathlib import Path; time.sleep(5); "
                        f"Path({str(output)!r}).touch()",
                    ], cancel)
            finally:
                timer.cancel()
                timer.join()
            self.assertFalse(output.exists())

    def test_canceled_subprocess_exits_and_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cancel = root / "cancel"
            output = root / "unwanted"
            cancel.touch()
            with self.assertRaises(InterruptedError):
                _run_command([
                    sys.executable, "-c",
                    f"from pathlib import Path; Path({str(output)!r}).touch()",
                ], cancel)
            self.assertFalse(output.exists())

    def test_publish_preserves_output_created_while_separating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "new.mp4"
            output = root / "existing.mp4"
            temporary.write_bytes(b"new audio")
            output.write_bytes(b"existing audio")
            with self.assertRaises(FileExistsError):
                _publish_output(temporary, output, False)
            self.assertEqual(output.read_bytes(), b"existing audio")
            _publish_output(temporary, output, True)
            self.assertEqual(output.read_bytes(), b"new audio")

    def test_mux_maps_metadata_without_mapping_original_audio_or_video(self):
        raw_paths = [Path(f"{index}.f32") for index in range(5)]
        with patch("demucs_onnx_to_stem.subprocess.run") as run, patch(
                "demucs_onnx_to_stem._copy_extra_metadata") as copy, patch(
                "demucs_onnx_to_stem.MP4") as mp4:
            mp4.return_value.tags = {}
            _mux_stems("ffmpeg", raw_paths, Path("original.flac"), Path("out.mp4"), "alac")
        command = run.call_args.args[0]
        self.assertEqual(
            [command[index + 1] for index, value in enumerate(command) if value == "-map"],
            [f"{index}:a:0" for index in range(5)],
        )
        self.assertEqual(command[command.index("-map_metadata") + 1], "5")
        self.assertIn("original.flac", command)
        mp4.assert_called_once_with(Path("out.mp4"))
        copy.assert_called_once_with(Path("original.flac"), mp4.return_value)
        self.assertEqual(mp4.return_value.tags["\xa9nam"], ["[STEM] original"])
        mp4.return_value.save.assert_called_once_with()

    def test_contract_matches_fixed_htdemucs_model(self):
        contract = {
            "model": "htdemucs",
            "sample_rate": 44100,
            "sources": ["drums", "bass", "other", "vocals"],
            "input": {"name": "audio", "dtype": "float32", "shape": [1, 2, 343980]},
            "output": {
                "name": "stems",
                "dtype": "float32",
                "shape": [1, 4, 2, 343980],
            },
        }
        validate_contract(ContractSession(contract), contract)

    def test_overlapped_chunks_reconstruct_sum_without_gaps(self):
        segment_frames = 32
        overlap_frames = 8
        audio = np.arange(96 * 2, dtype=np.float32).reshape(96, 2) / 100
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.f32"
            audio.tofile(source_path)
            output_path = root / "separated.f32"
            separate_to_raw(
                source_path,
                output_path,
                FakeSession([0.1, 0.2, 0.3, 0.4]),
                segment_frames,
                overlap_frames,
            )
            result = np.memmap(
                output_path,
                dtype=np.float32,
                mode="r",
                shape=(4, len(audio), 2),
            )
            np.testing.assert_allclose(result.sum(axis=0), audio, atol=2e-6)
            del result

    def test_separation_reports_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.f32"
            np.ones((96, 2), dtype=np.float32).tofile(source)
            progress = []
            separate_to_raw(
                source, root / "out.f32", FakeSession([0.1, 0.2, 0.3, 0.4]),
                32, 8, progress.append,
            )
            self.assertGreater(len(progress), 1)
            self.assertEqual(progress[-1], 100)
            self.assertEqual(progress, sorted(progress))

    def test_manifest_is_added_to_existing_udta_without_moving_media(self):
        payload = b"sample-media-data"
        mdat = atom(b"mdat", payload)
        moov = atom(
            b"moov",
            atom(b"udta", atom(b"meta", b"existing")) + atom(b"trak", b"after"),
        )
        original = atom(b"ftyp", b"isom") + mdat + moov
        manifest = {
            "version": 1,
            "stems": [{"name": "Drums", "color": "#fd4a4a"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mix.stem.mp4"
            path.write_bytes(original)
            embed_stem_manifest(path, manifest)
            result = path.read_bytes()

        original_mdat = next(
            offset for atom_type, offset, _ in child_atoms(result, 0, len(result))
            if atom_type == b"mdat"
        )
        self.assertEqual(
            result[original_mdat + 8:original_mdat + 8 + len(payload)],
            payload,
        )
        top_moov = next(
            (offset, size) for atom_type, offset, size in child_atoms(result, 0, len(result))
            if atom_type == b"moov"
        )
        moov_offset, moov_size = top_moov
        udta_offset, udta_size = next(
            (offset, size)
            for atom_type, offset, size in child_atoms(
                result, moov_offset + 8, moov_offset + moov_size
            )
            if atom_type == b"udta"
        )
        stem_offset, stem_size = next(
            (offset, size)
            for atom_type, offset, size in child_atoms(
                result, udta_offset + 8, udta_offset + udta_size
            )
            if atom_type == b"stem"
        )
        self.assertEqual(
            json.loads(result[stem_offset + 8:stem_offset + stem_size]),
            manifest,
        )
        child_types = [
            atom_type
            for atom_type, _, _ in child_atoms(
                result, moov_offset + 8, moov_offset + moov_size
            )
        ]
        self.assertEqual(child_types, [b"udta", b"trak"])

    def test_manifest_creates_udta_when_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mix.stem.mp4"
            path.write_bytes(atom(b"ftyp", b"isom") + atom(b"mdat", b"x") + atom(b"moov"))
            embed_stem_manifest(path, {"version": 1, "stems": []})
            data = path.read_bytes()
        moov = next(
            (offset, size) for atom_type, offset, size in child_atoms(data, 0, len(data))
            if atom_type == b"moov"
        )
        offset, size = moov
        udta = next(
            (child_offset, child_size)
            for atom_type, child_offset, child_size in child_atoms(
                data, offset + 8, offset + size
            )
            if atom_type == b"udta"
        )
        self.assertEqual(
            [atom_type for atom_type, _, _ in child_atoms(
                data, udta[0] + 8, udta[0] + udta[1]
            )],
            [b"stem"],
        )


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "FFmpeg and ffprobe are required for metadata integration tests")
class MetadataIntegrationTest(unittest.TestCase):
    cover = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    )

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw_path = self.root / "audio.f32"
        np.zeros((4410, 2), dtype=np.float32).tofile(self.raw_path)

    def make_source(self, suffix, codec):
        path = self.root / f"original{suffix}"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", "44100", "-ac", "2", "-i", str(self.raw_path),
            "-c:a", codec,
            "-metadata", "title=Canci\u00f3n original",
            "-metadata", "artist=Artista",
            "-metadata", "album=\u00c1lbum",
            "-metadata", "album_artist=Artista del \u00e1lbum",
            "-metadata", "date=2024",
            "-metadata", "genre=Pop",
            "-metadata", "track=3/12",
            "-metadata", "disc=1/2",
            "-metadata", "composer=Compositor",
            "-metadata", "copyright=Copyright de prueba",
            "-metadata", "comment=Comentario",
            str(path),
        ], check=True)
        return path

    def generate(self, source, codec="alac"):
        output = self.root / f"{source.stem}-{codec}.stem.mp4"
        original_bytes = source.read_bytes()
        _mux_stems("ffmpeg", [self.raw_path] * 5, source, output, codec)
        embed_stem_manifest(output, _manifest())
        self.assertEqual(source.read_bytes(), original_bytes)
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output),
        ], check=True, capture_output=True, text=True)
        streams = json.loads(result.stdout)["streams"]
        audio_streams = [stream for stream in streams if stream["codec_type"] == "audio"]
        self.assertEqual(len(audio_streams), 5)
        self.assertEqual([stream["index"] for stream in audio_streams], list(range(5)))
        self.assertEqual([stream["codec_name"] for stream in audio_streams], [codec] * 5)
        for stream in streams[5:]:
            self.assertEqual(stream["disposition"]["attached_pic"], 1)
        data = output.read_bytes()
        moov_offset, moov_size = next(
            (offset, size) for atom_type, offset, size in child_atoms(data, 0, len(data))
            if atom_type == b"moov"
        )
        self.assertEqual(
            sum(atom_type == b"trak" for atom_type, _, _ in child_atoms(
                data, moov_offset + 8, moov_offset + moov_size)),
            5,
        )
        udta_offset, udta_size = next(
            (offset, size) for atom_type, offset, size in child_atoms(
                data, moov_offset + 8, moov_offset + moov_size)
            if atom_type == b"udta"
        )
        stem_offset, stem_size = next(
            (offset, size) for atom_type, offset, size in child_atoms(
                data, udta_offset + 8, udta_offset + udta_size)
            if atom_type == b"stem"
        )
        self.assertEqual(
            json.loads(data[stem_offset + 8:stem_offset + stem_size]), _manifest()
        )
        self.assertEqual(
            [stream["tags"].get("title", stream["tags"].get("name"))
             for stream in audio_streams],
            ["Mix", "Drums", "Bass", "Other", "Vocals"],
        )
        subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a",
            "-f", "null", "-",
        ], check=True, stdout=subprocess.DEVNULL)
        return MP4(output)

    def assert_song_tags(self, output):
        for key, value in (
                ("\xa9nam", "[STEM] Canci\u00f3n original"),
                ("\xa9ART", "Artista"),
                ("\xa9alb", "\u00c1lbum"),
                ("aART", "Artista del \u00e1lbum"),
                ("\xa9day", "2024"),
                ("\xa9gen", "Pop"),
                ("\xa9wrt", "Compositor"),
                ("cprt", "Copyright de prueba"),
                ("\xa9cmt", "Comentario")):
            self.assertEqual(output.tags[key], [value])
        self.assertEqual(output.tags["trkn"], [(3, 12)])
        self.assertEqual(output.tags["disk"], [(1, 2)])

    def assert_dj_tags_and_cover(self, output):
        self.assertEqual(output.tags["----:com.apple.iTunes:BPM"], [b"128.5"])
        self.assertEqual(output.tags["----:com.apple.iTunes:initialkey"], [b"8A"])
        self.assertEqual(output.tags["covr"][0], self.cover)
        self.assertEqual(output.tags["covr"][0].imageformat, MP4Cover.FORMAT_PNG)

    def test_flac_metadata_and_cover_survive_alac_and_aac_mux_and_manifest(self):
        source = self.make_source(".flac", "flac")
        tags = FLAC(source)
        tags["bpm"] = "128.5"
        tags["initialkey"] = "8A"
        tags["replaygain_track_gain"] = "-6.2 dB"
        cover = Picture()
        cover.type = 3
        cover.mime = "image/png"
        cover.data = self.cover
        tags.add_picture(cover)
        tags.save()
        for codec in ("alac", "aac"):
            with self.subTest(codec=codec):
                output = self.generate(source, codec)
                self.assert_song_tags(output)
                self.assert_dj_tags_and_cover(output)
                self.assertEqual(
                    output.tags["----:com.apple.iTunes:replaygain_track_gain"], [b"-6.2 dB"]
                )

    def test_id3_metadata_and_cover(self):
        source = self.make_source(".mp3", "libmp3lame")
        tags = ID3(source)
        tags.add(TBPM(encoding=3, text=["128.5"]))
        tags.add(TKEY(encoding=3, text=["8A"]))
        tags.add(APIC(encoding=3, mime="image/png", type=3, data=self.cover))
        tags.save(source)
        output = self.generate(source)
        self.assert_song_tags(output)
        self.assert_dj_tags_and_cover(output)

    def test_mp4_preserves_freeform_tags_and_cover(self):
        source = self.make_source(".m4a", "alac")
        tags = MP4(source)
        tags["----:com.apple.iTunes:BPM"] = [b"128.5"]
        tags["----:com.apple.iTunes:initialkey"] = [b"8A"]
        tags["----:com.apple.iTunes:custom"] = [b"Original custom tag"]
        tags["covr"] = [MP4Cover(self.cover, imageformat=MP4Cover.FORMAT_PNG)]
        tags.save()
        output = self.generate(source)
        self.assert_song_tags(output)
        self.assert_dj_tags_and_cover(output)
        self.assertEqual(
            output.tags["----:com.apple.iTunes:custom"], [b"Original custom tag"]
        )

    def test_untagged_input_still_generates_five_audio_streams(self):
        source = self.root / "untagged.wav"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "f32le", "-ar", "44100",
            "-ac", "2", "-i", str(self.raw_path), str(source),
        ], check=True)
        output = self.generate(source)
        self.assertEqual(output.tags["\xa9nam"], ["[STEM] untagged"])
        self.assertNotIn("covr", output.tags)

    def test_stem_title_prefix_is_not_duplicated(self):
        source = self.make_source(".m4a", "alac")
        tags = MP4(source)
        tags["\xa9nam"] = ["[STEM] Canci\u00f3n original"]
        tags.save()
        output = self.generate(source)
        self.assertEqual(output.tags["\xa9nam"], ["[STEM] Canci\u00f3n original"])

    def test_blank_title_uses_original_filename(self):
        source = self.make_source(".m4a", "alac")
        tags = MP4(source)
        tags["\xa9nam"] = ["   "]
        tags.save()
        output = self.generate(source)
        self.assertEqual(output.tags["\xa9nam"], ["[STEM] original"])

    def test_unsupported_supplemental_metadata_keeps_ffmpeg_title(self):
        source = self.make_source(".m4a", "alac")
        with patch("demucs_onnx_to_stem.mutagen.File", return_value=None):
            with self.assertWarnsRegex(UserWarning, "only FFmpeg-supported metadata"):
                output = self.generate(source)
        self.assert_song_tags(output)


if __name__ == "__main__":
    unittest.main()
