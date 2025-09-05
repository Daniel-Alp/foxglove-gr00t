# from https://github.com/Daniel-Alp/py-mp4-to-mcap/

import argparse
import av
import subprocess
from pathlib import Path
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer
from tempfile import NamedTemporaryFile

def mp4_to_mcap(input_path: Path, output_path: Path, topic: str, frame_id: str):
    with av.open(input_path, "r") as container:
        video_stream = container.streams.video[0]
        codec_context = video_stream.codec_context
        codec_name = codec_context.name
        if codec_name not in ["h264", "h265", "hevc"]:
            raise ValueError(f"Unsupported codec: {codec_name}")
    
    with NamedTemporaryFile(suffix=".ts") as temp_output:
        temp_output_path = Path(temp_output.name)
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-c:v", "libx264" if codec_name == "h264" else "libx265",
            "-bf", "0",
            "-bsf:v", "h264_mp4toannexb" if codec_name == "h264" else "hevc_mp4toannexb",
            str(temp_output_path)
        ]
        subprocess.run(cmd, check=True)
        
        with av.open(temp_output_path, "r") as container, open(output_path, "wb") as stream, Writer(stream) as writer:
            video_stream = container.streams.video[0]

            format = "h264" if codec_name == "h264" else "h265"

            for packet in container.demux(video_stream):
                if packet.pts is None:
                    continue
                data = bytes(packet)
                # assumes that 1 packet corresponds to 1 frame https://ffmpeg.org/doxygen/2.0/structAVPacket.html
                timestamp_ns = int(packet.pts * 1_000_000_000 * packet.time_base.numerator / packet.time_base.denominator)
                message = CompressedVideo(
                    timestamp   = Timestamp(seconds=timestamp_ns // 1_000_000_000, nanos=timestamp_ns % 1_000_000_000),
                    data        = data,
                    format      = format
                )
                writer.write_message(
                    topic        = topic,
                    message      = message,
                    publish_time = timestamp_ns,
                    log_time     = timestamp_ns
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, help="root directory of dataset")
    parser.add_argument("chunk", help="chunk number")
    parser.add_argument("episode", help="episode number")
    args = parser.parse_args()

    name = f"{args.data_root.name}-{args.chunk}-{args.episode}"

    path_left = args.data_root / f"videos/chunk-{args.chunk}/observation.images.left_view/episode_{args.episode}.mp4"
    path_right = args.data_root / f"videos/chunk-{args.chunk}/observation.images.right_view/episode_{args.episode}.mp4"
    path_wrist = args.data_root / f"videos/chunk-{args.chunk}/observation.images.wrist_view/episode_{args.episode}.mp4"

    mp4_to_mcap(path_left, f"{name}-left_view.mcap", "/camera/left", "/camera/left")
    mp4_to_mcap(path_right, f"{name}-right_view.mcap", "/camera/right", "/camera/right")
    mp4_to_mcap(path_wrist, f"{name}-wrist_view.mcap", "/camera/wrist", "/camera/wrist")