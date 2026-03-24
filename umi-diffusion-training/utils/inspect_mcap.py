#!/usr/bin/env python3
"""Inspect the contents of an MCAP file."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.protobuf.message import Message as ProtoMessage
from google.protobuf.text_format import MessageToString
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


@dataclass
class TopicStats:
    schema_name: str
    schema_encoding: str
    message_encoding: str
    count: int = 0
    first_log_time: int | None = None
    last_log_time: int | None = None

    def update(self, log_time: int) -> None:
        self.count += 1
        if self.first_log_time is None or log_time < self.first_log_time:
            self.first_log_time = log_time
        if self.last_log_time is None or log_time > self.last_log_time:
            self.last_log_time = log_time


def ns_to_str(ns: int | None) -> str:
    if ns is None:
        return "-"
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds")


def summarize_message(decoded: Any, max_lines: int) -> str:
    if isinstance(decoded, ProtoMessage) and hasattr(decoded, "data"):
        data = getattr(decoded, "data")
        if isinstance(data, (bytes, bytearray)):
            lines = [f"data_bytes: {len(data)}"]
            if hasattr(decoded, "format"):
                lines.append(f'format: "{decoded.format}"')
            if hasattr(decoded, "frame_id") and decoded.frame_id:
                lines.append(f'frame_id: "{decoded.frame_id}"')
            if hasattr(decoded, "header"):
                lines.append("header {")
                header_text = MessageToString(decoded.header, as_one_line=False).strip()
                for line in header_text.splitlines():
                    lines.append(f"  {line}")
                lines.append("}")
            if len(lines) <= max_lines:
                return "\n".join(lines)
            return "\n".join(lines[:max_lines] + ["..."])

    if isinstance(decoded, ProtoMessage):
        text = MessageToString(decoded, as_one_line=False).strip()
    else:
        text = str(decoded).strip()
    lines = []
    for line in text.splitlines():
        if len(line) > 160:
            line = f"{line[:160]}... [truncated, {len(line)} chars]"
        lines.append(line)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["..."])


def inspect_mcap(path: Path, sample_count: int, sample_lines: int) -> None:
    topic_stats: dict[str, TopicStats] = {}
    samples: dict[str, list[tuple[int, str]]] = defaultdict(list)
    attachment_count = 0
    metadata_count = 0

    with path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()

        for schema, channel, message, decoded in reader.iter_decoded_messages():
            schema_name = schema.name if schema else "-"
            schema_encoding = schema.encoding if schema else "-"
            if channel.topic not in topic_stats:
                topic_stats[channel.topic] = TopicStats(
                    schema_name=schema_name,
                    schema_encoding=schema_encoding,
                    message_encoding=channel.message_encoding,
                )
            topic_stats[channel.topic].update(message.log_time)

            if len(samples[channel.topic]) < sample_count:
                samples[channel.topic].append(
                    (message.log_time, summarize_message(decoded, sample_lines))
                )

        for _ in reader.iter_attachments():
            attachment_count += 1

        for _ in reader.iter_metadata():
            metadata_count += 1

    print(f"File: {path}")
    print()

    if summary and summary.statistics:
        stats = summary.statistics
        print("Summary")
        print(f"  message_count: {stats.message_count}")
        print(f"  channel_count: {stats.channel_count}")
        print(f"  schema_count: {stats.schema_count}")
        print(f"  chunk_count: {stats.chunk_count}")
        print(f"  attachment_count: {stats.attachment_count}")
        print(f"  metadata_count: {stats.metadata_count}")
        print(f"  start_time: {ns_to_str(stats.message_start_time)}")
        print(f"  end_time:   {ns_to_str(stats.message_end_time)}")
        print()

    print("Topics")
    for topic in sorted(topic_stats):
        stats = topic_stats[topic]
        print(f"- {topic}")
        print(f"  schema: {stats.schema_name} ({stats.schema_encoding})")
        print(f"  message_encoding: {stats.message_encoding}")
        print(f"  count: {stats.count}")
        print(f"  first_log_time: {ns_to_str(stats.first_log_time)}")
        print(f"  last_log_time:  {ns_to_str(stats.last_log_time)}")
        for idx, (log_time, sample) in enumerate(samples[topic], start=1):
            print(f"  sample_{idx} @ {ns_to_str(log_time)}:")
            for line in sample.splitlines():
                print(f"    {line}")
        print()

    if attachment_count or metadata_count:
        print("Extra Records")
        print(f"  attachments: {attachment_count}")
        print(f"  metadata: {metadata_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an MCAP file.")
    parser.add_argument("mcap_path", type=Path, help="Path to the .mcap file")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="Number of sample decoded messages to print per topic",
    )
    parser.add_argument(
        "--sample-lines",
        type=int,
        default=12,
        help="Maximum lines to print for each sample message",
    )
    args = parser.parse_args()

    inspect_mcap(args.mcap_path, args.sample_count, args.sample_lines)


if __name__ == "__main__":
    main()
