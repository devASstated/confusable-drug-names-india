from pathlib import Path
import shutil
import sys


SHARD_DIR = Path("results/candidate_pairs")
OUTPUT_CSV = Path("results/candidate_pairs.csv")


def main():
    if not SHARD_DIR.exists():
        raise SystemExit(f"Missing shard directory: {SHARD_DIR}")

    shards = sorted(SHARD_DIR.glob("part_*.csv"))

    if not shards:
        raise SystemExit(f"No candidate shard files found in: {SHARD_DIR}")

    if OUTPUT_CSV.exists():
        raise SystemExit(
            f"Output already exists: {OUTPUT_CSV}\n"
            "Delete or rename it before running this merge."
        )

    print(f"Found {len(shards):,} shard files")
    print(f"Output: {OUTPUT_CSV}")
    print()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0

    with OUTPUT_CSV.open("wb") as out:
        for index, shard in enumerate(shards, start=1):
            print(
                f"[{index:,}/{len(shards):,}] "
                f"Merging {shard.name} ..."
            )

            with shard.open("rb") as src:
                if index == 1:
                    # Preserve the header from the first shard.
                    shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
                else:
                    # Skip the header line in every subsequent shard.
                    src.readline()
                    shutil.copyfileobj(src, out, length=16 * 1024 * 1024)

            out.flush()

            # Delete only after this shard has been copied successfully.
            size = shard.stat().st_size
            shard.unlink()
            total_bytes += size

    try:
        SHARD_DIR.rmdir()
    except OSError:
        print(
            f"Warning: {SHARD_DIR} was not empty after merging, "
            "so it was not removed."
        )

    final_size = OUTPUT_CSV.stat().st_size

    print()
    print("Merge complete")
    print("=" * 60)
    print(f"Shards merged:          {len(shards):,}")
    print(f"Input bytes consumed:   {total_bytes:,}")
    print(f"Final CSV bytes:        {final_size:,}")
    print(f"Final CSV GiB:          {final_size / (1024 ** 3):,.3f}")
    print(f"Output:                 {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
