import glob
import json
import os

from shared.models import DocumentBlock


DATA_DIR = "/data/processed_json"


def main():
    files = [
        path
        for path in glob.glob(os.path.join(DATA_DIR, "*_blocks.json"))
        if "-checkpoint_blocks.json" not in os.path.basename(path)
    ]

    print(f"Files found: {len(files)}")

    valid_files = 0
    failed_files = []
    total_blocks = 0

    for index, path in enumerate(files, start=1):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                blocks = data
            elif isinstance(data, dict):
                blocks = data.get(
                    "blocks",
                    data.get("document_blocks", [])
                )
            else:
                raise ValueError("Unexpected JSON structure")

            for block in blocks:
                DocumentBlock.model_validate(block)

            valid_files += 1
            total_blocks += len(blocks)

            if index % 100 == 0:
                print(
                    f"Progress: {index}/{len(files)} | "
                    f"Valid: {valid_files} | "
                    f"Blocks: {total_blocks}"
                )

        except Exception as exc:
            failed_files.append((path, str(exc)))

    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Files found:   {len(files)}")
    print(f"Valid files:   {valid_files}")
    print(f"Failed files:  {len(failed_files)}")
    print(f"Total blocks:  {total_blocks}")

    if failed_files:
        print()
        print("FAILED FILES:")
        for path, error in failed_files:
            print(f"- {os.path.basename(path)}")
            print(f"  {error}")
    else:
        print()
        print("ALL FILES PASSED DocumentBlock VALIDATION")


if __name__ == "__main__":
    main()