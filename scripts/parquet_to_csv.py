import os
import pandas as pd


def convert_parquet_to_csv_and_delete(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".parquet"):
                parquet_path = os.path.join(dirpath, file)
                csv_path = os.path.splitext(parquet_path)[0] + ".csv"

                try:
                    print(f"Converting: {parquet_path}")
                    df = pd.read_parquet(parquet_path)
                    df.to_csv(csv_path, index=False)
                    print(f"Saved:      {csv_path}")

                    # Delete original parquet file
                    os.remove(parquet_path)
                    print(f"Deleted:    {parquet_path}")
                except Exception as e:
                    print(f"❌ Failed to convert {parquet_path}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parquet_to_csv_and_delete.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    convert_parquet_to_csv_and_delete(directory)
