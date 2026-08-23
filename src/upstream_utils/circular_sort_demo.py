"""
Small standalone demonstration of one_day.py's circular_sort() behavior,
built for presentation purposes to show a concrete before/after example
without needing to run the full download pipeline.
"""

from one_day import circular_sort

if __name__ == "__main__":
    example_files = [
        "X_120000_spec.fit.gz",
        "X_000000_spec.fit.gz",
        "X_060000_spec.fit.gz",
        "X_180000_spec.fit.gz",
    ]
    offset = "060000"

    result = circular_sort(example_files, offset=offset, url="example")

    print("Input files (unsorted):")
    for f in example_files:
        print(f"  {f}")

    print(f"\nLocal-day UTC offset: {offset}")

    print("\nOutput (circularly sorted, starting from offset):")
    for f in result:
        print(f"  {f}")