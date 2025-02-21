#!/usr/bin/env python
"""
This script scans the current dataset directory for specific signal files (.mp3, .raw, and .csv),
processes them to extract metadata and signal data, and compiles the results into a single
Pandas DataFrame. The final compiled dataset is saved as a pickle (.pkl) file.

This script is to be run inside the 'DatasetCHVNGE' folder.

For some cases, the ECG and PCG signals corresponding to the same ID and auscultation point
may be split across different files. This script merges such entries so that each record
has both an ECG and a PCG signal, with no empty fields.

Sources:
- Rijuven: Files in the root directory with filenames in the format:
      [ID]_[Auscultation_Point].mp3  (PCG signal)
      [ID]_[Auscultation_Point].raw  (ECG signal)
- INESCTEC: CSV files located in any subfolder within the directory, with filenames that contain:
      P[ID]_[Extended_Auscultation_Point]_[Signal]_[Date].csv
  For example:
      "P110_Tricúspide_PCG_2023-11-29;11-44-54.1280.csv"
  From this filename:
      ID = 110
      Auscultation_Point = "Tricúspide" (mapped to standard abbreviation)
      Signal = "PCG"
      Date = "2023-11-29"   (only the part before the semicolon is used)
  The Extended_Auscultation_Point is mapped to a standard abbreviation:
      "Aórtica"/"Aortica" → "AV"
      "Mitral" → "MV"
      "Pulmonar" → "PV"
      "Tricúspide"/"Tricuspide" → "TV"
  Duplicate CSV files (i.e., same ID, auscultation point, and signal) are sorted by date
  (oldest to newest) and subsequent ones are renamed with suffixes (_2, _3, etc.).

Dependencies: numpy, pandas, pydub, os, re, csv

@author: Daniel Proaño-Guevara
"""

import os
import re
import csv
import numpy as np
import pandas as pd
from pydub import AudioSegment


def read_ecg_pcg_file(csv_file):
    """
    Simplified CSV reader for ECG/PCG files.

    Reads an ECG/PCG CSV file, skipping header rows and extracting only the signal data.
    It returns a list of integer samples extracted from the file.

    Args:
        csv_file (str): Path to the CSV file.


    Returns:
        list: Extracted signal samples as integers.
    """
    data = []
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row and date_pattern.match(row[0]):
                try:
                    data.extend([int(value) for value in row[1:]])
                except ValueError:
                    pass
    return data


def process_rijuven_files():
    """
    Process .mp3 and .raw files in the current directory.
    - .mp3 files are read as PCG signals using pydub.
    - .raw files are read as ECG signals using numpy.
    Returns a list of dictionaries with keys: [ID, Auscultation_Point, Source, ECG, PCG].
    """
    data = {}  # Key: (ID, Auscultation_Point)
    excluded_ids = {"118", "119", "121", "122"}  # Set of IDs to exclude

    for file in os.listdir('.'):
        if os.path.isfile(file) and file.lower().endswith(('.mp3', '.raw')):
            # Expecting format: [ID]_[Auscultation_Point].<ext>
            match = re.match(
                r"^(\d+)_([^\.]+)\.(mp3|raw)$", file, re.IGNORECASE)
            if not match:
                continue

            id_ = match.group(1)
            if id_ in excluded_ids:
                continue  # Skip processing for excluded IDs

            ausc_point = match.group(2)
            ext = match.group(3).lower()
            key = (id_, ausc_point)

            if key not in data:
                data[key] = {
                    "ID": id_,
                    "Auscultation_Point": ausc_point,
                    "Source": "Rijuven",
                    "ECG": None,
                    "PCG": None
                }

            if ext == "mp3":
                try:
                    audio = AudioSegment.from_mp3(file)
                    pcg_signal = np.array(audio.get_array_of_samples())
                    data[key]["PCG"] = pcg_signal
                except Exception as e:
                    print(f"Error processing MP3 file {file}: {e}")
            elif ext == "raw":
                try:
                    ecg_signal = np.loadtxt(file, delimiter=",", dtype=int)
                    data[key]["ECG"] = ecg_signal
                except Exception as e:
                    print(f"Error processing RAW file {file}: {e}")

    return list(data.values())


def map_ausc_point(extended_point):
    """
    Map extended auscultation points to standard abbreviations.
    """
    mapping = {
        "aortica": "AV",
        "aórtica": "AV",
        "mitral": "MV",
        "pulmonar": "PV",
        "tricuspide": "TV",
        "tricúspide": "TV"
    }
    return mapping.get(extended_point.lower(), extended_point)


def process_inesctec_csv_files():
    """
    Process all CSV files found in any subfolder within the directory.
    For each CSV file, attempt to extract parameters from the filename.

    Expected filename example:
      "P110_Tricúspide_PCG_2023-11-29;11-44-54.1280.csv"
    Extraction:
      - ID: from "P110" → "110"
      - Auscultation_Point: from "Tricúspide" (mapped to standard abbreviation)
      - Signal: from "PCG" (converted to uppercase)
      - Date: from "2023-11-29;11-44-54.1280" → "2023-11-29" (only the part before the semicolon is used)

    The CSV is read using the read_ecg_pcg_file() function with a specified sample rate.
    Duplicate files (same ID, auscultation point, and signal) are sorted by date and renamed
    with suffixes for subsequent occurrences.

    Returns a list of dictionaries with keys: [ID, Auscultation_Point, Source, ECG, PCG].
    """
    csv_entries = {}  # Key: (ID, base_abbr, signal), Value: list of entries

    # Walk through all subfolders in the current directory
    for root, dirs, files in os.walk('.'):
        # Skip processing CSV files in the root directory
        if root == '.':
            continue
        for file in files:
            if file.lower().endswith('.csv'):
                try:
                    # Remove the .csv extension and split the filename by underscore
                    base = file[:-4]
                    parts = base.split("_")
                    if len(parts) < 4:
                        print(
                            f"Filename '{file}' does not have enough parts. Skipping.")
                        continue
                    # Extract ID: Remove leading 'P' if present
                    if parts[0].startswith("P"):
                        id_ = parts[0][1:]
                    else:
                        id_ = parts[0]
                    ausc_point = parts[1]
                    signal_type = parts[2].upper()  # Expected: ECG or PCG
                    # Extract date from the fourth part: take the substring before the semicolon, if any.
                    date_str = parts[3].split(";")[0]

                    base_abbr = map_ausc_point(ausc_point)
                    key = (id_, base_abbr, signal_type)
                    file_path = os.path.join(root, file)

                    # Use the custom CSV reading function to get the signal data
                    data_samples = read_ecg_pcg_file(file_path)

                    entry = {
                        "ID": id_,
                        "Auscultation_Point": base_abbr,  # May be updated for duplicates
                        "Source": "INESCTEC",
                        "ECG": data_samples if signal_type == "ECG" else None,
                        "PCG": data_samples if signal_type == "PCG" else None,
                        "date": date_str,
                        "signal": signal_type
                    }
                    csv_entries.setdefault(key, []).append(entry)
                except Exception as e:
                    print(f"Error processing filename '{file}': {e}")
                    continue

    # Handle duplicates: sort entries by date and adjust Auscultation_Point for subsequent entries.
    results = []
    for key, entries in csv_entries.items():
        sorted_entries = sorted(entries, key=lambda x: x["date"])
        for i, entry in enumerate(sorted_entries):
            if i > 0:
                entry["Auscultation_Point"] = f"{entry['Auscultation_Point']}_{i+1}"
            # Remove temporary fields used for sorting
            entry.pop("date", None)
            entry.pop("signal", None)
            results.append(entry)
    return results


def merge_entries(entries):
    """
    Merge entries with the same ID and Auscultation_Point.
    If one entry has an empty ECG or PCG and another entry (with the same key)
    has that field populated, they are merged into a single record.
    The 'Source' field is combined (unique sources joined by a comma).
    Only entries with both ECG and PCG populated are kept.
    """
    merged = {}
    for rec in entries:
        key = (rec["ID"], rec["Auscultation_Point"])
        if key not in merged:
            # Initialize with current record and store source as a set for merging
            merged[key] = rec.copy()
            merged[key]["Source"] = {rec["Source"]}
        else:
            # Merge ECG if missing
            if merged[key]["ECG"] is None and rec["ECG"] is not None:
                merged[key]["ECG"] = rec["ECG"]
            # Merge PCG if missing
            if merged[key]["PCG"] is None and rec["PCG"] is not None:
                merged[key]["PCG"] = rec["PCG"]
            # Merge sources
            merged[key]["Source"].add(rec["Source"])
    # Convert merged entries to list and filter out records with any missing field
    final_entries = []
    for rec in merged.values():
        if rec["ECG"] is not None and rec["PCG"] is not None:
            # Convert the source set to a sorted, comma-separated string
            rec["Source"] = ", ".join(sorted(rec["Source"]))
            final_entries.append(rec)
    return final_entries


def main():
    """
    Main function to scan, process, merge, and compile signal data into a DataFrame,
    and then save it as a pickle file.
    """
    all_rows = []

    # Process Rijuven files (.mp3 and .raw) in the root directory
    rijuven_data = process_rijuven_files()
    all_rows.extend(rijuven_data)

    # Process INESCTEC CSV files in all subfolders
    inesctec_data = process_inesctec_csv_files()
    all_rows.extend(inesctec_data)

    # Merge entries with the same ID and Auscultation_Point to ensure no empty ECG or PCG fields
    merged_rows = merge_entries(all_rows)

    # Create DataFrame with the specified columns
    df = pd.DataFrame(merged_rows, columns=[
                      "ID", "Auscultation_Point", "Source", "ECG", "PCG"])

    # Save the DataFrame as a .pkl file
    output_file = "compiled_dataset.pkl"
    try:
        df.to_pickle(output_file)
        print(f"Compiled dataset successfully saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving the DataFrame to pickle: {e}")


if __name__ == "__main__":
    main()
