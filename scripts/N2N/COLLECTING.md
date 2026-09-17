# Collect LDH videos from network drives

`collect_videos.py` collects exact measurement names from a UTF-8 `.txt` list.
It copies existing AVI files byte-for-byte. With `--h5`, it extracts the dataset
`moment0ff` into a lossless NumPy array, without normalization or compression.

## Inputs

`measures.txt` contains measurement folder names (not full paths), one per line:

```text
# Selected recordings
260310_AUZ0752_4
260622_LEC0430_L_4
```

Blank lines, whole-line `#` comments, UTF-8 BOMs, and duplicate names are handled.
Names are matched exactly, case-insensitively; substring matches are not used.
Use the full measure name as it appears in the directory, without adding `_HD`.

Expected layout inside each search folder:

```text
folder1/
  measure/
    measure_HD/
      avi/measure_HD_M0.avi
      h5/measure_HD_output.h5
```

## Copy AVI files

Run from the repository root:

```powershell
python scripts/N2N/collect_videos.py --measures "D:/lists/measures.txt" --folders "Y:/folder1" "Y:/folder2" "Z:/folder3" --output "D:/N2T/videos"
```

Output names are `<measure>_HD_M0.avi`. AVI mode needs only the Python standard
library; it does not decode or recompress videos.

Instead of a long command, put search folders in `folders.txt`, one path per line:

```text
Y:/folder1
Y:/folder2
Z:/folder3
```

```powershell
python scripts/N2N/collect_videos.py --measures "D:/lists/measures.txt" --folders-file "D:/lists/folders.txt" --output "D:/N2T/videos"
```

`--folders` and `--folders-file` can be combined. Paths in the text file are
literal paths: do not surround them with quotes. Relative paths resolve from
the working directory. UNC paths such as `//server/share/folder1` also work.
Mapped drives must be visible to the account/session running Python.

## Search depth and network performance

- Default `--max-depth 2`: searches up to `Y:/folder1/measure` from `Y:/`.
- `--max-depth 1`: faster when each supplied folder directly contains measurements.
- Set a larger value to search deeper directory structures.
- A supplied measurement folder itself is also accepted.

Example starting at drive roots:

```powershell
python scripts/N2N/collect_videos.py --measures "D:/lists/measures.txt" --folders "Y:/" "Z:/" --max-depth 2 --output "D:/N2T/videos"
```

Supply the narrowest known parent folders for best performance. The script lists
each searched directory once, uses directory-entry information, and never
recursively walks the AVI/H5 contents. It stops descending when it finds a
requested measurement and checks only the exact expected file path. It searches
all supplied roots to detect duplicate sources rather than silently choosing one.
Directory entries are not followed through symbolic links.

There are four scan workers and two copy workers by default. These are bounded
to avoid flooding SMB with parallel operations. Tune `--scan-workers` and
`--copy-workers` conservatively; more workers do not necessarily increase
throughput. H5 extraction uses h5py, whose library calls may serialize across
threads. Local filesystem tests verify the bounded search behavior; network
throughput has not been benchmarked on your mapped drives.

## Extract H5 videos

Requires `numpy` and `h5py` (already project dependencies):

```powershell
python scripts/N2N/collect_videos.py --measures "D:/lists/measures.txt" --folders "Y:/folder1" "Y:/folder2" --output "D:/N2T/arrays" --h5 --frame-axis 2
```

Specify the actual time axis; the script does not guess:

| Source `moment0ff` shape | Argument |
|---|---|
| `(frames, height, width)` | `--frame-axis 0` |
| `(height, frames, width)` | `--frame-axis 1` |
| `(height, width, frames)` | `--frame-axis 2` |

The output is `<measure>_HD_M0.npy`, shaped `(frames, height, width)`. The two
remaining spatial axes keep their original order. Data type and values are
preserved, including floating-point values outside [0,1]. No automatic scaling,
transposition of the spatial axes, clipping, or 8-bit conversion takes place.
The collection report records source/output shapes, dtype, and frame axis.

Extraction reads temporal slabs with a target buffer size of 64 MiB per worker
(`--chunk-mb`). At least one frame is read at once; HDF5 internal chunk caches,
decompression buffers, and OS-mapped output pages add to memory use. All input
files in one run must use the selected frame-axis convention. Missing keys and
invalid shapes/dtypes are reported as errors.

**Noise2Time preparation:** the training preparation script accepts uint8 or
float arrays in [0,1]. Raw H5 moments may have different units/ranges, so apply an
explicit, scientifically chosen scale before preparation if needed. The
collector intentionally does not invent that scale. NPY does not carry an
acquisition frame rate; supply the actual `--fps` when preparing later.

## Preview, reruns, and reports

Add `--dry-run` to locate files and write a report without copying/extracting
them. A dry run does not open the H5 dataset or verify its key/shape.

Every run writes a uniquely named `collection_*.json` report in the output
folder, with one result per requested measurement and any directory access
errors. Status values:

- `copied`: file copied or array extracted successfully.
- `planned`: found during a dry run.
- `missing`: measurement or expected source file absent within the search depth.
- `ambiguous`: more than one matching source file; narrow the search folders.
- `exists`: output left untouched; existing content has **not** been verified.
- `error`: permission, I/O, H5, or other failure.

Existing outputs are not overwritten unless `--overwrite` is explicitly given.
Files are written to uniquely named temporary files in the output folder and
published only after successful completion; failed temporary files are removed.
Windows uses an atomic same-directory rename for normal publication; POSIX uses
a hard link for no-clobber publication (the output filesystem must support it).
Source size and modification time are checked before/after reading, and AVI
copy size is checked. No expensive second full network read for hashing occurs;
these are consistency checks, not cryptographic content verification.

The exit code is 1 if any measurement is missing, ambiguous, or failed, or a
directory could not be searched; other measurements are still processed.
`exists` is a nonfatal skip, not a claim that a previous copy is valid.

The script collects videos only. It does not launch preparation or training.

## Tests

```powershell
python -m pytest tests/test_collect_videos.py -q
```
