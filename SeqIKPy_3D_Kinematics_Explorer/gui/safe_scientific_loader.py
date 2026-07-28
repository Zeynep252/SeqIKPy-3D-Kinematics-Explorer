"""
safe_scientific_loader.py

Controlled loading and scientific validation for Streamlit uploads used with
SeqIKPy-style data.

Supported formats: .pkl, .npy, .h5, .hdf

Public functions
----------------
1. safe_load_uploaded_file(...)
2. validate_scientific_data(...)
"""
from __future__ import annotations

import builtins
import collections
import datetime as _datetime
import decimal
import fractions
import io
import pickle
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

try:
    import pandas as pd
except ImportError:  # Pandas support is optional
    pd = None


class _RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted reader for passive scientific data objects.

    The allow-list includes basic Python containers, NumPy arrays/scalars,
    common pandas objects, date/time values, Decimal/Fraction, and selected
    collections. Unknown application classes, callables, modules and operating
    system/process functions remain forbidden.

    This substantially reduces pickle risk, but it is not a replacement for an
    operating-system sandbox when accepting arbitrary public uploads.
    """

    _SAFE_BUILTINS = {
        "dict", "list", "tuple", "set", "frozenset", "str", "bytes",
        "bytearray", "int", "float", "complex", "bool", "slice", "range",
    }

    _SAFE_GLOBALS = {
        # NumPy arrays, scalars, dtypes and masked arrays
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.ma.core", "MaskedArray"),
        ("numpy.ma.core", "_mareconstruct"),

        # Passive standard-library data objects
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "deque"),
        ("datetime", "date"),
        ("datetime", "datetime"),
        ("datetime", "time"),
        ("datetime", "timedelta"),
        ("datetime", "timezone"),
        ("decimal", "Decimal"),
        ("fractions", "Fraction"),

        # pandas core containers and reconstruction helpers
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.series", "Series"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.managers", "SingleBlockManager"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),

        # pandas arrays and dtypes
        ("pandas.core.arrays.datetimes", "DatetimeArray"),
        ("pandas.core.arrays.timedeltas", "TimedeltaArray"),
        ("pandas.core.arrays.categorical", "Categorical"),
        ("pandas.core.dtypes.dtypes", "CategoricalDtype"),
        ("pandas.core.dtypes.dtypes", "DatetimeTZDtype"),
        ("pandas.core.dtypes.dtypes", "PeriodDtype"),
        ("pandas.core.dtypes.dtypes", "IntervalDtype"),

        # pandas indexes
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.indexes.multi", "MultiIndex"),
        ("pandas.core.indexes.datetimes", "DatetimeIndex"),
        ("pandas.core.indexes.timedeltas", "TimedeltaIndex"),
        ("pandas.core.indexes.period", "PeriodIndex"),
        ("pandas.core.indexes.interval", "IntervalIndex"),
        ("pandas.core.indexes.category", "CategoricalIndex"),

        # pandas scalar-like passive values
        ("pandas._libs.tslibs.timestamps", "Timestamp"),
        ("pandas._libs.tslibs.timedeltas", "Timedelta"),
        ("pandas._libs.tslibs.period", "Period"),
        ("pandas._libs.interval", "Interval"),
        ("pandas._libs.missing", "NAType"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in self._SAFE_BUILTINS:
            return getattr(builtins, name)

        if (module, name) in self._SAFE_GLOBALS:
            # pandas entries are accepted only when pandas is installed.
            if module.startswith("pandas") and pd is None:
                raise pickle.UnpicklingError(
                    "This file contains pandas objects, but pandas is not installed."
                )

            imported_module = __import__(module, fromlist=[name])
            return getattr(imported_module, name)

        raise pickle.UnpicklingError(
            f"Forbidden or unknown object in pickle data: {module}.{name}"
        )


def _validate_passive_loaded_object(obj: Any, *, max_depth: int = 30) -> None:
    """
    Perform a post-load type check on the resulting object graph.

    Construction safety is provided by _RestrictedUnpickler's global allow-list.
    This second check rejects unexpected callable/module/custom objects that may
    be nested inside an otherwise accepted container.
    """
    seen: set[int] = set()

    scalar_types = (
        type(None), bool, int, float, complex, str, bytes, bytearray,
        np.generic, _datetime.date, _datetime.datetime, _datetime.time,
        _datetime.timedelta, _datetime.timezone, decimal.Decimal,
        fractions.Fraction,
    )

    container_types = (
        dict, list, tuple, set, frozenset,
        collections.OrderedDict, collections.defaultdict, collections.deque,
    )

    pandas_types: tuple[type, ...] = ()
    if pd is not None:
        pandas_types = (
            pd.DataFrame, pd.Series, pd.Index, pd.Categorical,
            pd.Timestamp, pd.Timedelta, pd.Period, pd.Interval,
        )

    def walk(value: Any, depth: int) -> None:
        if depth > max_depth:
            raise ValueError("Loaded object is nested too deeply.")

        if isinstance(value, scalar_types):
            return

        if callable(value) or isinstance(value, type):
            raise ValueError(
                f"Callable/type objects are not accepted: {type(value).__name__}"
            )

        object_id = id(value)
        if object_id in seen:
            return
        seen.add(object_id)

        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                for item in value.flat:
                    walk(item, depth + 1)
            return

        if isinstance(value, np.ma.MaskedArray):
            walk(np.asarray(value.data), depth + 1)
            walk(np.asarray(value.mask), depth + 1)
            return

        if pandas_types and isinstance(value, pandas_types):
            # Object columns may contain nested passive Python values.
            if isinstance(value, pd.DataFrame):
                for column in value.columns:
                    series = value[column]
                    if series.dtype == object:
                        for item in series.array:
                            walk(item, depth + 1)
            elif isinstance(value, pd.Series) and value.dtype == object:
                for item in value.array:
                    walk(item, depth + 1)
            return

        if isinstance(value, dict):
            for key, item in value.items():
                walk(key, depth + 1)
                walk(item, depth + 1)
            return

        if isinstance(value, container_types):
            for item in value:
                walk(item, depth + 1)
            return

        raise ValueError(
            f"Unsupported loaded object type: {type(value).__module__}."
            f"{type(value).__name__}"
        )


def _as_bytes(uploaded_file: Any) -> bytes:
    if isinstance(uploaded_file, bytes):
        return uploaded_file
    if isinstance(uploaded_file, bytearray):
        return bytes(uploaded_file)
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):
        position = None
        try:
            position = uploaded_file.tell()
        except Exception:
            pass
        data = uploaded_file.read()
        if position is not None:
            try:
                uploaded_file.seek(position)
            except Exception:
                pass
        return data
    raise TypeError(
        "uploaded_file must be bytes, Streamlit UploadedFile, or a binary file object."
    )


def _filename(uploaded_file: Any, filename: str | None) -> str:
    if filename:
        return filename
    inferred = getattr(uploaded_file, "name", None)
    if inferred:
        return str(inferred)
    raise ValueError("Filename could not be inferred; pass filename explicitly.")


def _read_npy_with_restricted_pickle(raw: bytes) -> Any:
    """
    Load an NPY file without calling np.load(..., allow_pickle=True).

    Numeric arrays are read by NumPy with pickle disabled. Object arrays are
    parsed by reading the NPY header first and then loading the pickle payload
    with _RestrictedUnpickler. This permits dictionaries and NumPy arrays while
    rejecting arbitrary globals/functions embedded in the pickle stream.
    """
    stream = io.BytesIO(raw)

    try:
        version = np.lib.format.read_magic(stream)

        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream,
                max_header_size=10_000,
            )
        elif version in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream,
                max_header_size=10_000,
            )
        else:
            raise ValueError(f"Unsupported NPY format version: {version}")
    except Exception as exc:
        raise ValueError(f"Invalid NPY header: {exc}") from exc

    # Plain numeric/string arrays do not need pickle at all.
    if not dtype.hasobject:
        try:
            return np.load(
                io.BytesIO(raw),
                allow_pickle=False,
                max_header_size=10_000,
            )
        except Exception as exc:
            raise ValueError(f"Invalid NPY array: {exc}") from exc

    # Object-array payloads are pickle streams. Use the same restricted loader
    # used for .pkl instead of NumPy's unrestricted allow_pickle=True path.
    try:
        loaded = _RestrictedUnpickler(stream).load()
        _validate_passive_loaded_object(loaded)
    except pickle.UnpicklingError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Invalid or unsupported object content in NPY file: {exc}"
        ) from exc

    if not isinstance(loaded, np.ndarray):
        raise ValueError("Object NPY payload did not produce a NumPy array.")

    if loaded.shape != shape:
        raise ValueError(
            f"NPY payload shape {loaded.shape} does not match header shape {shape}."
        )

    if bool(loaded.flags.f_contiguous) != bool(fortran_order) and loaded.size > 1:
        # This is an integrity check, not a scientific restriction.
        # C-contiguous one-dimensional arrays can also be F-contiguous.
        if not (loaded.flags.c_contiguous and loaded.flags.f_contiguous):
            raise ValueError("NPY payload memory order does not match its header.")

    # np.save(dict_obj) creates a zero-dimensional object ndarray. Returning
    # .item() restores the original dictionary expected by SeqIKPy workflows.
    if loaded.dtype.hasobject and loaded.shape == ():
        return loaded.item()

    return loaded


def _read_hdf5_tree(h5_file: h5py.File) -> dict[str, Any]:
    """
    Read an HDF5 tree after rejecting external links.

    The loader intentionally does not enforce scientific shape or dtype rules;
    those checks belong to validate_scientific_data(). Groups are represented
    as nested dictionaries so dictionary-like HDF5 files keep their structure.
    """

    def read_group(group: h5py.Group) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for name in group.keys():
            link = group.get(name, getlink=True)

            if isinstance(link, h5py.ExternalLink):
                raise ValueError(
                    f"External HDF5 links are not allowed: {group.name}/{name}"
                )

            obj = group.get(name)

            if isinstance(obj, h5py.Group):
                result[name] = read_group(obj)
            elif isinstance(obj, h5py.Dataset):
                result[name] = obj[()]

        return result

    return read_group(h5_file)


def _detect_file_format(raw: bytes) -> str:
    """
    Detect the actual binary format from file signatures rather than extension.

    Returns one of: "npy", "hdf5", "pickle", "unknown".
    """
    if raw.startswith(b"\x93NUMPY"):
        return "npy"

    hdf5_signature = b"\x89HDF\r\n\x1a\n"
    offsets = [0]
    offset = 512
    while offset <= min(len(raw) - len(hdf5_signature), 1024 * 1024):
        offsets.append(offset)
        offset *= 2

    for candidate in offsets:
        if raw[candidate : candidate + len(hdf5_signature)] == hdf5_signature:
            return "hdf5"

    if raw.startswith(b"\x80"):
        return "pickle"

    return "unknown"


def safe_load_uploaded_file(
    uploaded_file: Any,
    *,
    filename: str | None = None,
    max_file_mb: float = 50.0,
    max_hdf5_datasets: int = 100,
    max_hdf5_elements: int = 50_000_000,
    max_hdf5_dimensions: int = 4,
) -> Any:
    """
    Open supported scientific files according to their actual binary format.

    Security behavior
    -----------------
    - HDF5: reject ExternalLink objects.
    - NPY: object payloads are opened only through _RestrictedUnpickler.
    - Pickle: opened only through _RestrictedUnpickler.
    - Unknown content: tested only with _RestrictedUnpickler; there is no
      unrestricted fallback.

    Important
    ---------
    External links are an HDF5-specific concept. They cannot be checked in a
    truly format-independent way. For pickle/NPY, the corresponding security
    control is restriction of executable/global object references.
    """
    _ = (max_hdf5_datasets, max_hdf5_elements, max_hdf5_dimensions)

    name = _filename(uploaded_file, filename)
    declared_suffix = Path(name).suffix.lower()

    raw = _as_bytes(uploaded_file)
    if not raw:
        raise ValueError("Uploaded file is empty.")

    max_bytes = int(max_file_mb * 1024 * 1024)
    if len(raw) > max_bytes:
        raise ValueError(
            f"File is {len(raw)/(1024*1024):.2f} MB; "
            f"limit is {max_file_mb:.2f} MB."
        )

    detected_format = _detect_file_format(raw)

    if detected_format == "hdf5":
        try:
            with h5py.File(io.BytesIO(raw), "r") as h5_file:
                return _read_hdf5_tree(h5_file)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Invalid or unsupported HDF5 file: {exc}"
            ) from exc

    if detected_format == "npy":
        return _read_npy_with_restricted_pickle(raw)

    if detected_format == "pickle":
        try:
            loaded = _RestrictedUnpickler(io.BytesIO(raw)).load()
            _validate_passive_loaded_object(loaded)
            return loaded
        except pickle.UnpicklingError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Invalid or unsupported pickle file: {exc}"
            ) from exc

    try:
        loaded = _RestrictedUnpickler(io.BytesIO(raw)).load()
        _validate_passive_loaded_object(loaded)
        return loaded
    except Exception as pickle_exc:
        suffix_message = (
            f" The filename extension is '{declared_suffix}', but the content "
            "does not match a supported NPY, HDF5, or restricted-pickle file."
            if declared_suffix
            else ""
        )
        raise ValueError(
            "Could not determine a supported safe file format."
            + suffix_message
        ) from pickle_exc

def validate_scientific_data(
    data: Any,
    *,
    data_type: Literal["auto", "pose", "forward_kinematics", "joint_angles"] = "auto",
    max_frames: int = 20_000,
    max_keypoints: int = 100,
    max_arrays: int = 100,
    max_total_elements: int = 50_000_000,
    require_finite: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Validate SeqIKPy-style pose/FK or joint-angle data.

    pose / forward_kinematics:
        dict[str, ndarray] or one ndarray, each shape (frames, keypoints, 3)
    joint_angles:
        dict[str, ndarray], each shape (frames,), equal lengths, keys Angle_*
    """
    allowed = {"auto", "pose", "forward_kinematics", "joint_angles"}
    if data_type not in allowed:
        raise ValueError(f"Unknown data_type: {data_type}")

    if isinstance(data, np.ndarray):
        items = {"array": data}
        input_was_array = True
    elif isinstance(data, dict):
        if not data:
            raise ValueError("Scientific data dictionary is empty.")
        if len(data) > max_arrays:
            raise ValueError(f"Array count exceeds the limit ({max_arrays}).")
        items = data
        input_was_array = False
    else:
        raise TypeError("Data must be a NumPy array or dictionary of arrays.")

    if not all(isinstance(k, str) for k in items):
        raise ValueError("All dictionary keys must be strings.")

    inferred = data_type
    if data_type == "auto":
        inferred = (
            "joint_angles"
            if not input_was_array and all(k.startswith("Angle_") for k in items)
            else "pose"
        )

    validated: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    total_elements = 0
    frame_count: int | None = None

    if inferred == "joint_angles":
        if input_was_array:
            raise ValueError("Joint-angle data must be a dictionary with Angle_* keys.")
        for key, value in items.items():
            if not key.startswith("Angle_"):
                raise ValueError(f"Unexpected joint-angle key '{key}'.")
            arr = np.asarray(value)
            if arr.ndim != 1:
                raise ValueError(f"'{key}' must have shape (frames,), got {arr.shape}.")
            if arr.shape[0] == 0 or arr.shape[0] > max_frames:
                raise ValueError(f"'{key}' has an invalid frame count: {arr.shape[0]}.")
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"'{key}' must contain numeric values.")
            if require_finite and not np.all(np.isfinite(arr)):
                raise ValueError(f"'{key}' contains NaN or infinite values.")
            if frame_count is None:
                frame_count = arr.shape[0]
            elif arr.shape[0] != frame_count:
                raise ValueError("All joint-angle arrays must have equal length.")
            total_elements += arr.size
            if total_elements > max_total_elements:
                raise ValueError("Scientific data exceed the total element limit.")
            validated[key] = arr.astype(np.float64, copy=False)
            shapes[key] = arr.shape
    else:
        for key, value in items.items():
            arr = np.asarray(value)
            if arr.ndim != 3:
                raise ValueError(
                    f"'{key}' must have shape (frames, keypoints, 3), got {arr.shape}."
                )
            if arr.shape[2] != 3:
                raise ValueError(f"'{key}' last dimension must be 3 (x, y, z).")
            if arr.shape[0] == 0 or arr.shape[1] == 0:
                raise ValueError(f"'{key}' contains an empty dimension.")
            if arr.shape[0] > max_frames:
                raise ValueError(f"'{key}' exceeds {max_frames} frames.")
            if arr.shape[1] > max_keypoints:
                raise ValueError(f"'{key}' exceeds {max_keypoints} keypoints.")
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"'{key}' must contain numeric values.")
            if require_finite and not np.all(np.isfinite(arr)):
                raise ValueError(f"'{key}' contains NaN or infinite values.")
            if frame_count is None:
                frame_count = arr.shape[0]
            elif arr.shape[0] not in {1, frame_count} and frame_count != 1:
                raise ValueError("Pose/FK arrays have incompatible frame counts.")
            elif frame_count == 1 and arr.shape[0] > 1:
                frame_count = arr.shape[0]
            total_elements += arr.size
            if total_elements > max_total_elements:
                raise ValueError("Scientific data exceed the total element limit.")
            validated[key] = arr.astype(np.float64, copy=False)
            shapes[key] = arr.shape

    output: Any = validated["array"] if input_was_array else validated
    summary = {
        "data_type": inferred,
        "frames": frame_count,
        "array_count": len(validated),
        "total_elements": total_elements,
        "shapes": shapes,
    }
    return output, summary
