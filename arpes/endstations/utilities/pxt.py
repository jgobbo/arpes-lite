"""Implements Igor <-> xarray interop, notably loading Igor waves and packed experiment files."""

import warnings
from pathlib import Path

import xarray as xr

from arpes.utilities import rename_keys, safe_decode

from pygor import load, Wave


__all__ = ("read_single_pxt",)


def read_header(header_bytes: bytes):
    header_as_string = safe_decode(header_bytes)

    lines = [x for x in header_as_string.replace("\r", "\n").split("\n") if x]
    lines = [x for x in lines if "=" in x]

    header = {}
    for line in lines:
        fragments = line.split("=")
        first, rest = fragments[0], "=".join(fragments[1:])

        try:
            rest = int(rest)
        except ValueError:
            try:
                rest = float(rest)
            except ValueError:
                pass

        header[first.lower().replace(" ", "_")] = rest

    return rename_keys(
        header,
        {
            "sample_x": "x",
            "sample_y_(vert)": "y",
            "sample_y": "y",
            "sample_z": "z",
            "bl_energy": "hv",
        },
    )


def wave_to_xarray(wave: Wave) -> xr.DataArray:
    """Convert a wave to an `xr.DataArray`.

    Units, if present on the wave, are used to furnish the dimension names. If dimension
    names are not present, placeholder names ("X", "Y", "Z", "W", as in Igor) are used
    for each unitless dimension.

    Args:
        wave: The input wave, an `igor.Wave` instance.

    Returns:
        The converted `xr.DataArray` instance.
    """
    # only need four because Igor only supports four dimensions!
    extra_names = iter(["W", "X", "Y", "Z"])
    n_dims = len([a for a in wave.axis if len(a)])

    def get_axis_name(index: int) -> str:
        unit = wave.axis_units[index]
        if unit:
            return {
                "eV": "eV",
                "deg": "phi",
                "Pwr Supply V": "volts",
                "K2200 V": "volts",
            }.get(unit, unit)

        return next(extra_names)

    axis_names = [get_axis_name(i) for i in range(n_dims)]
    coords = dict(zip(axis_names, wave.axis))

    return xr.DataArray(
        wave.data,
        coords=coords,
        dims=axis_names,
        attrs=read_header(wave.notes),
    )


def read_single_pxt(
    reference_path: Path | str,
) -> xr.DataArray:
    """Use igor.igorpy to load a single .PXT or .PXP file."""

    if isinstance(reference_path, Path):
        reference_path = str(reference_path.absolute())

    loaded = None
    for try_byte_order in [">", "=", "<"]:
        try:
            loaded = load(reference_path, initial_byte_order=try_byte_order)
            break
        except Exception:  # pylint: disable=broad-except
            # bad byte ordering probably
            pass

    children = [c for c in loaded.children if isinstance(c, Wave)]

    if len(children) == 1:
        return wave_to_xarray(children[0])

    warnings.warn(
        f"Igor PXT file contained {len(children)} waves. Ignoring all but first."
    )
    return wave_to_xarray(children[0])
