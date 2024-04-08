"""Utility functions for extracting ARPES information from the HDF5 file conventions."""

import numpy as np
import h5py
from .common import safe_reshape

__all__ = (
    "construct_coords",
    "get_attrs",
    "dataset_to_array",
)


def get_scan_coords(
    scan_info: dict[str, str], scalar_data: h5py.Dataset
) -> dict[str, np.ndarray]:
    """Gets the scan coordinates from the scan information in the headers"""

    n_loops = int(scan_info["LWLVLPN"])
    scan_coords = {}
    for loop in range(n_loops):
        n_scan_coords = int(scan_info[f"NMSBDV{loop}"])
        n_scan_dimensions = 0
        for i in range(n_scan_coords):
            if f"ST_{loop}_{i}" in scan_info:
                n_scan_dimensions += 1

        shape = tuple(
            [int(scan_info[f"N_{loop}_{i}"]) for i in range(n_scan_dimensions)][::-1]
        )

        for scan_dimension in range(n_scan_dimensions):
            if f"ST_{loop}_{scan_dimension}" not in scan_info:
                continue
            name = scan_info[f"NM_{loop}_{scan_dimension}"]
            raw_data = dataset_to_array(scalar_data[name])
            reshaped_data = np.moveaxis(
                safe_reshape(raw_data, shape), scan_dimension, 0
            )
            averaged_data = reshaped_data.mean(axis=tuple(range(n_scan_dimensions - 1)))
            scan_coords[name] = averaged_data

    return scan_coords


def construct_coords(
    hdf5: h5py.File,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str]], tuple[str]]:
    """
    Constructs all coordinates from the HDF5 file, including the scan coordinates and the detector coordinates.
    Returns a dictionary of the coordinates and a dictionary of the dimensions for each data variable
    """

    scan_header = hdf5["Headers"]["Scan"] if "Scan" in hdf5["Headers"] else []
    low_level_scan_header = hdf5["Headers"]["Low_Level_Scan"]
    scan_info = {}
    for item in list(scan_header) + list(low_level_scan_header):
        item: list[bytes]
        try:
            scan_info[item[1].decode("utf-8").strip()] = (
                item[2].decode("utf-8").replace("'", "")
            )
        except UnicodeDecodeError:
            pass

    scan_coords = get_scan_coords(scan_info, hdf5["0D_Data"])
    scan_coord_names = tuple(scan_coords.keys())
    data_dimensions = {}
    constructed_coords = {}

    for scalar_data in hdf5["0D_Data"]:
        if scalar_data in scan_coords:
            continue
        data_dimensions[scalar_data] = tuple(scan_coords.keys())

    for n_dims in range(1, 3):
        dataset = hdf5[f"{n_dims}D_Data"]
        for data_name in dataset:
            attrs = dataset[data_name].attrs
            offsets = attrs["scaleOffset"][::-1]
            deltas = attrs["scaleDelta"][::-1]
            coord_names = attrs["unitNames"][::-1]
            coord_lengths = dataset[data_name].shape[:n_dims]

            all_coords = ()
            for dim in range(n_dims):
                coord_name = coord_names[dim]
                coord = {
                    coord_name: np.linspace(
                        offsets[dim],
                        offsets[dim] + deltas[dim] * coord_lengths[dim],
                        coord_lengths[dim],
                    )
                }

                if coord_name in constructed_coords:
                    if np.array_equal(
                        coord[coord_name], constructed_coords[coord_name]
                    ):
                        continue
                    else:
                        coord_name = f"{coord_name}_{data_name}"
                        coord[f"{coord_name}_{data_name}"] = coord.pop(coord_name)

                constructed_coords.update(coord)
                all_coords += (set(coord).pop(),)
            all_coords += tuple(scan_coords.keys())

            data_dimensions[data_name] = all_coords

    for coord_name, values in scan_coords.items():
        if coord_name in constructed_coords:
            constructed_coords[f"{coord_name}_scan"] = values
        else:
            constructed_coords[coord_name] = values

    return constructed_coords, data_dimensions, scan_coord_names


def get_attrs(hdf5: h5py.File) -> dict[str, str]:
    """Gets the relevant attributes from the HDF5 file"""

    attrs = {}
    try:
        comments = hdf5["Comments"]["PreScan"]
        attrs["comment"] = comments[0][0].decode("ascii")
    except KeyError:
        pass

    def clean_attr(value: bytes) -> float | str:
        value = value.decode("ascii")
        try:
            return float(value)
        except ValueError:
            return value.strip("'")

    column_for_name = {"Beamline": 3}
    skip_headers = ["Low_Level_Scan", "Scan", "Switch"]
    headers = hdf5["Headers"]
    for header in headers:
        if header in skip_headers:
            continue

        header_attrs = {
            item[column_for_name.get(header, 0)].decode("ascii"): clean_attr(item[2])
            for item in headers[header]
        }
        attrs.update(header_attrs)

    return attrs


def dataset_to_array(dataset: h5py.Dataset, type: str = "float64") -> np.ndarray:
    """Quickly converts a h5py dataset to a numpy array"""
    # Converting to a numpy array without using .astype takes ~1000x longer
    return dataset.astype(type)[:]


# def construct_scan_coords(scan_info: Dict[str, str]) -> Dict[str, np.ndarray]:
#     """Gets the scan coordinates from the scan information in the headers"""

#     n_loops = int(scan_info["LWLVLPN"])
#     scan_coords = {}
#     for loop in range(n_loops):
#         n_scan_dimensions = int(scan_info[f"NMSBDV{loop}"])

#         for scan_dimension in range(n_scan_dimensions):
#             if f"ST_{loop}_{scan_dimension}" not in scan_info:
#                 continue
#             name, start, end, n = (
#                 scan_info[f"NM_{loop}_{scan_dimension}"],
#                 float(scan_info[f"ST_{loop}_{scan_dimension}"]),
#                 float(scan_info[f"EN_{loop}_{scan_dimension}"]),
#                 int(scan_info[f"N_{loop}_{scan_dimension}"]),
#             )
#             scan_coords[name] = np.linspace(start, end, n, endpoint=True)

#     return scan_coords

# def extract_coords(
#     scan_info: Dict[str, Any],
# ) -> Tuple[CoordsDict, List[Dimension], List[int]]:
#     """Does the hard work of extracting coordinates from the scan description.

#     Args:
#         attrs:
#         dimension_renamings:
#         trace: A Trace instance used for debugging. You can pass True or False (including to the originating load_data call)
#             to enable execution tracing.

#     Returns:
#         A tuple consisting of the coordinate arrays, the dimension names, and their shapes
#     """

#     n_loops = scan_info["LWLVLPN"]

#     scan_dimension = []
#     scan_shape = []
#     scan_coords = {}
#     for loop in range(n_loops):
#         n_scan_dimensions = scan_info[f"NMSBDV{loop}"]
#         if scan_info[f"SCNTYP{loop}"] == 0:
#             for i in range(n_scan_dimensions):
#                 name, start, end, n = (
#                     scan_info[f"NM_{loop}_{i}"],
#                     float(scan_info[f"ST_{loop}_{i}"]),
#                     float(scan_info[f"EN_{loop}_{i}"]),
#                     int(scan_info[f"N_{loop}_{i}"]),
#                 )

#                 scan_dimension.append(name)
#                 scan_shape.append(n)
#                 scan_coords[name] = np.linspace(start, end, n, endpoint=True)
#         else:  # tabulated scan, this is more complicated
#             # In the past this branch has been especially tricky.
#             # I know of at least two client pieces of data:
#             #    * Tabulated scans which include angle-compensated scans
#             #    * Multi region scans at MAESTRO
#             #
#             # Remarkably, I'm having a hard time figuring out how this code ever worked
#             # in the past for beta compensated scans which appear to be stored with a literal table.
#             # I think in the past I probably neglected to unfreeze the tabulated coordinates which were
#             # attached since they do not matter much from the perspective of analysis.
#             #
#             # As of 2021, that is the perspective we are taking on the issue.
#             if n_scan_dimensions > 1:
#                 for i in range(n_scan_dimensions):
#                     name = scan_info[f"NM_{loop}_{i}"]
#                     if f"ST_{loop}_{i}" not in scan_info and f"PV_{loop}_{i}_0" in scan_info:
#                         continue
#                     start, end, n = (
#                         float(scan_info[f"ST_{loop}_{i}"]),
#                         float(scan_info[f"EN_{loop}_{i}"]),
#                         int(scan_info[f"N_{loop}_{i}"]),
#                     )

#                     scan_dimension.append(name)
#                     scan_shape.append(n)
#                     scan_coords[name] = np.linspace(start, end, n, endpoint=True)

#             else:
#                 name, n = (
#                     scan_info[f"NM_{loop}_0"],
#                     scan_info[f"NMPOS_{loop}"],
#                 )

#                 try:
#                     n_regions_key = {"Delay": "DS_NR"}.get(name, "DS_NR")
#                     n_regions = scan_info[n_regions_key]
#                 except KeyError:
#                     if "ST_{}_1".format(loop) in scan_info:
#                         warnings.warn("More than one region detected but unhandled.")

#                     n_regions = 1

#                 coord = np.array(())
#                 for region in range(n_regions):
#                     start, end, n = (
#                         scan_info[f"ST_{loop}_{region}"],
#                         scan_info[f"EN_{loop}_{region}"],
#                         scan_info[f"N_{loop}_{region}"],
#                     )

#                     coord = np.concatenate((coord, np.linspace(start, end, n, endpoint=True)))

#                 scan_dimension.append(name)
#                 scan_shape.append(len(coord))
#                 scan_coords[name] = coord
#     return scan_coords, scan_dimension, scan_shape


# def construct_coords(
#     hdf5: h5py.File,
#     spectra: Optional[Any] = None,
#     mode: str = "ToF",
# ) -> Tuple[CoordsDict, Dict[str, List[Dimension]], Dict[str, Any]]:
#     """Determines the scan degrees of freedom, and reads coordinates.

#     To do this, we also extract the shape of the actual "spectrum" before reading and parsing
#     the coordinates from the header information in the recorded scan.

#     Note: because different scan configurations can have different values of the detector coordinates, such as
#     for instance when you record in two different angular modes of the spectrometer or when you record XPS spectra
#     at different binding energies, we need to be able to provide separate coordinates for each of the different scans.

#     In the case where there is a unique coordinate, we will return only that coordinate, under the anticipated name,
#     such as 'eV'. Otherwise, we will return the coordinates that different between the scan configurations under the
#     spectrum name, and with unique names, such as 'eV-Swept_Spectra0'.

#     Args:
#         spectra
#         hdu
#         attrs
#         mode: Available modes are "ToF", "MC". This customizes the read process
#         dimension_renamings

#     Returns:
#         A tuple consisting of
#         (coordinates, dimensions, np shape of actual spectrum)
#     """

#     def extract_scan_info(hdf5: h5py.File):
#         scan_header = hdf5["Headers"]["Scan"]
#         low_level_scan_header = hdf5["Headers"]["Low_Level_Scan"]

#         scan_info = {}
#         for item in list(scan_header) + list(low_level_scan_header):
#             scan_info[item[1].decode("ascii").strip()] = item[2].decode("ascii").replace("'", "")

#         return scan_info

#     scan_coords, scan_dimension, scan_shape = extract_coords(
#         extract_scan_info(hdf5),
#     )

#     # bit of a hack to deal with the internal motor used for the swept spectra being considered as a cycle
#     # if "cycle" in scan_coords and len(scan_coords["cycle"]) > 200:
#     #     idx = scan_dimension.index("cycle")

#     #     real_data_for_cycle = hdf5.data.columns["null"].array

#     #     scan_coords["cycle"] = real_data_for_cycle
#     #     scan_shape[idx] = len(real_data_for_cycle)

#     scan_dimension = scan_dimension[::-1]
#     extra_coords = {}
#     scan_shape = scan_shape[::-1]

#     spectrum_shapes = {}
#     dimensions_for_spectra = {}

#     for scalar_data_var in hdf5["0D_Data"]:
#         spectrum_shapes[scalar_data_var] = scan_shape
#         dimensions_for_spectra[scalar_data_var] = scan_dimension

#     for spectrum_key in spectra:

#         if isinstance(spectrum_key, str):
#             spectrum_key = hdf5.columns.names.index(spectrum_key) + 1

#         spectrum_name = hdf5.columns.names[spectrum_key - 1]
#         loaded_shape_from_header = False
#         desc = None

#         spectrum_shapes[spectrum_name] = scan_shape
#         dimensions_for_spectra[spectrum_name] = scan_dimension

#         should_skip = False
#         for skipped in skip_names:
#             if callable(skipped) and skipped(spectrum_name):
#                 should_skip = True
#             elif skipped == spectrum_name:
#                 should_skip = True
#         if should_skip:
#             continue

#         try:
#             offset = hdf5.header[f"TRVAL{spectrum_key}"]
#             delta = hdf5.header[f"TDELT{spectrum_key}"]
#             offset = literal_eval(offset) if isinstance(offset, str) else offset
#             delta = literal_eval(delta) if isinstance(delta, str) else delta

#             try:
#                 shape = hdf5.header[f"TDIM{spectrum_key}"]
#                 shape = literal_eval(shape) if isinstance(shape, str) else shape
#                 loaded_shape_from_header = True
#             except:
#                 shape = hdf5.data.field(spectrum_key - 1).shape

#             try:
#                 desc = hdf5.header[f"TDESC{spectrum_key}"]
#                 if "(" in desc:
#                     # might be a malformed tuple, we can't use literal_eval unfortunately
#                     desc = desc.replace("(", "").replace(")", "").split(",")

#                 if isinstance(desc, str):
#                     desc = (desc,)
#             except KeyError:
#                 pass

#             if not isinstance(delta, Iterable):
#                 delta = (delta,)
#             if not isinstance(offset, Iterable):
#                 offset = (offset,)

#         if not scan_shape and shape[0] == 1:
#             # the ToF pads with ones on single EDCs
#             shape = shape[1:]

#         if mode == "ToF":
#             rest_shape = shape[len(scan_shape) :]
#         else:
#             if isinstance(desc, tuple):
#                 rest_shape = shape[-len(desc) :]
#             elif not loaded_shape_from_header:
#                 rest_shape = shape[1:]
#             else:
#                 rest_shape = shape

#         assert len(offset) == len(delta) and len(delta) == len(rest_shape)

#         # Build the actually coordinates
#         coords = [
#             np.linspace(o, o + s * d, s, endpoint=False)
#             for o, d, s in zip(offset, delta, rest_shape)
#         ]

#         # We need to do smarter inference here
#         def infer_hemisphere_dimensions() -> List[Dimension]:
#             # scans can be two dimensional per frame, or a
#             # scan can be either E or K integrated, or something I've never seen before
#             # try to get the description or the UNIT
#             if desc is not None:
#                 RECOGNIZED_DESCRIPTIONS = {
#                     "eV": "eV",
#                     "pixels": "pixel",
#                     "pixel": "pixel",
#                 }

#                 if all(d in RECOGNIZED_DESCRIPTIONS for d in desc):
#                     return [RECOGNIZED_DESCRIPTIONS[d] for d in desc]

#             try:
#                 # TODO read above like desc
#                 unit = hdf5.header["TUNIT{}".format(spectrum_key)]
#                 RECOGNIZED_UNITS = {
#                     # it's probably 'arb' which doesn't tell us anything...
#                     # because all spectra have arbitrary absolute intensity
#                 }
#                 if all(u in RECOGNIZED_UNITS for u in unit):
#                     return [RECOGNIZED_UNITS[u] for u in unit]
#             except KeyError:
#                 pass

#             # Need to fall back on some human in the loop to improve the read here
#             import pdb

#             pdb.set_trace()

#         # TODO for cleanup in future, these should be provided by the implementing endstation class, so they do not
#         # get so cluttered, best way will be to make this function a class method, and use class attributes for
#         # each of `coord_names_for_spectrum`, etc. For now, patching to avoid error with the microscope camera images
#         # at BL7
#         coord_names_for_spectrum = {
#             "Time_Spectra": ["time"],
#             "Energy_Spectra": ["eV"],
#             # MC hemisphere image, this can still be k-integrated, E-integrated, etc
#             "wave": ["time"],
#             "targetPlus": ["time"],
#             "targetMinus": ["time"],
#             "Energy_Target_Up": ["eV"],
#             "Energy_Target_Down": ["eV"],
#             "Energy_Up": ["eV"],
#             "Energy_Down": ["eV"],
#             "Energy_Pol": ["eV"],
#         }

#         spectra_types = {
#             "Fixed_Spectra",
#             "Swept_Spectra",
#         }

#         time_spectra_type = {
#             "Time_Target",
#         }
#         coord_names = None
#         if spectrum_name not in coord_names_for_spectrum:
#             # Don't remember what the MC ones were, so I will wait to do those again
#             # Might have to add new items for new spectrometers as well
#             if any(s in spectrum_name for s in spectra_types):
#                 coord_names = infer_hemisphere_dimensions
#             elif any(s in spectrum_name for s in time_spectra_type):
#                 coord_names = [
#                     "time",
#                 ]
#             else:
#                 import pdb

#                 pdb.set_trace()
#         else:
#             coord_names = coord_names_for_spectrum[spectrum_name]

#         if callable(coord_names):
#             coord_names = coord_names()
#             if len(coord_names) > 1 and mode == "MC":
#                 # for whatever reason, the main chamber records data
#                 # in nonstandard byte order
#                 coord_names = coord_names[::-1]
#                 rest_shape = list(rest_shape)[::-1]
#                 coords = coords[::-1]

#         coords_for_spectrum = dict(zip(coord_names, coords))
#         # we need to store the coordinates that were kept in a table separately, because they are allowed to differ
#         # between different scan configurations in the same file
#         if mode == "ToF":
#             extra_coords.update(coords_for_spectrum)
#         else:
#             extra_coords[spectrum_name] = coords_for_spectrum
#         dimensions_for_spectra[spectrum_name] = tuple(scan_dimension) + tuple(coord_names)
#         spectrum_shapes[spectrum_name] = tuple(scan_shape) + tuple(rest_shape)
#         coords_for_spectrum.update(scan_coords)

#     extra_coords.update(scan_coords)

#     if mode != "ToF":
#         detector_coord_names = [k for k, v in extra_coords.items() if isinstance(v, dict)]

#         from collections import Counter

#         c = Counter(item for name in detector_coord_names for item in extra_coords[name])
#         conflicted = [k for k, v in c.items() if v != 1 and k != "cycle"]

#         flat_coordinates = collect_leaves(extra_coords)

#         def can_resolve_conflict(c):
#             coordinates = flat_coordinates[c]

#             if not isinstance(coordinates, list) or len(coordinates) < 2:
#                 return True

#             # check if list of arrays is all equal
#             return functools.reduce(
#                 lambda x, y: (np.array_equal(x[1], y) and x[0], y),
#                 coordinates,
#                 (True, coordinates[0]),
#             )[0]

#         conflicted = [c for c in conflicted if not can_resolve_conflict(c)]

#         def clarify_dimensions(dims: List[Dimension], sname: str) -> List[Dimension]:
#             return [d if d not in conflicted else d + "-" + sname for d in dims]

#         def clarify_coordinate(
#             coordinates: Union[CoordsDict, ndarray], sname: str
#         ) -> Union[CoordsDict, ndarray]:
#             if not isinstance(coordinates, dict):
#                 return coordinates

#             return {
#                 k if k not in conflicted else k + "-" + sname: v for k, v in coordinates.items()
#             }

#         dimensions_for_spectra = {
#             k: clarify_dimensions(v, k) for k, v in dimensions_for_spectra.items()
#         }
#         extra_coords = {k: clarify_coordinate(v, k) for k, v in extra_coords.items()}
#         extra_coords = dict(iter_leaves(extra_coords))

#     return extra_coords, dimensions_for_spectra, spectrum_shapes
