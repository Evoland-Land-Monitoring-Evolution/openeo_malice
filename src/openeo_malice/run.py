# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
MALICE embeddings with OpenEO
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

from openeo import DataCube

DEPENDENCIES_URL: str = (
    "https://artifactory.vgt.vito.be:443/auxdata-public/openeo/onnx_dependencies.zip"
)

import openeo

from openeo_malice import __version__

__author__ = "Ekaterina Kalinicheva"
__copyright__ = "Ekaterina Kalinicheva"
__license__ = "AGPL-3.0-or-later"

_logger = logging.getLogger(__name__)


def default_bands_list(satellite: str) -> list[str]:
    """Default bands for each satellite"""
    if satellite.lower() == "s2":
        return ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    return ["VV", "VH"]


@dataclass(frozen=True)
class Parameters:
    """Collection parameters"""
    spatial_extent: Dict[str, float]
    start_date: str
    end_date: str
    output_file: str
    max_cloud_cover: int = 30
    openeo_instance: str = "openeo.vito.be"
    collection: str = "SENTINEL2_L2A"
    satellite: str = "s2"
    patch_size: int = 128
    overlap: Optional[int] = 0


def process(parameters: Parameters, output: str) -> None:
    """
    Main processing function
    """
    # First connect to OpenEO instance
    connection = openeo.connect(parameters.openeo_instance).authenticate_oidc()

    # Search for the S2 datacube
    if parameters.satellite.lower() == "s2":
        MODEL_URL: str = \
            "https://artifactory.vgt.vito.be/artifactory/evoland/malice_models/malice_s2.zip"
        sat_cube: DataCube = connection.load_collection(
            parameters.collection,
            spatial_extent=parameters.spatial_extent,
            temporal_extent=[parameters.start_date, parameters.end_date],
            bands=default_bands_list(parameters.satellite),
            max_cloud_cover=parameters.max_cloud_cover,
            fetch_metadata=True
        )
    else:
        MODEL_URL: str = \
            "https://artifactory.vgt.vito.be/artifactory/evoland/malice_models/malice_s1.zip"
        sat_cube: DataCube = connection.load_collection(
            parameters.collection,
            spatial_extent=parameters.spatial_extent,
            temporal_extent=[parameters.start_date, parameters.end_date],
            bands=default_bands_list(parameters.satellite),
            fetch_metadata=True,
            properties={"sat:orbit_state": lambda od: od == "ASCENDING"}
        ).sar_backscatter(
            coefficient="gamma0-terrain",
            elevation_model=None,
            mask=False,
            contributing_area=False,
            local_incidence_angle=False,
            ellipsoid_incidence_angle=False,
            noise_removal=True,
            options=None,
        )

    udf_file = os.path.join(os.path.dirname(__file__), f"udf_{parameters.satellite}.py")
    udf = openeo.UDF.from_file(udf_file, runtime="Python-Jep")

    # Handle optional overlap parameter
    overlap = []
    if parameters.overlap is not None:
        overlap = [
            {"dimension": "x", "value": parameters.overlap, "unit": "px"},
            {"dimension": "y", "value": parameters.overlap, "unit": "px"},
        ]

    # Process the cube with the UDF
    malice_sat_cube = sat_cube.apply_neighborhood(
        udf,
        size=[
            {"dimension": "x",
             "value": parameters.patch_size - parameters.overlap * 2,
             "unit": "px"},
            {"dimension": "y",
             "value": parameters.patch_size - parameters.overlap * 2,
             "unit": "px"},
        ],
        overlap=overlap,
        # context={"satellite": parameters.satellite.lower()}
    )
    job_options = {
        "udf-dependency-archives": [
            f"{DEPENDENCIES_URL}#tmp/extra_venv",
            f"{MODEL_URL}#tmp/extra_files",
        ],
        "executor-memory": "10G",
        "executor-memoryOverhead": "20G",  # default 2G
        "executor-cores": 2,
        "task-cpus": 1,
        "executor-request-cores": "400m",
        "max-executors": "100",
        "driver-memory": "16G",
        "driver-memoryOverhead": "16G",
        "driver-cores": 5,
    }
    download_job1 = malice_sat_cube.save_result("netCDF").create_job(
        title=f"malice_{parameters.satellite}", job_options=job_options
    )

    download_job1.start_and_wait()
    os.makedirs(output, exist_ok=True)
    download_job1.get_results().download_files(output)

    download_job2 = malice_sat_cube.save_result("netCDF").create_job(title="sits-orig")
    download_job2.start_and_wait()
    os.makedirs(os.path.join(output, "original"), exist_ok=True)
    download_job2.get_results().download_files(output)


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Run Multi-Temporal MALICE embedding model on OpenEO and download results"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"openeo_malice {__version__}",
    )
    parser.add_argument(
        "--satellite", help="Modality we want to encode. Should be s1_asc or s2",
        type=str, required=True
    )
    parser.add_argument(
        "--start_date", help="Start date (format: YYYY-MM-DD)", type=str, required=True
    )
    parser.add_argument(
        "--end_date", help="End date (format: YYYY-MM-DD)", type=str, required=True
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        help="Extent (west lat, east lat, south lon, north lon)",
        required=True,
    )
    parser.add_argument(
        "--output", type=str, help="Path to ouptput NetCDF file", required=True
    )
    parser.add_argument(
        "--overlap",
        required=False,
        default=10,
        type=int,
        help="Overlap between patches to avoid border effects",
    )

    parser.add_argument(
        "--instance",
        type=str,
        default="openeo.vito.be",
        help="OpenEO instance on which to run the MALICE algorithm",
    )
    return parser.parse_args(args)


def main(args):
    """
    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)

    # Build parameters
    parameters = Parameters(
        spatial_extent={
            "west": args.extent[0],
            "east": args.extent[1],
            "south": args.extent[2],
            "north": args.extent[3],
        },
        openeo_instance=args.instance,
        start_date=args.start_date,
        end_date=args.end_date,
        collection="SENTINEL2_L2A"
        if args.satellite.lower() == "s2" else "SENTINEL1_GRD",
        satellite=args.satellite.lower(),
        output_file=args.output,
        overlap=args.overlap,
    )
    print(parameters)

    _logger.info(f"Parameters : {parameters}")
    process(parameters, args.output)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m openeo_malice.skeleton 42
    #
    run()
