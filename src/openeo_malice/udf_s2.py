#!/usr/bin/env python

# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call MALICE model
for Sentinel-2 time series embedding
"""
import sys
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube

NEW_BANDS = [f"F0{i}" for i in range(10)] + [f"F{i}" for i in range(10, 64)]

NEW_DATES = [f"D0{i}" for i in range(10)]


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    metadata = metadata.rename_labels(
        dimension="band",
        target=NEW_BANDS
    )
    return metadata.rename_labels(
        dimension="t",
        target=NEW_DATES
    )


def check_datacube(cube: xr.DataArray):
    """ """
    if cube.data.ndim != 4:
        raise RuntimeError("DataCube dimensions should be (t,bands, y, x)")

    if cube.data.shape[0] > 2:
        raise RuntimeError(
            "DataCube should have at least 3 days of temporal series)"
        )


def normalize_s2(input_data: np.ndarray) -> np.ndarray:
    """Clip and normalize s2 data"""
    med = np.array([414.0, 675.0, 570.0, 1132.0, 2374.0, 2744.0, 2885.0, 3016.0, 2003.0, 1117.0])
    qmin = np.array([130.0, 247.0, 168.0, 442.0, 907.0, 1044.0, 1069.0, 1160.0, 493.0, 299.0])
    qmax = np.array([3780.0, 3704.0, 3554.0, 3885.0, 4644.0, 5296.0, 5478.0, 5494.0, 3747.0, 2892.0])
    scale = np.array([float(x) - float(y) for x, y in zip(qmax, qmin)])
    input_data = input_data.clip(qmin[None, :, None, None], qmax[None, :, None, None])
    return np.nan_to_num((input_data - med[None, :, None, None]) / scale[None, :, None, None])


def run_inference(input_data: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """
    Inference function for Sentinel-2 embeddings with MALICE.
    The input should be in shape (B, C, H, W)
    The output shape is (10, 64, H, W)
    """
    # First get virtualenv
    sys.path.insert(0, "tmp/extra_venv")
    import onnxruntime as ort

    model_file = "tmp/extra_files/malice_s2.onnx"

    # ONNX inference session options
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.use_deterministic_compute = True

    # Execute on cpu only
    ep_list = ["CPUExecutionProvider"]

    # Create inference session
    ort_session = ort.InferenceSession(model_file, sess_options=so, providers=ep_list)

    ro = ort.RunOptions()
    ro.add_run_config_entry("log_severity_level", "3")

    # We transform input data in right format
    input_data = normalize_s2(input_data)

    reference_date = "2014-03-03"

    doy = (pd.to_datetime(doy) - pd.to_datetime(reference_date)).days

    input = {"sits": input_data.astype(np.float32)[None, ...],
             "tpe": np.array(doy)[None, ...].astype(np.float32),
             "padd_mask": np.zeros((1, input_data.shape[0]), dtype=bool)}

    # Get the ouput of the exported model
    res = ort_session.run(None, input, run_options=ro)[0][0]
    print(res.shape)

    return res


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    Apply UDF function to datacube
    """
    # We get data from datacube
    cubearray: xr.DataArray
    if isinstance(cube, xr.DataArray):
        cubearray = cube
    else:
        cubearray = cube.get_array().copy()

    cube_collection = run_inference(cubearray.data, cubearray.t.values)
    # Build output data array
    predicted_cube = xr.DataArray(
        cube_collection,
        dims=["t", "bands", "y", "x"],
        coords=dict(x=cubearray.coords["x"], y=cubearray.coords["y"]),
    )

    return XarrayDataCube(predicted_cube)
