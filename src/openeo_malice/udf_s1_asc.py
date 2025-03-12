#!/usr/bin/env python

# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call MALICE model
for Sentinel-1 Ascending time series embedding
"""
import sys
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube, inspect

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


def normalize_s1(input_data: np.ndarray) -> np.ndarray:
    med = np.array([-3.7551794052124023, -2.246965169906616, -1.5142076015472412])
    qmin = np.array([-5.7809295654296875, -3.768258810043335, -3.211071252822876])
    qmax = np.array([-2.4399311542510986, -1.0318946838378906, -0.174086332321167])
    scale = np.array([float(x) - float(y) for x, y in zip(qmax, qmin)])
    input_data = input_data.clip(qmin[None, :, None, None], qmax[None, :, None, None])
    return np.nan_to_num((input_data - med[None, :, None, None]) / scale[None, :, None, None])


def run_inference(input_data: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """
    Inference function for Sentinel-1 embeddings with MALICE.
    The input should be in shape (B, C, H, W)
    The output shape is (10, 64, H, W)
    """

    # First get virtualenv
    sys.path.insert(0, "tmp/extra_venv")
    import onnxruntime as ort

    model_file = "tmp/extra_files/malice_s1.onnx"
    # model_file = "../../models/malice_models/malice_s1.onnx"
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

    if input_data.shape[-1] % 32 == 0 and input_data.shape[-2] % 32 == 0: #or input_data.shape[-1] != input_data.shape[-2]:
        return np.full([10, 64, *input_data.shape[-2:]], np.nan)


    # We transform input data in right format
    vv, vh = input_data[:, 0], input_data[:, 1]

    input_data = np.log(np.stack([vh, vv, vh / vv], axis=1))

    input_data = normalize_s1(input_data)

    #
    # if input_data.shape[-1] != input_data.shape[-2]:
    #         min(np.floor(input_data.shape[-2:] / 32))

    reference_date = "2014-03-03"

    doy = (pd.to_datetime(doy) - pd.to_datetime(reference_date)).days

    input = {"sits": input_data.astype(np.float32)[None, ...],   #[..., :128, :128],
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
    inspect(data=[cubearray.data.shape], message="Shape of input cubearray")

    cube_collection = run_inference(cubearray.data, cubearray.t.values)
    # Build output data array
    predicted_cube = xr.DataArray(
        cube_collection,
        dims=["t", "bands", "y", "x"],
        # coords=dict(x=cubearray.coords["x"][:122], y=cubearray.coords["y"][:122]),
        coords=dict(x=cubearray.coords["x"], y=cubearray.coords["y"]),

    )

    return XarrayDataCube(predicted_cube)
