"""
STEP 0 -- extract the raw ENIA variables used downstream into chile_raw.mat

Input : chile_original.dta
        (from Raval (2023)'s replication package:
         "Replication Files.zip" -> Data Cleaning/Chile/chile_original.dta;
         if the .dta is not in this folder it is read straight from the zip)
Output: chile_raw.mat, read by step1_prepare_data.m (T = p = 4 design)
        and step1b_prepare_data_T5.m (T = p = 5 design)

Variables kept (all real series are in thousands of 1985 pesos):
    id, year, ciiu_3d          plant id, year, 3-digit industry
    routput, rva               real gross output, real value added
    realmats, renerg           real materials, real energy
    rmidbldg, rmidmach, rmidveh   mid-year real capital stocks
    totalcnt                   total employment headcount

Run from the project root:  python export_raw.py
"""
from __future__ import annotations

import os
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio

DTA = "chile_original.dta"
ZIP = "Replication Files.zip"
ZIP_MEMBER = "Data Cleaning/Chile/chile_original.dta"
OUT = "chile_raw.mat"

COLS = ["id", "year", "ciiu_3d", "routput", "rva", "realmats", "renerg",
        "rmidbldg", "rmidmach", "rmidveh", "totalcnt"]


def read_raw() -> pd.DataFrame:
    if os.path.exists(DTA):
        return pd.read_stata(DTA, columns=COLS, convert_categoricals=False)
    with zipfile.ZipFile(ZIP) as z, z.open(ZIP_MEMBER) as f:
        return pd.read_stata(f, columns=COLS, convert_categoricals=False)


if __name__ == "__main__":
    df = read_raw()
    print(f"read {len(df)} plant-years, {df.id.nunique()} plants, "
          f"{int(df.year.min())}-{int(df.year.max())}")
    sio.savemat(OUT, {c: df[c].to_numpy(dtype=float)[:, None] for c in COLS},
                do_compression=True)
    print(f"wrote {OUT}")
