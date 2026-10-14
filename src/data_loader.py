import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions
import streamlit as st

@st.cache_resource
def load_fits_data(image_file):
    image_data = fits.getdata(image_file, ext=0)
    hdu = fits.open(image_file)[0]
    wcs = WCS(hdu.header)
    return image_data, hdu, wcs

@st.cache_resource
def load_and_clean_radio(radio_image_file):
    hdu_radio = fits.open(radio_image_file)[0]
    radio_data = fits.getdata(radio_image_file, ext=0).squeeze()

    high_dimm_keys = [
        "CRPIX3", "CTYPE3", "CDELT3", "CRVAL3", "CRPIX4", "CTYPE4", "CDELT4", "CRVAL4",
        "CUNIT3", "CUNIT4", "NAXIS3", "NAXIS4", "PC3_1", "PC3_2", "PC3_3", "PC3_4",
        "PC4_1", "PC4_2", "PC4_3", "PC4_4", "PC1_3", "PC1_4", "PC2_3", "PC2_4"
    ]
    for key in high_dimm_keys:
        if key in hdu_radio.header:
            del hdu_radio.header[key]
    wcs_radio = WCS(hdu_radio.header, naxis=2)
    return hdu_radio, radio_data, wcs_radio

@st.cache_resource
def load_catalog_and_regions(src_list_path, region_file_path, multi_wave_path=None):
    df = pd.read_csv(src_list_path)
    regions = Regions.read(region_file_path, format='ds9')
    multi_df = None
    if multi_wave_path:
        try:
            multi_df = pd.read_csv(multi_wave_path)
        except Exception:
            pass
    return df, regions, multi_df