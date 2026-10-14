import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval

# Enforcing non-interactive Agg backend for multiprocessing workers.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import aplpy

def get_zscale_limits(data):
    clean_data = data[~np.isnan(data) & ~np.isinf(data)]
    if clean_data.size == 0:
        return np.nanmin(data), np.nanmax(data)
    interval = ZScaleInterval()
    return interval.get_limits(clean_data)

def get_background_variance(data):
    clean_data = data[~np.isnan(data) & ~np.isinf(data)]
    if clean_data.size == 0:
        return 1.0
    med = np.median(clean_data)
    mad = np.median(np.abs(clean_data - med))
    return (1.4826 * mad) ** 2

def process_single_cutout(i, source_id_series, ra_series, dec_series, image_data, hdu_header, wcs, radio_data, hdu_radio_header, wcs_radio, cutouts_dir, multi_df=None):
    try:
        os.makedirs(cutouts_dir, exist_ok=True)
        source_id = source_id_series.iloc[i]
        ra = ra_series.iloc[i]
        dec = dec_series.iloc[i]
        position = SkyCoord(ra, dec, frame='icrs', unit='deg')

        # Generating Radio Cutout FITS
        cutout_radio = Cutout2D(radio_data, position, (1800, 1800), wcs=wcs_radio)
        radio_file_out = os.path.join(cutouts_dir, f"Radiocutout{source_id}.fits")
        hdu_radio = fits.PrimaryHDU(data=cutout_radio.data, header=hdu_radio_header.copy())
        hdu_radio.header.update(cutout_radio.wcs.to_header())
        hdu_radio.writeto(radio_file_out, overwrite=True)

        # Generating Optical Cutout FITS
        cutout_optical = Cutout2D(image_data, position, (3200, 3200), wcs=wcs)
        optical_file_out = os.path.join(cutouts_dir, f"Irac_Mosaic_cutout{source_id}.fits")
        hdu = fits.PrimaryHDU(data=cutout_optical.data, header=hdu_header.copy())
        hdu.header.update(cutout_optical.wcs.to_header())
        hdu.writeto(optical_file_out, overwrite=True)

        # Creating Radio PNG Cutout
        r_data_loaded = fits.getdata(radio_file_out, ext=0)
        r_hdu_loaded = fits.open(radio_file_out)[0]
        r_wcs_loaded = WCS(r_hdu_loaded.header, naxis=2, fix=True)
        vmin_r, vmax_r = get_zscale_limits(r_data_loaded)
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=r_wcs_loaded)
        ax.imshow(r_data_loaded, cmap='gray', origin='lower', vmin=vmin_r, vmax=vmax_r)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.set_title(f"Radio Cutout: {source_id}", fontweight='bold', pad=20)
        radio_png = os.path.join(cutouts_dir, f"Radiocutout{source_id}.png")
        plt.savefig(radio_png, bbox_inches='tight')
        plt.close(fig)
        plt.close('all')

        # Creating Optical PNG Cutout
        opt_data_loaded = fits.getdata(optical_file_out, ext=0)
        opt_hdu_loaded = fits.open(optical_file_out)[0]
        opt_wcs_loaded = WCS(opt_hdu_loaded.header, naxis=2, fix=True)
        vmin_opt, vmax_opt = get_zscale_limits(opt_data_loaded)
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=opt_wcs_loaded)
        ax.imshow(opt_data_loaded, cmap='gray', origin='lower', vmin=vmin_opt, vmax=vmax_opt)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.set_title(f"Mosaic Cutout: {source_id}", fontweight='bold', pad=20)
        opt_png = os.path.join(cutouts_dir, f"Mosaic_cutout{source_id}.png")
        plt.savefig(opt_png, bbox_inches='tight')
        plt.close(fig)
        plt.close('all')

        # APLpy Overlay or Postage Stamp with Counterpart Detection Circles
        sigma = np.sqrt(get_background_variance(r_data_loaded.flatten()))
        adfs_levels = np.logspace(np.log10(max(sigma, 1e-5)), np.log10(0.5 * np.nanmax(r_data_loaded)), num=10)

        overlay_png = os.path.join(cutouts_dir, f"overlay_{source_id}.png")
        fig_apl = plt.figure(figsize=(10, 10))
        f = aplpy.FITSFigure(optical_file_out, figure=fig_apl, north=True)
        f.set_title(f"ID #{source_id}: Radio Contour & Counterpart Overlay", size=16, pad=20)
        f.show_grayscale(stretch='sqrt', invert=True)
        f.recenter(ra, dec, radius=0.5 / 60.0) # 0.5 arcmin view
        
        # Primary or main source marker
        f.show_markers(ra, dec, edgecolor='g', marker='*', s=70, linewidths=2.0, zorder=10)
        
        # Search / detection radius circle (e.g. 2 arcsec)
        f.show_circles(ra, dec, 2.0 / 3600.0, edgecolor='cyan', linewidth=1.5, linestyle='--', zorder=9)

        # Multi-wavelength counterpart detections (if provided)
        if multi_df is not None and not multi_df.empty and 'RA' in multi_df.columns and 'DEC' in multi_df.columns:
            m_ras = multi_df['RA'].to_numpy()
            m_decs = multi_df['DEC'].to_numpy()
            f.show_circles(m_ras, m_decs, 1.5 / 3600.0, edgecolor='orange', linewidth=1.2, zorder=8)

        # Radio Contours
        f.show_contour(radio_file_out, levels=adfs_levels, colors='green', overlap=True, smooth=1, linewidths=1.2)
        f.save(overlay_png)
        f.close()
        plt.close('all')

        return radio_png, opt_png, overlay_png

    except Exception as e:
        plt.close('all')
        raise e