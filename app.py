import streamlit as st
import time
import os
import tempfile
import numpy as np
from multiprocessing import Pool
from astropy.coordinates import SkyCoord
import astropy.units as u

from config import (
    DEFAULT_SRC_LIST,
    DEFAULT_MULTI_LAMBDA,
    DEFAULT_REGION_FILE,
    DEFAULT_CUTOUTS_DIR,
    DEFAULT_BACKGROUND_FITS,
    DEFAULT_RADIO_FITS
)
from src.data_loader import load_fits_data, load_and_clean_radio, load_catalog_and_regions
from src.cutouts import process_single_cutout

st.set_page_config(page_title="Astronomical XID Cutout generator & Aladin Pipeline", layout="wide")

st.title("🌌 SMART AST XID PIPELINE: CUTOUT & POSTAGE STAMP MAKER")
st.markdown("Dynamic percentage slicing, single live preview, APLpy counterpart overlays, and interactive XID cross-match tools.")

def handle_input(label, default_path, file_type=None):
    mode = st.sidebar.radio(f"Input method for {label}", ("Upload File", "Server Path"), key=f"mode_{label}")
    if mode == "Upload File":
        uploaded = st.sidebar.file_uploader(f"Choose {label}", type=file_type, key=f"upload_{label}")
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.getbuffer())
            tmp.close()
            return tmp.name
    return st.sidebar.text_input(f"{label} Path", value=default_path, key=f"path_{label}")

# Sidebar Configuration
st.sidebar.header("📁 Data Inputs")
src_list = handle_input("Radio Source List CSV", DEFAULT_SRC_LIST, file_type=["csv"])
multiwave_catalog = handle_input("Multi-Wavelength Data Catalog CSV (OPT)", DEFAULT_MULTI_LAMBDA, file_type=["csv"])
region_file = handle_input("Region File", DEFAULT_REGION_FILE, file_type=["reg", "txt"])
cutouts_dir = st.sidebar.text_input("Cutouts Output Directory", value=DEFAULT_CUTOUTS_DIR)
background_image_file = handle_input("Background Mosaic FITS", DEFAULT_BACKGROUND_FITS, file_type=["fits"])
radio_image_file = handle_input("Radio Image FITS", DEFAULT_RADIO_FITS, file_type=["fits"])

st.sidebar.header("⚙️ Processing Options")
max_cores = os.cpu_count() or 4
safe_default = max(1, max_cores // 2)
nw = st.sidebar.slider("Number of CPU Processes (NW)", min_value=1, max_value=max_cores, value=min(4, safe_default))
cutout_percentage = st.sidebar.slider("Percentage of sources to process", min_value=1, max_value=100, value=30)

if "results" not in st.session_state:
    st.session_state.results = None
if "source_info" not in st.session_state:
    st.session_state.source_info = None

_worker_data = {}

def init_worker(source_id, ra, dec, image_data, hdu_header, wcs, radio_data, hdu_radio_header, wcs_radio, cutouts_dir, multi_df):
    global _worker_data
    _worker_data['source_id'] = source_id
    _worker_data['ra'] = ra
    _worker_data['dec'] = dec
    _worker_data['image_data'] = image_data
    _worker_data['hdu_header'] = hdu_header
    _worker_data['wcs'] = wcs
    _worker_data['radio_data'] = radio_data
    _worker_data['hdu_radio_header'] = hdu_radio_header
    _worker_data['wcs_radio'] = wcs_radio
    _worker_data['cutouts_dir'] = cutouts_dir
    _worker_data['multi_df'] = multi_df

def global_worker(i):
    d = _worker_data
    return process_single_cutout(
        i, d['source_id'], d['ra'], d['dec'],
        d['image_data'], d['hdu_header'], d['wcs'],
        d['radio_data'], d['hdu_radio_header'], d['wcs_radio'],
        d['cutouts_dir'], d['multi_df']
    )

# Setup Tabs for Pipeline and XID Cross-Match Module
tab_pipeline, tab_xid = st.tabs(["🚀 Cutout Pipeline & Aladin", "🎯 XID Cross-Match Reviewer"])

with tab_pipeline:
    if st.button("🚀 Start Pipeline", type="primary"):
        start_time = time.time()
        with st.status("Initializing pipeline...", expanded=True) as status:
            try:
                for path_name, path_val in [("Source List", src_list), ("Region File", region_file), ("Background Mosaic", background_image_file), ("Radio Image", radio_image_file)]:
                    if not os.path.exists(path_val):
                        status.update(label=f"❌ Error: {path_name} not found!", state="error")
                        st.error(f"Could not find **{path_name}** at: `{path_val}`.")
                        st.stop()

                status.update(label="📁 Loading Background Mosaic FITS...", state="running")
                image_data, hdu, wcs = load_fits_data(background_image_file)
                status.update(label="📻 Loading and cleaning Radio FITS image...", state="running")
                hdu_radio, radio_data, wcs_radio = load_and_clean_radio(radio_image_file)
                status.update(label="📋 Loading catalog, regions, and multi-wavelength data...", state="running")
                df, regions, multi_df = load_catalog_and_regions(src_list, region_file, multiwave_catalog)
                
                source_id = df['ID_S9']
                ra = df['RA_S9']
                dec = df['DEC_S9']
                polygon_sky = regions[0]
                
                status.update(label="🔍 Filtering targets within polygon region...", state="running")
                skycoord = SkyCoord(ra.to_numpy() * u.deg, dec.to_numpy() * u.deg, frame='icrs')
                truth_val = polygon_sky.contains(skycoord, wcs)
                src_index_in = [i for i, val in enumerate(truth_val) if val]
                total_valid_sources = len(src_index_in)
                num_to_process = int(np.ceil(total_valid_sources * (cutout_percentage / 100.0)))
                indices_to_process = [src_index_in[i] for i in range(num_to_process)]
                status.update(label=f"✅ Initialization complete! {len(indices_to_process)} sources selected.", state="complete")
            except Exception as e:
                status.update(label="❌ Initialization failed!", state="error")
                st.error(f"Error during initialization: {e}")
                st.stop()

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        single_preview_slot = st.empty()

        results = []
        total_tasks = len(indices_to_process)
        
        with Pool(nw, initializer=init_worker, initargs=(
            source_id, ra, dec,
            image_data, hdu.header, wcs,
            radio_data, hdu_radio.header, wcs_radio,
            cutouts_dir, multi_df
        )) as p:
            for idx, res in enumerate(p.imap(global_worker, indices_to_process)):
                orig_idx = indices_to_process[idx]
                curr_id = source_id.iloc[orig_idx]
                radio_png, opt_png, overlay_png = res
                results.append((orig_idx, res))
                
                progress_pct = (idx + 1) / total_tasks
                
                # Throttle UI updates to prevent WebSocket flooding
                if (idx + 1) % 5 == 0 or idx == total_tasks - 1:
                    progress_bar.progress(progress_pct)
                    status_text.text(f"⚡ Processing: [{idx + 1}/{total_tasks}] (ID: {curr_id}) — {progress_pct * 100:.1f}%")
                
                if idx == 0:
                    with single_preview_slot.container():
                        st.image(overlay_png, caption=f"Live Postage Stamp Overlay - ID: {curr_id}", use_container_width=True)

        elapsed = time.time() - start_time
        st.session_state.results = results
        st.session_state.source_info = {"id": source_id, "ra": ra, "dec": dec}
        st.balloons()
        st.success(f"✨ Job Completed in {elapsed:.2f} seconds!")

    # Render Results
    if st.session_state.results is not None:
        st.header("📊 Cutout Results & Postage Stamp Overlays")
        source_id = st.session_state.source_info["id"]
        ra = st.session_state.source_info["ra"]
        dec = st.session_state.source_info["dec"]
        
        for orig_idx, (r_png, opt_png, overlay_png) in st.session_state.results:
            curr_id = source_id.iloc[orig_idx]
            curr_ra = ra.iloc[orig_idx]
            curr_dec = dec.iloc[orig_idx]
            with st.expander(f"Source ID: {curr_id} (RA: {curr_ra:.4f}, Dec: {curr_dec:.4f})", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(r_png, caption=f"Radio Cutout", use_container_width=True)
                with col2:
                    st.image(opt_png, caption=f"Optical Cutout", use_container_width=True)
                with col3:
                    st.image(overlay_png, caption=f"APLpy Contour & Counterpart Overlay", use_container_width=True)

with tab_xid:
    st.header("🎯 XID Cross-Match Review Module")
    st.markdown("Interactive classification tool styled after your `XMATCHIT` notebook pipeline.")
    
    classifier_name = st.text_input("Classifier Initials", value="Mfu")
    output_dat_file = st.text_input("Output Batch Log File", value="Batch00MFU.dat")
    
    if st.session_state.results is not None:
        res_list = st.session_state.results
        if "xid_index" not in st.session_state:
            st.session_state.xid_index = 0
            
        curr_step = st.session_state.xid_index
        if curr_step < len(res_list):
            orig_idx, (r_png, opt_png, overlay_png) = res_list[curr_step]
            source_id = st.session_state.source_info["id"]
            curr_id = source_id.iloc[orig_idx]
            
            st.subheader(f"Reviewing Source Index [{curr_step + 1} / {len(res_list)}] — Radio ID: {curr_id}")
            st.image(overlay_png, caption=f"Postage Stamp Overlay for ID {curr_id}", width=600)
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("✅ Match Confirmed ('y')"):
                    with open(output_dat_file, 'a') as f:
                        f.write(f"{curr_step} {curr_id} {classifier_name} match_ok \"\"\n")
                    st.success("Saved 'y' response!")
                    st.session_state.xid_index += 1
                    st.rerun()
            with col_btn2:
                if st.button("🔄 Recheck Required ('r')"):
                    with open(output_dat_file, 'a') as f:
                        f.write(f"{curr_step} {curr_id} {classifier_name} -2 \"RECHECK\"\n")
                    st.warning("Saved 'r' response!")
                    st.session_state.xid_index += 1
                    st.rerun()
            with col_btn3:
                if st.button("❌ No Match ('n')"):
                    with open(output_dat_file, 'a') as f:
                        f.write(f"{curr_step} {curr_id} {classifier_name} -1 \"NO MATCH\"\n")
                    st.error("Saved 'n' response!")
                    st.session_state.xid_index += 1
                    st.rerun()
        else:
            st.balloons()
            st.info("🎉 All processed sources have been reviewed in the XID Cross-Match pipeline!")
    else:
        st.info("Please run the cutout pipeline in the first tab to populate targets for XID cross-matching review.")