astronomical_crossmatching_pipeline/
├── app.py                      # Streamlit UI, state management, and tab orchestration
├── config.py                   # Centralized configuration and default file paths
└── src/
    ├── __init__.py
    ├── data_loader.py          # FITS loading, caching, catalog reading, and regions
    └── cutouts.py              # Cutout2D extraction, ZScale, and APLpy rendering with circles