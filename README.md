# Land Cap Analysis

Computes the smallest spherical caps containing 90% and 100% of land points from paleogeographic datasets. Produces AEQD map visualizations and summary statistics for each time period.

## Data

This code operates on paleogeographic land-point CSV files derived from:

Scotese, C.R. (2016). PALEOMAP PaleoAtlas for GPlates and the PaleoData Plotter Program. PALEOMAP Project. DOI: 10.13140/RG2.2.34367.00166

Input CSV files are expected to contain:

lon, lat, elev

where positive elevation values are treated as land points.

The PALEOMAP-derived CSV files used in this study are not included in this repository. Users should obtain the underlying paleogeographic reconstructions from the original source and generate the corresponding CSV files in accordance with the data provider's licensing terms.
