# Land Cap Analysis
Computes the smallest spherical caps containing 90% and 100% of land points 
from paleogeographic datasets. Produces AEQD map visualizations and summary 
statistics for each time period.

## Data
This code operates on paleogeographic land-point CSV files derived from:

Scotese, C.R. & Wright, N. (2018). PALEOMAP Paleodigital Elevation Models 
(PaleoDEMS) for the Phanerozoic. Available at: 
https://www.earthbyte.org/paleodem-resource-scotese-and-wright-2018/

Input CSV files are expected to contain:

lon, lat, elev

where positive elevation values are treated as land points.

The PALEOMAP-derived CSV files used in this study are not included in this 
repository. Users should obtain the underlying paleogeographic reconstructions 
from the original source and generate the corresponding CSV files in accordance 
with the data provider's licensing terms.

The code output images have been put together as a chronological video, shown here as Landcaps.gif

If you use this code or the animation, please cite:
M. Sade, A. Tsarfati and O. Birnholtz (2026), "Gamma Ray Bursts Effects on Extinction and Survivability in the Galaxy", arXiv:2603.18250, https://doi.org/10.48550/arXiv.2603.18250
