# dopplerview/fix_imports.py
# Force eager loading of lazy-imported modules before frozen bootstrap runs.
# Must be imported before anything that touches skimage or scipy.stats.

import scipy.stats._distn_infrastructure 
import scipy.stats._stats_py 
import scipy.stats.distributions 
import scipy.stats 
import skimage.filters.ridges  
import skimage.feature.corner