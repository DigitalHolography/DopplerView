# Changelog

## 1.8.0
* Add unit tests for DAG engine execution, cache behavior and output worker
* Ensure DAG determinism

## 1.7.4
* GUI
    * Remove top menu bar
* Restore steps caching following pipeline execution

## 1.7.3
* Add HD and DV app versions in the h5
* Update UI

## 1.7.2
* Update keys for moments reading

## 1.7.1
* Update high frequency and low frequency band reading with the new naming convention *band_<band_nb>_<low_freq>_<high_freq>*, while keeping retro-compatibility

## 1.7.0
* GUI
    * Enable light/dark theme switching, add release version in app title and status below progress bar
    * Enable / disable steps outputs in output folder (disabled by default)
    * Stop at retinal artery/vein segmentation and optic disc detection by default
* Handle errors cleanly : step failures don't interrupt run_batch, and a crash does not affect GUI
* First baseline for choroid artery / vein segmentation, using correlation maps with retinal arterial signal in different frequency bands

## 1.6.0
* GUI
    * Move configuration in a specific window, and allow dynamic modification of config files
    * Add visualization of to-be-processed inputs while highlighting the currently processed input, with a dedicated progress bar 
* CLI
    * Take config into account to validate/invalidate cache in debug mode
* Add version.txt at DV_folder root
* Add parameters for pre-mask method and branch signal correction
* Improve retinal masks cleaning

## 1.5.0
* Pre-classify branch signals by clustering of their first harmonic feature in the complex domain
* Fix choroid mask diaphragm removal, by using image center instead of optic disc center
* Accelerate pipeline by handling output saving in a separate thread
* Save debug cache in user ~/.cache/dopplerview folder, to avoid heavy reading on NAS, and automatically invalidate steps when the release version does not match.

## 1.4.1
* Ensure stacktrace when a step fails
* If optic disc segmentation confidence is too low, fallback to detection, and infer mask from bounding box. If detection confidence is also too low, fallback to M1 argmax to estimate optic disc center
* Add output shape to model specs

## 1.4.0
* Fix cache saving in debug mode
* Enable multiple inputs loading as batch folder (automatic search of .holo files) and .txt file with list of .holo
* Clean retinal artery/vein masks

## 1.3.0
* Handle multiple inputs
* Resize advanced window when images are displayed
* Add eye laterality classification
* Enable parallel step execution
* Add optic disc segmentation
* Clean arterial signal (used for artery/vein segmentation): remove heartbeats badly correlated with median beat

## 1.2.0
* Take into account *NumberOfWorkers* params. By default, set at 0.5 : take half of the available workers
* Enable loading of doppler_view config
* Choose to use local configs or default config
* Enable modification of model_registry, h5_schema and output_config in advanced_ui
* Log last run in %AppData%

## 1.1.0

* Interactive Tkinter GUI
* Progress bar and real-time validation of steps

## 1.0.0

* DAG-like pipeline with Preprocessing, Optic disc detection, Binary vessel segmentation, Pulse analysis, retinal Artery/vein segmentation, Velocity estimation and Arterial Waveform analysis
* Dynamic configuration of models fetched from huggingface, of .h5 output format and debug outputs
* Takes a .holo/holodoppler folder as input
* Outputs a DV_folder with .h5, outputs, config and cache for debugging
* CLI, Streamlit GUI and Tkinter App
* Automatic installer creation