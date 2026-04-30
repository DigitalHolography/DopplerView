# Changelog

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