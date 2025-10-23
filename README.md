# Simple Bioacoustic Classifier (ESC-50)
This is a small project demonstrating a pipeline for classifying environmental sounds from the [ESC-50 dataset](https://github.com/karoldvl/ESC-50).

It includes:
- Data exploration and visualization (`01_explore_data.ipynb`)
- Baseline classical models using aggregated features (`02_baseline_model.ipynb`)
- Frame-level feature extraction and modeling (`03_frame_level_model.ipynb`)
- Modular Python source code (`src/` folder) for reusable processing and modeling
- Pre-processed datasets in `data/processed_data/`

## Project Structure
bioacoustic-classifier/
├── data/
│   ├── processed_data/
│   │   ├── baseline_model.csv # computed via 02_baseline_model.ipynb
│   │   └── frame_level_model.csv # computed via 03_frame_level_model.ipynb
│   └── ESC-50-master/          # downloaded via 01_explore_data.ipynb
├── notebooks/
│   ├── 01_explore_data.ipynb 
│   ├── 02_baseline_model.ipynb 
│   ├── 03_frame_level_model.ipynb
├── src/
│   ├── data_utils.py
│   ├── eda.py
│   ├── baseline_model.py
│   ├── frame_level_model.py
│   └── model_performance.py
├── requirements.txt
└── README.md

## Usage
```bash
git clone https://github.com/yourusername/bioacoustic-classifier.git
cd bioacoustic-classifier
conda create -n bioaudio python=3.10
conda activate bioaudio
pip install -r requirements.txt
```

## Findings Summary
`02_baseline_model.ipynb` gave a reasonable baseline clearly better than random guessing with both random forest and logistic regression compared. In `03_frame_level_model.ipynb` we observe a marked increase with both random forest and logistic regression showing the features engineered using aggregated statistics based on overlapping 25ms frames contain far more predictive power than those based on global statistics. In `03_frame_level_model.ipynb` there is also an illustration of hyper-parameter selection via grid-search and cross validation for final insights into the quality of the final model.

In both `02_baseline_model.ipynb` and `03_frame_level_model.ipynb` we do model on a single feature vector per clip, and there is no extensive consideration of the temporal component of the data (although at the frame level, standard deviation givs an indication of it). Next steps would be to...