# Math-Parser

A lightweight modular math OCR pipeline that processes scanned equations and reconstructs them into LaTeX.

## Modules
1. **Symbol Classifier** (CNN-based)
2. **Symbol Segmentation**

## Datasets
For symbol classification training, we use the [HASYv2 dataset](https://www.kaggle.com/datasets/guru001/hasyv2?select=symbols.csv).


## Setup
```bash
git clone https://github.com/yourusername/math-ocr.git
cd math-ocr
pip install -r requirements.txt
```

Folder Structure
- `data/`: Store your raw, processed, and synthetic images
- `models/`: Trained model checkpoints
- `src/`: All source code modules

To download HASYv2 dataset:
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("guru001/hasyv2")

print("Path to dataset files:", path)
```
Then move files as needed.
