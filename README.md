# Skin Lesion Classification Project

## Project Structure

```
Dimitris Kaltsios Project/
├── README.md
├── .gitignore
├── Stage1/
│   ├── main.py                  # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── create_csv.py            # CSV generation script
│   ├── requirements.txt         # Python dependencies
│   ├── notes.txt                # Project notes
│   ├── skin_lesions.csv         # Dataset metadata
│   ├── Belignmetadata.csv       # Benign metadata
│   ├── MalignMetadata.csv       # Malignant metadata
│   ├── all_imgs/                # Image dataset (NOT in Git - download from Google Cloud)
│   ├── MalignImages/            # Malignant images (NOT in Git - download from Google Cloud)
│   ├── NEVSEKimages/            # Benign images (NOT in Git - download from Google Cloud)
│   ├── best_model.pth          # Trained model (NOT in Git)
│   └── training_history.png     # Training history plot (NOT in Git)
└── Stage2/
    (empty)
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Dimitris Kaltsios Project"
```

2. Set up Python environment:
```bash
cd Stage1
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or: .venv\Scripts\activate  # Windows CMD
# or: source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install torch torchvision pandas scikit-learn matplotlib tqdm
```

4. **Download image datasets from Google Cloud Storage**:
   - Download the image directories (`all_imgs/`, `MalignImages/`, `NEVSEKimages/`) from Google Cloud Storage
   - Place them in the `Stage1/` directory

## Usage

### Create Dataset CSV
```bash
python create_csv.py
```

### Train Model
```bash
python main.py --csv skin_lesions.csv --img_dir images --epochs 8 --batch_size 16
```

### Evaluate Model
```bash
python evaluate.py --model best_model.pth --csv skin_lesions.csv --img_dir images
```

## Dataset Information

- **Total Images:** ~4,596 images
- **Malignant:** 1,819 images
- **Benign:** 479 images (NEVSEK)
- **All Images:** 2,298 images (all_imgs)

## Notes

- The image datasets are NOT stored in Git due to size (7.8 GB). Download them from Google Cloud Storage.
- Training outputs are excluded from Git
- Only source code and small metadata files are version controlled
