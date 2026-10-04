# Privacy-Preserving Explainable Brain Tumor Detection

This version is specifically corrected for the Mendeley dataset structure:

Epic and CSCR hospital Dataset/
- Train/
  - glioma/
  - meningioma/
  - pituitary/
  - notumor/
- Test/
  - glioma/
  - meningioma/
  - pituitary/
  - notumor/

## Important
Do NOT merge Train and Test.

The official Test folder remains completely untouched. Only the official Train folder is split:
- 90% actual training
- 10% validation
- 100% official Test = final evaluation

## Setup

1. Edit `config.py` and set:
   `DATASET_ROOT = Path(r"C:\...\Epic and CSCR hospital Dataset")`

2. Install:
   `pip install -r requirements.txt`

3. Train DP model:
   `python train.py`

4. Train non-private baseline:
   `python train_baseline.py`

5. Evaluate:
   `python evaluate.py`

6. Grad-CAM:
   `python gradcam.py --image "path\\to\\mri.jpg"`

7. Streamlit:
   `streamlit run app.py`

The project is a research prototype, not a clinical diagnostic device. Do not fabricate metrics or privacy epsilon values; use the values produced by the actual run.
