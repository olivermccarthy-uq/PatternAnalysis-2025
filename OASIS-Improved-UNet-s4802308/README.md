# Improved UNet for OASIS Brain MRI Segmentation

**Student Name:** Oliver McCarthy
**Student Number:** 48023083

## Introduction

This assignment presents a solution to the problem of biomedical image segmentation, specifically segmenting the 2D OASIS brain dataset using an Improved UNet architecture in PyTorch. The task demonstrates practical skills in deep learning for medical imaging, software engineering in version control, and reproducible research reporting. The goal is to achieve accurate segmentation on preprocessed brain MRI slices while documenting the workflow and outcomes.

---

## Problem Description

The task is to segment anatomical regions from 2D brain MRI slices. Images are sourced from the OASIS dataset, with corresponding ground-truth segmentation masks provided. Accurate segmentation of medical images is critical for quantitative analysis in neuroimaging research and clinical diagnosis.

---

## Project Structure

recognition/
└── OASIS-ImprovedUNet/
    ├── dataset.py        # Loads and preprocesses OASIS MRI & label data
    ├── modules.py        # Improved UNet model architecture (PyTorch)
    ├── train.py          # Model training and validation loop
    ├── predict.py        # Example inference script + result visualisations
    └── README.md         # Documentation and usage instructions

---

## Algorithm Summary

The Improved UNet model is a convolutional neural network with an encoder–decoder structure optimised for semantic segmentation tasks. It processes grey-scale brain MRI slices and produces corresponding binary segmentation masks. The model is trained using binary cross-entropy loss with logits and tracks training and validation loss across epochs.

---

## Workflow and Usage Instructions

### Code Structure

- **`modules.py`** – Implements the Improved UNet architecture components.  
- **`dataset.py`** – Loads and preprocesses OASIS image and mask data from PNG files.  
- **`train.py`** – Handles model training, validation, checkpoint saving, and loss plotting.  
- **`predict.py`** – Loads trained checkpoints and generates segmentation predictions with visualisations.

### Pipeline

1. **Data Preparation**  
   - OASIS 2D slice images and segmentation masks (stored as PNGs) are loaded and normalised.  
   - Training, validation, and testing splits follow the assignment convention.

2. **Training**  
   - The model trains for a fixed number of epochs.  
   - Training and validation losses are tracked to monitor overfitting.

3. **Prediction & Visualisation**  
   - The best checkpoint is used to generate segmentation masks on test images.  
   - Predicted outputs are visualised alongside the ground truth.

### Reproducible Setup

1. Install and activate a Python 3.11+ virtual environment inside the project directory.

python3 -m venv myenv
source myenv/bin/activate

2. Install dependencies
pip install torch matplotlib imageio numpy

3. Run Model Training
python3 train.py

4. Run prediction and generate output images
python3 predict.py

5. Plots and prediction images will be saved in the directory; download as needed.

---

## Data & Splitting Rationale
- Dataset: The OASIS 2D slice data is provided and preprocessed for direct usage.
- Splits: Training, validation and test sets follow the supplied partitioning to ensure consistency and fair evaluation.

---

## Dependencies
- Python 3.11+
- PyTorch
- matplotlib
- numpy
- imageio

---

## Limitations & Improvements

- The current implementation applies the Improved UNet to the supplied OASIS dataset only, which meets the easy difficulty criteria for this assignment.
- The dataset is normalised and reshaped for single-channel input; no additional data augmentation is used.
- The project could be extended to more challenging tasks such as segmentation on the HipMRI dataset, use of 3D models, or implementation of advanced architectures.
- Additional evaluation metrics (e.g., Dice coefficient) could be implemented for quantitative assessment.

---

## How to Run

- Activate your Python environment.
- Install required packages.
- Run scripts as described above.
- Visualisations and plots are produced automatically; check your working directory for output files.

---

## Results

### Training and Validation Loss

![Loss Plot](loss_plot.png)

The training and validation loss values, tracked over 15 epochs, are visualised in the loss plot above. The training loss remains consistently low from the start, indicating that the model is rapidly able to fit the training data. In contrast, the validation loss begins substantially higher, reflecting the challenge the model initially faces when generalising to new data.

Notably, the validation loss decreases sharply after the first few epochs, with the largest improvement occurring between epochs 4 and 5. By epoch 9, the validation loss stabilises at a considerably lower value—much closer to the training loss—demonstrating that the model's performance on unseen data has improved and is no longer dramatically different from its performance during training.

The convergence and parallel decline of both losses toward the end suggest successful learning, effective model generalisation, and little evidence of overfitting. This pattern, with high initial validation loss followed by rapid convergence, is typical for deep learning models as they transition from learning broad patterns to refining their segmentation boundary details over successive epochs.

Monitoring both curves ensures the model is not simply memorising the training set, but rather learning generalisable features essential for robust MRI segmentation.

#### Dice Similarity Coefficient

The Dice coefficient for segmentation predictions on the test set was **0.6323**, which does not meet the assignment threshold (≥ 0.9). This score reflects that, while the model segments the main brain regions effectively, it struggles with accurately capturing fine anatomical details and boundaries, as seen in both the visual results and the quantitative metric.


### Sample Prediction 1
![Prediction 0](prediction_0.png)

In Prediction 1, the input MRI slice (left) is clearly segmented with distinct anatomical structures in the ground truth mask (centre). The predicted mask (right) successfully outlines major brain regions and excludes most of the non-brain background. The segmentation captures general tissue contours and shows good agreement with the shape and location of structures in the ground truth.

However, there are noticeable artifacts, such as ring-like patterns near the outer edge of the mask and some discontinuities inside the brain area. These artifacts likely result from thresholding imperfections, limited post-processing, or model bias toward certain regions seen during training. Fine internal features appear less precisely defined in the prediction—some small structures are missed or blurred compared to the ground truth.

Overall, the model provides a reasonable binary segmentation for core brain regions but could be improved to reduce edge artifacts and sharpen internal anatomical boundaries. Incorporating additional metrics or data augmentation may help in addressing these issues in future iterations.

### Sample Prediction 2
![Prediction 1](prediction_1.png)

This figure again illustrates the input MRI, the ground truth, and the model’s predicted segmentation. The prediction displays strong overall agreement in identifying the brain region, with the major lobes and boundaries reasonably well matched to the ground truth.

However, closer inspection reveals several limitations: the model misses some of the finer internal details, leading to thicker and less contoured segmentation of internal regions compared to the true mask. There are still faint ring artifacts, and some central features appear overly connected or blurred. Despite these artifacts, the core region of the brain is successfully segmented, and the model avoids large false positives outside the brain.

Interpreting these results, the model clearly generalises the broad brain shape but could benefit from additional regularisation, improved post-processing, or refined loss functions to increase the precision of boundary and internal structure segmentation.

### Sample Prediction 3
![Prediction 2](prediction_2.png)

This prediction shows strong alignment between the predicted mask and the main brain tissue in the ground truth, with the overall shape and location of the brain accurately delineated. The segmentation correctly excludes most of the background and captures the gross anatomical structure.

Nonetheless, similar to prior predictions, the model does not capture every fine internal contour, with some subtle boundaries and intricate regions blurred or missed entirely in the prediction. Artefactual ring patterns remain present at the periphery, and some thin structures present in the ground truth are not fully segmented, possibly due to the network’s preference for large, contiguous foreground regions and limitations in spatial resolution.

In summary, the result is a clean, faithful segmentation of the brain region, appropriate for a baseline model, but still limited in detail—highlighting opportunities for more advanced modelling or post-processing if higher precision is needed.

### Prediction Summary

Each example displays the input brain MRI slice, its ground truth segmentation mask, and the model's predicted mask. The predicted segmentation masks are qualitatively similar to the ground truth, capturing major anatomical structures and providing good separation between tissue and background.

- Strengths:
    - The Improved UNet achieves crisp, binary mask boundaries and extracts the main brain regions with high similarity to the provided ground truth. Most large-scale features and shapes are very well captured, reflecting effective training and model architecture.

- Limitations:
    - Some fine details and small structures are missed, and there are occasional artifacts (e.g., ring-like shapes) at the periphery of the predicted masks. These may result from limitations in model depth, preprocessing, or thresholding method. Additionally, some regions of the brain boundaries look slightly thicker or thinner in predictions than ground truth, likely due to imperfect probability thresholding.

- Overall:
    - The model provides a robust segmentation baseline, with room for improvement through enhanced data augmentation, post-processing, or advanced architectures

#### Limitations and Troubleshooting

Despite retraining the model for 35 epochs (instead of 15), monitoring the validation loss, and experimenting with hyperparameters, the Dice similarity coefficient remained below the required value. Potential causes include:
- Insufficient data augmentation or model regulariaation
- Suboptimal thresholding for mask binariaation
- The model’s ability to generalize to fine structures and small regions within the brain masks

Further improvements could include more extensive data preprocessing, different model architectures, and advanced training techniques. Given time constraints, these enhancements were not implemented, but would be the logical next steps for improving segmentation quality.







