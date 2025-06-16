# Analyzing the impact of Topic Framing, Emotional Tone, and Demographic Interactions in Political Microtargeting

This repository contains the data and Python scripts used for the paper 'Analyzing the impact of Topic Framing, Emotional Tone, and Demographic Interactions in Political Microtargeting'.

## How to Reproduce the Analysis

There are two ways to reproduce the results.

### Option 1: Analyze the Pre-processed Data (Recommended)

This is the fastest and simplest method. It uses the final, processed dataset to run the regression model directly, bypassing the time-consuming data preparation steps.

1.  Download the repository.
2.  Ensure the file `Final_dataset_cleaned.csv` is in the root directory.
3.  Run the OLS model script from your terminal:

    ```bash
    python OLS_model.py
    ```

This will print the OLS regression summary to the console and display the forest plots from the analysis.

### Option 2: Run the Full Data Preparation Pipeline

This option regenerates the final dataset from the raw `GPTarget.csv` data.

**Important Considerations:**
* **Llama 2 Model Access**: This pipeline uses the `meta-llama/Llama-2-7b-chat-hf` model. To run it, you must first request access to the model on its [Hugging Face page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and be logged into the Hugging Face CLI on your machine.
* **Hardware**: This process is computationally intensive and **requires a CUDA-enabled GPU**.

To run the pipeline:

1.  Make sure the initial dataset `GPTarget.csv` is located inside the `data/` folder.
2.  Run the data preparation script:

    ```bash
    python data_preparation_pipeline.py
    ```
    The script will create `Final_dataset.csv`. You can then run the `column_remover.py` script to generate the cleaned version used in the final analysis.
