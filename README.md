# ChemGenomicsKp: A Bacterial Colony Analysis Application

## Overview

**ChemGenomicsKp** is a Streamlit-based application designed for the comprehensive analysis of bacterial colonies. It integrates two powerful functionalities into a single, user-friendly interface:

-   **Colony Picker**: for morphological analysis of colony images.
-   **Machine Learning Predictions**: for predicting colony characteristics from genomic data (FASTA files).

---

## Background and Purpose

*Klebsiella pneumoniae* is a bacterium of significant global medical importance, responsible for a range of infections, particularly in healthcare settings. Understanding the interplay between a colony's observable traits (phenotype) and its genetic makeup (genotype) is crucial for combating antibiotic resistance and tracking infections.

This application bridges the gap between laboratory observations and genomic data by:

-   **Streamlining High-Throughput Screening**: Automating the analysis of colony images.
-   **Linking Genotype to Phenotype**: Using machine learning to predict colony morphology from genomic sequences, providing insights into the genetic drivers of antimicrobial resistance.

---

## Features

### Colony Picker

The Colony Picker module extracts and analyzes images of bacterial colonies grown on agar plates. It measures key morphological features—such as size, circularity, and opacity—which are critical indicators of a strain’s behavior and response to different growth conditions.

-   **Analyze Colonies**: Extract colonies by strain name or grid coordinates.
-   **Calculate Metrics**: Automatically compute opacity, circularity, and size.
-   **Visualize Results**: View colony images and metric distributions directly within the app.

### ML Prediction (Genomic Data Analysis)

This module analyzes genomic sequences from FASTA files to predict colony characteristics using pre-trained XGBoost and TabNet models.

-   **Predict Colony Traits**: Upload a FASTA file to predict opacity, circularity, and size.
-   **Interpretability**: Visualize Principal Component Analysis (PCA) and SHAP (SHapley Additive exPlanations) values to understand model predictions.
-   **Gene Ontology (GO) Enrichment Analysis**: Identify the biological significance of genes influencing the predictions.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/amr_genomics_kp.git
cd amr_genomics_kp
```

### 2. Set Up a Virtual Environment

**Windows:**

```bash
python3 -m venv venv
venv\Scripts\activate
```

**Unix/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Files (Required for ML Prediction)

Download the following files and place them in the `Models/` directory:

-   [pca_transformer.joblib](https://mega.nz/file/YHdF1Z4J#ejON7zilFjXF2xR9po-8OWuvmMbomJ-BJBdHaxrplMM)
-   [unitig_to_index_xg.pkl](https://mega.nz/file/AGNi1ILZ#lkPcN6Gb0Yo3ndXlxvMQTpzIUfsu5E1SwOLGyL-k1Yc)

> **Note on Model File Size:** The PCA model files are large due to the high dimensionality of the genomic data (500 components × ~2.5M features), exceeding GitHub's file size limit.

---

## Usage

### Web Application

To run the Streamlit web application:

```bash
streamlit run app/main.py
```

The application will be accessible at `http://localhost:8501`.

### Command-Line Interface (CLI)

The CLI provides a way to run the analysis pipelines without the graphical interface.

#### Colony Picker

**By Row/Column Coordinates:**

```bash
python cli/amr_genomics_cli.py colony_picker --config config/config.yaml --row 31 --col 48 --condition "Colistin-0.8ugml"
```

**By Strain Name:**

```bash
python cli/amr_genomics_cli.py colony_picker --config config/config.yaml --strain H150 --condition "Colistin-0.8ugml"
```

#### ML Prediction

```bash
python cli/amr_genomics_cli.py ml_prediction --config config/config.yaml --fasta /path/to/your/fasta.fasta --condition "Colistin_0.8ugml"
```

> **Important Notes:**
>
> -   Valid conditions are listed in `cli/condition_names.txt`.
> -   Valid strain names are listed in `cli/strain_names.txt`.
> -   Condition names are case-sensitive and must be an exact match.

---

## Docker Installation (Recommended)

For a hassle-free setup, a pre-configured Docker image is available.

### Prerequisites

-   [Docker Desktop](https://www.docker.com/products/docker-desktop) installed on your system.

### Run with Docker

1.  **Pull the Docker Image:**

    ```bash
    docker pull hinkovn/app_final
    ```

2.  **Start the Container:**

    ```bash
    docker run -p 8501:8501 hinkovn/app_final
    ```

3.  **Access the Application:**

    Open your web browser and go to `http://localhost:8501`.

> **Note:** It is recommended to increase the Docker container's memory allocation to 16GB in Docker Desktop's settings (Settings -> Resources).

### Docker Advantages

-   **No Manual Downloads**: All model files are included.
-   **Pre-configured**: The environment is ready to use out of the box.
-   **Consistency**: Ensures the application runs the same way across different operating systems.
-   **Isolation**: The application and its dependencies are isolated from your system packages.

---

## Troubleshooting

### `pyarrow` Installation Issues on macOS

Some users, particularly on macOS, may encounter an `ImportError` related to the `pyarrow` library. This is often due to incompatibilities with system-level libraries.

If you encounter such issues, please try the following:

1.  **Use a Clean Virtual Environment**: Ensure you are in a new, clean virtual environment to prevent conflicts with other installed packages.
2.  **Use Conda**: Consider using the [Conda](https://docs.conda.io/en/latest/) package manager, as it can be more robust at managing complex binary dependencies.
3.  **Use Docker**: The recommended and most reliable method is to use the provided Docker container, which guarantees a compatible and consistent environment.
