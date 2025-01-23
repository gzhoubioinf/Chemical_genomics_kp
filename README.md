	•	n_components = 500
	•	n_features = 2,500,000 

    thats why the joblib file is so big and has to be downloaded - github has file size upload limit 

\text{Size} \approx (500 \times 2,500,000 + 2,500,000 + 1,500) \times 8 = 1,252,501,500 \times 8 \approx 10,020,012,000 \text{ bytes} \approx 10 \text{ GB}


ChemGenomicsKp Streamlit Application

Overview

ChemGenomicsKp is a Streamlit-based application designed for chemogenomics analysis. It provides functionalities for colony picking and machine learning-based predictions from FASTA files. The application leverages various data processing, image handling, and machine learning techniques to deliver insightful analyses.

Features
	•	ColonyPicker: Extract and analyze colonies based on strain names or grid positions.
	•	ML Prediction: Predict traits like opacity, circularity, and size from uploaded FASTA files using pre-trained machine learning models.
	•	PCA and SHAP Analysis: Visualize principal component analysis and SHAP values for model interpretability.





ChemGenomicsKp/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── colony_picker.py
│   ├── ml_prediction.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loading.py
│       ├── image_handling.py
│       ├── ml_models.py
│       └── gbff_processing.py
├── config/
│   └── config.yaml
├── data/
│   ├── resistance_genes_seq.fsa
│   ├── virulence_genes_seq.fas
│   ├── reference.gbk.gb
│   └── ... (other data files)
├── models/
│   ├── pca_500_fitted_model_xg.joblib
│   ├── unitig_to_index_xg.pkl
│   └── ... (other model files)
├── newfigs/
│   └── ... (image files)
├── iris_measurements/
│   └── ... (IRIS measurement files)
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE







How to run

git clone https://github.com/notnat9/ChemGenomicsKp.git
cd ChemGenomicsKp


It’s recommended to use a virtual environment to manage dependencies.
python3 -m venv venv (on windows)
source venv/bin/activate (unix/mac)


Install dependencies

pip install -r requirements.txt




Download and add


Models/pca_500_fitted_model_xg.joblib
Models/unitig_to_index_xg.pkl





---



Configure the Application:

Ensure that the config/config.yaml file has the correct paths to your data and models. Modify it to match your project’s file locations.





Run the app: streamlit run app/main.py

Streamlit will start the server and provide a local URL (e.g., http://localhost:8501) in the terminal.
	•	Open your web browser and navigate to the provided URL to interact with the application.




Run the CLI

- colony picker - with row/column coordinates

python /ibex/user/hinkovn/backup_Project_File/cli/amr_genomics_cli.py     colony_picker     --config /ibex/user/hinkovn/backup_Project_File/config/config.yaml     --row 31     --col 48     --condition "Colistin-0.8ugml"

- - colony picker - with strain names - refer to strain_names.txt

python /ibex/user/hinkovn/backup_Project_File/cli/amr_genomics_cli.py colony_picker \
    --config /ibex/user/hinkovn/backup_Project_File/config/config.yaml \
    --strain H150 \
    --condition "Colistin-0.8ugml"

- ML prediction 

python /ibex/user/hinkovn/backup_Project_File/cli/amr_genomics_cli.py \
    ml_prediction \
    --config /ibex/user/hinkovn/backup_Project_File/config/config.yaml \
    --fasta /ibex/user/hinkovn/test_fasta_files/30.fasta \
    --condition "Colistin_0.8ugml"

conditions are in conditiona_names.txt

strain and condtions names must be pasted exactly as in the .txt files otherwisse an error will arise