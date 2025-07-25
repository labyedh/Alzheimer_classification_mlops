# ALZHEIMER_CLASSIFICATION_MLOPS

> Transforming Alzheimer Detection with Seamless Automation

<!-- 
===================================================
BADGES
===================================================
-->
<p align="center">
  <img src="https://img.shields.io/github/last-commit/labyedh/Alzheimer_classification_mlops?style=for-the-badge" alt="last commit">
  <img src="https://img.shields.io/github/languages/top/labyedh/Alzheimer_classification_mlops?style=for-the-badge" alt="top language">
  <img src="https://img.shields.io/github/languages/count/labyedh/Alzheimer_classification_mlops?style=for-the-badge" alt="language count">
  <img src="https://img.shields.io/github/license/labyedh/Alzheimer_classification_mlops?style=for-the-badge" alt="license">
</p>

**Built with the tools and technologies:**

<!-- 
===================================================
TECHNOLOGY STACK
===================================================
-->
<p align="center">
  <a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://scikit-learn.org/" target="_blank"><img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"></a>
  <a href="https://mlflow.org/" target="_blank"><img src="https://img.shields.io/badge/Mlflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLFlow"></a>
  <a href="https://dvc.org/" target="_blank"><img src="https://img.shields.io/badge/DVC-8E44AD?style=for-the-badge&logo=dvc&logoColor=white" alt="DVC"></a>
  <a href="https://docs.github.com/en/actions" target="_blank"><img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions"></a>
  <a href="https://www.json.org/json-en.html" target="_blank"><img src="https://img.shields.io/badge/json-5E5C5C?style=for-the-badge&logo=json&logoColor=white" alt="JSON"></a>
  <a href="https://www.markdownguide.org/" target="_blank"><img src="https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white" alt="Markdown"></a>
  <a href="https://yaml.org/" target="_blank"><img src="https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white" alt="YAML"></a>
</p>

---

## Table of Contents

- [Overview](#overview)
  - [Key Features](#key-features)
  - [MLOps Workflow](#mlops-workflow)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

`Alzheimer_classification_mlops` is a comprehensive MLOps toolkit designed to streamline the development, training, and deployment of **Alzheimer's disease classification models**. It uses audio features like **MFCC** and **Log-Mel Spectrograms** extracted from speech data to power its models. The project integrates automated pipeline orchestration, environment setup, and experiment tracking to ensure reproducibility and efficiency from start to finish.

### Key Features

-   🎯 **Automated pipeline execution** that handles dependencies, data versioning, and artifact management.
-   🚀 **Seamless environment setup and deployment**, including secret management and repository cloning.
-   🔬 **Reproducible experiments** with DVC and MLflow, supporting robust cross-validation and performance tracking.
-   📊 **Consistent audio feature extraction** using MFCC and LogMel, ensuring reliable data for model training.
-   ⚙️ **Orchestrated training and evaluation workflows**, facilitating scalable model development and deployment.

### MLOps Workflow

This project is built on a modern MLOps stack to automate and manage the entire machine learning lifecycle:

-   🚀 **Automated CI Workflow with GitHub Actions & Kaggle:** The core of our automation is a Continuous Integration (CI) pipeline powered by GitHub Actions. This workflow is uniquely configured to:
    1.  **Utilize Kaggle Kernels as Runners:** Instead of standard GitHub runners, our jobs execute on Kaggle's free GPU environment, which is ideal for training ML models.
    2.  **Run the Full Pipeline:** The action automatically checks out the code, installs dependencies, and runs the complete DVC pipeline for training and evaluation.
    3.  **Commit Results Back:** Upon successful completion, the workflow automatically commits the new results—such as updated metric files, evaluation plots, and DVC pointers—back to the Git repository. This creates a fully automated loop where new code triggers a run, and its results are immediately versioned and available.

-   📦 **Data & Model Versioning with DVC:** We use Data Version Control (DVC) to manage large datasets, intermediate files, and trained models. This ensures that every experiment is fully reproducible by versioning not just the code (with Git), but also the exact data, parameters, and models used in each run.

-   🔬 **Experiment Tracking with MLflow:** Every training run is meticulously logged with MLflow. It captures hyperparameters, performance metrics (e.g., accuracy, loss), and model artifacts for each fold in our cross-validation. The MLflow UI provides a clear, centralized dashboard to compare different experiments and identify the best-performing models.

<p align="right">(<a href="#alzheimer_classification_mlops">back to top</a>)</p>

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project requires the following dependencies:

-   **Programming Language:** Python 3.8+
-   **Package Manager:** Pip

### Installation

Build `Alzheimer_classification_mlops` from the source and install dependencies:

1.  **Clone the repository:**
    ```sh
    git clone git@github.com:labyedh/Alzheimer_classification_mlops.git
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd Alzheimer_classification_mlops
    ```

3.  **Install the dependencies:**
    
    Using `pip`:
    ```sh
    pip install -r requirements.txt
    ```

<p align="right">(<a href="#alzheimer_classification_mlops">back to top</a>)</p>

### Usage

Run the main project pipeline using `main.py`. You must specify a stage (`train`, `evaluate`, or `full-pipeline`). You can also temporarily override the feature type defined in `params.yaml`.

**1. Run the full pipeline (Training and Evaluation):**
```sh
python main.py full-pipeline
```

**2. Run only the training stage:**
```sh
python main.py train
```

**3. Run only the evaluation stage:**
```sh
python main.py evaluate
```

**4. Override the feature type for a run:**
This command will run the full pipeline using `mfcc` features, temporarily overriding the value in `params.yaml`.
```sh
python main.py full-pipeline --feature mfcc
```
This command will run the full pipeline using `log_mel` features, temporarily overriding the value in `params.yaml`.
```sh
python main.py full-pipeline --feature log_mel
```


<p align="right">(<a href="#alzheimer_classification_mlops">back to top</a>)</p>

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

<p align="right">(<a href="#alzheimer_classification_mlops">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#alzheimer_classification_mlops">back to top</a>)</p>

## Acknowledgements

-   [Shields.io](https://shields.io)
-   [Devicon](https://devicon.dev/)

<p align="right">(<a href="#alzheimer_classification_mlops">back to top</a>)</p>