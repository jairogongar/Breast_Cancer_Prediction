# Breast Cancer Prediction

## Project Overview

This project aims to support oncologists by providing an interpretable machine learning model for breast cancer diagnosis. The model classifies a tumor as either benign or malignant and aims to achieve an F1 score greater than 0.95. The emphasis is on both high accuracy and model interpretability to facilitate clinical decision-making.

## Table of Contents

1. [Installation](#installation)
2. [Data](#data)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)

## Installation

To clone the repository, run the following command in your terminal:

\`\`\`
git clone https://github.com/your_username/Breast_Cancer_Prediction.git
\`\`\`

The code is written in Python and requires Jupyter Notebook for execution.

### Required Libraries:

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

Install the libraries using `pip`:

\`\`\`
pip install -r requirements.txt
\`\`\`

## Data

The dataset contains various features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. Features include aspects like texture, perimeter, area, and smoothness. There are no missing attribute values, and the class distribution is 357 benign, 212 malignant.

## Methodology

The project follows the CRISP-DM methodology and includes data preprocessing, feature selection, and model training steps. Various machine learning models like Logistic Regression, Random Forest, and k-NN were experimented with, focusing on model interpretability.

## Results

- **Accuracy Score**: 0.97
- **Precision Score**: 0.95
- **Recall Score**: 0.98
- **F1 Score**: 0.964

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

- [CRISP-DM Methodology](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
