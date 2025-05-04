# Breast Cancer Risk Assessment Tool

A comprehensive machine learning-based tool for assessing breast cancer risk using cell nucleus characteristics from fine needle aspirate (FNA) biopsies.

## Project Overview

This project combines advanced data science techniques with an intuitive user interface to provide:
- Individual risk assessment for single biopsy samples
- Batch analysis for multiple samples
- Comprehensive data insights and visualizations
- Model performance evaluation and interpretation

## Features

### 1. Individual Risk Assessment
- Enter measurements for a single biopsy sample
- Get immediate risk assessment with confidence levels
- View feature importance for the prediction
- Interactive visualization of results

### 2. Batch Analysis
- Upload multiple samples in CSV format
- Download analysis results
- View summary statistics
- Export results for further analysis

### 3. Data Insights
- Explore dataset characteristics
- View feature distributions
- Analyze feature correlations
- Understand model feature importance

## Technical Stack

- **Python 3.9+**
- **Machine Learning**: scikit-learn, Random Forest, Gradient Boosting, SVM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Interface**: Streamlit
- **Model Persistence**: pickle

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd breast-cancer-risk-assessment
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Required Files

The application requires the following files in the project directory:
- `final_rf_model.pkl`: Trained Random Forest model
- `features.pkl`: Feature information
- `ranges.pkl`: Feature value ranges
- `data (1).csv`: Dataset
- `breast_cancer_awareness.jpg`: Home page image

## Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run breast_cancer_app.py
   ```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

## Project Structure

```
breast-cancer-risk-assessment/
├── breast_cancer_app.py      # Streamlit application
├── breast_cancer_analysis.ipynb  # Jupyter notebook with EDA and model development
├── data (1).csv             # Dataset
├── final_rf_model.pkl       # Trained model
├── features.pkl             # Feature information
├── ranges.pkl               # Feature value ranges
├── breast_cancer_awareness.jpg  # Home page image
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Data Analysis

The Jupyter notebook (`breast_cancer_analysis.ipynb`) contains:
- Comprehensive EDA
- Feature engineering
- Model development and comparison
- Performance evaluation
- Model interpretation

## Model Performance

The Random Forest model achieves:
- High accuracy in risk prediction
- Robust feature importance analysis
- Good generalization across different samples

## Usage Guide

### Individual Risk Assessment
1. Navigate to "Risk Assessment" in the sidebar
2. Enter measurements for each feature
3. Click "Assess Risk" to get results
4. View detailed analysis and feature importance

### Batch Analysis
1. Go to "Batch Analysis" in the sidebar
2. Download the template CSV file
3. Fill in your data following the template
4. Upload the completed CSV file
5. View and download results

### Data Insights
1. Select "Data Insights" from the sidebar
2. Explore dataset characteristics
3. View feature distributions and correlations
4. Analyze model feature importance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Support

For support, please open an issue in the repository or contact the development team. 