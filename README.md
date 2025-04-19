# Beta-Bank-Supervised-Machine-Learning
Write prediction code for when a customer is likely to leave a bank based on customer history

🏦 Beta Bank – Supervised Machine Learning Project

This project uses supervised machine learning techniques to predict customer churn for Beta Bank. The goal is to identify clients likely to leave the bank, enabling proactive retention efforts. The project includes data preprocessing, model selection, evaluation, and final recommendations.

📚 Table of Contents
About the Project
Installation
Usage
Project Structure
Technologies Used
Results & Insights
Screenshots
Contributing
License

📌 About the Project
This notebook walks through the complete machine learning pipeline:

Exploring and cleaning the Beta Bank dataset
Handling class imbalance
Training multiple models (Logistic Regression, Random Forest, etc.)
Optimizing hyperparameters
Evaluating performance using metrics like accuracy, precision, recall, and F1-score
Delivering actionable business insights to reduce customer churn

🛠 Installation
Clone the repository or download the .ipynb file

Install dependencies with pip:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
Launch Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
🚀 Usage
Open the Beta Bank Supervised Machine Learning.ipynb notebook in Jupyter and run cells step-by-step. It will walk you through:
Exploratory Data Analysis (EDA)
Feature scaling and encoding
Train/test splits and model evaluation
Interpretation of results and next steps

📁 Project Structure
bash
Copy
Edit
Beta Bank Supervised Machine Learning.ipynb  # Main notebook
README.md                                   # Documentation
images/                                     # Screenshots of results/plots (optional)

⚙️ Technologies Used
Python 3.8+
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook

📊 Results & Insights
Editing the data to be examined had an intense shift in the data analysis. 

Initial model testing was performed without altering the source data.  The following information compares the results form the test test.  
Decision Tree Test F1 Score  was 0.555.  
F1 Score of Logistic Regression was 0.555. 
Highest F1 score was Random Forest Classifier Test Accuracy at 0.556.

Downsample Summary:
Downsampled Decision Tress F1 Score: 0.70. 
Downsampled Logistic Regression F1 Score: 0.70. 
Downsampled Random Forest F1 Score: 0.68. 
Decision Tree and Logistic Regression maintain same pattern of equal scores.  Downsampling scores achieved higher f1 scores so performed on test data.  

Upsampled Summary:
Upsampled Decision Tress F1 Score: 0.57. 
Upsampled Logistic Regression F1 Score: 0.57. 
Upsampled Random Forest F1 Score: 0.60. 
Decision Tree and Logistic Regression maintain same pattern of equal scores. 

Final model computed was the Downsampled Random Forest on Test Data since the training and validation data achieved the highest F1.  Final F1 Score is 0.62.  

Interestingly the Decision Tree and Logistic Regression F1 score was the same through unbalanced, downsampled, and upsampled testing.  Random Forest remained the best model. The best F1 score was achieved of 0.615 when the test data was run on the upsampled data set.  The closer the F1 score is to 1 is an indication of how accurately the prediction of exiting customers has been.  

Beta Bank indeed should develop more strategies to retain their current clients.  

📸 Screenshots
### 📈 Correlation Heatmap  
![Heatmap](images/image_1.png)

### 📉 Model Evaluation Metrics  
![Metrics](images/image_2.png)

### 🧠 Feature Importance Plot  
![Feature Importance](images/image_3.png)

### 📊 Class Distribution After SMOTE  
![SMOTE Results](images/image_4.png)

### 🧪 Confusion Matrix  
![Confusion Matrix](images/image_5.png)


🤝 Contributing
Contributions are welcome! If you have ideas to improve modeling, visualizations, or want to try deploying the model, feel free to fork and submit a pull request.

🪪 License
This project is licensed under the MIT License.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-JupyterLab%20%7C%20Notebook-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Exploratory-blueviolet.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

