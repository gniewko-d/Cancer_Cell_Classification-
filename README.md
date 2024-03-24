Project Title: Normoxia_hypoxia_Cancer Cell Classification using Machine Learning and Artificial Neural Networks (ANN)

Description:
This project aims to develop a robust classification system for different types of cell cancer into  Normoxia and Hypoxia conditions using machine learning techniques as well as artificial neural networks (ANN). The goal is to create a model capable of accurately classifying and distinguishing between two different physiological conditions namely normoxia and hypoxia 
Dataset:
The dataset used in this project consists of features extracted from cell images of different cancer types such as GL261, LN229, PANC-2, PANC-1, U-87. The dataset was divided into training, validation, and test sets to facilitate model development, evaluation, and testing.

Approach:
1. Data Preprocessing:
   - Data cleaning and normalization.
   - Feature extraction: Extracting relevant features from cell images and other available data.
   - Data augmentation (if necessary) to increase the diversity of the training set and improve model generalization.

2. Model Development:
   - Designing an artificial neural network architecture suitable for the classification task.
   - Training the SVC, LOGISTIC REGRESSION, RANDOM FOREST, XGBOOST, and ANN models using the preprocessed data.
   - Fine-tuning hyperparameters using grid search to optimize model performance.
   - Evaluating the model's performance on the validation set to ensure it generalizes well to unseen data.

3. Model Evaluation:
   - Assessing the performance of the trained model on the test set.
   - Calculating metrics such as accuracy, precision, recall, and F1-score to measure classification performance.
   - Visualizing model predictions and analyzing any misclassifications to identify areas for improvement.

4. Results and Discussion:
   - Presenting the results of the classification experiments.
   - Discussing the strengths and limitations of the developed model.
   - Suggestions for future improvements and extensions to the project.

Dependencies:
- Python 3.x
- TensorFlow or PyTorch (for implementing ANN)
- Scikit-learn (for data preprocessing and evaluation)
- Matplotlib and Seaborn (for data visualization)

Usage:
1. Clone the repository to your local machine.
2. Install the required dependencies listed in the 'requirements.txt' file.
3. Run the main script or Jupyter notebook to train and evaluate the classification model.
4. Customize the model architecture, hyperparameters, and preprocessing steps as needed for your specific dataset and requirements.

Contributing:
Contributions to this project are welcome! Feel free to submit bug reports, feature requests, or pull requests via GitHub.

License:
This project is licensed under the [insert license type, e.g., MIT License]. See the LICENSE file for more details.

Contact:
For any inquiries or questions regarding this project, please contact [gniewosz.drwiega@doctoral.uj.edu.pl].
