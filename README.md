
# BERT Multi-class Classification Project

This project aims to fine-tune a BERT model for multi-class text classification, specifically for sentiment analysis in the Turkish language. It includes data preprocessing, model training, evaluation, and result visualization. The project is designed for data scientists and NLP enthusiasts who are interested in applying transformer models to Turkish text data.



## Installation

1-Clone the repository:

```bash 
  git clone https://github.com/yourusername/your-repo-name.git
  cd your-repo-name
```
2-Create a virtual environment and activate it:

```bash 
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3-Install the required packages:

```bash 
  pip install -r requirements.txt
```
## Usage

To run the project, use the following command:
```bash 
python main.py --data_file "path/to/your/cleaned_data.csv" --model_name "dbmdz/bert-base-turkish-uncased" --output_dir "./FineTuneModel" --num_train_epochs 3
```
Replace "path/to/your/cleaned_data.csv" with the path to your preprocessed data file.

## Project Structure

```bash 
.
├── data_loader.py
├── data_preprocess.py
├── main.py
├── model.py
├── utils.py
├── requirements.txt
└── README.md

```
data_loader.py: Contains the DataLoader class for loading and batching the data.

data_preprocess.py: Contains functions for data cleaning and preprocessing.

main.py: The main script to run the model training and evaluation.

model.py: Contains the create_model function for initializing the BERT model.

utils.py: Utility functions for computing metrics, plotting, and generating reports.

requirements.txt: Lists the Python packages required to run the project.
## Data Preprocessing
Before running the training script, ensure your data is preprocessed correctly. The data_preprocess.py script provides functions to clean the text data by removing emojis, URLs, mentions, hashtags, numbers, and expanding abbreviations.
## Training and Evaluation

The training process includes fine-tuning the BERT model and evaluating it on a validation set. The main.py script handles these tasks, using the Trainer class from the Hugging Face Transformers library.

Key arguments for the script:

--data_file: Path to the preprocessed CSV file.

--model_name: Name of the pre-trained BERT model to use.

--output_dir: Directory to save the fine-tuned model.

--num_train_epochs: Number of epochs to train the model.
## Ekran Görüntüleri
After training, the script evaluates the model and generates a classification report, confusion matrix, and visualizations.

![Uygulama Ekran Görüntüsü](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

  
## Acknowledgements

Hugging Face for providing the Transformers library.

Contributors and the open-source community for their invaluable support.
