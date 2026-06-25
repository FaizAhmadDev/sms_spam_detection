# SMS Spam Detection with DistilBERT

This repository contains the code for an SMS spam detection project utilizing the DistilBERT transformer model. The goal is to accurately classify incoming SMS messages as either 'ham' (legitimate) or 'spam' (unwanted/malicious).

## Project Overview

This project demonstrates a complete machine learning pipeline for text classification:

1.  **Data Loading & Preprocessing**: Loading the SMS Spam Collection Dataset, cleaning text data, handling duplicates, and encoding labels.
2.  **Model Selection**: Using the pre-trained `distilbert-base-uncased` model from Hugging Face.
3.  **Fine-tuning**: Adapting the DistilBERT model to the SMS spam detection task using the Hugging Face `Trainer` API.
4.  **Evaluation**: Assessing model performance using standard metrics like accuracy, precision, recall, F1-score, and a confusion matrix.
5.  **Inference**: Demonstrating how to use the fine-tuned model to predict new messages.
6.  **Deployment**: Saving the trained model and publishing it to the Hugging Face Hub.

## Repository Structure

-   `sms_spam_detection.ipynb`: The main Jupyter/Colab notebook containing all the code for data preprocessing, training, evaluation, and publishing.
-   `spam.csv`: The dataset used for training and evaluation.
-   `./spam_classifier_model/`: Directory where the trained model and tokenizer files are saved locally.

## Getting Started

### Prerequisites

To run this project, you'll need Python 3.x and the following libraries:

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `torch`
*   `transformers`
*   `huggingface_hub`

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch transformers huggingface_hub
```

### Running the Notebook

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/FaizAhmadDev/sms-spam-detection-distilbert.git # Replace with your repo URL
    cd sms-spam-detection-distilbert
    ```
2.  **Download the Dataset:** Ensure `spam.csv` is in the root directory of the project.
3.  **Open the Notebook:** You can open `sms_spam_detection.ipynb` in Google Colab (recommended for GPU access) or a local Jupyter environment.
4.  **Execute Cells:** Run all cells in the notebook sequentially. The notebook guides you through each step from data loading to model evaluation and saving.

### Accessing the Trained Model

The trained model has been uploaded to the Hugging Face Hub. You can access and use it directly without needing to retrain it. Visit the model page here:

➡️ **[FaizAhmadDev/sms-spam-detection-distilbert on Hugging Face](https://huggingface.co/FaizAhmadDev/sms-spam-detection-distilbert)** ⬅️

To load the model for inference in your Python code:

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

model_name = "FaizAhmadDev/sms-spam-detection-distilbert" # Your repo_id
loaded_tokenizer = DistilBertTokenizer.from_pretrained(model_name)
loaded_model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Example prediction (as shown in the Hugging Face Model Card)
def predict_spam(message):
    inputs = loaded_tokenizer(
        message,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    # Move inputs to appropriate device (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {key: val.to(device) for key, val in inputs.items()}

    loaded_model.to(device)
    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()

    label = 'SPAM' if prediction == 1 else 'HAM'
    return label, confidence

message = "Congratulations! You've won a $1000 gift card. Click here to claim now!"
label, conf = predict_spam(message)
print(f"Message: '{message}'\nPrediction: {label}, Confidence: {conf:.2f}")
```

## Results

The fine-tuned DistilBERT model achieved excellent performance on the test set:

| Metric               | Value   |
| :------------------- | :------ |
| **Accuracy**         | 99.23%  |
| **Precision (Spam)** | 0.9767  |
| **Recall (Spam)**    | 0.9618  |
| **F1-Score (Spam)**  | 0.9692  |

This indicates strong capability in identifying spam messages while minimizing false positives and negatives.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
