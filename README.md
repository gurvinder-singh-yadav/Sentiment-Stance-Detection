# Sentiment Stance Detection
The following project contains code and demo of CS455 course where we have prepared a sentiment and stance prediction models respectively to effectively analyse the standing and emotion of the person on Climate Change Data

## Evironment Setup
```
conda create -n nlp -f environment.yaml
```

## Run
Local Mode
```
streamlit run app.py
```
Deployment Mode
```
streamlit run app.py --server.port=8001
```


## Model Architecture

![File Not Found](assets/sentiment_model_arch.png "Model Arch.")

## Results

### Sentiment
![Loss file Not found](assets/sentiment_accuracy.png "Sentiment Training Accuracy")

![Loss file Not found](assets/sentiment_loss.png "Sentiment Training Loss")

### Stance
![Loss file Not found](assets/stance_accuracy.png "Stance Training Accuracy")

![Loss file Not found](assets/stance_loss.png "Stance Training Loss")

