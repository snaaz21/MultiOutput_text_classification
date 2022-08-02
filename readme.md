**Model Architecture:**

1. By using LSTM model, I created 3 input layers such as action's classes, object's classes and location's classes.
2. Created output layer for 3 label predictions by passing above 3 input layers.


**Training:**

    Train the model by passing input such as "train.csv" into **train.py** and no of desired epochs.

    Text Pre-processing:
        Do text pre-processing before training such as lowercasing the text.
        Convert sentence to tokens then convert them to sequence of integers before training.

    while training model:
        Model's config file will be created named as "model_config.yaml"
        Model's weigths will be saved inside "checkpoints/weights.h5"
        Model's logs will be saved inside "logs" directory

**Inferencing:**

    Run the "inference.py" by passing input text such "switch on the lights", 
    it will generate output as given below:
    
    (inference output on saved weights trained on 2 epochs)
    action:  activate
    object:  lamp
    location: bedroom

**Model's accuracy achieved on 2 epochs:**

    Training Acc: 0.45
    Val Acc: 0.42



