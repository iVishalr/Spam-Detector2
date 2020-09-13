# Spam Dectector2
---

A simple implementation of a Spam Detector in Python Programming Language using Scikit Learn and Natural Language Toolkit.

## Requirements
---

1. Python 3.7+
2. Sklearn
3. nltk
4. pandas
5. numpy

### Installation Guide
---

Clone this repository onto your local machine using the following command 

```bash
$ git clone 
```

Go to the project directory and type 

```bash
$ pip install -r requirements.txt
```
This command will install all the required dependencies to run the project.

### Project Execution
---

To run the project, type the following command in the terminal

```bash
$ python3 test.py "message"
```

Replace `"message"` with a random sentence of your choice and make sure its enclosed within double quotes.

This will run the classifer model and classifies `"message"` as either a valid message or spam.

### Training the model
---

To train your own classifier, type the follwing command in terminal

```bash
$ python3 train.py
```

You have to make necessary changes in the model parameters before training it.

### Note
---

*The dataset being used to train the model isn't diverse enough, which may result in classifing most messages as a valid message. Thus this project is only for educational purposes and not meant for deployment.*
