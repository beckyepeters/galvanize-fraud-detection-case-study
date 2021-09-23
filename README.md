# Galvanize DSI Fraud Detection Case Study
[Becky Peters](https://github.com/beckyepeters)
[Matthew Best](https://github.com/bestmatthew)
[Tony Nguyen](https://github.com/tonyn513)
August 2021

[Presentation Slides](https://docs.google.com/presentation/d/1dpa3oKvp2XuNfvDFgcGies4ufp-jwB7H2DN3XJqLj2A/edit?usp=sharing)
[Presentation Video](https://youtu.be/w7iBL8Y5tfc)

## Background and Motivation 
As a final learning project in the Galvanize part-time Data Science Immersive course, we tackled the problem of fraudulent event detection in a proprietary dataset provided by our instructors. This was a supervised machine learning project, where our task was to clean the data, select and / or engineer important features, predict fraudulent events using those features, and use other skills in the DSI to present our findings to a fictional business user. The dataset includes fraudulent and legitimate events from a virtual event host. 

Roughly 91% of the events in the dataset were legitimate (Fraud = False), while 9.1% of the events were fraudulent (Fraud = True). This introduces a problem of imbalanced classes for our classificaiton algorithm. 

Figure 1: Imbalanced Representation of Fraud vs. Non-Fraud Classes
![classes](images/classes.png)

The dataset began with 14,337 entries and 44 features of varied types, including both numeric and textual information.   

Figure 2: Event Types Included in Dataset
![event_types](images/event_types.png)

Initial Exploratory Data Analysis indicated relatively even distribution for features such as body length (length of text in description field), channels, location, etc. top5

Figure 3: Event Types in Top 5 Countries Represented in the Data
![top5](images/top5.png)

Figure 4: Event Name Length Distributions among Fraudulent and Legitimate Events
![name_length](images/name_length)

## Cleaning and Modeling the Data
Categorical features (such as Country, Currency type, Email domain, etc.) were re-labeled with sklearn's Label Encoder and NaN values were dropped, leaving us with 14,132 events to classify. 

We engineered some additional features to enhance our model's performance. Text entries from both the name and description of the event were evaluated for presence of punctuation and capitalization (both as a percentage of characters).   

Using a stratified train-test-split, we then began experimenting with different algorithms for fraud classification. The Random Forest classifier performed the best with an Accuracy of 97.6%, Precision at 92.7% and Recall at 76.0%. (Baseline metrics for this classification would be 91% accuracy, 0% precision and 0% recall if we were to classify all events as legitimate.)

In order to balance false negatives and false positive predictions for our investigators, we fine-tuned our classification model threshold to bring both precision and recall to 84% while maintaining a 97% accuracy. 

We also created a Flask app (in local development) which will assign events a high-, medium-, or low-risk label for our investigators. 

Figure 5: Flask app Screen Shot of prediction
![flask_screenshot](images/flask_screenshot.png)

We then did some Natural Language Processing on the Fraudulent event corpus and the Legitimate event corpus to see if further determinations could be made using cosine similarity for a new event under investigation. Word clouds from the descriptions of the two types of events showed marked differences: 

Figure 6: Word Clouds for Event Descriptions
![wordclouds](images/wordclouds.png)

We then used cosine similarity to evaluate a new event description's similarity to the fraud or non-fraud corpus, as an additional data point for investigators to consider. 

Figure 7: Cosine Similarity example
![cossim](images/cossim.png)

## Conclusions and Future Work 
The high / medium / low risk threshold seems a good starting point for investigators of fraud. For future additions to this project, we could add the cosine similarity feature to the dashboard, make the threshold customizable to modify the prediction classification for the user, and make the model updatable with new event information. 

