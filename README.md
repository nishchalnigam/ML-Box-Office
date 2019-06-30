# ML-Box-Office

BACKGROUND
As an avid movie lover, I would like to use my knowledge of machine learning in predicting the performance of a movie, even before it is released. Predicting performance can be a lot of things for movie, like average critic review, ratings, revenue of its prequels and awards, but quantifying them as is kind of a pain. In my case I shall use the revenue the movie generated as parameter to gauge my models performance under the hypothesis that good movies always have high revenue (not always true but most of the time it is true).
DATASET
The dataset is TMDB dataset, taken from Kaggle (https://www.kaggle.com/c/tmdb-box-office-prediction/data), which consists of 7398 rows. Out of these, the revenue (y) for 3000 is given, and the revenue for 4398 movies has to be predicted. The number of features in the database are 22.

DATA PRE-PROCESSING
One-hot Encoding:
A lot of the columns are given in JSON format, so I had to clean them. I also had to apply one-hot encoding to a lot of data.
Example:
The genre column contains data like:
[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}, {'id': 10751, 'name': 'Family'}, {'id': 10749, 'name': 'Romance'}]
So I made a column for the number of genres, and then used one-hot encoding for the 15 most common genres.
I did the same for the top most common production companies, countries, spoken languages.
Splitting the release date:
The date column was divided into year, month, days, quarter etc.
Cast and crew:
Since all movies have different cast and crew, it did not make sense to use one-hot encoding for cast and crew. Hence I just used the total number of people casted in the movie, and the total number of crew who worked on the movie.
DATA VISUALIZATION
I made a lot of graphs to compare impact of numerical data on movies.
GRADIENT BOOSTING
model = ensemble.GradientBoostingRegressor(
n_estimators=1000,
learning_rate=0.1,
max_depth=6,
min_samples_leaf=9,
max_features=0.1,
loss='huber'
)
• n estimators: number of decision trees to build • Learning rate: controls how much each additional decision tree influences the overall prediction • Max depth: controls how deep each individual decision tree can be • Min samples leaf: Controls how many times a value must appear in our training set for a decision tree to make a decision based • Max features: percentage of features in our model that we randomly choose to consider each time we create a branch in our decision tree • Loss: controls how SciKit-learn calculates the model's error rate or cost as it learns I tried using the following parameters and ran a grid search cross-validation to determine the best combination of params.
After running the GridSearchCV function with the above grid, I found out that this was the best combination 'learning_rate': 0.05 'loss': 'huber' 'max_depth': 4 'max_features': 0.1 'min_samples_leaf': 3 'n_estimators': 1000 Submission on Kaggle: Analysis of Result:
After running the model with best combination of parameters, I got a Training Set Mean Absolute Error of $17.16 Million, which maybe very good for a successful movie as they gross around a $500 million to $1.5 billion. However, this may not be so accurate for movies which make less than $50 million.
My Test Set Mean Absolute Error was $41.65 million, which was a little more than twice of the training error. This shows that the overfitting is significantly high for movies which gross less money. Conclusion:
The model I have made works well for movies which have high revenues, because most of those movies are produced by major production houses, have higher budget and popularity, and are mostly English movies made in the united states. However, it is much more difficult to predict the revenue for a Portuguese language movie produced in Brazil, because of lesser number of Brazilian movies in the dataset.
The same is true for low budget Hollywood movies and other independent production, as some of them like Paranormal Activity earn as much as a Superhero movie, although there is no particular criteria other than popularity to predict that, and popularity is available only after a movie is released.
Another important factor which is not available in this dataset is franchise. This could help make better predictions, because of the fact that many movies belong to a franchise, and movies of the same franchise tend to earn revenues in the same range. Further scope for improving the accuracy: There were a lot of data which I was unable to use, which could have been pre-processed in a better way. • Crew: The crew column had a lot of details, which I did not use. E.g.: for every crew member, the following details were present: 'department': 'Sound', 'gender': 2, 'id': 23486, 'job': 'Original Music Composer', 'name': 'Christophe Beck' The only parameter I used was the total number of crew members in a movie, which is not sufficient • Keywords: I had dropped the Keywords column completely, because I had used genre. However, using NLP, or using just the top few keywords, the ranking could’ve improved. E.g.: The keywords like “independent film” could have been used to determine figure out that independent movies usually make less money. • New feature creation: A lot of new features can be generated by combining existing features. A very simple example would be a budget to runtime ratio of a movie, which would indicate the average money invested per minute on a movie. Related Work:
Similar approaches have been used to solve problems related to predicting the cost of houses in a locality. Just like movies, housing data also contains a lot of categorical data, which can be used to make a decision tree. An example of this is the street name. A house which is located in an affluent street like the Park Avenue in New York City will always have high cost. This is very similar to a movie which is being produced by a major studio like Walt Disney. Also, newer houses have higher value than older houses, if they are in the same locality. Other numerical values that affect the price of a house will be basic features like the number of bedrooms, the area, original purchase price, etc.
Here’s a link to a similar problem:
https://towardsdatascience.com/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f
REFERENCES [1] Prediction of a Movie’s Success Using Data Mining Techniques - https://link.springer.com/chapter/10.1007/978-981-13-1742-2_22 [2] Gradient Boosted Regression Trees in Scikit-Learn - https://orbi.uliege.be/handle/2268/163521
