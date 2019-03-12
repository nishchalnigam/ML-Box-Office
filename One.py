import pandas as pd
import webbrowser
import os

box_office_train = pd.read_csv("Data/train.csv")

#print(len(box_office_train)) 3000 Movies

html = box_office_train[0:100].to_html()

#print(html.encode("utf-8"))
# Save the html to a temporary file
with open("data.html", "w", encoding="utf-8") as f:
    f.write(html)

# with open("data.html", "w") as f:
#     f.write(html.encode("utf-8"))

# Open the web page in our web browser
full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))


# Remove the fields from the data set that we don't want to include in our model
del box_office_train['id']
del box_office_train['genres']
del box_office_train['belongs_to_collection']
del box_office_train['homepage']
del box_office_train['imdb_id']
del box_office_train['original_title']
del box_office_train['overview']
del box_office_train['poster_path']
del box_office_train['status']
del box_office_train['tagline']
del box_office_train['title']
del box_office_train['crew']

# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(box_office_train, columns=['garage_type', 'city'])

# Remove the sale price from the feature data
del features_df['sale_price']

# Create the X and y arrays
X = features_df.as_matrix()
y = box_office_train['revenue'].as_matrix()
