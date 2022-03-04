# best-city-to-live-in

### Work done by: Orish Jindal and Sangeetha VS 
Date: Feb 13, 2022.

What we were given:
- Normalized data of all the columns 

Our progress:
- Found the co-relations of the given data to club those 17 columns.
- Did hierarchical clustering to make 4 group based on positive co-relation index (shown
in the heatmap below). The data driven categories are as follows:
  1) Housing, Cost of Living.
  2) Startups, Venture Capital, Travel Connectivity, Commute, Leisure & Culture,
Outdoors.
  3) Business freedom, Healthcare, Education, Environmental Quality, Economy, Internet
Access, Tolerance.
  4) Safety, taxation.

- We also prepared another list by grouping the columns that may be prioritized in the same categories by a particular type of user.
  1) Vacation Lovers:
Travel Connectivity, Commute, Leisure & Culture, Internet Access.
  2) Entrepreneurs & Businesspersons:
Startups, Venture Capital, Business Freedom, Taxation, Economy.
  3) Stability Seekers:
Housing, Cost of Living, Tolerance, Outdoors.
  4) Family:Healthcare, Education, Safety, Environmental Quality.


We then clubbed the data by multiplying entries of the columns in the group and then normalized it for better visualization and removing bias for the four groups each in both the cases mentioned above (Data driven and Subjective). The code of our progress is uploaded on GitHub.

For further actions: Run two parallel cases (Data driven and Manual categorization) and then assign weights for the person to make a better choice.

---
 
### Work done by: Zhuomin Zhang and Janish Chauhan
Date: Feb 21, 2022.

What we were given:
- df6 and df7 are 2 parallel cases
- df6 for category1-4
- df7 for manually selected category

Our progress:
- Plot bar figures for each category on continents 
- normalize each column, change the value to 0-10, adjust the mean of each column to  5, keep the variance unmodified
- value for each category=avg(columns in this category)

Date: Feb 22, 2022 (Jainish)
- Kmeanse clustering on original data to cluster given cities into four categories
- Trained xgboost classifier on these labeled data 
- visualization of classification model
- visualization of how much importance each feature has in categorizing given city into one of four categories
- weights of each feature in decding category of city
...

---

### Work done by: Boyu Chen and Sangeetha
Date: Mar 2, 2022.
Added pie chart showing distributions of cities; visualized top cities for each category.