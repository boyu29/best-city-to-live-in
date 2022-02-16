# best-city-to-live-in

- **city-quality-worldwide** contains .csv data, just run the jupyter notebook to do the data cleaning.


- we also provide a back up dataset **livingwage-in-us**, added more columns to the original data, if you decide not to use this one, just discard it.


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
  