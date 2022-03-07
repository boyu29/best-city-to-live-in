#Best City To Live In Based Upon Preference
ECE 143 final project by group2

## Collaborators
Zhuomin Zhang<br>
Sangeetha Viswanathan-Sakthivel <br>
Jainish Chauhan<br>
Orish Jindal<br>
George Liu<br>
Boyu Chen<br>


# Overview
The real world application of this solution is that both companies as well as individuals can use this to choose a location to build their company or live. For example, individuals can find a city based upon weather, salary, etc depending on their personal preferences. Using the general population’s preferences as well as the most popular cities, companies can strategically locate their locations to align with customers which can include financial status, weather, education, etc.

# Data Set
The data set is collected from [Kaggle](https://www.kaggle.com/orhankaramancode/city-quality-of-life-dataset) containing 265 cities and 21 columns.

# File Structure
```
root
├── data
│   └──uaScoreDataFrame.csv  
├── src
│   ├── main.py
│   ├── assign_weight.py
│   ├── data_cleaning.py
│   ├── set_category.py.py
│   ├── visualization.py
│   ├── requirements.txt
│   └── ...
├── plots
│   └── ...
├── final_notebook.ipynb
├── slides.pdf
└── README.md

```

# Run the code
Please download/import all third-party modules before executing the code，then do the following：

```
python src/main.py
``` 
It will run the whole project including read in .csv data, preprocess and clean data, analyze and assign weight, plot all figures and save to ```plots``` file.

# Third-party 
pandas<br>
numpy<br>
matplotlib.pyplot<br>
sklearn<br>
seaborn<br>
scipy<br>
xgboost<br>
folium<br>
IPython.display<br>