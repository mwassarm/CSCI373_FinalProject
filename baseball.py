import sys 
import pandas as pd
import numpy as np
import pybaseball
from pybaseball import batting_stats
pybaseball.cache.enable()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler


#Create helper function to apply to each player in the batting database
def next_season(p):
    p = p.sort_values("Season")
    p["Next_WAR"] = p["WAR"].shift(-1)
    return p

#load the csv from pybaseball, only from the last few years, and only for players with more than 150 ABs
def load():
    batting_stats(2017, end_season=2023, qual=150, ind=1).to_csv('batting_stats_150.csv', index=False)
    dataset = pd.read_csv('batting_stats_150.csv')
    #makes it only players with multiple seasons
    dataset = dataset.groupby("IDfg", group_keys=False).filter(lambda x: x.shape[0] > 1)
    null_variables = dataset.isnull().sum() #find nulls (cant have for ML)
    full_variables = list(dataset.columns[null_variables == 0]) #gets list of all non nulls
    batting = dataset[full_variables].copy() #updates to only non nulls
    #drop unnecessary identifiers
    batting = batting.drop('Dol', axis=1)
    batting = batting.drop('Team', axis=1)
    batting = batting.drop('Age Rng', axis=1)
    #batting = batting.drop('Unnamed: 0', axis=1)
    batting = batting.drop('IDfg', axis=1)
    batting = batting.drop('L-WAR', axis=1)
    #create a row for next WAR
    batting = batting.groupby("Name", group_keys=False).apply(next_season)
    return batting

def split(batting):
    #adjust the dataset to split by season
    bat_copy=batting.copy()
    bat_copy2=batting.copy()
    batting.sort_values("Season")
    X_train_temp=bat_copy[bat_copy["Season"]<2023]
    X_train_temp=X_train_temp[X_train_temp["PA"]>300]
    #split dataset to get training X and y
    X_train_names = X_train_temp.dropna()
    X_train=X_train_names.drop(['Name','Next_WAR'],axis=1)
    y_train=X_train_names['Next_WAR']
    #do the same as above for the test sets.
    X_test_temp=bat_copy2[bat_copy2["Season"]>2022]
    X_test_temp=X_test_temp[X_test_temp["PA"]>300]
    players_2023=X_test_temp.drop(['Next_WAR'],axis=1)
    X_test=players_2023.drop(['Name'],axis=1)
    y_test=X_test_temp['Next_WAR']
    return X_train, X_test, y_train, y_test, players_2023

def train_pred(X_train, X_test, y_train, y_test, players, increase, age, war):
    #create and train the lasso model using our determinded best alpha
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    #make predictions for 2024 based on the 2023 stats
    y_pred = lasso.predict(X_test)
    #add the predictions to the 2023 dataset along with the improvement or regression in WAR
    players['Prediction'] = y_pred.tolist()
    diff=np.subtract(y_pred,players['WAR'])
    players['Increase'] = diff.tolist()
    #find our breakout players
    breakouts=players.copy()
    breakouts=breakouts[breakouts["Increase"]>increase]
    breakouts=breakouts[breakouts["Age"]<age]
    breakouts=breakouts[breakouts["WAR"]<war]
    breakouts.sort_values("Increase")
    return breakouts
    
def main():
    try:
        increase=float(sys.argv[1])
        age=float(sys.argv[2])
        war=float(sys.argv[3])
    except:
        increase=1.25
        age=27
        war=3
    dataset = load()
    X_train, X_test, y_train, y_test, players_23 = split(dataset)
    out = train_pred(X_train, X_test, y_train, y_test, players_23, increase, age, war)
    
    out1 = out.copy()
    out1.sort_values("Increase")
    out1 = out.loc[:, out.columns.intersection(["Season","Name","Age", "WAR", "Next_WAR", "Prediction", "Increase"])]

    print(out1)

if __name__ == "__main__":
    main()