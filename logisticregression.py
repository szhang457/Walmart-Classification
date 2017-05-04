from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import sklearn
import pandas as pd
import numpy as np
mydata = pd.read_csv("Data_Trans_3.csv")
mytrain, mytest = cross_validation.train_test_split(mydata, test_size = .4)
mytrain.TripClass = mytrain.TripClass.astype(float)
mytrain.WeekdayNumber = mytrain.WeekdayNumber.astype(float)
mytrain.TotalPurchase = mytrain.TotalPurchase.astype(float)
mytrain.TotalReturns = mytrain.TotalReturns.astype(float)
mytrain.UpcCount = mytrain.UpcCount.astype(float)
mytrain.Top_UPC = mytrain.Top_UPC.astype(float)
mytrain.DepartCount = mytrain.DepartCount.astype(float)
mytrain.upcdepart = mytrain.upcdepart.astype(float)
mytest.TripClass = mytest.TripClass.astype(float)
mytest.WeekdayNumber = mytest.WeekdayNumber.astype(float)
mytest.TotalPurchase = mytest.TotalPurchase.astype(float)
mytest.TotalReturns = mytest.TotalReturns.astype(float)
mytest.UpcCount = mytest.UpcCount.astype(float)
mytest.Top_UPC = mytest.Top_UPC.astype(float)
mytest.DepartCount = mytest.DepartCount.astype(float)
mytest.upcdepart = mytest.upcdepart.astype(float)
features = ["WeekdayNumber", "TotalPurchase", "TotalReturns", 'UpcCount','Top_UPC','DepartCount','ACCESSORIES','AUTOMOTIVE', 'BAKERY', 'BATH.AND.SHOWER', 'BEAUTY', 'BEDDING','BOOKS.AND.MAGAZINES', 'BOYS.WEAR', 'BRAS...SHAPEWEAR','CAMERAS.AND.SUPPLIES', 'CANDY..TOBACCO..COOKIES', 'CELEBRATION','COMM.BREAD', 'CONCEPT.STORES', 'COOK.AND.DINE', 'DAIRY', 'DSD.GROCERY','ELECTRONICS', 'FABRICS.AND.CRAFTS', 'FINANCIAL.SERVICES','FROZEN.FOODS', 'FURNITURE', 'GIRLS.WEAR..4.6X..AND.7.14','GROCERY.DRY.GOODS', 'HARDWARE', 'HOME.DECOR','HOME.MANAGEMENT', 'HORTICULTURE.AND.ACCESS','HOUSEHOLD.CHEMICALS.SUPP', 'HOUSEHOLD.PAPER.GOODS','IMPULSE.MERCHANDISE', 'INFANT.APPAREL', 'INFANT.CONSUMABLE.HARDLINES','JEWELRY.AND.SUNGLASSES', 'LADIES.SOCKS', 'LADIESWEAR','LARGE.HOUSEHOLD.GOODS', 'LAWN.AND.GARDEN', 'LIQUOR.WINE.BEER','MEAT...FRESH...FROZEN', 'MEDIA.AND.GAMING', 'MENS.WEAR', 'MENSWEAR','OFFICE.SUPPLIES', 'OPTICAL...FRAMES', 'OPTICAL...LENSES','OTHER.DEPARTMENTS', 'PAINT.AND.ACCESSORIES', 'PERSONAL.CARE','PETS.AND.SUPPLIES', 'PHARMACY.OTC', 'PHARMACY.RX','PLAYERS.AND.ELECTRONICS', 'PLUS.AND.MATERNITY', 'PRE.PACKED.DELI','PRODUCE', 'SEAFOOD', 'SEASONAL', 'SERVICE.DELI', 'SHEER.HOSIERY','SHOES', 'SLEEPWEAR.FOUNDATIONS', 'SPORTING.GOODS','SWIMWEAR.OUTERWEAR', 'TOYS', 'WIRELESS','upcdepart']

lr = LogisticRegression()
lr.fit(X = np.asarray(mytrain[features]), y = np.asarray(mytrain.TripClass))

predictions = lr.predict_proba(np.asarray(mytest[features]))
predictions_notproba = lr.predict(np.asarray(mytest[features]))

print("Log loss Percentage: {}".format(log_loss(mytest.TripClass, predictions).round(5)))

print("Accuracy Score: {}".format(sklearn.metrics.accuracy_score(mytest.TripClass, predictions_notproba)))
