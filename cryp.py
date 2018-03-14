import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv('All_Crypto2.csv')
df2 = pd.read_csv('predict.csv')

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['BitCoin_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting BitCoin Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)) , columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('BitCoin Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()


#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['Ethereum_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting Ethereum Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Ethereum Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************



#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['Dash_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting Dash Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Dash Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************


#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['EthereumClassic_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting EthereumClassic Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('EthereumClassic Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************


#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['LiteCoin_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting LiteCoin Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('LiteCoin Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************


#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['Monero_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting Monero Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Monero Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************


#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['Nem_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting Nem Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Nem Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************


#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['Neo_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting Neo Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Neo Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************


#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['Ripple_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting Ripple Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Ripple Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************


#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['Stratis_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting Stratis Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Stratis Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************


#***********************************

X = pd.DataFrame(df['ID'])
y = pd.DataFrame(df['Waves_Price'])
Z = pd.DataFrame(df2['ID2'])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print("------------------------------------------")
print("------------------------------------------")

print ("Predicting Waves Price for next 15 days")
print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
print("------------------------------------------")
print("------------------------------------------")

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Waves Price Predictor')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

#*****************************************