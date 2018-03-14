import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from flask import Flask , redirect , url_for , request , render_template , jsonify , json

app = Flask(__name__)

@app.route("/")
def hello():

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
    test1 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)) , columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)) , columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('BitCoin Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    


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
    test2 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Ethereum Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    

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
    test3 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Dash Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    

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
    test4 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('EthereumClassic Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    

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
    test5 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('LiteCoin Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    

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
    test6 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Monero Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    

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
    test7 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Nem Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    

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
    test8 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Neo Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    

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
    test9 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Ripple Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    

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
    test10 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Stratis Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    test = pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)))
    #return 'Predicting Ethereum Price for next 15 days....%s' % test

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
    test11 = (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price'] ))
    print (pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(Z)), columns = ['Price']))
    print("------------------------------------------")
    print("------------------------------------------")

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Waves Price Predictor')
    plt.xlabel('Days')
    plt.ylabel('Price')
    

    #*****************************************
    #return 'Predicting BitCoin Price for next 15 days....%s' % test
    return  '<h1>Predicting BitCoin Price for next 15 days</h1><br><br><h3><h3>{}</h3></h3></><br/><br/><h1>Predicting Ethereum Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/><h1>Predicting Dash Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/><h1>Predicting Ethereum Classic Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/><h1>Predicting Litecoin Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/><h1>Predicting Monero Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/><h1>Predicting Nem Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/><h1>Predicting Neo Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/><h1>Predicting Ripple Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/><h1>Predicting Stratis Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/><h1>Predicting Waves Price for next 15 days</h1><br><br><h3>{}</h3><br/><br/>'.format(test1, test2,test3, test4,test5, test6, test7,test8, test9, test10,test11)
if __name__ == '__main__':
    app.run(debug = True)
    