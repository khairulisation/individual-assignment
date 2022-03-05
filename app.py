#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import request, render_template
import joblib

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        loan = request.form.get("loan")
        income = request.form.get("income")
        age = request.form.get("age")
        loan = float(loan)
        income = float(income)
        age = float(age)
        print(loan,income,age)
        #Decision Tree Model
        model1 = joblib.load("CCD_DT")
        pred1 = model1.predict([[loan,income,age]])
        s1 = "The score of credit card default based on Decision Tree Model is: " + str(pred1)
        #Linear Regression Tree Model 
        model2 = joblib.load("CCD_Reg")
        pred2 = model2.predict([[loan,income,age]])
        s2 = "The score of credit card default based on Linear Regression is: " + str(pred2)
        #Neural Network Model
        model3 = joblib.load("CCD_NN")
        pred3 = model3.predict([[loan,income,age]])
        s3 = "The score of credit card default based on Neural Network is: " + str(pred3)
        #Random Forest Model
        model4 = joblib.load("CCD_RF")
        pred4 = model4.predict([[loan,income,age]])
        s4 = "The score of credit card default based on Random Forest Model is:  " + str(pred4)
        #Gradient Boosting Model
        model5 = joblib.load("CCD_GB")
        pred5 = model5.predict([[loan,income,age]])
        s5 = "The score of credit card default based on Gradient Boosting is: " + str(pred5)
        return(render_template("index.html",result1=s1,result2=s2,result3=s3,result4=s4,result5=s5))
    else:
        return(render_template("index.html",result1="2",result2="2",result3="2",result4="2",result5="2"))


# In[ ]:


if __name__ == "__main__":
    app.run()

