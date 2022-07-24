

# data link - https://drive.google.com/file/d/1mSkKEe0SUJ7AZHiubxKSke7HWf75JA_Z/view




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("train_set.csv")
df




# checking is there any null value or not.
df.isnull().sum() 




df.dropna(how='any', axis=0, inplace=True)



df.head()




df.info()


# In[7]:


sns.heatmap(df.corr(), annot=True)




x = df.loc[:,['Salaries','Overtime','H/D']]
y = df.iloc[0:,-1]
x




df1 = df[df.Salaries < 350000]
x = df1.loc[:,['Salaries','Overtime','H/D']]
y = df1.iloc[0:,-1]




from sklearn.model_selection import train_test_split, GridSearchCV
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# # Linear Regressors
# 




from sklearn.linear_model import LinearRegression, Lasso, Ridge

#Linear regression
lin_reg = LinearRegression(normalize=False)
lin_reg.fit(x_train, y_train)
y_predict_lin_reg = lin_reg.predict(x_test)

#Lasso
lasso = Lasso(alpha=0.1, normalize=False)
lasso.fit(x_train, y_train)
y_predict_lasso = lasso.predict(x_test)

#Ridge
ridge = Ridge(alpha=0.1, normalize=False)
ridge.fit(x_train, y_train)
y_predict_ridge = ridge.predict(x_test)

print(lin_reg.score(x_test, y_test),'\n', lasso.score(x_test, y_test),'\n', ridge.score(x_test, y_test))


# In[12]:


from sklearn.metrics import mean_squared_error

rms_lin_reg = mean_squared_error(y_test, y_predict_lin_reg, squared=False)
rms_lasso = mean_squared_error(y_test, y_predict_lasso, squared=False)
rms_ridge = mean_squared_error(y_test, y_predict_ridge, squared=False)

print(rms_lin_reg,'\n', rms_lasso,'\n', rms_ridge)



from sklearn.ensemble import RandomForestRegressor

randForest = RandomForestRegressor()
randForest.fit(x_train, y_train)

y_predict_randForest = randForest.predict(x_test)
print(randForest.score(x_test, y_test),'\n',mean_squared_error(y_test, y_predict_randForest, squared=False))


from sklearn.neural_network import MLPRegressor

NN = MLPRegressor(alpha=1e-5, random_state=42)
NN.fit(x_train, y_train)

y_predict_NN = NN.predict(x_test)
print(NN.score(x_test, y_test),'\n',mean_squared_error(y_test, y_predict_NN))




testData = pd.read_csv("train_set.csv")
testData




test_x = testData.loc[:,['Salaries','Overtime','H/D']]
test_x



f, ax = plt.subplots(3, 1, figsize=(16,10))
f.tight_layout()
ax1,ax2,ax3 = ax.flatten()
sns.boxplot(x = test_x['Salaries'], ax=ax1)
#ax1.set_title('Slaries')
sns.boxplot(x = test_x['Overtime'], ax=ax2)
#ax2.set_title('Overtime')
sns.boxplot(x = test_x['H/D'], ax=ax3)
#ax3.set_title('H/D')


# In[ ]:


pred_y_xgb = lin_reg.predict(test_x)

res = pd.DataFrame(pred_y_xgb) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = testData.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["Compensation"]
res.to_csv("prediction_lin_reg.csv", index = False)



pd.read_csv("prediction_set.csv")




