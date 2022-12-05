#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
data = pd.read_csv('E:/codeclause internship/project 1/creditcard.csv')


# In[33]:


data.head()


# Pre-processing

# In[34]:


from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)


# In[35]:


data.head()


# In[36]:


data = data.drop(['Time'],axis=1)


# In[37]:


data.head()


# In[38]:


X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']


# In[39]:


y.head()


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)


# In[41]:


X_train.shape


# In[42]:


X_test.shape


# Decision trees

# In[43]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()


# In[44]:


decision_tree.fit(X_train,y_train.values.ravel())


# In[45]:


y_pred = decision_tree.predict(X_test)


# In[46]:


decision_tree.score(X_test,y_test)


# In[47]:


import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[48]:


y_pred = decision_tree.predict(X)


# In[49]:


y_expected = pd.DataFrame(y)


# In[50]:


cnf_matrix = confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# Random Forest
# 

# In[51]:


from sklearn.ensemble import RandomForestClassifier


# In[52]:


random_forest = RandomForestClassifier(n_estimators=100)


# In[53]:


random_forest.fit(X_train,y_train.values.ravel())


# In[54]:


y_pred = random_forest.predict(X_test)


# In[55]:


random_forest.score(X_test,y_test)


# In[56]:


import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[57]:


cnf_matrix = confusion_matrix(y_test,y_pred)


# In[58]:


plot_confusion_matrix(cnf_matrix,classes=[0,1])


# In[59]:


plt.show()


# In[60]:


y_pred = random_forest.predict(X)


# In[61]:


cnf_matrix = confusion_matrix(y,y_pred.round())


# In[62]:


plot_confusion_matrix(cnf_matrix,classes=[0,1])


# In[63]:


plt.show()


# Neural Network approach
# 

# In[64]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,200,30))


# In[68]:


mlp.fit(X_train,y_train)


# In[69]:


predictions = mlp.predict(X_test)
print("Size of training set: ", X_test.shape)
print(predictions.shape)


# In[70]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[71]:


print(classification_report(y_test,predictions))


# In[ ]:




