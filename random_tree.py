from sklearn import svm
from sklearn.metrics import classification_report
import import_handler

x_train, x_test, y_train, y_test = import_handler.get_normalized()
x_train = x_train.reshape((-1, 400))
x_test = x_test.reshape((-1, 400))
y_train = y_train.argmax(axis=1)
y_test = y_test.argmax(axis=1)


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=20)
clf = clf.fit(x_train, y_train)


pred = clf.predict(x_test)


corr = pred==y_test
print(sum(corr)/len(corr))