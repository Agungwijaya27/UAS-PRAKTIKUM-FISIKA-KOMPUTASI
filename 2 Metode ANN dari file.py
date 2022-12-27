from sklearn.neural_network import MLPClassifier
import pandas as pd
#Data base : Gerbang Logika AND
#Membaca data dari file
FileDB = 'logikaand.csv'
Database = pd.read_csv(FileDB, sep=",", header=0)
print ("---------------------")
print (Database)
#x : Data, y : Target
x = Database[[u'Feature1', u'Feature2']] #ciri1, ciri2, dst
y = Database.Target

#Training and classify
clf = MLPClassifier(solver='lbfgs', alpha=1e-2,
                    hidden_layer_sizes=(10, 5),
                    random_state=1, max_iter=1000,
                    warm_start=True)

clf.fit(x,y)

#Prediksi
print("LOGIKA AND METODE Artificial Neural Network (ANN)")
print("Logika = Prediksi")
print("0 0 = ", clf.predict([[0, 0]]))
print("0 1 = ", clf.predict([[0, 1]]))
print("1 0 = ", clf.predict([[1, 0]]))
print("1 1 = ", clf.predict([[1, 1]]))
