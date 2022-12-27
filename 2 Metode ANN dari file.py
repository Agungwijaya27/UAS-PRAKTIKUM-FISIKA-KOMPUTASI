from sklearn.neural_network import MLPClassifier
import pandas as pd
#Data base : Gerbang Logika AND
#Membaca data dari file
FileDB = 'Database.txt'
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
print("1 10 = ", clf.predict([[0.36787941633460847]]))
print("2 20 = ", clf.predict([[0.045789196146081367]]))
print("3 30 = ", clf.predict([[0.0006170591435823976]]))
print("4 40 = ", clf.predict([[9.56605389460488e-07]]))
print("2 10 = ")clf.predict([[0.04578911676242501]])
print("3 10 = ")clf.predict([[0.0006170497008659222]])
print("4 10 = ")clf.predict([[9.565505519162414e-07]])
print("5 10 = ")clf.predict([[1.8054361027771023e-10]])
print("6 20 = ")clf.predict([[4.2912113620677254e-15]])
print("7 20 = ")clf.predict([[1.3107557938098917e-20]])
print("8 10 = ")clf.predict([[5.212389671966898e-27]])
print("9 20 = ")([[2.7223538935493817e-34]])
print("10 20 = ")clf.predict([[1.8786994509649057e-42]])
