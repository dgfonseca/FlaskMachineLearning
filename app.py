from flask import Flask
import pandas as pd
from sklearn.cluster import KMeans
import json
from firebase_admin import credentials, firestore, initialize_app

app = Flask(__name__)

cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
todo_ref = db.collection(u'surveys')



@app.route('/parejas')
def machine_learning_clustering_parejas():
    df = pd.read_csv(r'proyecto.csv', encoding="ISO-8859-1")
    clusters = algoritmo(df)
    save_all = []
    for item in clusters:
        dictionary = {}
        dictionary['cluster'] = str(item)
        parejas = []
        for i in range(len(clusters[item])):
            array = {}
            if i+1 > len(clusters):
                array['user1'] = str(clusters[item][i][0])
                array['user2'] = "NA"
                parejas.append(array)
                i += 1
            else:
                array['user1'] = str(clusters[item][i][0])
                array['user2'] = str(clusters[item][i+1][0])
                parejas.append(array)
                i += 1
        dictionary['parejas'] = parejas
        save_all.append(dictionary)
    return json.dumps(save_all)

@app.route('/grupos')
def machine_learning_clustering():
    df = pd.read_csv(r'proyecto.csv', encoding="ISO-8859-1")
    clusters = algoritmo(df)

    save_all = []
    for item in clusters:
        dictionary = {}
        dictionary['cluster'] = str(item)
        array = []
        for i in clusters[item]:
            array.append(i[0])
        dictionary['people'] = array
        save_all.append(dictionary)

    return json.dumps(save_all)


def algoritmo(df):
    llaves = ["compañeros", "hablar", "estresarme", "me importan los demas", "desordenado", "comodidad",
              "malas palabras", "empatico", "social", "conversaciones", "orden y limpieza", "divertirme",
              "estudiar con otros", "videojuegos", "leer", "deportes", "anime", "futbol", "maquillaje", "musica",
              "acostarme tarde", "Describase a si mismo", "objetivo", "puntuacion"]
    info = todo_ref.stream()
    for doc in info:
        add = {}
        info = doc.to_dict()
        respuestas = info["questions"]
        add["correo"] = info["userEmail"]
        add["sexo"] = "N/A"
        for i in range(len(llaves)):
            add[llaves[i]] = respuestas[i]
        df = df.append(add, ignore_index=True)

    # K means
    x = df.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]].values
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit_predict(x)
    arreglo = kmeans.predict(x)
    clusters = {}
    n = 0
    for item in arreglo:
        if item in clusters:
            clusters[item].append(df.to_numpy()[n])
            n += 1
        else:
            clusters[item] = [df.to_numpy()[n]]
            n += 1
    return clusters


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
