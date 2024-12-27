from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
from werkzeug.utils import secure_filename

import os
import pandas as pd
import numpy as np
import cv2 as cv
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

# machineLearningModels
import sklearn.linear_model as lm
import sklearn.tree as tree # criterion = ['gini','entropy']
import sklearn.ensemble as ens
import sklearn.naive_bayes as nb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.metrics as met

# _____________________________________________________
app = Flask(__name__)
root = os.getcwd()
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = 'TA_wanDinulAqli'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# inisiasiDeteksiWajah
cc_wajah = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
if cc_wajah.empty():
    raise IOError('File XML tidak ditemukan')
# inisiasiPengenalanWajah
lbphfaces_recognizer = cv.face.LBPHFaceRecognizer_create(radius=2, neighbors=2)
lbphfaces_recognizer.read('training.xml') # readModel

# _____________________________________________________
def get_plot(wajah_predict, lbp, hist, m):
    # tampilkanHasil
    plt.figure(figsize=(10,5))
    # citraAsli
    plt.subplot(1,3,1)
    plt.imshow(wajah_predict, cmap='gray')
    plt.title('citraGrayscale_wajah'+str(m))
    plt.axis('off')
    # citraLBP
    plt.subplot(1,3,2)
    plt.imshow(lbp, cmap='gray')
    plt.title('citraLBP_wajah'+str(m))
    plt.axis('off')
    # histogram
    plt.subplot(1,3,3)
    plt.bar(range(len(hist)), hist, color='gray')
    plt.title('histogramLBP_wajah'+str(m))
    plt.xlabel('polaBiner')
    plt.ylabel('frekuensi')
    plt.tight_layout()
    # plt.show()
    return plt

# _____________________________________________________
def hPredict_CV(foto_gray):
    tags = ['laki-laki','perempuan']
    label = lbphfaces_recognizer.predict(foto_gray)
    label_text = tags[label[0]]
    print('label:',label[0], label_text, 'confidence:',label[1], 'typeLabel:',type(label))
    confidence = label[1]
    return label_text, confidence

def predictWajah_CV(dirGambar, filename):
    foto = cv.imread(dirGambar)
    foto_gray = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
    hasil = cc_wajah.detectMultiScale(foto_gray, scaleFactor=1.2, minNeighbors=5)
    print('terdeksi ',len(hasil),' wajah didalam gambar..')
    if len(hasil) == 0:
        return

    n = 0
    m = 1
    deepPink=147,20,255 # female
    mediumBlue=255,0,0 # male

    for (x,y,w,h) in hasil:
        wajah_predict = foto_gray[y:y+h, x:x+w]
        # hitungLBP
        lbp = local_binary_pattern(wajah_predict, 8, 1, method='uniform')
        # hitungHistogramLBP
        n_bins = int(lbp.max()+1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        # normalisasiHistogram
        hist = hist / hist.sum()
        print(hist, type(hist))
        pb0 = round(hist[0],3)
        pb1 = round(hist[1],3)
        pb2 = round(hist[2],3)
        pb3 = round(hist[3],3)
        pb4 = round(hist[4],3)
        pb5 = round(hist[5],3)
        pb6 = round(hist[6],3)
        pb7 = round(hist[7],3)
        pb8 = round(hist[8],3)
        pb9 = round(hist[9],3)
        # save
        plot = get_plot(wajah_predict, lbp, hist, m)
        pathPLT = os.path.join(root, 'static/hasilLBPH/pltLBPH_wajah'+str(m)+'_'+filename)
        plot.savefig(pathPLT)
        # saveFaces
        facePath = os.path.join(root, 'static/faces/lbpFace'+str(m)+'_'+filename)
        cv.imwrite(facePath, wajah_predict)

        # runPredictFaces
        label_text, confidence =  hPredict_CV(wajah_predict)
        if n == 0:
            df = pd.DataFrame({'faces':['wajah'+str(m)], 'pb0':[pb0], 'pb1':[pb1], 'pb2':[pb2], 'pb3':[pb3], 'pb4':[pb4], 'pb5':[pb5], 'pb6':[pb6], 'pb7':[pb7], 'pb8':[pb8], 'pb9':[pb9], 'hPredictCV':[label_text], 'confidence':[round(confidence,3)]})
        else:
            df.loc[n] = ['wajah'+str(m), pb0, pb1, pb2, pb3, pb4, pb5, pb6, pb7, pb8, pb9, label_text, round(confidence,3)]

        if label_text == 'perempuan':
            warna = deepPink
        else:
            warna = mediumBlue

        # runSetBingkaiFaces
        ukuranBingkai = w+h
        if (ukuranBingkai >= 100)and(ukuranBingkai < 150):
            size = 'Small'
            UB = 0.2
            SB = 7

        elif (ukuranBingkai >= 150)and(ukuranBingkai < 250):
            size = 'Medium'
            UB = 0.3
            SB = 9

        elif (ukuranBingkai >= 300):
            size = 'Maximum'
            UB = 0.5
            SB = 13

        elif (ukuranBingkai < 300)and(ukuranBingkai >= 250):
            size = 'Large'
            UB = 0.4
            SB = 11

        elif (ukuranBingkai < 100):
            size = 'Minimum'
            UB = 0.2 # 0.1
            SB = 7 # 5

        cv.rectangle(foto, (x,y), (x+w, y+h), color=(warna), thickness=2)
        cv.putText(foto, label_text, (x,y-4), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        cv.putText(foto,'Wajah '+str(m), (x,(y+h+SB)), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        m +=1
        n +=1

    # save_hPredictCV_filename
    path = os.path.join(root, 'static/hPredict/'+'hPredictCV_'+filename)
    cv.imwrite(path, foto)
    print('predictWajah_CV Selesai..')
    return df

# _____________________________________________________
def hPredict_LR(df, nWajah=1):
    dataTrainLBP = pd.read_excel('dataTrainLBP.xlsx')
    X = dataTrainLBP[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    y = dataTrainLBP['label']
    # # splitDataTrain&TestModel_1
    X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=1, random_state=0) #test_size=0.1 (10%_dariDataSet),
    # print('X_train.shape: ',X_train.shape)
    # print('y_train.shape: ',y_train.shape)
    # print('-------------------------------')
    Xtest = df
    X_test = df[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    # print('X_test.shape: ',X_test.shape)

    # _______CreatingModelsMachineLearning_____
    # LogisticRegression <======
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)
    # print(X_train.min(), X_train.max())
    # print(X_test.min(), X_test.max())
    model=lm.LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predictFace-n
    y_prediksi = model.predict(X_test)
    Xtest['hPredictLR'] = y_prediksi
    # replaceKolomjenisKelamin[hPredictLR]
    jenisKelamin = {'hPredictLR':{0:'laki-laki', 1:'perempuan'}}
    Xtest.replace(jenisKelamin, inplace=True)

    if y_prediksi[0] == 1:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': perempuan')
        return 'perempuan', Xtest
    else:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': laki-laki')
        return 'laki-laki', Xtest

def predictWajah_LR(dirGambar, filename):
    foto = cv.imread(dirGambar)
    foto_gray = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
    hasil = cc_wajah.detectMultiScale(foto_gray, scaleFactor=1.2, minNeighbors=5)
    print('terdeksi ',len(hasil),' wajah didalam gambar..')

    n = 0
    m = 1
    deepPink=147,20,255 # female
    mediumBlue=255,0,0 # male

    for (x,y,w,h) in hasil:
        wajah_predict = foto_gray[y:y+h, x:x+w]
        # hitungLBP
        lbp = local_binary_pattern(wajah_predict, 8, 1, method='uniform')
        # hitungHistogramLBP
        n_bins = int(lbp.max()+1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        # normalisasiHistogram
        hist = hist / hist.sum()
        print(hist, type(hist))
        pb0 = round(hist[0],3)
        pb1 = round(hist[1],3)
        pb2 = round(hist[2],3)
        pb3 = round(hist[3],3)
        pb4 = round(hist[4],3)
        pb5 = round(hist[5],3)
        pb6 = round(hist[6],3)
        pb7 = round(hist[7],3)
        pb8 = round(hist[8],3)
        pb9 = round(hist[9],3)

        # runPredictFace-n
        df = pd.DataFrame({'faces':['wajah'+str(m)], 'pb0':[pb0], 'pb1':[pb1], 'pb2':[pb2], 'pb3':[pb3], 'pb4':[pb4], 'pb5':[pb5], 'pb6':[pb6], 'pb7':[pb7], 'pb8':[pb8], 'pb9':[pb9]})
        label_text, dfN =  hPredict_LR(df, m)

        # predictFull
        if n == 0:
            dfFullHpredict = dfN
        else:
            dfV = dfN.values
            dfFullHpredict.loc[n] = [dfV[0][0], dfV[0][1], dfV[0][2], dfV[0][3], dfV[0][4], dfV[0][5], dfV[0][6], dfV[0][7], dfV[0][8], dfV[0][9], dfV[0][10], dfV[0][11]]

        if label_text == 'perempuan':
            warna = deepPink
        else:
            warna = mediumBlue

        ukuranBingkai = w+h
        if (ukuranBingkai >= 100)and(ukuranBingkai < 150):
            size = 'Small'
            UB = 0.2
            SB = 7

        elif (ukuranBingkai >= 150)and(ukuranBingkai < 250):
            size = 'Medium'
            UB = 0.3
            SB = 9

        elif (ukuranBingkai >= 300):
            size = 'Maximum'
            UB = 0.5
            SB = 13

        elif (ukuranBingkai < 300)and(ukuranBingkai >= 250):
            size = 'Large'
            UB = 0.4
            SB = 11

        elif (ukuranBingkai < 100):
            size = 'Minimum'
            UB = 0.2 # 0.1
            SB = 7 # 5

        cv.rectangle(foto, (x,y), (x+w, y+h), color=(warna), thickness=2)
        cv.putText(foto, label_text, (x,y-4), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        cv.putText(foto,'Wajah '+str(m), (x,(y+h+SB)), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        m += 1
        n += 1

    # save_hPredictLR_filename
    path = os.path.join(root, 'static/hPredict/'+'hPredictLR_'+filename)
    cv.imwrite(path, foto)
    print('predictWajah_LR Selesai..')
    return dfFullHpredict

# _____________________________________________________
def hPredict_DS(df, nWajah=1):
    dataTrainLBP = pd.read_excel('dataTrainLBP.xlsx')
    X = dataTrainLBP[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    y = dataTrainLBP['label']
    # # splitDataTrain&TestModel_1
    X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=1, random_state=0) #test_size=0.1 (10%_dariDataSet),
    # print('X_train.shape: ',X_train.shape)
    # print('y_train.shape: ',y_train.shape)
    # print('-------------------------------')
    Xtest = df
    X_test = df[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    # print('X_test.shape: ',X_test.shape)

    # _______CreatingModelsMachineLearning_____
    # DecisionTreeClassifier <======
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)
    # print(X_train.min(), X_train.max())
    # print(X_test.min(), X_test.max())
    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
    model.fit(X_train, y_train)

    # predictFace-n
    y_prediksi = model.predict(X_test)
    Xtest['hPredictDS'] = y_prediksi
    # replaceKolomjenisKelamin[hPredictLR]
    jenisKelamin = {'hPredictDS':{0:'laki-laki', 1:'perempuan'}}
    Xtest.replace(jenisKelamin, inplace=True)

    if y_prediksi[0] == 1:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': perempuan')
        return 'perempuan', Xtest
    else:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': laki-laki')
        return 'laki-laki', Xtest

def predictWajah_DS(dirGambar, filename):
    foto = cv.imread(dirGambar)
    foto_gray = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
    hasil = cc_wajah.detectMultiScale(foto_gray, scaleFactor=1.2, minNeighbors=5)
    print('terdeksi ',len(hasil),' wajah didalam gambar..')

    n = 0
    m = 1
    deepPink=147,20,255 # female
    mediumBlue=255,0,0 # male

    for (x,y,w,h) in hasil:
        wajah_predict = foto_gray[y:y+h, x:x+w]
        # hitungLBP
        lbp = local_binary_pattern(wajah_predict, 8, 1, method='uniform')
        # hitungHistogramLBP
        n_bins = int(lbp.max()+1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        # normalisasiHistogram
        hist = hist / hist.sum()
        print(hist, type(hist))
        pb0 = round(hist[0],3)
        pb1 = round(hist[1],3)
        pb2 = round(hist[2],3)
        pb3 = round(hist[3],3)
        pb4 = round(hist[4],3)
        pb5 = round(hist[5],3)
        pb6 = round(hist[6],3)
        pb7 = round(hist[7],3)
        pb8 = round(hist[8],3)
        pb9 = round(hist[9],3)

        # runPredictFace-n
        df = pd.DataFrame({'faces':['wajah'+str(m)], 'pb0':[pb0], 'pb1':[pb1], 'pb2':[pb2], 'pb3':[pb3], 'pb4':[pb4], 'pb5':[pb5], 'pb6':[pb6], 'pb7':[pb7], 'pb8':[pb8], 'pb9':[pb9]})
        label_text, dfN =  hPredict_DS(df, m)

        # predictFull
        if n == 0:
            dfFullHpredict = dfN
        else:
            dfV = dfN.values
            dfFullHpredict.loc[n] = [dfV[0][0], dfV[0][1], dfV[0][2], dfV[0][3], dfV[0][4], dfV[0][5], dfV[0][6], dfV[0][7], dfV[0][8], dfV[0][9], dfV[0][10], dfV[0][11]]

        if label_text == 'perempuan':
            warna = deepPink
        else:
            warna = mediumBlue

        ukuranBingkai = w+h
        if (ukuranBingkai >= 100)and(ukuranBingkai < 150):
            size = 'Small'
            UB = 0.2
            SB = 7

        elif (ukuranBingkai >= 150)and(ukuranBingkai < 250):
            size = 'Medium'
            UB = 0.3
            SB = 9

        elif (ukuranBingkai >= 300):
            size = 'Maximum'
            UB = 0.5
            SB = 13

        elif (ukuranBingkai < 300)and(ukuranBingkai >= 250):
            size = 'Large'
            UB = 0.4
            SB = 11

        elif (ukuranBingkai < 100):
            size = 'Minimum'
            UB = 0.2 # 0.1
            SB = 7 # 5

        cv.rectangle(foto, (x,y), (x+w, y+h), color=(warna), thickness=2)
        cv.putText(foto, label_text, (x,y-4), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        cv.putText(foto,'Wajah '+str(m), (x,(y+h+SB)), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        m += 1
        n += 1

    # save_hPredictLR_filename
    path = os.path.join(root, 'static/hPredict/'+'hPredictDS_'+filename)
    cv.imwrite(path, foto)
    print('predictWajah_DS Selesai..')
    return dfFullHpredict

# _____________________________________________________
def hPredict_RF(df, nWajah=1):
    dataTrainLBP = pd.read_excel('dataTrainLBP.xlsx')
    X = dataTrainLBP[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    y = dataTrainLBP['label']
    # # splitDataTrain&TestModel_1
    X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=1, random_state=0) #test_size=0.1 (10%_dariDataSet),
    # print('X_train.shape: ',X_train.shape)
    # print('y_train.shape: ',y_train.shape)
    # print('-------------------------------')
    Xtest = df
    X_test = df[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    # print('X_test.shape: ',X_test.shape)

    # _______CreatingModelsMachineLearning_____
    # RandomForestClassifier <======
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)
    # print(X_train.min(), X_train.max())
    # print(X_test.min(), X_test.max())
    model = ens.RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # predictFace-n
    y_prediksi = model.predict(X_test)
    Xtest['hPredictRF'] = y_prediksi
    # replaceKolomjenisKelamin[hPredictLR]
    jenisKelamin = {'hPredictRF':{0:'laki-laki', 1:'perempuan'}}
    Xtest.replace(jenisKelamin, inplace=True)

    if y_prediksi[0] == 1:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': perempuan')
        return 'perempuan', Xtest
    else:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': laki-laki')
        return 'laki-laki', Xtest

def predictWajah_RF(dirGambar, filename):
    foto = cv.imread(dirGambar)
    foto_gray = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
    hasil = cc_wajah.detectMultiScale(foto_gray, scaleFactor=1.2, minNeighbors=5)
    print('terdeksi ',len(hasil),' wajah didalam gambar..')

    n = 0
    m = 1
    deepPink=147,20,255 # female
    mediumBlue=255,0,0 # male

    for (x,y,w,h) in hasil:
        wajah_predict = foto_gray[y:y+h, x:x+w]
        # hitungLBP
        lbp = local_binary_pattern(wajah_predict, 8, 1, method='uniform')
        # hitungHistogramLBP
        n_bins = int(lbp.max()+1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        # normalisasiHistogram
        hist = hist / hist.sum()
        print(hist, type(hist))
        pb0 = round(hist[0],3)
        pb1 = round(hist[1],3)
        pb2 = round(hist[2],3)
        pb3 = round(hist[3],3)
        pb4 = round(hist[4],3)
        pb5 = round(hist[5],3)
        pb6 = round(hist[6],3)
        pb7 = round(hist[7],3)
        pb8 = round(hist[8],3)
        pb9 = round(hist[9],3)

        # runPredictFace-n
        df = pd.DataFrame({'faces':['wajah'+str(m)], 'pb0':[pb0], 'pb1':[pb1], 'pb2':[pb2], 'pb3':[pb3], 'pb4':[pb4], 'pb5':[pb5], 'pb6':[pb6], 'pb7':[pb7], 'pb8':[pb8], 'pb9':[pb9]})
        label_text, dfN =  hPredict_RF(df, m)

        # predictFull
        if n == 0:
            dfFullHpredict = dfN
        else:
            dfV = dfN.values
            dfFullHpredict.loc[n] = [dfV[0][0], dfV[0][1], dfV[0][2], dfV[0][3], dfV[0][4], dfV[0][5], dfV[0][6], dfV[0][7], dfV[0][8], dfV[0][9], dfV[0][10], dfV[0][11]]

        if label_text == 'perempuan':
            warna = deepPink
        else:
            warna = mediumBlue

        ukuranBingkai = w+h
        if (ukuranBingkai >= 100)and(ukuranBingkai < 150):
            size = 'Small'
            UB = 0.2
            SB = 7

        elif (ukuranBingkai >= 150)and(ukuranBingkai < 250):
            size = 'Medium'
            UB = 0.3
            SB = 9

        elif (ukuranBingkai >= 300):
            size = 'Maximum'
            UB = 0.5
            SB = 13

        elif (ukuranBingkai < 300)and(ukuranBingkai >= 250):
            size = 'Large'
            UB = 0.4
            SB = 11

        elif (ukuranBingkai < 100):
            size = 'Minimum'
            UB = 0.2 # 0.1
            SB = 7 # 5

        cv.rectangle(foto, (x,y), (x+w, y+h), color=(warna), thickness=2)
        cv.putText(foto, label_text, (x,y-4), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        cv.putText(foto,'Wajah '+str(m), (x,(y+h+SB)), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        m += 1
        n += 1

    # save_hPredictLR_filename
    path = os.path.join(root, 'static/hPredict/'+'hPredictRF_'+filename)
    cv.imwrite(path, foto)
    print('predictWajah_RF Selesai..')
    return dfFullHpredict

# _____________________________________________________
def hPredict_NB(df, nWajah=1):
    dataTrainLBP = pd.read_excel('dataTrainLBP.xlsx')
    X = dataTrainLBP[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    y = dataTrainLBP['label']
    # # splitDataTrain&TestModel_1
    X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=1, random_state=0) #test_size=0.1 (10%_dariDataSet),
    # print('X_train.shape: ',X_train.shape)
    # print('y_train.shape: ',y_train.shape)
    # print('-------------------------------')
    Xtest = df
    X_test = df[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    # print('X_test.shape: ',X_test.shape)

    # _______CreatingModelsMachineLearning_____
    # NaiveBayes <======
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)
    # print(X_train.min(), X_train.max())
    # print(X_test.min(), X_test.max())
    model = nb.GaussianNB()
    model.fit(X_train, y_train)

    # predictFace-n
    y_prediksi = model.predict(X_test)
    Xtest['hPredictNB'] = y_prediksi
    # replaceKolomjenisKelamin[hPredictLR]
    jenisKelamin = {'hPredictNB':{0:'laki-laki', 1:'perempuan'}}
    Xtest.replace(jenisKelamin, inplace=True)

    if y_prediksi[0] == 1:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': perempuan')
        return 'perempuan', Xtest
    else:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': laki-laki')
        return 'laki-laki', Xtest

def predictWajah_NB(dirGambar, filename):
    foto = cv.imread(dirGambar)
    foto_gray = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
    hasil = cc_wajah.detectMultiScale(foto_gray, scaleFactor=1.2, minNeighbors=5)
    print('terdeksi ',len(hasil),' wajah didalam gambar..')

    n = 0
    m = 1
    deepPink=147,20,255 # female
    mediumBlue=255,0,0 # male

    for (x,y,w,h) in hasil:
        wajah_predict = foto_gray[y:y+h, x:x+w]
        # hitungLBP
        lbp = local_binary_pattern(wajah_predict, 8, 1, method='uniform')
        # hitungHistogramLBP
        n_bins = int(lbp.max()+1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        # normalisasiHistogram
        hist = hist / hist.sum()
        print(hist, type(hist))
        pb0 = round(hist[0],3)
        pb1 = round(hist[1],3)
        pb2 = round(hist[2],3)
        pb3 = round(hist[3],3)
        pb4 = round(hist[4],3)
        pb5 = round(hist[5],3)
        pb6 = round(hist[6],3)
        pb7 = round(hist[7],3)
        pb8 = round(hist[8],3)
        pb9 = round(hist[9],3)

        # runPredictFace-n
        df = pd.DataFrame({'faces':['wajah'+str(m)], 'pb0':[pb0], 'pb1':[pb1], 'pb2':[pb2], 'pb3':[pb3], 'pb4':[pb4], 'pb5':[pb5], 'pb6':[pb6], 'pb7':[pb7], 'pb8':[pb8], 'pb9':[pb9]})
        label_text, dfN =  hPredict_NB(df, m)

        # predictFull
        if n == 0:
            dfFullHpredict = dfN
        else:
            dfV = dfN.values
            dfFullHpredict.loc[n] = [dfV[0][0], dfV[0][1], dfV[0][2], dfV[0][3], dfV[0][4], dfV[0][5], dfV[0][6], dfV[0][7], dfV[0][8], dfV[0][9], dfV[0][10], dfV[0][11]]

        if label_text == 'perempuan':
            warna = deepPink
        else:
            warna = mediumBlue

        ukuranBingkai = w+h
        if (ukuranBingkai >= 100)and(ukuranBingkai < 150):
            size = 'Small'
            UB = 0.2
            SB = 7

        elif (ukuranBingkai >= 150)and(ukuranBingkai < 250):
            size = 'Medium'
            UB = 0.3
            SB = 9

        elif (ukuranBingkai >= 300):
            size = 'Maximum'
            UB = 0.5
            SB = 13

        elif (ukuranBingkai < 300)and(ukuranBingkai >= 250):
            size = 'Large'
            UB = 0.4
            SB = 11

        elif (ukuranBingkai < 100):
            size = 'Minimum'
            UB = 0.2 # 0.1
            SB = 7 # 5

        cv.rectangle(foto, (x,y), (x+w, y+h), color=(warna), thickness=2)
        cv.putText(foto, label_text, (x,y-4), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        cv.putText(foto,'Wajah '+str(m), (x,(y+h+SB)), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        m += 1
        n += 1

    # save_hPredictLR_filename
    path = os.path.join(root, 'static/hPredict/'+'hPredictNB_'+filename)
    cv.imwrite(path, foto)
    print('predictWajah_NB Selesai..')
    return dfFullHpredict

# _____________________________________________________
def hPredict_KNN(df, nWajah=1):
    dataTrainLBP = pd.read_excel('dataTrainLBP.xlsx')
    X = dataTrainLBP[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    y = dataTrainLBP['label']
    # # splitDataTrain&TestModel_1
    X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=1, random_state=0) #test_size=0.1 (10%_dariDataSet),
    # print('X_train.shape: ',X_train.shape)
    # print('y_train.shape: ',y_train.shape)
    # print('-------------------------------')
    Xtest = df
    X_test = df[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    # print('X_test.shape: ',X_test.shape)

    # _______CreatingModelsMachineLearning_____
    # KNeighborsClassifier <======
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)
    # print(X_train.min(), X_train.max())
    # print(X_test.min(), X_test.max())
    model = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='auto', metric='euclidean')
    model.fit(X_train, y_train)

    # predictFace-n
    y_prediksi = model.predict(X_test)
    Xtest['hPredictKNN'] = y_prediksi
    # replaceKolomjenisKelamin[hPredictLR]
    jenisKelamin = {'hPredictKNN':{0:'laki-laki', 1:'perempuan'}}
    Xtest.replace(jenisKelamin, inplace=True)

    if y_prediksi[0] == 1:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': perempuan')
        return 'perempuan', Xtest
    else:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': laki-laki')
        return 'laki-laki', Xtest

def predictWajah_KNN(dirGambar, filename):
    foto = cv.imread(dirGambar)
    foto_gray = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
    hasil = cc_wajah.detectMultiScale(foto_gray, scaleFactor=1.2, minNeighbors=5)
    print('terdeksi ',len(hasil),' wajah didalam gambar..')

    n = 0
    m = 1
    deepPink=147,20,255 # female
    mediumBlue=255,0,0 # male

    for (x,y,w,h) in hasil:
        wajah_predict = foto_gray[y:y+h, x:x+w]
        # hitungLBP
        lbp = local_binary_pattern(wajah_predict, 8, 1, method='uniform')
        # hitungHistogramLBP
        n_bins = int(lbp.max()+1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        # normalisasiHistogram
        hist = hist / hist.sum()
        print(hist, type(hist))
        pb0 = round(hist[0],3)
        pb1 = round(hist[1],3)
        pb2 = round(hist[2],3)
        pb3 = round(hist[3],3)
        pb4 = round(hist[4],3)
        pb5 = round(hist[5],3)
        pb6 = round(hist[6],3)
        pb7 = round(hist[7],3)
        pb8 = round(hist[8],3)
        pb9 = round(hist[9],3)

        # runPredictFace-n
        df = pd.DataFrame({'faces':['wajah'+str(m)], 'pb0':[pb0], 'pb1':[pb1], 'pb2':[pb2], 'pb3':[pb3], 'pb4':[pb4], 'pb5':[pb5], 'pb6':[pb6], 'pb7':[pb7], 'pb8':[pb8], 'pb9':[pb9]})
        label_text, dfN =  hPredict_KNN(df, m)

        # predictFull
        if n == 0:
            dfFullHpredict = dfN
        else:
            dfV = dfN.values
            dfFullHpredict.loc[n] = [dfV[0][0], dfV[0][1], dfV[0][2], dfV[0][3], dfV[0][4], dfV[0][5], dfV[0][6], dfV[0][7], dfV[0][8], dfV[0][9], dfV[0][10], dfV[0][11]]

        if label_text == 'perempuan':
            warna = deepPink
        else:
            warna = mediumBlue

        ukuranBingkai = w+h
        if (ukuranBingkai >= 100)and(ukuranBingkai < 150):
            size = 'Small'
            UB = 0.2
            SB = 7

        elif (ukuranBingkai >= 150)and(ukuranBingkai < 250):
            size = 'Medium'
            UB = 0.3
            SB = 9

        elif (ukuranBingkai >= 300):
            size = 'Maximum'
            UB = 0.5
            SB = 13

        elif (ukuranBingkai < 300)and(ukuranBingkai >= 250):
            size = 'Large'
            UB = 0.4
            SB = 11

        elif (ukuranBingkai < 100):
            size = 'Minimum'
            UB = 0.2 # 0.1
            SB = 7 # 5

        cv.rectangle(foto, (x,y), (x+w, y+h), color=(warna), thickness=2)
        cv.putText(foto, label_text, (x,y-4), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        cv.putText(foto,'Wajah '+str(m), (x,(y+h+SB)), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        m += 1
        n += 1

    # save_hPredictLR_filename
    path = os.path.join(root, 'static/hPredict/'+'hPredictKNN_'+filename)
    cv.imwrite(path, foto)
    print('predictWajah_KNN Selesai..')
    return dfFullHpredict

# _____________________________________________________
def hPredict_SVM(df, nWajah=1):
    dataTrainLBP = pd.read_excel('dataTrainLBP.xlsx')
    X = dataTrainLBP[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    y = dataTrainLBP['label']
    # # splitDataTrain&TestModel_1
    X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=1, random_state=0) #test_size=0.1 (10%_dariDataSet),
    # print('X_train.shape: ',X_train.shape)
    # print('y_train.shape: ',y_train.shape)
    # print('-------------------------------')
    Xtest = df
    X_test = df[['pb0','pb1','pb2','pb3','pb4','pb5','pb6','pb7','pb8','pb9']]
    # print('X_test.shape: ',X_test.shape)

    # _______CreatingModelsMachineLearning_____
    # SupportVectorMachine <======
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)
    # print(X_train.min(), X_train.max())
    # print(X_test.min(), X_test.max())
    model = LinearSVC(C = 1.0)
    model.fit(X_train, y_train)

    # predictFace-n
    y_prediksi = model.predict(X_test)
    Xtest['hPredictSVM'] = y_prediksi
    # replaceKolomjenisKelamin[hPredictLR]
    jenisKelamin = {'hPredictSVM':{0:'laki-laki', 1:'perempuan'}}
    Xtest.replace(jenisKelamin, inplace=True)

    if y_prediksi[0] == 1:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': perempuan')
        return 'perempuan', Xtest
    else:
        print('>>',model,'__[hPredict]jenisKelaminWajah'+str(nWajah)+': laki-laki')
        return 'laki-laki', Xtest

def predictWajah_SVM(dirGambar, filename):
    foto = cv.imread(dirGambar)
    foto_gray = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
    hasil = cc_wajah.detectMultiScale(foto_gray, scaleFactor=1.2, minNeighbors=5)
    print('terdeksi ',len(hasil),' wajah didalam gambar..')

    n = 0
    m = 1
    deepPink=147,20,255 # female
    mediumBlue=255,0,0 # male

    for (x,y,w,h) in hasil:
        wajah_predict = foto_gray[y:y+h, x:x+w]
        # hitungLBP
        lbp = local_binary_pattern(wajah_predict, 8, 1, method='uniform')
        # hitungHistogramLBP
        n_bins = int(lbp.max()+1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        # normalisasiHistogram
        hist = hist / hist.sum()
        print(hist, type(hist))
        pb0 = round(hist[0],3)
        pb1 = round(hist[1],3)
        pb2 = round(hist[2],3)
        pb3 = round(hist[3],3)
        pb4 = round(hist[4],3)
        pb5 = round(hist[5],3)
        pb6 = round(hist[6],3)
        pb7 = round(hist[7],3)
        pb8 = round(hist[8],3)
        pb9 = round(hist[9],3)

        # runPredictFace-n
        df = pd.DataFrame({'faces':['wajah'+str(m)], 'pb0':[pb0], 'pb1':[pb1], 'pb2':[pb2], 'pb3':[pb3], 'pb4':[pb4], 'pb5':[pb5], 'pb6':[pb6], 'pb7':[pb7], 'pb8':[pb8], 'pb9':[pb9]})
        label_text, dfN =  hPredict_SVM(df, m)

        # predictFull
        if n == 0:
            dfFullHpredict = dfN
        else:
            dfV = dfN.values
            dfFullHpredict.loc[n] = [dfV[0][0], dfV[0][1], dfV[0][2], dfV[0][3], dfV[0][4], dfV[0][5], dfV[0][6], dfV[0][7], dfV[0][8], dfV[0][9], dfV[0][10], dfV[0][11]]

        if label_text == 'perempuan':
            warna = deepPink
        else:
            warna = mediumBlue

        ukuranBingkai = w+h
        if (ukuranBingkai >= 100)and(ukuranBingkai < 150):
            size = 'Small'
            UB = 0.2
            SB = 7

        elif (ukuranBingkai >= 150)and(ukuranBingkai < 250):
            size = 'Medium'
            UB = 0.3
            SB = 9

        elif (ukuranBingkai >= 300):
            size = 'Maximum'
            UB = 0.5
            SB = 13

        elif (ukuranBingkai < 300)and(ukuranBingkai >= 250):
            size = 'Large'
            UB = 0.4
            SB = 11

        elif (ukuranBingkai < 100):
            size = 'Minimum'
            UB = 0.2 # 0.1
            SB = 7 # 5

        cv.rectangle(foto, (x,y), (x+w, y+h), color=(warna), thickness=2)
        cv.putText(foto, label_text, (x,y-4), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        cv.putText(foto,'Wajah '+str(m), (x,(y+h+SB)), cv.FONT_HERSHEY_DUPLEX, UB, color=(warna), thickness=1,) #cv.LINE_AA
        m += 1
        n += 1

    # save_hPredictLR_filename
    path = os.path.join(root, 'static/hPredict/'+'hPredictSVM_'+filename)
    cv.imwrite(path, foto)
    print('predictWajah_SVM Selesai..')
    return dfFullHpredict

# _____________________________________________________
# _____________________________________________________
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('noFilePath..')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('noImageSelectedForUploading..')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: '+filename)

        path = os.path.join(root, 'static/uploads/'+filename)

        # runPredictCV
        dfCv = predictWajah_CV(path, filename)
        if dfCv is None:
            flash('tidakAdaWajahTerdeteksi_didalamGambar: '+filename)
            return redirect(request.url)

        print(dfCv)
        print('-----------------------------------------------')

        # runPredictLR
        dfLR = predictWajah_LR(path, filename)
        if dfLR is None:
            flash('tidakAdaWajahTerdeteksi_didalamGambar: '+filename)
            return redirect(request.url)

        print(dfLR)
        print('-----------------------------------------------')

        # runPredictDS
        dfDS = predictWajah_DS(path, filename)
        if dfDS is None:
            flash('tidakAdaWajahTerdeteksi_didalamGambar: '+filename)
            return redirect(request.url)

        print(dfDS)
        print('-----------------------------------------------')

        # runPredictRF
        dfRF = predictWajah_RF(path, filename)
        if dfRF is None:
            flash('tidakAdaWajahTerdeteksi_didalamGambar: '+filename)
            return redirect(request.url)

        print(dfRF)
        print('-----------------------------------------------')

        # runPredictNB
        dfNB = predictWajah_NB(path, filename)
        if dfNB is None:
            flash('tidakAdaWajahTerdeteksi_didalamGambar: '+filename)
            return redirect(request.url)

        print(dfNB)
        print('-----------------------------------------------')

        # runPredictKNN
        dfKNN = predictWajah_KNN(path, filename)
        if dfKNN is None:
            flash('tidakAdaWajahTerdeteksi_didalamGambar: '+filename)
            return redirect(request.url)

        print(dfKNN)
        print('-----------------------------------------------')

        # runPredictSVM
        dfSVM = predictWajah_SVM(path, filename)
        if dfSVM is None:
            flash('tidakAdaWajahTerdeteksi_didalamGambar: '+filename)
            return redirect(request.url)

        print(dfSVM)
        print('-----------------------------------------------')

        # flash('imageSuccesFullyUploadedAndDisplay..')
        arr = []
        for iarr in range(0,len(dfCv)):
            niarr = iarr + 1
            arr.append(str(niarr))

        print(arr, type(arr))
        # runPredictCV, runPredictLR, runPredictDS, runPredictRF, runPredictNB, runPredictKNN, runPredictSVM
        return render_template('display.html', filename=filename, dfCv=dfCv, dfLR=dfLR, dfDS=dfDS, dfRF=dfRF, dfNB=dfNB, dfKNN=dfKNN, dfSVM=dfSVM, arr=arr)

    else:
        flash('allowedImageTypesAre-png,jpg,jpeg..')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: '+filename)
    return redirect(url_for('static', filename='hPredict/'+filename), code=301)

@app.route('/displayFace/<filename>')
def display_faceImage(filename):
    print('display_faceImage filename: '+filename)
    return redirect(url_for('static', filename='faces/'+filename), code=301)

@app.route('/displayPlt/<filename>')
def display_plt(filename):
    print('display_plt filename: '+filename)
    return redirect(url_for('static', filename='hasilLBPH/'+filename), code=301)

# # _____________________________________________________
# def ekstraksi_lbp(image, radius=1, n_points=8):
#     # hitungLBP
#     lbp = local_binary_pattern(image, n_points, radius, method='uniform')
#     # hitungHistogramLBP
#     n_bins = int(lbp.max()+1)
#     hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
#     # normalisasiHistogram
#     hist = hist / hist.sum()
#     # print(hist, type(hist))
#
#     pb0,pb1,pb2,pb3,pb4,pb5,pb6,pb7,pb8,pb9 = [round(hist[0],3)],[round(hist[1],3)],[round(hist[2],3)],[round(hist[3],3)],[round(hist[4],3)],[round(hist[5],3)],[round(hist[6],3)],[round(hist[7],3)],[round(hist[8],3)],[round(hist[9],3)]
#     dataTest = pd.DataFrame({'pb0':pb0, 'pb1':pb1, 'pb2':pb2, 'pb3':pb3, 'pb4':pb4, 'pb5':pb5, 'pb6':pb6, 'pb7':pb7, 'pb8':pb8, 'pb9':pb9})
#     # print(dataTest)
#     return lbp, dataTest

if __name__=='__main__':
    app.run(debug=True)
