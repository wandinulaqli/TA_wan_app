# import os
# import pandas as pd
# import cv2 as cv
# import numpy as np
# from skimage.feature import local_binary_pattern
# import matplotlib.pyplot as plt
#
# def ekstraksi_lbp(image_path, radius=1, n_points=8):
#     # bacaCitraDalamSkalaAbu-abu
#     image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#     if image is None:
#         print('imagePath_tidakDitemukan!')
#         return
#
#     # hitungLBP
#     lbp = local_binary_pattern(image, n_points, radius, method='uniform')
#     # hitungHistogramLBP
#     n_bins = int(lbp.max()+1)
#     hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
#     # normalisasiHistogram
#     hist = hist / hist.sum()
#     # print(hist, type(hist))
#
#     # # tampilkanHasil
#     # plt.figure(figsize=(10,5))
#     # # citraAsli
#     # plt.subplot(1,3,1)
#     # plt.imshow(image, cmap='gray')
#     # plt.title('citraGrayscale')
#     # plt.axis('off')
#     # # citraLBP
#     # plt.subplot(1,3,2)
#     # plt.imshow(lbp, cmap='gray')
#     # plt.title('citraLBP')
#     # plt.axis('off')
#     # # histogram
#     # plt.subplot(1,3,3)
#     # plt.bar(range(len(hist)), hist, color='gray')
#     # plt.title('histogramLBP')
#     # plt.xlabel('polaBiner')
#     # plt.ylabel('frekuensi')
#     # plt.tight_layout()
#     return hist
#
# # histogram = ekstraksi_lbp('d:/ft.jpeg')
# # print('histogramLBP:', histogram)
# # plt.show()
#
# # _____________________________________________________setDataTrain
# pb0,pb1,pb2,pb3,pb4,pb5,pb6,pb7,pb8,pb9,arlb = [],[],[],[],[],[],[],[],[],[],[]
# root = os.getcwd()
#
# pathImg = input('pathGambarInFolder: ')
# image_path = os.path.join(root,pathImg)
#
# labels = os.listdir(image_path) # type(int)
#
# imageLabels = [os.path.join(image_path, fl) for fl in os.listdir(image_path)]
# i = 0
# for labelPath in imageLabels:
#     label = labels[i]
#     imageName = [os.path.join(labelPath, fn) for fn in os.listdir(labelPath)]
#     for fnm in imageName:
#         histogram = ekstraksi_lbp(fnm)
#         print('histogramLBP_'+fnm+':', histogram)
#         pb0.append(round(histogram[0],3))
#         pb1.append(round(histogram[1],3))
#         pb2.append(round(histogram[2],3))
#         pb3.append(round(histogram[3],3))
#         pb4.append(round(histogram[4],3))
#         pb5.append(round(histogram[5],3))
#         pb6.append(round(histogram[6],3))
#         pb7.append(round(histogram[7],3))
#         pb8.append(round(histogram[8],3))
#         pb9.append(round(histogram[9],3))
#         arlb.append(int(label))
#
#     i += 1
#
# df = pd.DataFrame({'pb0':pb0,'pb1':pb1,'pb2':pb2,'pb3':pb3,'pb4':pb4,'pb5':pb5,'pb6':pb6,'pb7':pb7,'pb8':pb8,'pb9':pb9,'label':arlb})
# path = os.path.join(root,'dataTrainLBP.xlsx')
# df.to_excel(path, sheet_name='polaBiner', index=False)
#
# print('----------------------------------------------------------selesaiBoss')
# dataTrainLBP = pd.read_excel('dataTrainLBP.xlsx')
# print(dataTrainLBP.head())
# # print(df.describe())
# print(dataTrainLBP.info())
# print(dataTrainLBP['label'].value_counts())

# import pandas as pd
# a = 1
# b = 2.10
# c = 3.45
# d = 4
#
# df = pd.DataFrame({'f1':[a], 'f2':[b], 'f3':[c], 'f4':[d]})
# print(df)
