# EEG data analysis
#
# Author: ROBALDO Axel for PI-Psy Institute
# Date: july 2020
#
# Description: 
# This programm extract several features from preprocessed data tensor obtained via EEG_preprocessing 
# script. The resulting features are stored in another features tensor which is finally serialized 
# to .npy file.
#
# EntroPy was created and is maintained by Raphael Vallat.



import numpy as np
import scipy as sp
import itertools
from entropy import katz_fd

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    def __init__(self, parent=None, **kwargs):
        self.path = False
        self.exctracted = False
        

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(264, 320)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 211, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 40, 221, 31))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")

        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.buttonFile = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.buttonFile.setObjectName("buttonFile")
        self.gridLayout.addWidget(self.buttonFile, 0, 1, 1, 1)
        self.comboFile = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboFile.setObjectName("comboFile")
        self.gridLayout.addWidget(self.comboFile, 0, 0, 1, 1)
        
        self.buttonExtract = QtWidgets.QPushButton(self.centralwidget)
        self.buttonExtract.setGeometry(QtCore.QRect(30, 150, 221, 51))
        self.buttonExtract.setObjectName("buttonExtract")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 120, 211, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.buttonSave = QtWidgets.QPushButton(self.centralwidget)
        self.buttonSave.setGeometry(QtCore.QRect(30, 210, 221, 51))
        self.buttonSave.setObjectName("buttonSave")
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 264, 21))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.showMessage('Ready')
        MainWindow.setStatusBar(self.statusbar)

        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")

        self.actionInformations = QtWidgets.QAction(MainWindow)
        self.actionInformations.setObjectName("actionInformations")

        self.menuMenu.addAction(self.actionQuit)
        self.menuMenu.addAction(self.actionInformations)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.actionQuit.triggered.connect(self.quit)

        self.actionInformations.triggered.connect(self.infos)

        self.buttonFile.clicked.connect(self.comboFile.clearEditText)
        self.buttonFile.clicked.connect(self.setPath)

        self.buttonExtract.clicked.connect(self.extract)
        self.buttonSave.clicked.connect(self.save)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "EEG_time_features"))
        self.label.setText(_translate("MainWindow", "Select the data tensor"))
        self.buttonFile.setText(_translate("MainWindow", "Browse .npy file"))
        self.buttonExtract.setText(_translate("MainWindow", "Extract time features"))
        self.label_2.setText(_translate("MainWindow", "Time features extraction"))
        self.buttonSave.setText(_translate("MainWindow", "Save features tensor to .npy"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Esc"))
        self.actionInformations.setText(_translate("MainWindow", "Informations"))

    def quit(self):
        sys.exit()

    def infos(self):
        print("""# EEG time features extractor
                    #
                    # Author: ROBALDO Axel for PI-Psy Institute
                    # Date: july 2020
                    #
                    # Description: 
                    # This programm extract several time features from preprocessed data tensor obtained via 
                    # EEG_preprocessing script. The resulting features are stored in another features tensor 
                    # which is finally serialized to .npy file.
                    #
                    # EntroPy was created and is maintained by Raphael Vallat.
                    """)

    def setPath(self):
        self.pathNpy, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Numpy Tensor File", "", "Npy Files (*.npy)")
        self.comboFile.addItem(self.pathNpy)
        self.path = True

    
    def extract(self):
        self.statusbar.showMessage('Features extraction, please wait...')
        
        if self.path == True:

            dt = 0.002              # sample rate of the original datas
            c = 3e8

            data_tensor = np.load(self.pathNpy)
            self.feat_tensor = np.empty((data_tensor.shape[0], data_tensor.shape[1], data_tensor.shape[2], 12), dtype = np.float32)


            for epoch in np.ndindex(data_tensor.shape[0]):
                for wave in np.ndindex(data_tensor.shape[1]):
                    for channel in np.ndindex(data_tensor.shape[2]):
            
                        values = data_tensor[epoch[0], wave[0], channel[0], :]
                        diff_values = np.diff(values)                                                               # derivate the values to find leading coefficient between each values
                        enveloppe = np.abs(sp.signal.hilbert(values))                                               # find the hilbert enveloppe
            

                        self.feat_tensor[epoch[0], wave[0], channel[0], 0]  = np.mean(values)                             #mean
                        self.feat_tensor[epoch[0], wave[0], channel[0], 1]  = sum(abs(values)**2.0)*dt                    #energy
                        self.feat_tensor[epoch[0], wave[0], channel[0], 2]  = sp.stats.kurtosis(values)                   #kurtosis
                        self.feat_tensor[epoch[0], wave[0], channel[0], 3]  = sp.stats.skew(values)                       #skewness
                        self.feat_tensor[epoch[0], wave[0], channel[0], 4]  = katz_fd(values)                             #katz fractal
                        
            
                        self.feat_tensor[epoch[0], wave[0], channel[0], 5]  = np.mean(enveloppe)                          #enveloppe mean
                        self.feat_tensor[epoch[0], wave[0], channel[0], 6]  = np.std(enveloppe)                           #enveloppe standard deviation
                        self.feat_tensor[epoch[0], wave[0], channel[0], 7]  = sp.stats.kurtosis(enveloppe)                #enveloppe kurtosis
                        self.feat_tensor[epoch[0], wave[0], channel[0], 8]  = sp.stats.skew(enveloppe)                    #enveloppe skewness
            

                        self.feat_tensor[epoch[0], wave[0], channel[0], 9]  = ((values[:-1] * values[1:]) < 0).sum()      #zero crossing rate: we summing 1 each time a value cross the horizontal axis
                        self.feat_tensor[epoch[0], wave[0], channel[0], 10] = len(list(itertools.groupby(diff_values, lambda diff_values: diff_values > 0)))    #slope changes: we find each sign change in the 1st degree derivate of the values
             
            
                        spectrum = sp.fft.fft(values)                                   # we process a 1D fft
                        freq = sp.fft.fftfreq(len(spectrum), dt)                        # then we set the frequency axis
                        peak_freq = abs(freq[abs(spectrum) == max(abs(spectrum))][0])   # we find which frequency belong to the highest fft value
            
                        self.feat_tensor[epoch[0], wave[0], channel[0], 11] = (c/peak_freq)  # finally we obtain the wavelength by dividing c by this frequency


            print(self.feat_tensor)
            print(self.feat_tensor.shape)

            self.extracted = True
            self.statusbar.showMessage('Features extracted !')

        else:
            self.statusbar.showMessage('Select data tensor first')

    def save(self):
        if self.extracted == True:
            np.save("time_features_tensor", self.feat_tensor) 
            self.statusbar.showMessage('Saved !')

        else:
            self.statusbar.showMessage('Extract data first')
    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
