# EEG data preprocessing
#
# Author: ROBALDO Axel for PI-Psy Institute
# Date: july 2020
#
# Description: 
# This algorithm preprocess EEG signals by proceeding to artifacts repair/rejection, epoching in desired fixed time
# duration and then filter by Cuda GPU compatible in 5 differents frequency bands. Metadata from acquisition are also 
# readen and used to get sample rate and channel names.
# Finally epochs are converted to classics Numpy 4D arrays.
#
#       MNE-Python package developped by:
# A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, 
# T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data analysis with MNE-Python, 
# Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X, [DOI]
#
#       Autoreject algorithm written by:
# Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort,
# "Automated rejection and repair of bad trials in MEG/EEG." In 6th International Workshop on
# Pattern Recognition in Neuroimaging (PRNI). 2016.
# Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017.
# Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.
############################################################################################################



import this
import os
import numpy as np
import mne
import tqdm

from autoreject import AutoReject, get_rejection_threshold
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt



class EmittingStream(QtCore.QObject):

    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class Ui_MainWindow(object):

    def __init__(self, parent=None, **kwargs):
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.setEEG = False
        self.setInfo = False
        self.loaded = False
        self.repaired = False
        self.filtred = False


    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__
        

    def normalOutputWritten(self, text):
        cursor = self.textConsole.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textConsole.setTextCursor(cursor)
        self.textConsole.ensureCursorVisible()

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1023, 897)
        MainWindow.setToolTip("")
        MainWindow.setWhatsThis("")
        MainWindow.setAccessibleName("")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 40, 251, 161))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.buttonFile = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.buttonFile.setObjectName("buttonFile")
        self.gridLayout.addWidget(self.buttonFile, 0, 1, 1, 1)

        self.comboInfo = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboInfo.setEditable(True)
        self.comboInfo.setObjectName("comboInfo")
        self.gridLayout.addWidget(self.comboInfo, 1, 0, 1, 1)

        self.buttonInfo = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.buttonInfo.setObjectName("buttonInfo")
        self.gridLayout.addWidget(self.buttonInfo, 1, 1, 1, 1)

        self.buttonExtract = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.buttonExtract.setStatusTip("")
        self.buttonExtract.setObjectName("buttonExtract")
        self.gridLayout.addWidget(self.buttonExtract, 3, 0, 1, 1)

        self.labelEpoch = QtWidgets.QLabel(self.gridLayoutWidget)
        self.labelEpoch.setObjectName("labelEpoch")
        self.gridLayout.addWidget(self.labelEpoch, 2, 1, 1, 1)

        self.comboFile = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboFile.setAutoFillBackground(False)
        self.comboFile.setEditable(True)
        self.comboFile.setFrame(True)
        self.comboFile.setObjectName("comboFile")
        self.gridLayout.addWidget(self.comboFile, 0, 0, 1, 1)

        self.lineEpoch = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEpoch.setAcceptDrops(False)
        self.lineEpoch.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEpoch.setText("0.5")
        self.lineEpoch.setObjectName("lineEpoch")
        self.gridLayout.addWidget(self.lineEpoch, 2, 0, 1, 1)

        self.buttonRepair = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.buttonRepair.setStatusTip("")
        self.buttonRepair.setObjectName("buttonRepair")
        self.gridLayout.addWidget(self.buttonRepair, 4, 0, 1, 1)

        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(280, 0, 20, 591))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(30, 200, 251, 91))
        self.formLayoutWidget.setObjectName("formLayoutWidget")

        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")

        self.labelEBefore = QtWidgets.QLabel(self.formLayoutWidget)
        self.labelEBefore.setObjectName("labelEBefore")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.labelEBefore)

        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label_2)

        self.labelEAfter = QtWidgets.QLabel(self.formLayoutWidget)
        self.labelEAfter.setObjectName("labelEAfter")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.labelEAfter)

        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.label_4)

        self.labelChannels = QtWidgets.QLabel(self.formLayoutWidget)
        self.labelChannels.setObjectName("labelChannels")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelChannels)

        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_7)

        self.labelPoints = QtWidgets.QLabel(self.formLayoutWidget)
        self.labelPoints.setObjectName("labelPoints")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.labelPoints)

        self.label_9 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_9)

        self.labelSamples = QtWidgets.QLabel(self.formLayoutWidget)
        self.labelSamples.setObjectName("labelSamples")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.labelSamples)

        self.label_11 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_11.setObjectName("label_11")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_11)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 241, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 310, 241, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")


        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 340, 160, 121))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.checkDelta = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkDelta.setObjectName("checkDelta")
        self.verticalLayout.addWidget(self.checkDelta)

        self.checkTheta = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkTheta.setObjectName("checkTheta")
        self.verticalLayout.addWidget(self.checkTheta)

        self.checkAlpha = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkAlpha.setObjectName("checkAlpha")
        self.verticalLayout.addWidget(self.checkAlpha)

        self.checkBeta = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBeta.setObjectName("checkBeta")
        self.verticalLayout.addWidget(self.checkBeta)

        self.checkGamma = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkGamma.setObjectName("checkGamma")
        self.verticalLayout.addWidget(self.checkGamma)

        self.buttonFilter = QtWidgets.QPushButton(self.centralwidget)
        self.buttonFilter.setGeometry(QtCore.QRect(190, 340, 91, 121))
        self.buttonFilter.setObjectName("buttonFilter")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(300, 10, 711, 571))
        self.tabWidget.setObjectName("tabWidget")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 480, 241, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")

        self.buttonGraphics = QtWidgets.QPushButton(self.centralwidget)
        self.buttonGraphics.setGeometry(QtCore.QRect(30, 510, 161, 28))
        self.buttonGraphics.setObjectName("buttonGraphics")

        self.buttonSave = QtWidgets.QPushButton(self.centralwidget)
        self.buttonSave.setGeometry(QtCore.QRect(30, 550, 161, 28))
        self.buttonSave.setObjectName("buttonSave")

        self.textConsole = QtWidgets.QTextEdit(self.centralwidget)
        self.textConsole.setGeometry(QtCore.QRect(10, 600, 1001, 251))
        self.textConsole.setObjectName("textConsole")

        MainWindow.setCentralWidget(self.centralwidget)



        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1023, 21))
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
        self.tabWidget.setCurrentIndex(0)

        self.actionQuit.triggered.connect(self.quit)

        self.actionInformations.triggered.connect(self.infos)

        self.buttonFile.clicked.connect(self.comboFile.clearEditText)
        self.buttonFile.clicked.connect(self.setEEGpath)

        self.buttonInfo.clicked.connect(self.comboInfo.clearEditText)
        self.buttonInfo.clicked.connect(self.setINFOpath)

        self.buttonExtract.clicked.connect(self.load_data)

        self.buttonRepair.clicked.connect(self.repair)
        self.buttonFilter.clicked.connect(self.filter)
        self.buttonSave.clicked.connect(self.save)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "EEG_preprocessing"))
        self.buttonFile.setText(_translate("MainWindow", "Browse .easy file"))
        self.comboInfo.setToolTip(_translate("MainWindow", "Select your .info file"))
        self.buttonInfo.setText(_translate("MainWindow", "Browse .info file"))
        self.comboFile.setToolTip(_translate("MainWindow", "Select your .easy file"))
        self.buttonRepair.setToolTip(_translate("MainWindow", "Apply signal processing treatment"))
        self.buttonRepair.setText(_translate("MainWindow", "Repair signals"))
        self.buttonExtract.setToolTip(_translate("MainWindow", "Extract signal and info from files"))
        self.buttonExtract.setText(_translate("MainWindow", "Extract EEG signal"))
        self.labelEpoch.setText(_translate("MainWindow", "Epoch duration (s)"))
        self.lineEpoch.setToolTip(_translate("MainWindow", "Enter fixed epoch duration"))
        self.labelEBefore.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", "Epochs before treatment"))
        self.labelEAfter.setText(_translate("MainWindow", ""))
        self.label_4.setText(_translate("MainWindow", "Epochs after treatment"))
        self.labelChannels.setText(_translate("MainWindow", ""))
        self.label_7.setText(_translate("MainWindow", "Channels detected"))
        self.labelPoints.setText(_translate("MainWindow", ""))
        self.label_9.setText(_translate("MainWindow", "Points detected"))
        self.labelSamples.setText(_translate("MainWindow", ""))
        self.label_11.setText(_translate("MainWindow", "Sample frequency (Hz)"))
        self.label.setText(_translate("MainWindow", "Import and clean the data first"))
        self.label_3.setText(_translate("MainWindow", "Extract interests waves"))
        self.checkDelta.setText(_translate("MainWindow", "Delta"))
        self.checkTheta.setText(_translate("MainWindow", "Theta"))
        self.checkAlpha.setText(_translate("MainWindow", "Alpha"))
        self.checkBeta.setText(_translate("MainWindow", "Beta"))
        self.checkGamma.setText(_translate("MainWindow", "Gamma"))
        self.buttonFilter.setText(_translate("MainWindow", "Filter"))
        self.buttonFilter.setToolTip(_translate("MainWindow", "Apply filters for the selectonned wavelenghts"))

        self.label_5.setText(_translate("MainWindow", "Export data"))
        self.buttonGraphics.setText(_translate("MainWindow", "Save graphics"))
        self.buttonGraphics.setToolTip(_translate("MainWindow", "Save current graphics to .png files"))
        self.buttonSave.setText(_translate("MainWindow", "Serialize tensors to .npy"))
        self.buttonSave.setToolTip(_translate("MainWindow", "Save the filtered signals to .pkl files"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.actionQuit.setIconText(_translate("MainWindow", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Escape"))
        self.actionInformations.setText(_translate("MainWindow", "Informations"))

    def quit(self):
        sys.exit()

    def infos(self):
        print("""EEG data preprocessing \n
              
                Author: ROBALDO Axel for PI-Psy Institute
                Date: july 2020
                
                Description: 
                This algorithm preprocess EEG signals by proceeding to artifacts repair/rejection, epoching in desired fixed time
                duration and then filter by Cuda GPU compatible in 5 differents frequency bands. Metadata from acquisition are also 
                readen and used to get sample rate and channel names.
                Finally epochs are converted to classics Numpy 4D arrays.
                
                       MNE-Python package developped by:
               A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, 
               T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data analysis with MNE-Python, 
               Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X, [DOI]
                
                       Autoreject algorithm written by:
                Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort,
               "Automated rejection and repair of bad trials in MEG/EEG." In 6th International Workshop on
               Pattern Recognition in Neuroimaging (PRNI). 2016.
               Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017.
               Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.)""")
    
    def setEEGpath(self):
        self.eeg_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open EEG File", "", "Easy Files (*.easy)")
        self.comboFile.addItem(self.eeg_path)
        self.setEEG = True

    
    def setINFOpath(self):
        self.info_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Info File", "", "Info Files (*.info)")
        self.comboInfo.addItem(self.info_path)
        self.setInfo = True

    def load_data(self):
        self.statusbar.showMessage('Extracting, please wait...')

        # Load the data and infos ----------------------------------------------------------------------------------------------------------------------------------
        if self.setEEG == True & self.setInfo == True:
            self.tEpoch = float(self.lineEpoch.text())
            self.lineEpoch.setReadOnly(True)

            ch_names = []
            unit = []
            sfreq = []

            array_file = np.loadtxt(fname = self.eeg_path)     #load in numpy array
            array_file = np.transpose(array_file)                                           #transpose to match shape (n_channel, n_time)



            with open (self.info_path, 'rt') as myfile:         #parsing the info file to find channels name and smapling rate
    
                for myline in myfile:                               # For each line, read it to a string 

                    if (myline.find("G sampling rate") > 0):              # For each string seek for...
                        sfreq.append(myline[-19:-16])       
                        print("sampling rate "+ str(sfreq))

                    if  (myline.find("G units") > 0):
                        unit.append(myline[-3:-1])
                        print("unit "+ str(unit))

                    if myline.find("Channel") > 0:
                        ch_name = myline[-4:-1]
                        ch_name = ch_name.replace(" ", "")                    #remove space from name string
                        ch_names.append(ch_name)

            self.labelSamples.setText(sfreq[0])
            sfreq = float(sfreq[0])
            self.nSampleEpoch = int(sfreq*self.tEpoch)+1
        

            if unit[0] == 'µV': 
                array_file = array_file/1e6
            elif unit[0] == 'nV':
                array_file = array_file/1e9

            info = mne.create_info(ch_names, sfreq, ch_types='eeg')     #creat info structure

            self.raw = mne.io.RawArray(array_file[0:8,:], info)              #create Raw object from Numpy array


            for ch in ch_names:                                         #loop to suppress empty channels
                data, times = self.raw[ch, :] 
                if data[0,0] == data[0,len(self.raw)-1]:                #comparing first and last values
                    self.raw.drop_channels(ch)
                    print("channel " + ch + " dropped: no data")
        
        
            nEpochs =int(len(self.raw)/sfreq/self.tEpoch)
            self.labelEBefore.setText(str(nEpochs))
            self.labelChannels.setText(str(len(self.raw.ch_names)))
            self.labelPoints.setText(str(len(self.raw)))


            montage_dir = os.path.join(os.path.dirname(mne.__file__), 'channels', 'data', 'montages')
            self.raw = self.raw.set_montage('standard_1020')

            print(self.raw)
            print(self.raw.info)


            # create tab with QtTabWidget
            self.rawTab = QtWidgets.QWidget()
            self.rawTab.setObjectName("rawTab")
            self.tabWidget.addTab(self.rawTab, "Raw")
                
            canvas = MplCanvas(self, data=self.raw, title="Raw")                #observe original EEG data
        
            self.rawTab.layout = QtWidgets.QVBoxLayout()
            self.rawTab.layout.addWidget(canvas)
            self.rawTab.setLayout(self.rawTab.layout)

            self.loaded = True
            self.statusbar.showMessage('Ready')

        
        else:
            self.statusbar.showMessage('Select data first')
        
        




    def repair(self):
        
        self.statusbar.showMessage('Repairing, please wait...')

        # First epoching for ICA---------------------------------------------------------------------------------------------------------------------------------------------------------- 
        if self.loaded ==True :  
            
            self.raw.load_data()                                             #we load the data in RAM

            filt_raw = self.raw.copy().filter(l_freq=1., h_freq=None)        #we high pass to suppress slow drifts

            events = mne.make_fixed_length_events(filt_raw, start=0, duration=self.tEpoch)
            epochs = mne.Epochs(filt_raw, events, tmin=0.0, baseline=(None, None), tmax=self.tEpoch, preload=True, verbose=True)


            # Repair artifacts w/ ICA-----------------------------------------------------------------------------------------------------------------------------------------
            ica = mne.preprocessing.ICA(random_state=97, max_iter=1600)

            reject = get_rejection_threshold(epochs)

            ica.fit(epochs, reject=reject, tstep=self.tEpoch)
            clean_raw = ica.apply(filt_raw)


            # Main epoching-----------------------------------------------------------------------------------------------------------------------------------------------------
            events = mne.make_fixed_length_events(clean_raw, start=0, duration=self.tEpoch)
            self.epochs_clean = mne.Epochs(clean_raw, events, tmin=0.0, baseline=(None, None), tmax=self.tEpoch, preload=True, verbose=True)

  
            # Bad epochs Autoreject---------------------------------------------------------------------------------------------------------------------------------------------
            ar = AutoReject(random_state=42, verbose='tqdm')

            self.epochs_clean = ar.fit_transform(self.epochs_clean)  

            self.labelEAfter.setText(str(len(self.epochs_clean)))


            # show graph in QtTabWidget
            self.epochsTab = QtWidgets.QWidget()
            self.epochsTab.setObjectName("epochsTab")
            self.tabWidget.addTab(self.epochsTab, "Cleaned Epochs")
        
            canvas = MplCanvas(self, data=self.epochs_clean, title="Epochs clean")

            self.epochsTab.layout = QtWidgets.QVBoxLayout()
            self.epochsTab.layout.addWidget(canvas)
            self.epochsTab.setLayout(self.epochsTab.layout)


            self.repaired = True
            self.statusbar.showMessage('Ready')
    
        else:
            self.statusbar.showMessage('Load data first')




    def filter(self):
        self.statusbar.showMessage('Filtering, please wait...')

        if self.repaired ==True:
            self.waves = []
            self.wavesTab = []

            if self.checkDelta.isChecked():
                delta = self.epochs_clean.copy().filter(l_freq=3, h_freq=4, n_jobs='cuda', method='fir')      #filtering executed by GPU
                self.waves.append(delta)
                self.wavesTab.append("Delta")

            if self.checkTheta.isChecked():
                theta = self.epochs_clean.copy().filter(l_freq=4, h_freq=8, n_jobs='cuda', method='fir')
                self.waves.append(theta)
                self.wavesTab.append("Theta")

            if self.checkAlpha.isChecked():
                alpha = self.epochs_clean.copy().filter(l_freq=8, h_freq=12, n_jobs='cuda', method='fir')
                self.waves.append(alpha)
                self.wavesTab.append("Alpha")

            if self.checkBeta.isChecked():
                beta = self.epochs_clean.copy().filter(l_freq=12, h_freq=25, n_jobs='cuda', method='fir')
                self.waves.append(beta)
                self.wavesTab.append("Beta")

            if self.checkGamma.isChecked():
                gamma = self.epochs_clean.copy().filter(l_freq=25, h_freq=35, n_jobs='cuda', method='fir')
                self.waves.append(gamma)
                self.wavesTab.append("Gamma")


            for i in range(len(self.waves)):
                self.waveTab = QtWidgets.QWidget()
                self.waveTab.setObjectName(self.wavesTab[i])
                self.tabWidget.addTab(self.waveTab, self.wavesTab[i])
        
                canvas = MplCanvas(self, data=self.waves[i], title=self.wavesTab[i])

                self.waveTab.layout = QtWidgets.QVBoxLayout()
                self.waveTab.layout.addWidget(canvas)
                self.waveTab.setLayout(self.waveTab.layout)

            self.filtred = True
            self.statusbar.showMessage('Ready')


        else:
            self.statusbar.showMessage('Repair data first')

        

    def save(self):
        if self.filtred == True:
            for i in range(len(self.waves)):
                df = self.waves[i].to_data_frame()
                df = df.drop(columns=['condition'], axis=1)                 
                df = df.drop(columns=['time'], axis=1)
                df = df.drop(columns=['epoch'], axis=1)
                df_array = np.asarray(df).astype(np.float32)

                if i == 0:
                    df_array1 = df_array
                elif i==1:
                    tensor = np.dstack((df_array1, df_array))
                else:
                    tensor = np.dstack((tensor, df_array))

            tensor = np.reshape(np.transpose(tensor), (1, 5, len(self.ch_names), len(self.epochs_clean)*self.nSampleEpoch))      #1serie*5waves*4sensors*n_samples

            
            for i in range(len(self.epochs_clean)):                              # epoch differenciation
                epoch = tensor[:,:,:,(self.nSampleEpoch*i):(self.nSampleEpoch*(i+1))]

                if i==0:
                    epoch1 = epoch
                elif i==1:
                    data_tensor = np.concatenate((epoch1, epoch))
                else:
                    data_tensor = np.concatenate((data_tensor, epoch))


            np.save("data_tensor_out", data_tensor) 

            self.statusbar.showMessage('Saved !')

        else:
            self.statusbar.showMessage('Filter data first!')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, data=None, title=None):
        fig = data.plot(title=title, show=False, block=False)
        super(MplCanvas, self).__init__(fig)

    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
