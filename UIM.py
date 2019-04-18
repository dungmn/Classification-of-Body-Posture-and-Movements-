# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\RandomForest_1.0.0.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from functools import partial
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
import matplotlib.pyplot as plt
import pickle
class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        self.delimiter=','
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(799, 497)
        mainWindow.setWindowTitle("")
        mainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 60, 791, 401))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(80, 50, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.txtPathFile = QtWidgets.QPlainTextEdit(self.tab)
        self.txtPathFile.setGeometry(QtCore.QRect(80, 90, 321, 31))
        self.txtPathFile.setObjectName("txtPathFile")
        self.btnExcute = QtWidgets.QPushButton(self.tab)
        self.btnExcute.setGeometry(QtCore.QRect(470, 90, 121, 31))
        self.btnExcute.setObjectName("btnExcute")
        self.btnLoadFile1 = QtWidgets.QPushButton(self.tab)
        self.btnLoadFile1.setGeometry(QtCore.QRect(400, 90, 51, 31))
        self.btnLoadFile1.setObjectName("btnLoadFile1")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 81, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lbStatus = QtWidgets.QLabel(self.tab_2)
        self.lbStatus.setGeometry(QtCore.QRect(200, 10, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.lbStatus.setFont(font)
        self.lbStatus.setAutoFillBackground(False)
        self.lbStatus.setObjectName("lbStatus")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(10, 70, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setGeometry(QtCore.QRect(10, 210, 341, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.txtQuery = QtWidgets.QPlainTextEdit(self.tab_2)
        self.txtQuery.setGeometry(QtCore.QRect(10, 110, 311, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.txtPathFile.setFont(font)
        self.txtQuery.setFont(font)
        self.txtQuery.setObjectName("txtQuery")
        self.txtPathTest = QtWidgets.QTextEdit(self.tab_2)
        self.txtPathTest.setGeometry(QtCore.QRect(13, 250, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.txtPathTest.setFont(font)
        self.txtPathTest.setObjectName("txtPathTest")
        self.btnTest1 = QtWidgets.QPushButton(self.tab_2)
        self.btnTest1.setGeometry(QtCore.QRect(250, 180, 75, 31))
        self.btnTest1.setObjectName("btnTest1")
        self.btnTest1_3 = QtWidgets.QPushButton(self.tab_2)
        self.btnTest1_3.setGeometry(QtCore.QRect(250, 290, 75, 31))
        self.btnTest1_3.setObjectName("btnTest1_3")
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setGeometry(QtCore.QRect(390, 10, 81, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        self.label_8.setGeometry(QtCore.QRect(390, 180, 81, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.btnLoadFile2 = QtWidgets.QPushButton(self.tab_2)
        self.btnLoadFile2.setGeometry(QtCore.QRect(270, 250, 51, 31))
        self.btnLoadFile2.setObjectName("btnLoadFile2")
        font = QtGui.QFont()
        font.setPointSize(10)

        self.txtResult = QtWidgets.QPlainTextEdit(self.tab_2)
        self.txtResult.setFont(font)
        self.txtResult.setGeometry(QtCore.QRect(393, 40, 371, 151))
        self.txtResult.setObjectName("txtResult")
        font = QtGui.QFont()
        font.setPointSize(10)
        self.txtAnalyze = QtWidgets.QPlainTextEdit(self.tab_2)
        self.txtAnalyze.setFont(font)
        self.txtAnalyze.setGeometry(QtCore.QRect(390, 210, 371, 151))
        self.txtAnalyze.setObjectName("txtAnalyze")
        self.tabWidget.addTab(self.tab_2, "")
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 799, 21))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)
        self.click()
        self.retranslateUi(mainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("mainWindow", "Đường dẫn file Trainning"))
        self.btnExcute.setText(_translate("mainWindow", "Thực hiện"))
        self.btnLoadFile1.setText(_translate("mainWindow", "File"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("mainWindow", "Prepare"))
        self.label_2.setText(_translate("mainWindow", "Trạng thái: "))
        self.lbStatus.setText(_translate("mainWindow", "Chưa sẵn sàng"))
        self.label_5.setText(_translate("mainWindow", "Phân lớp cho mẫu tự nhập"))
        self.label_6.setText(_translate("mainWindow", "Phân lớp cho nhiều mẫu (qua tập tin) và đánh giá"))
        self.txtPathTest.setPlaceholderText(_translate("mainWindow", "Đường dẫn tập tin test"))
        self.btnTest1.setText(_translate("mainWindow", "Phân lớp"))
        self.btnTest1_3.setText(_translate("mainWindow", "Phân lớp"))
        self.label_7.setText(_translate("mainWindow", "Kết quả:"))
        self.label_8.setText(_translate("mainWindow", "Đánh giá:"))
        self.btnLoadFile2.setText(_translate("mainWindow", "File"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("mainWindow", "Analyze"))
    def click(self):
        self.btnLoadFile1.clicked.connect(self.File_Selector)
        self.btnExcute.clicked.connect(self.change)
        # self.txtPathFile.toPlainText
        self.btnLoadFile2.clicked.connect(self.File_Selector2)
        self.btnTest1_3.clicked.connect(self.test_file)
        self.btnTest1.clicked.connect(self.test_query)
    def File_Selector(self):
        filenames,_ = QtWidgets.QFileDialog.getOpenFileName(None, "Select File Data", "", "*.csv")
        self.txtPathFile.setPlainText(str(filenames))
    def File_Selector2(self):
        filenames,_ = QtWidgets.QFileDialog.getOpenFileName(None, "Select File Test", "", "*.csv")
        self.txtPathTest.setPlainText(str(filenames))
    def change(self):
        self.tabWidget.setCurrentIndex(1)
        test(self)
    def test_file(self):
        classification_file(self.gl_columns,self.gl_features,self.gl_scaler,self.gl_clf,self.delimiter,self)
    def test_query(self):
        classification_query(self,self.gl_columns,self.gl_features,self.gl_scaler,self.gl_clf)
def load_csv(path,_delimiter):
    try:
        data = pd.read_csv(path,delimiter=_delimiter)
        return data
    except ValueError:
        return None


def normalizeTarget(dataTarget):
    global mydictarget
    mydictarget ={}
    result=[]
    val=1
    for i in dataTarget:
        if i not in mydictarget.keys():
            mydictarget[i]=val
            result.append(val)
            val+=1
        else:
            result.append(mydictarget[i])
    return result
def preprocessing(data):
    numColumns = len(data.columns)
    target = pd.Series(normalizeTarget(data['class']))
    features = pd.get_dummies(data.iloc[:,0:numColumns-1])
    np.savetxt('log.csv',features,delimiter=',')
    scaler = StandardScaler()
    data = scaler.fit_transform(features)
    return data,target,features,scaler
def cross_val(data,target,_cv):
    from sklearn import svm
    clf = svm.SVC(decision_function_shape='ovo',C=400,kernel='rbf')
    scores = cross_val_score(clf, data,target, cv=_cv)
    return scores
def get_train_test(data,target,testsize=0.1):
    x_train,x_test,y_train,y_test = train_test_split(data,target,test_size= testsize)
    return x_train,x_test,y_train,y_test
def excute_svm(data,target,testsize=0.1):
    clf = svm.SVC(decision_function_shape='ovo',C=400,kernel='rbf')
    clf.fit(data,target)
    return clf
def analyze_result(model,datatest,targettest):
    y_test=model.predict(datatest)
    report =classification_report(y_test,targettest)
    reportt=precision_recall_fscore_support(y_test,targettest,average='weighted')
    # acs=accuracy_score(y_test,targettest)
    # print(type(report))
    # print(report)
    # print(acs)
    return reportt,report
def get_number(s):
    try:
        float(s)
        return float(s)
    except ValueError:
        return s
def create_query(query,columns):
    data1 = [[get_number(i) for i in query.split(' ')]]
    return pd.DataFrame(data1, columns=columns)
def preprocessing_test(dataTrain,datatest,scaler):
    print('first',datatest.dtypes)
    datatest = pd.get_dummies(datatest)
    # Get missing columns in the training test
    missing_cols = set( dataTrain.columns ) - set( datatest.columns )
    # Add a missing column in test set with default value equal to 0
    print(datatest)
    for c in missing_cols:
        datatest[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    datatest = datatest[dataTrain.columns]
    print('datattest 1',datatest)
    datatest = scaler.transform(datatest)
    print(datatest)
    return datatest
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    # result = clf.predict(features)
    # print(result)
def classification_query(self,columns,features,scaler,clf):
    query = self.txtQuery.toPlainText()
    begin = time()
    query = create_query(query,columns[:-1])
    datatest=preprocessing_test(features,query,scaler)
    index = clf.predict(datatest).tolist()
    result =''
    for i in index:
        result +=str(list(mydictarget.keys())[i-1])+'\n'
    time2 = "Time:"+str(time()-begin)
    self.txtResult.setPlainText(time2+'\n'+result)
def read_csv_datatest(path,_delimiter):
    data = pd.read_csv(path,delimiter=_delimiter)
    return data
def classification_file(columns,features,scaler,clf,_delimiter,self):
    path = self.txtPathTest.toPlainText()
    print(path)
    print(columns)
    begin = time()
    queries = read_csv_datatest(path,_delimiter)
    datatest=preprocessing_test(features,queries,scaler)
    index = clf.predict(datatest).tolist()
    result = ''
    for i in index:
        result+=str(list(mydictarget.keys())[i-1])+'\n'
    time2="Time:"+str(time()-begin)
    self.txtResult.setPlainText(time2+'\n'+result)

def test_analys(datatest,targettest,clf):
    x_test=[]
    y_test=[]
    x_tr=datatest
    y_tr =targettest
    lentest=[]
    for i in range(1,4):
        x_tr,x_te,y_tr,y_te=get_train_test(x_tr,y_tr,i/10)
        x_test.append(x_te)
        y_test.append(y_te)
        lentest.append(len(x_te))
    x_test.append(x_tr)
    y_test.append(y_tr)
    lentest.append(len(x_test[3]))
    pre =[]
    rec =[]
    fmeasure =[]
    for i in range(0,4):
        report,_ =analyze_result(clf,x_test[i],y_test[i])
        pre.append(report[0])
        rec.append(report[1])
        fmeasure.append(report[2])
    # data to plot
    n_groups = len(pre)
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)+1
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, pre, bar_width,alpha=opacity,color='b',label='Precision')

    rects2 = plt.bar(index + bar_width, rec, bar_width,alpha=opacity,color='g',label='Recall')
    rects3 = plt.bar(index+ 2*bar_width, fmeasure, bar_width,alpha=opacity,color='r',label='Fmeasure')

    plt.xlabel('Chiều dài test')
    plt.ylabel('%')
    plt.title('Các bộ test')
    plt.xticks(index + bar_width, (lentest[0], lentest[1], lentest[2], lentest[3]))
    plt.legend()

    plt.tight_layout()
    plt.show()
def test(self):
    path = self.txtPathFile.toPlainText()
    self.tabWidget.setCurrentIndex(1)
    begin = time()
    if path=='':
        self.lbStatus.setText('File not found')
        return

    data = load_csv(path,self.delimiter)
    columns = data.columns.get_values()
    self.gl_columns  = columns
    data, target,features,scaler = preprocessing(data)
    self.gl_features = features
    self.gl_scaler=scaler
    x_train,x_test,y_train,y_test= get_train_test(data,target,0.1)
    clf = excute_svm(x_train,y_train)
    filename = 'finalized_model_deci.sav'
    # pickle.dump(clf, open(filename, 'wb'))
    # clf = pickle.load(open(filename,'rb'))
    self.gl_clf=clf
    #classification_query('katia;Woman;28;158;55;22;-11;97;-108;11;-12;-14;17;67;80;-154;-93;-148',columns,features,scaler,clf)
    #classification_file(columns,features,scaler,clf,'test.csv',';')
    #test_analys(x_test,y_test,clf)
    #print(mydictarget)
    _,report=analyze_result(clf,x_test,y_test)
    time1 = "Time "+str(round(time()-begin,2))+'\n'
    self.lbStatus.setText('Done! '+time1)
    self.txtAnalyze.insertPlainText(report)
    # print(clf.predict())
    # data2 = np.array((2,18))
    # data2[0,:] = data[0,:-1]
    # data2[1,:] =query
    # print(data2)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
