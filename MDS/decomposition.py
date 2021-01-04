# Imports
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox
import os
import sys
import numpy as np
import webbrowser
import pandas as pd
import sklearn
from monkey_pt import Table
from .utils import Data_Preview, quit_back, open_file, load_data, save_results

myfont = (None, 13)
myfont_b = (None, 13, 'bold')
myfont1 = (None, 11)
myfont1_b = (None, 11, 'bold')
myfont2 = (None, 10)
myfont2_b = (None, 10, 'bold')

class dcmp_app:
    class decomposition:
        def __init__(self):
            pass
    class prediction:
        def __init__(self):
            pass
    def __init__(self, parent):

        self.decomposition.data = None
        self.decomposition.sheet = tk.StringVar()
        self.decomposition.sheet.set('1')
        self.decomposition.header_var = tk.IntVar()
        self.decomposition.header_var.set(1)
        self.decomposition.x_from_var = tk.StringVar()
        self.decomposition.x_to_var = tk.StringVar()
        self.prediction.data = None
        self.prediction.sheet = tk.StringVar()
        self.prediction.sheet.set('1')
        self.prediction.header_var = tk.IntVar()
        self.prediction.header_var.set(1)
        self.prediction.x_from_var = tk.StringVar()
        self.prediction.x_to_var = tk.StringVar()
        w = 600
        h = 350
        
        #setting main window's parameters       
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        self.root = tk.Toplevel(parent)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.title('Monkey Decomposition')
        self.root.lift()
        self.root.tkraise()
        self.root.focus_force()
        self.root.resizable(False, False)

        parent.iconify()
        
        self.frame = ttk.Frame(self.root, width=w, height=h)
        self.frame.place(x=0, y=0)

        ttk.Label(self.frame, text='Data file:').place(x=30, y=10)
        e1 = ttk.Entry(self.frame, font=myfont1, width=40)
        e1.place(x=120, y=10)

        ttk.Button(self.frame, text='Choose file', command=lambda: open_file(self, e1)).place(x=490, y=10)

        ttk.Label(self.frame, text='List number:').place(x=120,y=50)
        dcmp_sheet_entry = ttk.Entry(self.frame, textvariable=self.decomposition.sheet, font=myfont1, width=3)
        dcmp_sheet_entry.place(x=215,y=52)

        ttk.Button(self.frame, text='Load data ', command=lambda: load_data(self, self.decomposition, e1, 'dcmp')).place(x=490, y=50)
        
        cb1 = ttk.Checkbutton(self.frame, text="header", variable=self.decomposition.header_var, takefocus=False)
        cb1.place(x=10, y=50)

        ttk.Label(self.frame, text='Data status:').place(x=10, y=95)
        self.dcmp_data_status = ttk.Label(self.frame, text='Not Loaded')
        self.dcmp_data_status.place(x=120, y=95)

        ttk.Button(self.frame, text='View/Change', 
                   command=lambda: Data_Preview(self, self.decomposition, 'dcmp', parent)).place(x=230, y=95)

        ttk.Label(self.frame, text='Number of features:', font=myfont1).place(x=30, y=140)
        self.n_features_var = tk.StringVar(value='All')
        self.n_features_entry = ttk.Entry(self.frame, textvariable=self.n_features_var, font=myfont1, width=3)
        self.n_features_entry.place(x=170, y=142)

        ttk.Button(self.frame, text="Methods' specifications", 
                  command=lambda: dcmp_mtds_specification(self, parent)).place(x=400, y=100)

        ttk.Label(self.frame, text='X from', font=myfont1).place(x=225, y=130)
        self.dcmp_x_from_combobox = ttk.Combobox(self.frame, textvariable=self.decomposition.x_from_var, width=14, values=[])
        self.dcmp_x_from_combobox.place(x=275, y=132)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=225, y=155)
        self.dcmp_x_to_combobox = ttk.Combobox(self.frame, textvariable=self.decomposition.x_to_var, width=14, values=[])
        self.dcmp_x_to_combobox.place(x=275, y=157)

        self.x_st_var = tk.StringVar(value='If needed')
        ttk.Label(self.frame, text='X Standartization', font=myfont1).place(x=30, y=180)
        self.combobox2 = ttk.Combobox(self.frame, textvariable=self.x_st_var, width=10,
                                        values=['No', 'If needed', 'Yes'])
        self.combobox2.place(x=150,y=185)

        self.dummies_var = tk.IntVar(value=0)
        cb2 = ttk.Checkbutton(self.frame, text="Dummies", variable=self.dummies_var, takefocus=False)
        cb2.place(x=270, y=185)

        ttk.Label(self.frame, text='Choose method', font=myfont1).place(x=30, y=230)
        self.dcmp_method = tk.StringVar(value='PCA')
        self.combobox9 = ttk.Combobox(self.frame, textvariable=self.dcmp_method, width=15, values=['PCA', 'Factor Analysis',
                                                                                                   'Incremental PCA', 'Kernel PCA',
                                                                                                   'Fast ICA'])
        self.combobox9.place(x=150,y=232)

        def decompose(method):
            try:
                x_from = self.decomposition.data.columns.get_loc(self.decomposition.x_from_var.get())
                x_to = self.decomposition.data.columns.get_loc(self.decomposition.x_to_var.get()) + 1
            except:
                x_from = int(self.decomposition.x_from_var.get())
                x_to = int(self.decomposition.x_to_var.get()) + 1
            if self.dummies_var.get()==0:
                X = self.decomposition.data.iloc[:,x_from : x_to]
            elif self.dummies_var.get()==1:
                X = pd.get_dummies(self.decomposition.data.iloc[:,x_from : x_to])
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            X_St = scaler.fit_transform(X)

            if method == 'PCA':
                from sklearn.decomposition import PCA
                pca = PCA()
                if self.x_st_var.get() == 'No':
                    pr_values = pca.fit_transform(X)
                else:
                    pr_values = pca.fit_transform(X_St)
            elif method == 'Factor Analysis':
                from sklearn.decomposition import FactorAnalysis
                fa = FactorAnalysis()
                if self.x_st_var.get() == 'No':
                    pr_values = fa.fit_transform(X)
                else:
                    pr_values = fa.fit_transform(X_St)
            elif method == 'Incremental PCA':
                from sklearn.decomposition import IncrementalPCA
                ipca = IncrementalPCA()
                if self.x_st_var.get() == 'No':
                    pr_values = ipca.fit_transform(X)
                else:
                    pr_values = ipca.fit_transform(X_St)
            elif method == 'Kernel PCA':
                from sklearn.decomposition import KernelPCA
                kpca = KernelPCA()
                if self.x_st_var.get() == 'No':
                    pr_values = kpca.fit_transform(X)
                else:
                    pr_values = kpca.fit_transform(X_St)
            elif method == 'Fast ICA':
                from sklearn.decomposition import FastICA
                fica = FastICA()
                if self.x_st_var.get() == 'No':
                    pr_values = fica.fit_transform(X)
                else:
                    pr_values = fica.fit_transform(X_St)

            self.decomposition.data = self.decomposition.data.join(pd.DataFrame(pr_values))

        def try_decompose(method):
            try:
                decompose(method)
            except ValueError as e:
                messagebox.showerror(message='error: "{}"'.format(e))

        ttk.Button(self.frame, text='Decompose', 
                  command=lambda: try_decompose(method=self.dcmp_method.get())).place(x=400, y=230)

        ttk.Button(self.frame, text='Save results', 
                  command=lambda: save_results(self, self.decomposition)).place(x=400, y=270)

        ttk.Button(self.frame, text='Quit', 
                  command=lambda: quit_back(self.root, parent)).place(x=400, y=310)