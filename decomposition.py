# Imports
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os
import sys
import numpy as np
import webbrowser
import pandas as pd
import sklearn
from monkey_pt import Table
from utils import Data_Preview, quit_back, open_file, load_data, save_results

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
    def __init__(self, parent):
        if not hasattr(self.decomposition, 'data'):
            self.decomposition.data = None
            self.decomposition.sheet = tk.StringVar()
            self.decomposition.sheet.set('1')
            self.decomposition.header_var = tk.IntVar()
            self.decomposition.header_var.set(1)
            self.decomposition.x_from_var = tk.StringVar()
            self.decomposition.x_to_var = tk.StringVar()
            self.decomposition.Viewed = tk.BooleanVar()
            self.decomposition.view_frame = None
            self.decomposition.pt = None
        
        w = 600
        h = 350
        
        #setting main window's parameters       
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        dcmp_app.root = tk.Toplevel(parent)
        dcmp_app.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        dcmp_app.root.title('Monkey Decomposition')
        dcmp_app.root.lift()
        dcmp_app.root.tkraise()
        dcmp_app.root.focus_force()
        dcmp_app.root.resizable(False, False)
        dcmp_app.root.protocol("WM_DELETE_WINDOW", lambda: quit_back(dcmp_app.root, parent))

        parent.iconify()
        
        self.frame = ttk.Frame(dcmp_app.root, width=w, height=h)
        self.frame.place(x=0, y=0)

        ttk.Label(self.frame, text='Data file:').place(x=30, y=10)
        e1 = ttk.Entry(self.frame, font=myfont1, width=40)
        e1.place(x=120, y=10)

        ttk.Button(self.frame, text='Choose file', 
            command=lambda: open_file(self, e1)).place(x=490, y=10)

        ttk.Label(self.frame, text='List number:').place(x=120,y=50)
        dcmp_sheet_entry = ttk.Entry(self.frame, 
            textvariable=self.decomposition.sheet, font=myfont1, width=3)
        dcmp_sheet_entry.place(x=215,y=52)

        ttk.Button(self.frame, text='Load data ', 
            command=lambda: load_data(self, self.decomposition, e1, 'dcmp')).place(x=490, y=50)
        
        cb1 = ttk.Checkbutton(self.frame, text="header", 
            variable=self.decomposition.header_var, takefocus=False)
        cb1.place(x=10, y=50)

        ttk.Label(self.frame, text='Data status:').place(x=10, y=95)
        self.decomposition.data_status = ttk.Label(self.frame, text='Not Loaded')
        self.decomposition.data_status.place(x=120, y=95)

        ttk.Button(self.frame, text='View/Change', 
            command=lambda: 
                Data_Preview(self, self.decomposition, 'dcmp', parent)).place(x=230, y=95)

        ttk.Label(self.frame, text='Number of features:', font=myfont1).place(x=30, y=140)
        self.n_features_var = tk.StringVar(value='All')
        self.n_features_entry = ttk.Entry(self.frame, 
            textvariable=self.n_features_var, font=myfont1, width=3)
        self.n_features_entry.place(x=170, y=142)

        ttk.Button(self.frame, text="Methods' specifications", 
            command=lambda: dcmp_mtds_specification(self, parent)).place(x=400, y=100)

        ttk.Label(self.frame, text='X from', font=myfont1).place(x=225, y=130)
        self.dcmp_x_from_combobox = ttk.Combobox(self.frame, 
            textvariable=self.decomposition.x_from_var, width=14, values=[])
        self.dcmp_x_from_combobox.place(x=275, y=132)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=225, y=155)
        self.dcmp_x_to_combobox = ttk.Combobox(self.frame, 
            textvariable=self.decomposition.x_to_var, width=14, values=[])
        self.dcmp_x_to_combobox.place(x=275, y=157)

        self.x_st_var = tk.StringVar(value='If needed')
        ttk.Label(self.frame, text='X Standartization', font=myfont1).place(x=30, y=180)
        self.combobox2 = ttk.Combobox(self.frame, textvariable=self.x_st_var, width=10,
                                        values=['No', 'If needed', 'Yes'])
        self.combobox2.place(x=150,y=185)

        self.dummies_var = tk.IntVar(value=0)
        cb2 = ttk.Checkbutton(self.frame, text="Dummies", 
            variable=self.dummies_var, takefocus=False)
        cb2.place(x=270, y=185)

        ttk.Label(self.frame, text='Choose method', font=myfont1).place(x=30, y=230)
        self.dcmp_method = tk.StringVar(value='PCA')
        self.combobox9 = ttk.Combobox(self.frame, textvariable=self.dcmp_method, width=15, 
            values=['PCA', 'Factor Analysis', 'Incremental PCA', 'Kernel PCA', 'Fast ICA'])
        self.combobox9.place(x=150,y=232)

        self.pca_copy = tk.BooleanVar(value=True)
        self.pca_whiten = tk.BooleanVar(value=False)
        self.pca_svd_solver = tk.StringVar(value='auto')
        self.pca_tol = tk.StringVar(value='0.0')
        self.pca_iterated_power = tk.StringVar(value='auto')
        self.pca_random_state = tk.StringVar(value='None')

        self.fa_tol = tk.StringVar(value='1e-2')
        self.fa_copy = tk.BooleanVar(value=True)
        self.fa_max_iter = tk.StringVar(value='1000')
        self.fa_svd_method = tk.StringVar(value='randomized')
        self.fa_iterated_power = tk.StringVar(value='3')
        self.fa_rotation = tk.StringVar(value='None')
        self.fa_random_state = tk.StringVar(value='0')

        self.ipca_whiten = tk.BooleanVar(value=False)
        self.ipca_copy = tk.BooleanVar(value=True)
        self.ipca_batch_size = tk.StringVar(value='None')

        self.kpca_kernel = tk.StringVar(value='linear')
        self.kpca_gamma = tk.StringVar(value='None')
        self.kpca_degree = tk.StringVar(value='3')
        self.kpca_coef0 = tk.StringVar(value='1.0')
        self.kpca_alpha = tk.StringVar(value='1.0')
        self.kpca_fit_inverse_transform = tk.BooleanVar(value=False)
        self.kpca_eigen_solver = tk.StringVar(value='auto')
        self.kpca_tol = tk.StringVar(value='0.0')
        self.kpca_max_iter = tk.StringVar(value='None')
        self.kpca_remove_zero_eig = tk.BooleanVar(value=False)
        self.kpca_random_state = tk.StringVar(value='None')
        self.kpca_copy_X = tk.BooleanVar(value=True)
        self.kpca_n_jobs = tk.StringVar(value='None')

        self.fica_algorithm = tk.StringVar(value='parallel')
        self.fica_whiten = tk.BooleanVar(value=True)
        self.fica_fun = tk.StringVar(value='logcosh')
        self.fica_max_iter = tk.StringVar(value='200')
        self.fica_tol = tk.StringVar(value='1e-4')
        self.fica_random_state = tk.StringVar(value='None')

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
                pca = PCA(
                    n_components=(None if self.n_features_var.get()=='All' 
                        else int(self.n_features_var.get())),
                    copy=self.pca_copy.get(), whiten=self.pca_whiten.get(), 
                    svd_solver=self.pca_svd_solver.get(),
                    tol=float(self.pca_tol.get()), 
                    iterated_power=('auto' if self.pca_iterated_power.get()=='auto' 
                        else int(self.pca_iterated_power.get())),
                    random_state=(None if self.pca_random_state.get()=='None' 
                        else int(self.pca_random_state.get())))
                if self.x_st_var.get() == 'No':
                    pr_values = pca.fit_transform(X)
                else:
                    pr_values = pca.fit_transform(X_St)
                if self.n_features_var.get()=='All':
                    column_names = ['PC{}'.format(i) for i in range(1, X.shape[1]+1)]
                else:
                    column_names = ['PC{}'.format(i) for i in range(1, 
                        int(self.n_features_var.get())+1)]
            elif method == 'Factor Analysis':
                from sklearn.decomposition import FactorAnalysis
                fa = FactorAnalysis(
                    n_components=(None if self.n_features_var.get()=='All' 
                        else int(self.n_features_var.get())),
                    tol=float(self.fa_tol.get()), copy=self.fa_copy.get(),
                    max_iter=int(self.fa_max_iter.get()),
                    svd_method=self.fa_svd_method.get(),
                    iterated_power=int(self.fa_iterated_power.get()),
                    rotation=(None if self.fa_rotation.get()=='None' 
                        else self.fa_rotation.get()),
                    random_state=int(self.fa_random_state.get())
                    )
                if self.x_st_var.get() == 'No':
                    pr_values = fa.fit_transform(X)
                else:
                    pr_values = fa.fit_transform(X_St)
                if self.n_features_var.get()=='All':
                    column_names = ['Factor{}'.format(i) for i in range(1, X.shape[1]+1)]
                else:
                    column_names = ['Factor{}'.format(i) for i in range(1, 
                        int(self.n_features_var.get())+1)]
            elif method == 'Incremental PCA':
                from sklearn.decomposition import IncrementalPCA
                ipca = IncrementalPCA(
                    n_components=(None if self.n_features_var.get()=='All' 
                        else int(self.n_features_var.get())),
                    whiten=self.ipca_whiten.get(), copy=self.ipca_copy.get(),
                    batch_size=(None if self.ipca_batch_size.get()=='None' 
                        else int(self.ipca_batch_size.get()))
                    )
                if self.x_st_var.get() == 'No':
                    pr_values = ipca.fit_transform(X)
                else:
                    pr_values = ipca.fit_transform(X_St)
                if self.n_features_var.get()=='All':
                    column_names = ['PC{}'.format(i) for i in range(1, X.shape[1]+1)]
                else:
                    column_names = ['PC{}'.format(i) for i in range(1, 
                        int(self.n_features_var.get())+1)]
            elif method == 'Kernel PCA':
                from sklearn.decomposition import KernelPCA
                kpca = KernelPCA(
                    n_components=(None if self.n_features_var.get()=='All' 
                        else int(self.n_features_var.get())),
                    kernel=self.kpca_kernel.get(),
                    gamma=(None if self.kpca_gamma.get()=='None' 
                        else float(self.kpca_gamma.get())),
                    degree=int(self.kpca_degree.get()), coef0=float(self.kpca_coef0.get()),
                    alpha=float(self.kpca_alpha.get()),
                    fit_inverse_transform=self.kpca_fit_inverse_transform.get(),
                    eigen_solver=self.kpca_eigen_solver.get(),
                    tol=float(self.kpca_tol.get()),
                    max_iter=(None if self.kpca_max_iter.get()=='None' 
                        else int(self.kpca_max_iter.get())),
                    remove_zero_eig=self.kpca_remove_zero_eig.get(),
                    random_state=(None if self.kpca_random_state.get()=='None' 
                        else int(self.kpca_random_state.get())),
                    copy_X=self.kpca_copy_X.get(),
                    n_jobs=(None if self.kpca_n_jobs.get()=='None' 
                        else int(self.kpca_n_jobs.get()))
                    )
                if self.x_st_var.get() == 'No':
                    pr_values = kpca.fit_transform(X)
                else:
                    pr_values = kpca.fit_transform(X_St)
                if self.n_features_var.get()=='All':
                    column_names = ['PC{}'.format(i) for i in range(1, X.shape[1]+1)]
                else:
                    column_names = ['PC{}'.format(i) for i in range(1, 
                        int(self.n_features_var.get())+1)]
            elif method == 'Fast ICA':
                from sklearn.decomposition import FastICA
                fica = FastICA(
                    n_components=(None if self.n_features_var.get()=='All' 
                        else int(self.n_features_var.get())),
                    algorithm=self.fica_algorithm.get(),
                    whiten=self.fica_whiten.get(), fun=self.fica_fun.get(),
                    max_iter=int(self.fica_max_iter.get()),
                    tol=float(self.fica_tol.get()),
                    random_state=(None if self.fica_random_state.get()=='None' 
                        else int(self.fica_random_state.get()))
                    )
                if self.x_st_var.get() == 'No':
                    pr_values = fica.fit_transform(X)
                else:
                    pr_values = fica.fit_transform(X_St)
                if self.n_features_var.get()=='All':
                    column_names = ['IC{}'.format(i) for i in range(1, X.shape[1]+1)]
                else:
                    column_names = ['IC{}'.format(i) for i in range(1, 
                        int(self.n_features_var.get())+1)]

            self.decomposition.data = self.decomposition.data.join(pd.DataFrame(pr_values, 
                columns=column_names), lsuffix='_left', rsuffix='_right')

            if self.decomposition.Viewed.get() == True:
                self.decomposition.pt = Table(self.decomposition.view_frame, 
                    dataframe=self.decomposition.data, showtoolbar=True, showstatusbar=True, 
                    height=350, notebook=Data_Preview.notebook.nb, dp_main=self.decomposition)
                self.decomposition.pt.show()
                self.decomposition.pt.redraw()
                Data_Preview.notebook.nb.select(self.decomposition.view_frame)
                Data_Preview.root.lift()

        def try_decompose(method):
            try:
                decompose(method)
            except ValueError as e:
                messagebox.showerror(parent=self.root, message='error: "{}"'.format(e))

        ttk.Button(self.frame, text='Decompose', 
                  command=lambda: try_decompose(method=self.dcmp_method.get())).place(x=400, y=230)

        ttk.Button(self.frame, text='Save results', 
            command=lambda: save_results(self, 
                self.decomposition, 'dcmp result')).place(x=400, y=270)

        ttk.Button(self.frame, text='Quit', 
                  command=lambda: quit_back(dcmp_app.root, parent)).place(x=400, y=310)

class dcmp_mtds_specification:
    def __init__(self, prev, parent):
        self.root = tk.Toplevel(parent)

        w = 450
        h = 520
        #setting main window's parameters       
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.focus_force()
        self.root.resizable(False, False)
        self.root.title('Decomposition methods specification')

        self.canvas = tk.Canvas(self.root)
        self.frame = ttk.Frame(self.canvas, width=690, height=640)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas_frame = self.canvas.create_window(0, 0, window=self.frame, anchor="nw")

        main_menu = tk.Menu(self.root)
        self.root.config(menu=main_menu)
        settings_menu = tk.Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label='Restore Defaults', 
            command=lambda: self.restore_defaults(prev))

        ttk.Label(self.frame, text='PCA', font=myfont_b).place(x=55, y=10)
        ttk.Label(self.frame, text='Copy', font=myfont2).place(x=5, y=40)
        pca_cb1 = ttk.Checkbutton(self.frame, variable=prev.pca_copy, takefocus=False)
        pca_cb1.place(x=110, y=40)
        ttk.Label(self.frame, text='Whiten', font=myfont2).place(x=5, y=60)
        pca_cb2 = ttk.Checkbutton(self.frame, variable=prev.pca_whiten, takefocus=False)
        pca_cb2.place(x=110, y=60)
        ttk.Label(self.frame, text='SVD solver', font=myfont2).place(x=5, y=80)
        pca_combobox1 = ttk.Combobox(self.frame, textvariable=prev.pca_svd_solver, width=8, 
            values=['auto', 'full', 'arpack', 'randomized'])
        pca_combobox1.place(x=100,y=80)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=5, y=100)
        pca_e1 = ttk.Entry(self.frame, textvariable=prev.pca_tol, font=myfont2, width=7)
        pca_e1.place(x=110, y=100)
        ttk.Label(self.frame, text='Iterated power', font=myfont2).place(x=5, y=120)
        pca_e2 = ttk.Entry(self.frame, textvariable=prev.pca_iterated_power, font=myfont2, width=7)
        pca_e2.place(x=110, y=120)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=5, y=140)
        pca_e3 = ttk.Entry(self.frame, textvariable=prev.pca_random_state, font=myfont2, width=7)
        pca_e3.place(x=110, y=140)

        ttk.Label(self.frame, text='Factor Analysis', font=myfont_b).place(x=30, y=170)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=5, y=200)
        fa_e1 = ttk.Entry(self.frame, textvariable=prev.fa_tol, font=myfont2, width=7)
        fa_e1.place(x=110, y=200)
        ttk.Label(self.frame, text='Copy', font=myfont2).place(x=5, y=220)
        fa_cb1 = ttk.Checkbutton(self.frame, variable=prev.fa_copy, takefocus=False)
        fa_cb1.place(x=110, y=220)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=5, y=240)
        fa_e2 = ttk.Entry(self.frame, textvariable=prev.fa_max_iter, font=myfont2, width=7)
        fa_e2.place(x=110, y=240)
        ttk.Label(self.frame, text='SVD method', font=myfont2).place(x=5, y=260)
        fa_combobox1 = ttk.Combobox(self.frame, textvariable=prev.fa_svd_method, width=8, 
            values=['lapack', 'randomized'])
        fa_combobox1.place(x=100,y=260)
        ttk.Label(self.frame, text='Iterated power', font=myfont2).place(x=5, y=280)
        fa_e3 = ttk.Entry(self.frame, textvariable=prev.fa_iterated_power, font=myfont2, width=7)
        fa_e3.place(x=110, y=280)
        ttk.Label(self.frame, text='Rotation', font=myfont2).place(x=5, y=300)
        fa_combobox2 = ttk.Combobox(self.frame, textvariable=prev.fa_rotation, width=8, 
            values=['varimax', 'quartimax'])
        fa_combobox2.place(x=100,y=300)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=5, y=320)
        fa_e4 = ttk.Entry(self.frame, textvariable=prev.fa_random_state, font=myfont2, width=7)
        fa_e4.place(x=110, y=320)

        ttk.Label(self.frame, text='Incremental PCA', font=myfont_b).place(x=20, y=350)
        ttk.Label(self.frame, text='Whiten', font=myfont2).place(x=5, y=380)
        ipca_cb1 = ttk.Checkbutton(self.frame, variable=prev.ipca_whiten, takefocus=False)
        ipca_cb1.place(x=110, y=380)
        ttk.Label(self.frame, text='Copy', font=myfont2).place(x=5, y=400)
        ipca_cb2 = ttk.Checkbutton(self.frame, variable=prev.ipca_copy, takefocus=False)
        ipca_cb2.place(x=110, y=400)
        ttk.Label(self.frame, text='Batch size', font=myfont2).place(x=5, y=420)
        ipca_e1 = ttk.Entry(self.frame, textvariable=prev.ipca_batch_size, font=myfont2, width=7)
        ipca_e1.place(x=110, y=420)

        ttk.Label(self.frame, text='Kernel PCA', font=myfont_b).place(x=205, y=10)
        ttk.Label(self.frame, text='Kernel', font=myfont2).place(x=185, y=40)
        kpca_combobox1 = ttk.Combobox(self.frame, textvariable=prev.kpca_kernel, width=8, 
            values=['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'])
        kpca_combobox1.place(x=280,y=40)
        ttk.Label(self.frame, text='Gamma', font=myfont2).place(x=185, y=60)
        kpca_e1 = ttk.Entry(self.frame, textvariable=prev.kpca_gamma, font=myfont2, width=7)
        kpca_e1.place(x=290, y=60)
        ttk.Label(self.frame, text='Degree', font=myfont2).place(x=185, y=80)
        kpca_e2 = ttk.Entry(self.frame, textvariable=prev.kpca_degree, font=myfont2, width=7)
        kpca_e2.place(x=290, y=80)
        ttk.Label(self.frame, text='coef0', font=myfont2).place(x=185, y=100)
        kpca_e3 = ttk.Entry(self.frame, textvariable=prev.kpca_coef0, font=myfont2, width=7)
        kpca_e3.place(x=290, y=100)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=185, y=120)
        kpca_e4 = ttk.Entry(self.frame, textvariable=prev.kpca_alpha, font=myfont2, width=7)
        kpca_e4.place(x=290, y=120)
        ttk.Label(self.frame, text='Fit inverse\ntransform', font=myfont2).place(x=185, y=140)
        kpca_cb1 = ttk.Checkbutton(self.frame, 
            variable=prev.kpca_fit_inverse_transform, takefocus=False)
        kpca_cb1.place(x=290, y=150)
        ttk.Label(self.frame, text='Eigen solver', font=myfont2).place(x=185, y=180)
        kpca_combobox2 = ttk.Combobox(self.frame, textvariable=prev.kpca_eigen_solver, width=7, 
            values=['auto', 'dense', 'arpack'])
        kpca_combobox2.place(x=285,y=180)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=185, y=200)
        kpca_e5 = ttk.Entry(self.frame, textvariable=prev.kpca_tol, font=myfont2, width=7)
        kpca_e5.place(x=290, y=200)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=185, y=220)
        kpca_e6 = ttk.Entry(self.frame, textvariable=prev.kpca_max_iter, font=myfont2, width=7)
        kpca_e6.place(x=290, y=220)
        ttk.Label(self.frame, text='Remove zero eig', font=myfont2).place(x=185, y=240)
        kpca_cb2 = ttk.Checkbutton(self.frame, 
            variable=prev.kpca_remove_zero_eig, takefocus=False)
        kpca_cb2.place(x=290, y=240)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=185, y=240)
        kpca_e7 = ttk.Entry(self.frame, textvariable=prev.kpca_random_state, font=myfont2, width=7)
        kpca_e7.place(x=290, y=240)
        ttk.Label(self.frame, text='Copy X', font=myfont2).place(x=185, y=260)
        kpca_cb3 = ttk.Checkbutton(self.frame, 
            variable=prev.kpca_copy_X, takefocus=False)
        kpca_cb3.place(x=290, y=260)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=185, y=280)
        kpca_e8 = ttk.Entry(self.frame, textvariable=prev.kpca_n_jobs, font=myfont2, width=7)
        kpca_e8.place(x=290, y=280)

        ttk.Label(self.frame, text='Fast ICA', font=myfont_b).place(x=205, y=310)
        ttk.Label(self.frame, text='Algorithm', font=myfont2).place(x=185, y=340)
        fica_combobox1 = ttk.Combobox(self.frame, textvariable=prev.fica_algorithm, width=8, 
            values=['parallel', 'deflation'])
        fica_combobox1.place(x=280,y=340)
        ttk.Label(self.frame, text='Whiten', font=myfont2).place(x=185, y=360)
        fica_cb1 = ttk.Checkbutton(self.frame, 
            variable=prev.fica_whiten, takefocus=False)
        fica_cb1.place(x=290, y=360)
        ttk.Label(self.frame, text='fun', font=myfont2).place(x=185, y=380)
        fica_combobox2 = ttk.Combobox(self.frame, textvariable=prev.fica_fun, width=7, 
            values=['logcosh', 'exp', 'cube'])
        fica_combobox2.place(x=285,y=380)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=185, y=400)
        fica_e1 = ttk.Entry(self.frame, textvariable=prev.fica_max_iter, font=myfont2, width=7)
        fica_e1.place(x=290, y=400)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=185, y=420)
        fica_e2 = ttk.Entry(self.frame, textvariable=prev.fica_tol, font=myfont2, width=7)
        fica_e2.place(x=290, y=420)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=185, y=440)
        fica_e3 = ttk.Entry(self.frame, textvariable=prev.fica_random_state, font=myfont2, width=7)
        fica_e3.place(x=290, y=440)

        ttk.Button(self.root, text='OK', 
            command=lambda: quit_back(self.root, dcmp_app.root)).place(relx=0.8, rely=0.92)

    def restore_defaults(self, prev):

        prev.pca_copy = tk.BooleanVar(value=True)
        prev.pca_whiten = tk.BooleanVar(value=False)
        prev.pca_svd_solver = tk.StringVar(value='auto')
        prev.pca_tol = tk.StringVar(value='0.0')
        prev.pca_iterated_power = tk.StringVar(value='auto')
        prev.pca_random_state = tk.StringVar(value='None')

        prev.fa_tol = tk.StringVar(value='1e-2')
        prev.fa_copy = tk.BooleanVar(value=True)
        prev.fa_max_iter = tk.StringVar(value='1000')
        prev.fa_svd_method = tk.StringVar(value='randomized')
        prev.fa_iterated_power = tk.StringVar(value='3')
        prev.fa_rotation = tk.StringVar(value='None')
        prev.fa_random_state = tk.StringVar(value='0')

        prev.ipca_whiten = tk.BooleanVar(value=False)
        prev.ipca_copy = tk.BooleanVar(value=True)
        prev.ipca_batch_size = tk.StringVar(value='None')

        prev.kpca_kernel = tk.StringVar(value='linear')
        prev.kpca_gamma = tk.StringVar(value='None')
        prev.kpca_degree = tk.StringVar(value='3')
        prev.kpca_coef0 = tk.StringVar(value='1.0')
        prev.kpca_alpha = tk.StringVar(value='1.0')
        prev.kpca_fit_inverse_transform = tk.BooleanVar(value=False)
        prev.kpca_eigen_solver = tk.StringVar(value='auto')
        prev.kpca_tol = tk.StringVar(value='0.0')
        prev.kpca_max_iter = tk.StringVar(value='None')
        prev.kpca_remove_zero_eig = tk.BooleanVar(value=False)
        prev.kpca_random_state = tk.StringVar(value='None')
        prev.kpca_copy_X = tk.BooleanVar(value=True)
        prev.kpca_n_jobs = tk.StringVar(value='None')

        prev.fica_algorithm = tk.StringVar(value='parallel')
        prev.fica_whiten = tk.BooleanVar(value=True)
        prev.fica_fun = tk.StringVar(value='logcosh')
        prev.fica_max_iter = tk.StringVar(value='200')
        prev.fica_tol = tk.StringVar(value='1e-4')
        prev.fica_random_state = tk.StringVar(value='None')

        quit_back(self.root, prev.root)