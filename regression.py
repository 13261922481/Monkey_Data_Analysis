# Imports
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from tkinter.filedialog import askopenfilename, asksaveasfilename
from threading import Thread
import os
import sys
import numpy as np
import webbrowser
import pandas as pd
import sklearn
from monkey_pt import Table
from utils import *

# fonts
myfont = (None, 12)
myfont_b = (None, 12, 'bold')
myfont1 = (None, 11)
myfont1_b = (None, 11, 'bold')
myfont2 = (None, 10)
myfont2_b = (None, 10, 'bold')

# App to do regression
class rgr_app:
    # sub-class for training-related data
    class training:
        def __init__(self):
            pass
    # sub-class for prediction-related data
    class prediction:
        def __init__(self):
            pass
    # App to show comparison results
    class Comp_results:
        def __init__(self, prev, parent):
            self.root = tk.Toplevel(parent)
            #setting main window's parameters     
            w = 250
            h = 500  
            x = (parent.ws/2) - (w/2)
            y = (parent.hs/2) - (h/2) - 30
            self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
            self.root.lift()
            self.root.grab_set()
            self.root.focus_force()
            self.root.resizable(False, False)
            self.root.title('Comparison results')

            self.frame = ttk.Frame(self.root, width=w, height=h)
            self.frame.place(x=0, y=0)

            ttk.Label(self.frame, text='The results of regression\n methods comparison', 
                font=myfont).place(x=35,y=10)
            ttk.Label(self.frame, text='Least Squares:', font=myfont1).place(x=10,y=60)
            ttk.Label(self.frame, text='Ridge:', font=myfont1).place(x=10,y=90)
            ttk.Label(self.frame, text='Lasso:', font=myfont1).place(x=10,y=120)
            ttk.Label(self.frame, text='Random Forest:', font=myfont1).place(x=10,y=150)
            ttk.Label(self.frame, text='Support Vector:', font=myfont1).place(x=10,y=180)
            ttk.Label(self.frame, text='SGD:', font=myfont1).place(x=10,y=210)
            ttk.Label(self.frame, text='Nearest Neighbor:', font=myfont1).place(x=10,y=240)
            ttk.Label(self.frame, text='Gaussian Process:', font=myfont1).place(x=10,y=270)
            ttk.Label(self.frame, text='Decision tree:', font=myfont1).place(x=10,y=300)
            ttk.Label(self.frame, text='Multi-layer Perceptron:', font=myfont1).place(x=10,y=330)
            ttk.Label(self.frame, text='XGB:', font=myfont1).place(x=10,y=360)
            ttk.Label(self.frame, text='CatBoost:', font=myfont1).place(x=10,y=390)
            ttk.Label(self.frame, text='LightGBM:', font=myfont1).place(x=10,y=420)

            if 'lr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['lr']), 
                    font=myfont1).place(x=200,y=60)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=60)
            if 'rr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['rr']), 
                    font=myfont1).place(x=200,y=90)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=90)
            if 'lassor' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['lassor']), 
                    font=myfont1).place(x=200,y=120)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=120)
            if 'rfr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['rfr']), 
                    font=myfont1).place(x=200,y=150)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=150)
            if 'svr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['svr']), 
                    font=myfont1).place(x=200,y=180)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=180)
            if 'sgdr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['sgdr']), 
                    font=myfont1).place(x=200,y=210)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=210)
            if 'knr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['knr']), 
                    font=myfont1).place(x=200,y=240)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=240)
            if 'gpr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['gpr']), 
                    font=myfont1).place(x=200,y=270)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=270)
            if 'dtr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['dtr']), 
                    font=myfont1).place(x=200,y=300)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=300)
            if 'mlpr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['mlpr']), 
                    font=myfont1).place(x=200,y=330)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=330)
            if 'xgbr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['xgbr']), 
                    font=myfont1).place(x=200,y=360)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=360)
            if 'cbr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['cbr']), 
                    font=myfont1).place(x=200,y=390)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=390)
            if 'lgbmr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['lgbmr']), 
                    font=myfont1).place(x=200,y=420)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=420)

            ttk.Button(self.frame, text='OK', 
                command=lambda: quit_back(self.root, prev.root)).place(x=90, y=460)

    # App to do grid search for hyperparameters
    class Grid_search:
        def __init__(self, prev, parent):
            self.root = tk.Toplevel(parent)
            #setting main window's parameters      
            w = 400
            h = 400 
            x = (parent.ws/2) - (w/2)
            y = (parent.hs/2) - (h/2) - 30
            self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
            self.root.lift()
            self.root.focus_force()
            self.root.resizable(False, False)
            self.root.title('Grid search')

            self.frame = ttk.Frame(self.root, width=w, height=h)
            self.frame.place(x=0, y=0)

            ttk.Label(self.frame, text='Method', font=myfont1).place(x=30, y=10)

            self.method_to_grid = tk.StringVar(value='Decision Tree')
            self.combobox1 = ttk.Combobox(self.frame, textvariable=self.method_to_grid, width=15, 
                values=['Least squares', 'Ridge', 'Lasso', 'Random Forest', 'Support Vector',
                    'SGD', 'Nearest Neighbor', 'Gaussian Process', 'Decision Tree',
                    'Multi-layer Perceptron', 'XGBoost', 'CatBoost', 'LightGBM'])
            self.combobox1.place(x=110,y=12)

            ttk.Label(self.frame, text='n jobs', font=myfont1).place(x=250, y=10)
            n_jobs_entry = ttk.Entry(self.frame, font=myfont1, width=4)
            n_jobs_entry.place(x=330, y=13)
            n_jobs_entry.insert(0, "-1")

            ttk.Label(self.frame, text='Number of folds:', font=myfont1).place(x=30, y=45)
            e2 = ttk.Entry(self.frame, font=myfont1, width=4)
            e2.place(x=150, y=47)
            e2.insert(0, "5")

            ttk.Label(self.frame, text='Number of repeats:', font=myfont1).place(x=200, y=45)
            rep_entry = ttk.Entry(self.frame, font=myfont1, width=4)
            rep_entry.place(x=330, y=48)
            rep_entry.insert(0, "1")

            ttk.Label(self.frame, text='param_grid dict:', font=myfont1).place(x=30, y=80)
            param_grid_entry = scrolledtext.ScrolledText(self.frame, width=50, height=7)
            param_grid_entry.place(x=20, y=100)

            ttk.Label(self.frame, text='Results:', font=myfont1).place(x=30, y=220)

            results_text = scrolledtext.ScrolledText(self.frame, width=50, height=5)
            results_text.place(x=20, y=240)

            # function to perform grid search
            def do_grid_search(method, main):
                try:
                    try:
                        x_from = main.data.columns.get_loc(main.x_from_var.get())
                        x_to = main.data.columns.get_loc(main.x_to_var.get()) + 1
                    except:
                        x_from = int(main.x_from_var.get())
                        x_to = int(main.x_to_var.get()) + 1
                    if prev.handle_cat_var.get()=='No':
                        try:
                            prev.X = main.data.iloc[:,x_from : x_to]
                            prev.y = main.data[prev.y_var.get()]
                        except:
                            prev.X = main.data.iloc[:,x_from : x_to]
                            prev.y = main.data[int(prev.y_var.get())]
                    elif prev.handle_cat_var.get()=='Dummies':
                        from sklearn.preprocessing import OneHotEncoder
                        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        try:
                            prev.X = main.data.iloc[:,x_from : x_to]
                            onehot_columns = prev.X.select_dtypes(include=['object']).columns
                            onehot_df = prev.X[onehot_columns]
                            onehot_df = pd.DataFrame(enc.fit_transform(onehot_df))
                            df_onehot_drop = prev.X.drop(onehot_columns, axis = 1)
                            prev.X = pd.concat([df_onehot_drop, onehot_df], axis = 1)

                            prev.y = main.data[prev.y_var.get()]
                        except:
                            prev.X = main.data.iloc[:,x_from : x_to]
                            onehot_columns = prev.X.select_dtypes(include=['object']).columns
                            onehot_df = prev.X[onehot_columns]
                            onehot_df = pd.DataFrame(enc.fit_transform(onehot_df))
                            df_onehot_drop = prev.X.drop(onehot_columns, axis = 1)
                            prev.X = pd.concat([df_onehot_drop, onehot_df], axis = 1)

                            prev.y = main.data[int(prev.y_var.get())]
                    elif prev.handle_cat_var.get()=='One Hot Encoding':
                        from sklearn.preprocessing import OneHotEncoder
                        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        try:
                            prev.X = pd.DataFrame(enc.fit_transform(main.data.iloc[:,x_from : x_to]))
                            prev.y = main.data[prev.y_var.get()]
                        except:
                            prev.X = pd.DataFrame(enc.fit_transform(main.data.iloc[:,x_from : x_to]))
                            prev.y = main.data[int(prev.y_var.get())]
                    elif prev.handle_cat_var.get()=='Ordinal Encoding':
                        from sklearn.preprocessing import OrdinalEncoder
                        enc = OrdinalEncoder(handle_unknown='ignore')
                        try:
                            prev.X = pd.DataFrame(enc.fit_transform(main.data.iloc[:,x_from : x_to]))
                            prev.y = main.data[prev.y_var.get()]
                        except:
                            prev.X = pd.DataFrame(enc.fit_transform(main.data.iloc[:,x_from : x_to]))
                            prev.y = main.data[int(prev.y_var.get())]
                    from sklearn import preprocessing
                    scaler = preprocessing.StandardScaler()
                    prev.X_St = scaler.fit_transform(prev.X)
                    from sklearn.model_selection import GridSearchCV, RepeatedKFold
                    folds = RepeatedKFold(n_splits = int(e2.get()), 
                        n_repeats=int(rep_entry.get()), random_state = None)
                    if method == 'Least squares':
                        from sklearn.linear_model import LinearRegression
                        gc = GridSearchCV(estimator=LinearRegression(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'Ridge':
                        from sklearn.linear_model import Ridge
                        gc = GridSearchCV(estimator=Ridge(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'Lasso':
                        from sklearn.linear_model import Lasso
                        gc = GridSearchCV(estimator=Lasso(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'Random Forest':
                        from sklearn.ensemble import RandomForestRegressor
                        gc = GridSearchCV(estimator=RandomForestRegressor(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'Support Vector':
                        from sklearn.svm import SVR
                        gc = GridSearchCV(estimator=SVR(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'SGD':
                        from sklearn.linear_model import SGDRegressor
                        gc = GridSearchCV(estimator=SGDRegressor(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'Nearest Neighbor':
                        from sklearn.neighbors import KNeighborsRegressor
                        gc = GridSearchCV(estimator=KNeighborsRegressor(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'Gaussian Process':
                        from sklearn.gaussian_process import GaussianProcessRegressor
                        gc = GridSearchCV(estimator=GaussianProcessRegressor(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'Decision Tree':
                        from sklearn.tree import DecisionTreeRegressor
                        gc = GridSearchCV(estimator=DecisionTreeRegressor(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'Multi-layer Perceptron':
                        from sklearn.neural_network import MLPRegressor
                        gc = GridSearchCV(estimator=MLPRegressor(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'XGBoost':
                        from xgboost import XGBRegressor
                        gc = GridSearchCV(estimator=XGBRegressor(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'CatBoost':
                        from catboost import CatBoostRegressor
                        gc = GridSearchCV(estimator=CatBoostRegressor(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'LightGBM':
                        from lightgbm import LGBMRegressor
                        gc = GridSearchCV(estimator=LGBMRegressor(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()), verbose=3)
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    best_params = gc.best_params_
                    results_text.insert('1.0', str(best_params))
                except ValueError as e:
                    messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))

                self.pb1.destroy()

            # function to initialize grid search in a particular thread
            def try_grid_search(method, main):
                try:
                    if hasattr(self, 'pb1'):
                        self.pb1.destroy()
                    results_text.delete("1.0",'end')
                    self.pb1 = ttk.Progressbar(self.frame, mode='indeterminate', length=150)
                    self.pb1.place(x=130, y=330)
                    self.pb1.start(10)

                    thread1 = Thread(target=do_grid_search, args=(method, main))
                    thread1.daemon = True
                    thread1.start()
                except ValueError as e:
                    self.pb1.destroy()
                    messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))

            ttk.Button(self.frame, text='Search', 
                command=lambda: try_grid_search(method=self.method_to_grid.get(), 
                main=prev.training)).place(x=30, y=360)
            ttk.Button(self.frame, text='Close', 
                command=lambda: quit_back(self.root, prev.root)).place(x=250, y=360)
    # Running regression app itself
    def __init__(self, parent):
        # initialize training and prediction data
        if not hasattr(self.training, 'data'):
            self.training.data = None
            self.training.sheet = tk.StringVar()
            self.training.sheet.set('1')
            self.training.header_var = tk.IntVar()
            self.training.header_var.set(1)
            self.training.x_from_var = tk.StringVar()
            self.training.x_to_var = tk.StringVar()
            self.training.Viewed = tk.BooleanVar()
            self.training.view_frame = None
            self.training.pt = None

        if not hasattr(self.prediction, 'data'):
            self.prediction.data = None
            self.prediction.sheet = tk.StringVar()
            self.prediction.sheet.set('1')
            self.prediction.header_var = tk.IntVar()
            self.prediction.header_var.set(1)
            self.prediction.x_from_var = tk.StringVar()
            self.prediction.x_to_var = tk.StringVar()
            self.prediction.Viewed = tk.BooleanVar()
            self.prediction.view_frame = None
            self.prediction.pt = None
        
        #setting main window's parameters   
        w = 620
        h = 500    
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        rgr_app.root = tk.Toplevel(parent)
        rgr_app.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        rgr_app.root.title('Monkey Regression')
        rgr_app.root.lift()
        rgr_app.root.tkraise()
        rgr_app.root.focus_force()
        rgr_app.root.resizable(False, False)
        rgr_app.root.protocol("WM_DELETE_WINDOW", lambda: quit_back(rgr_app.root, parent))

        parent.iconify()

        #methods parameters
        self.inc_scoring = tk.StringVar(value='r2')
        self.lr_include_comp = tk.BooleanVar(value=True)
        self.lr_function_type = tk.StringVar(value='Linear')
        self.lr_fit_intercept = tk.BooleanVar(value=True)
        self.lr_normalize = tk.BooleanVar(value=False)
        self.lr_copy_X = tk.BooleanVar(value=True)
        self.lr_n_jobs = tk.StringVar(value='None')
        self.lr_positive = tk.BooleanVar(value=False)
        
        self.rr_include_comp = tk.BooleanVar(value=True)
        self.rr_alpha = tk.StringVar(value='1.0')
        self.rr_fit_intercept = tk.BooleanVar(value=True)
        self.rr_normalize = tk.BooleanVar(value=False)
        self.rr_copy_X = tk.BooleanVar(value=True)
        self.rr_max_iter = tk.StringVar(value='None')
        self.rr_tol = tk.StringVar(value='1e-3')
        self.rr_solver = tk.StringVar(value='auto')
        self.rr_random_state = tk.StringVar(value='None')
        
        self.lassor_include_comp = tk.BooleanVar(value=True)
        self.lassor_alpha = tk.StringVar(value='1.0')
        self.lassor_fit_intercept = tk.BooleanVar(value=True)
        self.lassor_normalize = tk.BooleanVar(value=False)
        self.lassor_precompute = tk.BooleanVar(value=False)
        self.lassor_copy_X = tk.BooleanVar(value=True)
        self.lassor_max_iter = tk.StringVar(value='1000')
        self.lassor_tol = tk.StringVar(value='1e-4')
        self.lassor_warm_start = tk.BooleanVar(value=False)
        self.lassor_positive = tk.BooleanVar(value=False)
        self.lassor_random_state = tk.StringVar(value='None')
        self.lassor_selection = tk.StringVar(value='cyclic')
        
        self.rfr_include_comp = tk.BooleanVar(value=True)
        self.rfr_n_estimators = tk.StringVar(value='100')
        self.rfr_criterion = tk.StringVar(value='mse')
        self.rfr_max_depth = tk.StringVar(value='None')
        self.rfr_min_samples_split = tk.StringVar(value='2')
        self.rfr_min_samples_leaf = tk.StringVar(value='1')
        self.rfr_min_weight_fraction_leaf = tk.StringVar(value='0.0')
        self.rfr_max_features = tk.StringVar(value='auto')
        self.rfr_max_leaf_nodes = tk.StringVar(value='None')
        self.rfr_min_impurity_decrease = tk.StringVar(value='0.0')
        self.rfr_bootstrap = tk.BooleanVar(value=True)
        self.rfr_oob_score = tk.BooleanVar(value=False)
        self.rfr_n_jobs = tk.StringVar(value='3')
        self.rfr_random_state = tk.StringVar(value='None')
        self.rfr_verbose = tk.StringVar(value='0')
        self.rfr_warm_start = tk.BooleanVar(value=False)
        self.rfr_ccp_alpha = tk.StringVar(value='0.0')
        self.rfr_max_samples = tk.StringVar(value='None')
        
        self.svr_include_comp = tk.BooleanVar(value=True)
        self.svr_kernel = tk.StringVar(value='rbf')
        self.svr_degree = tk.StringVar(value='3')
        self.svr_gamma = tk.StringVar(value='scale')
        self.svr_coef0 = tk.StringVar(value='0.0')
        self.svr_tol = tk.StringVar(value='1e-3')
        self.svr_C = tk.StringVar(value='1.0')
        self.svr_epsilon = tk.StringVar(value='0.1')
        self.svr_shrinking = tk.BooleanVar(value=True)
        self.svr_cache_size = tk.StringVar(value='200')
        self.svr_verbose = tk.BooleanVar(value=False)
        self.svr_max_iter = tk.StringVar(value='-1')
        
        self.sgdr_include_comp = tk.BooleanVar(value=True)
        self.sgdr_loss = tk.StringVar(value='squared_loss')
        self.sgdr_penalty = tk.StringVar(value='l2')
        self.sgdr_alpha = tk.StringVar(value='0.0001')
        self.sgdr_l1_ratio = tk.StringVar(value='0.15')
        self.sgdr_fit_intercept = tk.BooleanVar(value=True)
        self.sgdr_max_iter = tk.StringVar(value='1000')
        self.sgdr_tol = tk.StringVar(value='1e-3')
        self.sgdr_shuffle = tk.BooleanVar(value=True)
        self.sgdr_verbose = tk.StringVar(value='0')
        self.sgdr_epsilon = tk.StringVar(value='0.1')
        self.sgdr_random_state = tk.StringVar(value='None')
        self.sgdr_learning_rate = tk.StringVar(value='invscaling')
        self.sgdr_eta0 = tk.StringVar(value='0.01')
        self.sgdr_power_t = tk.StringVar(value='0.25')
        self.sgdr_early_stopping = tk.BooleanVar(value=False)
        self.sgdr_validation_fraction = tk.StringVar(value='0.1')
        self.sgdr_n_iter_no_change = tk.StringVar(value='5')
        self.sgdr_warm_start = tk.BooleanVar(value=False)
        self.sgdr_average = tk.StringVar(value='False')
        
        self.knr_include_comp = tk.BooleanVar(value=True)
        self.knr_n_neighbors = tk.StringVar(value='5')
        self.knr_weights = tk.StringVar(value='uniform')
        self.knr_algorithm = tk.StringVar(value='auto')
        self.knr_leaf_size = tk.StringVar(value='30')
        self.knr_p = tk.StringVar(value='2')
        self.knr_metric = tk.StringVar(value='minkowski')
        self.knr_n_jobs = tk.StringVar(value='None')
        
        self.gpr_include_comp = tk.BooleanVar(value=True)
        self.gpr_alpha = tk.StringVar(value='1e-10')
        self.gpr_n_restarts_optimizer = tk.StringVar(value='0')
        self.gpr_normalize_y = tk.BooleanVar(value=True)
        self.gpr_copy_X_train = tk.BooleanVar(value=True)
        self.gpr_random_state = tk.StringVar(value='None')
        
        self.dtr_include_comp = tk.BooleanVar(value=True)
        self.dtr_criterion = tk.StringVar(value='mse')
        self.dtr_splitter = tk.StringVar(value='best')
        self.dtr_max_depth = tk.StringVar(value='None')
        self.dtr_min_samples_split = tk.StringVar(value='2')
        self.dtr_min_samples_leaf = tk.StringVar(value='1')
        self.dtr_min_weight_fraction_leaf = tk.StringVar(value='0.0')
        self.dtr_max_features = tk.StringVar(value='None')
        self.dtr_random_state = tk.StringVar(value='None')
        self.dtr_max_leaf_nodes = tk.StringVar(value='None')
        self.dtr_min_impurity_decrease = tk.StringVar(value='0.0')
        self.dtr_ccp_alpha = tk.StringVar(value='0.0')
        
        self.mlpr_include_comp = tk.BooleanVar(value=True)
        self.mlpr_hidden_layer_sizes = tk.StringVar(value='(100,)')
        self.mlpr_activation = tk.StringVar(value='relu')
        self.mlpr_solver = tk.StringVar(value='adam')
        self.mlpr_alpha = tk.StringVar(value='0.0001')
        self.mlpr_batch_size = tk.StringVar(value='auto')
        self.mlpr_learning_rate = tk.StringVar(value='constant')
        self.mlpr_learning_rate_init = tk.StringVar(value='0.001')
        self.mlpr_power_t = tk.StringVar(value='0.5')
        self.mlpr_max_iter = tk.StringVar(value='200')
        self.mlpr_shuffle = tk.BooleanVar(value=True)
        self.mlpr_random_state = tk.StringVar(value='None')
        self.mlpr_tol = tk.StringVar(value='1e-4')
        self.mlpr_verbose = tk.BooleanVar(value=False)
        self.mlpr_warm_start = tk.BooleanVar(value=False)
        self.mlpr_momentum = tk.StringVar(value='0.9')
        self.mlpr_nesterovs_momentum = tk.BooleanVar(value=True)
        self.mlpr_early_stopping = tk.BooleanVar(value=False)
        self.mlpr_validation_fraction = tk.StringVar(value='0.1')
        self.mlpr_beta_1 = tk.StringVar(value='0.9')
        self.mlpr_beta_2 = tk.StringVar(value='0.999')
        self.mlpr_epsilon = tk.StringVar(value='1e-8')
        self.mlpr_n_iter_no_change = tk.StringVar(value='10')
        self.mlpr_max_fun = tk.StringVar(value='15000')

        self.xgbr_include_comp = tk.BooleanVar(value=True)
        self.xgbr_n_estimators = tk.StringVar(value='1000')
        self.xgbr_eta = tk.StringVar(value='0.1')
        self.xgbr_min_child_weight = tk.StringVar(value='1')
        self.xgbr_max_depth = tk.StringVar(value='6')
        self.xgbr_gamma = tk.StringVar(value='1')
        self.xgbr_subsample = tk.StringVar(value='1.0')
        self.xgbr_colsample_bytree = tk.StringVar(value='1.0')
        self.xgbr_lambda = tk.StringVar(value='1.0')
        self.xgbr_alpha = tk.StringVar(value='0.0')
        self.xgbr_use_gpu = tk.BooleanVar(value=False)
        self.xgbr_eval_metric = tk.StringVar(value='rmse')
        self.xgbr_early_stopping = tk.BooleanVar(value=False)
        self.xgbr_n_iter_no_change = tk.StringVar(value='100')
        self.xgbr_validation_fraction = tk.StringVar(value='0.2')
        self.xgbr_cv_voting_mode = tk.BooleanVar(value=False)
        self.xgbr_cv_voting_folds = tk.StringVar(value='5')

        self.cbr_include_comp = tk.BooleanVar(value=True)
        self.cbr_eval_metric = tk.StringVar(value='RMSE')
        self.cbr_iterations = tk.StringVar(value='1000')
        self.cbr_learning_rate = tk.StringVar(value='None')
        self.cbr_depth = tk.StringVar(value='6')
        self.cbr_reg_lambda = tk.StringVar(value='None')
        self.cbr_subsample = tk.StringVar(value='None')
        self.cbr_colsample_bylevel = tk.StringVar(value='1.0')
        self.cbr_random_strength = tk.StringVar(value='1.0')
        self.cbr_use_gpu = tk.BooleanVar(value=False)
        self.cbr_cf_list = tk.StringVar(value='[]')
        self.cbr_early_stopping = tk.BooleanVar(value=False)
        self.cbr_n_iter_no_change = tk.StringVar(value='100')
        self.cbr_validation_fraction = tk.StringVar(value='0.2')
        self.cbr_cv_voting_mode = tk.BooleanVar(value=False)
        self.cbr_cv_voting_folds = tk.StringVar(value='5')

        self.lgbmr_include_comp = tk.BooleanVar(value=True)
        self.lgbmr_boosting_type = tk.StringVar(value='gbdt')
        self.lgbmr_num_leaves = tk.StringVar(value='31')
        self.lgbmr_max_depth = tk.StringVar(value='-1')
        self.lgbmr_learning_rate = tk.StringVar(value='0.1')
        self.lgbmr_n_estimators = tk.StringVar(value='100')
        self.lgbmr_subsample_for_bin = tk.StringVar(value='200000')
        self.lgbmr_min_split_gain = tk.StringVar(value='0.0')
        self.lgbmr_min_child_weight = tk.StringVar(value='1e-3')
        self.lgbmr_min_child_samples = tk.StringVar(value='20')
        self.lgbmr_subsample = tk.StringVar(value='1.0')
        self.lgbmr_subsample_freq = tk.StringVar(value='0')
        self.lgbmr_colsample_bytree = tk.StringVar(value='1.0')
        self.lgbmr_reg_alpha = tk.StringVar(value='0.0')
        self.lgbmr_reg_lambda = tk.StringVar(value='0.0')
        self.lgbmr_n_jobs = tk.StringVar(value='-1')
        self.lgbmr_silent = tk.BooleanVar(value=True)
        self.lgbmr_categorical_feature = tk.StringVar(value='auto')
        self.lgbmr_eval_metric = tk.StringVar(value='rmse')
        self.lgbmr_early_stopping = tk.BooleanVar(value=False)
        self.lgbmr_n_iter_no_change = tk.StringVar(value='100')
        self.lgbmr_validation_fraction = tk.StringVar(value='0.2')
        self.lgbmr_cv_voting_mode = tk.BooleanVar(value=False)
        self.lgbmr_cv_voting_folds = tk.StringVar(value='5')

        # flag for stopping comparison
        self.continue_flag = True

        # function to compare regression methods
        def compare_methods(prev, main):
            try:
                try:
                    x_from = main.data.columns.get_loc(main.x_from_var.get())
                    x_to = main.data.columns.get_loc(main.x_to_var.get()) + 1
                except:
                    x_from = int(main.x_from_var.get())
                    x_to = int(main.x_to_var.get()) + 1
                if self.handle_cat_var.get()=='No':
                    try:
                        prev.X = main.data.iloc[:,x_from : x_to]
                        prev.y = main.data[self.y_var.get()]
                    except:
                        prev.X = main.data.iloc[:,x_from : x_to]
                        prev.y = main.data[int(self.y_var.get())]

                    prev.X[prev.X.select_dtypes(['object']).columns] = \
                        prev.X.select_dtypes(['object']).apply(lambda x: x.astype('category'))
                elif self.handle_cat_var.get()=='Dummies':
                    from sklearn.preprocessing import OneHotEncoder
                    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    try:
                        prev.X = main.data.iloc[:,x_from : x_to]
                        onehot_columns = prev.X.select_dtypes(include=['object']).columns
                        onehot_df = prev.X[onehot_columns]
                        onehot_df = pd.DataFrame(enc.fit_transform(onehot_df))
                        df_onehot_drop = prev.X.drop(onehot_columns, axis = 1)
                        prev.X = pd.concat([df_onehot_drop, onehot_df], axis = 1)

                        prev.y = main.data[self.y_var.get()]
                    except:
                        prev.X = main.data.iloc[:,x_from : x_to]
                        onehot_columns = prev.X.select_dtypes(include=['object']).columns
                        onehot_df = prev.X[onehot_columns]
                        onehot_df = pd.DataFrame(enc.fit_transform(onehot_df))
                        df_onehot_drop = prev.X.drop(onehot_columns, axis = 1)
                        prev.X = pd.concat([df_onehot_drop, onehot_df], axis = 1)
                        
                        prev.y = main.data[int(self.y_var.get())]
                elif self.handle_cat_var.get()=='One Hot Encoding':
                    from sklearn.preprocessing import OneHotEncoder
                    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    try:
                        prev.X = pd.DataFrame(enc.fit_transform(main.data.iloc[:,x_from : x_to]))
                        prev.y = main.data[self.y_var.get()]
                    except:
                        prev.X = pd.DataFrame(enc.fit_transform(main.data.iloc[:,x_from : x_to]))
                        prev.y = main.data[int(self.y_var.get())]
                elif self.handle_cat_var.get()=='Ordinal Encoding':
                    from sklearn.preprocessing import OrdinalEncoder
                    enc = OrdinalEncoder(handle_unknown='ignore')
                    try:
                        prev.X = pd.DataFrame(enc.fit_transform(main.data.iloc[:,x_from : x_to]))
                        prev.y = main.data[self.y_var.get()]
                    except:
                        prev.X = pd.DataFrame(enc.fit_transform(main.data.iloc[:,x_from : x_to]))
                        prev.y = main.data[int(self.y_var.get())]
                from sklearn import preprocessing
                scaler = preprocessing.StandardScaler()
                try:
                    prev.X_St = scaler.fit_transform(prev.X)
                except:
                    self.x_st_var.set('No')
                self.scores = {}
                from sklearn.model_selection import cross_val_score
                from sklearn.model_selection import KFold, RepeatedKFold
                folds_list = []
                for i in range(int(rep_entry.get())):
                    folds_list.append(KFold(n_splits = int(e2.get()), shuffle=True))
                from sklearn.model_selection import train_test_split
                self.pb1.step(1)
                # Linear regression
                if self.lr_include_comp.get() and self.continue_flag:
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression(fit_intercept=self.lr_fit_intercept.get(), 
                        normalize=self.lr_normalize.get(),
                        copy_X=self.lr_copy_X.get(), 
                        n_jobs=(int(self.lr_n_jobs.get()) 
                            if (self.lr_n_jobs.get() != 'None') else None))
                    lr_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        if self.lr_function_type.get() == 'Linear':
                            for fold in folds_list:
                                for train_index, test_index in fold.split(prev.X_St):
                                    if self.continue_flag:
                                        lr.fit(prev.X_St[train_index], prev.y[train_index])
                                        score = sklearn.metrics.mean_squared_error(lr.predict(prev.X_St[test_index]), 
                                            prev.y[test_index])
                                        lr_scores = np.append(lr_scores, score)
                                        self.pb1.step(1)
                                        print('LR Done')
                        elif self.lr_function_type.get() == 'Polynomial':
                            from sklearn.preprocessing import PolynomialFeatures
                            for fold in folds_list:
                                if self.continue_flag:
                                    lr_scores = np.append(lr_scores, 
                                        cross_val_score(lr, 
                                            PolynomialFeatures(degree=2).fit_transform(prev.X_St), 
                                        prev.y, scoring=self.inc_scoring.get(), cv=fold))
                                    self.pb1.step(1)
                        elif self.lr_function_type.get() == 'Exponential':
                            for fold in folds_list:
                                if self.continue_flag:
                                    lr_scores = np.append(lr_scores, 
                                        cross_val_score(lr, prev.X_St, 
                                            np.log(prev.y), scoring=self.inc_scoring.get(), cv=fold))
                                    self.pb1.step(1)
                        elif self.lr_function_type.get() == 'Power':
                            for fold in folds_list:
                                if self.continue_flag:
                                    lr_scores = np.append(lr_scores, 
                                        cross_val_score(lr, np.log(prev.X_St), 
                                            np.log(prev.y), scoring=self.inc_scoring.get(), cv=fold))
                                    self.pb1.step(1)
                        elif self.lr_function_type.get() == 'Logarithmic':
                            for fold in folds_list:
                                if self.continue_flag:
                                    lr_scores = np.append(lr_scores, 
                                        cross_val_score(lr, np.log(prev.X_St), 
                                            prev.y, scoring=self.inc_scoring.get(), cv=fold))
                                    self.pb1.step(1)
                    else:
                        if self.lr_function_type.get() == 'Linear':
                            for fold in folds_list:
                                for train_index, test_index in fold.split(prev.X):
                                    if self.continue_flag:
                                        lr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                        score = sklearn.metrics.mean_squared_error(lr.predict(prev.X.iloc[test_index]), 
                                            prev.y[test_index])
                                        lr_scores = np.append(lr_scores, score)
                                        self.pb1.step(1)
                                        print('LR Done')
                        elif self.lr_function_type.get() == 'Polynomial':
                            from sklearn.preprocessing import PolynomialFeatures
                            for fold in folds_list:
                                if self.continue_flag:
                                    lr_scores = np.append(lr_scores, 
                                        cross_val_score(lr, 
                                            PolynomialFeatures(degree=2).fit_transform(prev.X), 
                                        prev.y, scoring=self.inc_scoring.get(), cv=fold))
                                    self.pb1.step(1)
                        elif self.lr_function_type.get() == 'Exponential':
                            for fold in folds_list:
                                if self.continue_flag:
                                    lr_scores = np.append(lr_scores, 
                                        cross_val_score(lr, prev.X, np.log(prev.y), 
                                            scoring=self.inc_scoring.get(), cv=fold))
                                    self.pb1.step(1)
                        elif self.lr_function_type.get() == 'Power':
                            for fold in folds_list:
                                if self.continue_flag:
                                    lr_scores = np.append(lr_scores, 
                                        cross_val_score(lr, np.log(prev.X), 
                                            np.log(prev.y), scoring=self.inc_scoring.get(), cv=fold))
                                    self.pb1.step(1)
                        elif self.lr_function_type.get() == 'Logarithmic':
                            for fold in folds_list:
                                if self.continue_flag:
                                    lr_scores = np.append(lr_scores, 
                                        cross_val_score(lr, np.log(prev.X), 
                                            prev.y, scoring=self.inc_scoring.get(), cv=fold))
                                    self.pb1.step(1)
                    self.scores['lr'] = lr_scores.mean()
                # Ridge regression
                if self.rr_include_comp.get() and self.continue_flag:
                    from sklearn.linear_model import Ridge
                    rr = Ridge(
                        alpha=float(self.rr_alpha.get()), 
                        fit_intercept=self.rr_fit_intercept.get(), 
                        normalize=self.rr_normalize.get(),
                        copy_X=self.rr_copy_X.get(), 
                        max_iter=(int(self.rr_max_iter.get()) 
                            if (self.rr_max_iter.get() != 'None') else None),
                        tol=float(self.rr_tol.get()), solver=self.rr_solver.get(),
                        random_state=(int(self.rr_random_state.get()) 
                            if (self.rr_random_state.get() != 'None') else None))
                    rr_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    rr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(rr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    rr_scores = np.append(rr_scores, score)
                                    self.pb1.step(1)
                                    print('Ridge Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    rr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(rr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    rr_scores = np.append(rr_scores, score)
                                    self.pb1.step(1)
                                    print('Ridge Done')
                    self.scores['rr'] = rr_scores.mean()
                # Lasso regression
                if self.lassor_include_comp.get() and self.continue_flag:
                    from sklearn.linear_model import Lasso
                    lassor = Lasso(
                        alpha=float(self.lassor_alpha.get()), 
                        fit_intercept=self.lassor_fit_intercept.get(),
                        normalize=self.lassor_normalize.get(), 
                        precompute=self.lassor_precompute.get(), 
                        copy_X=self.lassor_copy_X.get(), 
                        max_iter=(int(self.lassor_max_iter.get()) 
                            if (self.lassor_max_iter.get() != 'None') else None),
                        tol=float(self.lassor_tol.get()), warm_start=self.lassor_warm_start.get(),
                        positive=self.lassor_positive.get(),
                        random_state=(int(self.lassor_random_state.get()) 
                            if (self.lassor_random_state.get() != 'None') else None),
                        selection=self.lassor_selection.get())
                    lassor_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    lassor.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(lassor.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    lassor_scores = np.append(lassor_scores, score)
                                    self.pb1.step(1)
                                    print('Lasso Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    lassor.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(lassor.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    lassor_scores = np.append(lassor_scores, score)
                                    self.pb1.step(1)
                                    print('Lasso Done')
                    self.scores['lassor'] = lassor_scores.mean()
                # Decision Tree regression
                if self.dtr_include_comp.get() and self.continue_flag:
                    from sklearn.tree import DecisionTreeRegressor
                    dtr = DecisionTreeRegressor(
                        criterion=self.dtr_criterion.get(), splitter=self.dtr_splitter.get(), 
                        max_depth=(int(self.dtr_max_depth.get()) 
                            if (self.dtr_max_depth.get() != 'None') else None), 
                        min_samples_split=(float(self.dtr_min_samples_split.get()) 
                            if '.' in self.dtr_min_samples_split.get()
                            else int(self.dtr_min_samples_split.get())), 
                        min_samples_leaf=(float(self.dtr_min_samples_leaf.get()) 
                            if '.' in self.dtr_min_samples_leaf.get()
                            else int(self.dtr_min_samples_leaf.get())),
                        min_weight_fraction_leaf=float(self.dtr_min_weight_fraction_leaf.get()),
                        max_features=(float(self.dtr_max_features.get()) 
                            if '.' in self.dtr_max_features.get() 
                            else int(self.dtr_max_features.get()) 
                            if len(self.dtr_max_features.get()) < 4 
                            else self.dtr_max_features.get() 
                            if (self.dtr_max_features.get() != 'None') 
                            else None), 
                        random_state=(int(self.dtr_random_state.get()) 
                            if (self.dtr_random_state.get() != 'None') else None),
                        max_leaf_nodes=(int(self.dtr_max_leaf_nodes.get()) 
                            if (self.dtr_max_leaf_nodes.get() != 'None') else None),
                        min_impurity_decrease=float(self.dtr_min_impurity_decrease.get()),
                        ccp_alpha=float(self.dtr_ccp_alpha.get()))
                    dtr_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    dtr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(dtr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    dtr_scores = np.append(dtr_scores, score)
                                    self.pb1.step(1)
                                    print('DT Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    dtr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(dtr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    dtr_scores = np.append(dtr_scores, score)
                                    self.pb1.step(1)
                                    print('DT Done')
                    self.scores['dtr'] = dtr_scores.mean()
                # Random forest regression
                if self.rfr_include_comp.get() and self.continue_flag:
                    from sklearn.ensemble import RandomForestRegressor
                    rfr = RandomForestRegressor(
                        n_estimators=int(self.rfr_n_estimators.get()), 
                        criterion=self.rfr_criterion.get(),
                        max_depth=(int(self.rfr_max_depth.get()) 
                            if (self.rfr_max_depth.get() != 'None') else None), 
                        min_samples_split=(float(self.rfr_min_samples_split.get()) 
                            if '.' in self.rfr_min_samples_split.get()
                            else int(self.rfr_min_samples_split.get())),
                        min_samples_leaf=(float(self.rfr_min_samples_leaf.get()) 
                            if '.' in self.rfr_min_samples_leaf.get()
                            else int(self.rfr_min_samples_leaf.get())),
                        min_weight_fraction_leaf=float(self.rfr_min_weight_fraction_leaf.get()),
                        max_features=(float(self.rfr_max_features.get()) 
                            if '.' in self.rfr_max_features.get() 
                            else int(self.rfr_max_features.get()) 
                            if len(self.rfr_max_features.get()) < 4 
                            else self.rfr_max_features.get() 
                            if (self.rfr_max_features.get() != 'None') 
                            else None),
                        max_leaf_nodes=(int(self.rfr_max_leaf_nodes.get()) 
                            if (self.rfr_max_leaf_nodes.get() != 'None') else None),
                        min_impurity_decrease=float(self.rfr_min_impurity_decrease.get()),
                        bootstrap=self.rfr_bootstrap.get(), oob_score=self.rfr_oob_score.get(),
                        n_jobs=(int(self.rfr_n_jobs.get()) 
                            if (self.rfr_n_jobs.get() != 'None') else None),
                        random_state=(int(self.rfr_random_state.get()) 
                            if (self.rfr_random_state.get() != 'None') else None),
                        verbose=int(self.rfr_verbose.get()), 
                        warm_start=self.rfr_warm_start.get(),
                        ccp_alpha=float(self.rfr_ccp_alpha.get()),
                        max_samples=(float(self.rfr_max_samples.get()) 
                            if '.' in self.rfr_max_samples.get() 
                            else int(self.rfr_max_samples.get()) 
                            if (self.rfr_max_samples.get() != 'None') 
                            else None))
                    rfr_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    rfr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(rfr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    rfr_scores = np.append(rfr_scores, score)
                                    self.pb1.step(1)
                                    print('RF Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    rfr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(rfr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    rfr_scores = np.append(rfr_scores, score)
                                    self.pb1.step(1)
                                    print('RF Done')
                    self.scores['rfr'] = rfr_scores.mean()
                # Support vector
                if self.svr_include_comp.get() and self.continue_flag:
                    from sklearn.svm import SVR
                    svr = SVR(
                        kernel=self.svr_kernel.get(), degree=int(self.svr_degree.get()), 
                        gamma=(float(self.svr_gamma.get()) 
                          if '.' in self.svr_gamma.get() else self.svr_gamma.get()),
                        coef0=float(self.svr_coef0.get()), tol=float(self.svr_tol.get()), 
                        C=float(self.svr_C.get()), epsilon=float(self.svr_epsilon.get()), 
                        shrinking=self.svr_shrinking.get(), 
                        cache_size=float(self.svr_cache_size.get()), 
                        verbose=self.svr_verbose.get(), 
                        max_iter=int(self.svr_max_iter.get()))
                    svr_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    svr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(svr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    svr_scores = np.append(svr_scores, score)
                                    self.pb1.step(1)
                                    print('SV Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    svr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(svr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    svr_scores = np.append(svr_scores, score)
                                    self.pb1.step(1)
                                    print('SV Done')
                    self.scores['svr'] = svr_scores.mean()
                # Stochastic Gradient Descent
                if self.sgdr_include_comp.get() and self.continue_flag:
                    from sklearn.linear_model import SGDRegressor
                    sgdr = SGDRegressor(
                        loss=self.sgdr_loss.get(), penalty=self.sgdr_penalty.get(),
                        alpha=float(self.sgdr_alpha.get()), 
                        l1_ratio=float(self.sgdr_l1_ratio.get()),
                        fit_intercept=self.sgdr_fit_intercept.get(), 
                        max_iter=int(self.sgdr_max_iter.get()),
                        tol=float(self.sgdr_tol.get()), shuffle=self.sgdr_shuffle.get(), 
                        verbose=int(self.sgdr_verbose.get()), 
                        epsilon=float(self.sgdr_epsilon.get()),
                        random_state=(int(self.sgdr_random_state.get()) 
                            if (self.sgdr_random_state.get() != 'None') else None),
                        learning_rate=self.sgdr_learning_rate.get(), 
                        eta0=float(self.sgdr_eta0.get()),
                        power_t=float(self.sgdr_power_t.get()), 
                        early_stopping=self.sgdr_early_stopping.get(),
                        validation_fraction=float(self.sgdr_validation_fraction.get()),
                        n_iter_no_change=int(self.sgdr_n_iter_no_change.get()), 
                        warm_start=self.sgdr_warm_start.get(),
                        average=(True if self.sgdr_average.get()=='True' else False 
                            if self.sgdr_average.get()=='False' 
                            else int(self.sgdr_average.get())))
                    sgdr_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    sgdr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(sgdr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    sgdr_scores = np.append(sgdr_scores, score)
                                    self.pb1.step(1)
                                    print('SGD Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    sgdr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(sgdr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    sgdr_scores = np.append(sgdr_scores, score)
                                    self.pb1.step(1)
                                    print('SGD Done')
                    self.scores['sgdr'] = sgdr_scores.mean()
                # Nearest Neighbor
                if self.knr_include_comp.get() and self.continue_flag:
                    from sklearn.neighbors import KNeighborsRegressor
                    knr = KNeighborsRegressor(
                        n_neighbors=int(self.knr_n_neighbors.get()), 
                        weights=self.knr_weights.get(), algorithm=self.knr_algorithm.get(),
                        leaf_size=int(self.knr_leaf_size.get()), p=int(self.knr_p.get()),
                        metric=self.knr_metric.get(), 
                        n_jobs=(int(self.knr_n_jobs.get()) 
                            if (self.knr_n_jobs.get() != 'None') else None))
                    knr_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    knr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(knr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    knr_scores = np.append(knr_scores, score)
                                    self.pb1.step(1)
                                    print('KN Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    knr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(knr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    knr_scores = np.append(knr_scores, score)
                                    self.pb1.step(1)
                                    print('KN Done')
                    self.scores['knr'] = knr_scores.mean()
                # Gaussian Process
                if self.gpr_include_comp.get() and self.continue_flag:
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    gpr = GaussianProcessRegressor(
                        alpha=float(self.gpr_alpha.get()),
                        n_restarts_optimizer=int(self.gpr_n_restarts_optimizer.get()),
                        normalize_y=self.gpr_normalize_y.get(), 
                        copy_X_train=self.gpr_copy_X_train.get(),
                        random_state=(int(self.gpr_random_state.get()) 
                            if (self.gpr_random_state.get() != 'None') else None))
                    gpr_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    gpr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(gpr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    gpr_scores = np.append(gpr_scores, score)
                                    self.pb1.step(1)
                                    print('GP Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    gpr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(gpr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    gpr_scores = np.append(gpr_scores, score)
                                    self.pb1.step(1)
                                    print('GP Done')
                    self.scores['gpr'] = gpr_scores.mean()
                # MLP
                if self.mlpr_include_comp.get() and self.continue_flag:
                    from sklearn.neural_network import MLPRegressor
                    mlpr = MLPRegressor(
                        hidden_layer_sizes=eval(self.mlpr_hidden_layer_sizes.get()),
                        activation=self.mlpr_activation.get(), solver=self.mlpr_solver.get(),
                        alpha=float(self.mlpr_alpha.get()), 
                        batch_size=(int(self.mlpr_batch_size.get()) 
                            if (self.mlpr_batch_size.get() != 'auto') else 'auto'),
                        learning_rate=self.mlpr_learning_rate.get(), 
                        learning_rate_init=float(self.mlpr_learning_rate_init.get()),
                        power_t=float(self.mlpr_power_t.get()), 
                        max_iter=int(self.mlpr_max_iter.get()),
                        shuffle=self.mlpr_shuffle.get(),
                        random_state=(int(self.mlpr_random_state.get()) 
                            if (self.mlpr_random_state.get() != 'None') else None),
                        tol=float(self.mlpr_tol.get()), verbose=self.mlpr_verbose.get(),
                        warm_start=self.mlpr_warm_start.get(), 
                        momentum=float(self.mlpr_momentum.get()),
                        nesterovs_momentum=self.mlpr_nesterovs_momentum.get(),
                        early_stopping=self.mlpr_early_stopping.get(), 
                        validation_fraction=float(self.mlpr_validation_fraction.get()),
                        beta_1=float(self.mlpr_beta_1.get()), 
                        beta_2=float(self.mlpr_beta_2.get()),
                        epsilon=float(self.mlpr_epsilon.get()), 
                        n_iter_no_change=int(self.mlpr_n_iter_no_change.get()),
                        max_fun=int(self.mlpr_max_fun.get()))
                    mlpr_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    mlpr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(mlpr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    mlpr_scores = np.append(mlpr_scores, score)
                                    self.pb1.step(1)
                                    print('MLP Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    mlpr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(mlpr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    mlpr_scores = np.append(mlpr_scores, score)
                                    self.pb1.step(1)
                                    print('MLP Done')
                    self.scores['mlpr'] = mlpr_scores.mean()
                # XGB
                if self.xgbr_include_comp.get() and self.continue_flag:
                    from xgboost import XGBRegressor
                    if self.xgbr_use_gpu.get() == False:
                        xgbr = XGBRegressor(
                            learning_rate=float(self.xgbr_eta.get()), 
                            n_estimators=int(self.xgbr_n_estimators.get()), 
                            max_depth=int(self.xgbr_max_depth.get()),
                            min_child_weight=int(self.xgbr_min_child_weight.get()),
                            gamma=float(self.xgbr_gamma.get()), 
                            subsample=float(self.xgbr_subsample.get()),
                            colsample_bytree=float(self.xgbr_colsample_bytree.get()),
                            reg_lambda=float(self.xgbr_lambda.get()), 
                            reg_alpha=float(self.xgbr_alpha.get()),
                            verbosity=2)
                    else:
                        xgbr = XGBRegressor(
                            learning_rate=float(self.xgbr_eta.get()), 
                            n_estimators=int(self.xgbr_n_estimators.get()), 
                            max_depth=int(self.xgbr_max_depth.get()),
                            min_child_weight=int(self.xgbr_min_child_weight.get()),
                            gamma=float(self.xgbr_gamma.get()), 
                            subsample=float(self.xgbr_subsample.get()),
                            colsample_bytree=float(self.xgbr_colsample_bytree.get()),
                            reg_lambda=float(self.xgbr_lambda.get()), 
                            reg_alpha=float(self.xgbr_alpha.get()),
                            verbosity=2,
                            tree_method='gpu_hist', gpu_id=0)
                    xgbr_scores = np.array([])
                    if self.x_st_var.get() == 'Yes' and self.xgbr_early_stopping.get()==True:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    check_tr_X_St, check_val_X_St, check_tr_y, check_val_y = \
                                        train_test_split(prev.X_St[train_index], prev.y[train_index], 
                                            test_size=float(self.xgbr_validation_fraction.get()), random_state=22)
                                    xgbr.fit(X=check_tr_X_St, y=check_tr_y, eval_set=[(check_val_X_St, check_val_y)],
                                        early_stopping_rounds=int(self.xgbr_n_iter_no_change.get()),
                                        eval_metric=self.xgbr_eval_metric.get()
                                        )
                                    best_iteration = xgbr.get_booster().best_ntree_limit
                                    score = sklearn.metrics.mean_squared_error(xgbr.predict(prev.X_St[test_index], 
                                        ntree_limit=best_iteration), 
                                        prev.y[test_index])
                                    xgbr_scores = np.append(xgbr_scores, score)
                                    self.pb1.step(1)
                                    print('XGB Done')
                    elif self.xgbr_early_stopping.get()==True:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    check_tr_X, check_val_X, check_tr_y, check_val_y = \
                                        train_test_split(prev.X.iloc[train_index], prev.y[train_index], 
                                            test_size=float(self.xgbr_validation_fraction.get()), random_state=22)
                                    xgbr.fit(X=check_tr_X, y=check_tr_y, eval_set=[(check_val_X, check_val_y)],
                                        early_stopping_rounds=int(self.xgbr_n_iter_no_change.get()),
                                        eval_metric=self.xgbr_eval_metric.get()
                                        )
                                    best_iteration = xgbr.get_booster().best_ntree_limit
                                    score = sklearn.metrics.mean_squared_error(xgbr.predict(prev.X.iloc[test_index], 
                                        ntree_limit=best_iteration), 
                                        prev.y[test_index])
                                    xgbr_scores = np.append(xgbr_scores, score)
                                    self.pb1.step(1)
                                    print('XGB Done')
                    elif self.x_st_var.get() == 'Yes' and self.xgbr_early_stopping.get()==False:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    xgbr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(xgbr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    xgbr_scores = np.append(xgbr_scores, score)
                                    self.pb1.step(1)
                                    print('XGB Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    xgbr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(xgbr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    xgbr_scores = np.append(xgbr_scores, score)
                                    self.pb1.step(1)
                                    print('XGB Done')
                    self.scores['xgbr'] = xgbr_scores.mean()
                # CatBoost
                if self.cbr_include_comp.get() and self.continue_flag:
                    from catboost import CatBoostRegressor
                    if self.cbr_use_gpu.get() == False:
                        cbr = CatBoostRegressor(
                            eval_metric=self.cbr_eval_metric.get(),
                            cat_features=eval(self.cbr_cf_list.get()),
                            iterations=int(self.cbr_iterations.get()), 
                            learning_rate=(None if self.cbr_learning_rate.get()=='None' 
                                else float(self.cbr_learning_rate.get())),
                            depth=int(self.cbr_depth.get()), 
                            reg_lambda=(None if self.cbr_reg_lambda.get()=='None' 
                                else float(self.cbr_reg_lambda.get())),
                            subsample=(None if self.cbr_subsample.get()=='None' 
                                else float(self.cbr_subsample.get())), 
                            colsample_bylevel=float(self.cbr_colsample_bylevel.get()),
                            random_strength=float(self.cbr_random_strength.get()))
                    else:
                        cbr = CatBoostRegressor(
                            eval_metric=self.cbr_eval_metric.get(),
                            cat_features=eval(self.cbr_cf_list.get()),
                            iterations=int(self.cbr_iterations.get()), 
                            task_type="GPU", devices='0:1',
                            learning_rate=(None if self.cbr_learning_rate.get()=='None' 
                                else float(self.cbr_learning_rate.get())),
                            depth=int(self.cbr_depth.get()), 
                            reg_lambda=(None if self.cbr_reg_lambda.get()=='None' 
                                else float(self.cbr_reg_lambda.get())),
                            subsample=(None if self.cbr_subsample.get()=='None' 
                                else float(self.cbr_subsample.get())), 
                            colsample_bylevel=float(self.cbr_colsample_bylevel.get()),
                            random_strength=float(self.cbr_random_strength.get()))
                    cbr_scores = np.array([])
                    if self.x_st_var.get() == 'Yes' and self.cbr_early_stopping.get()==True:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    check_tr_X_St, check_val_X_St, check_tr_y, check_val_y = \
                                        train_test_split(prev.X_St[train_index], prev.y[train_index], 
                                            test_size=float(self.cbr_validation_fraction.get()), random_state=22)
                                    cbr.fit(X=check_tr_X_St, y=check_tr_y, eval_set=(check_val_X_St, check_val_y),
                                        early_stopping_rounds=int(self.cbr_n_iter_no_change.get()),
                                        use_best_model=True)
                                    score = sklearn.metrics.mean_squared_error(cbr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    cbr_scores = np.append(cbr_scores, score)
                                    self.pb1.step(1)
                                    print('CB Done')
                    elif self.cbr_early_stopping.get()==True:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    check_tr_X, check_val_X, check_tr_y, check_val_y = \
                                        train_test_split(prev.X.iloc[train_index], prev.y[train_index], 
                                            test_size=float(self.cbr_validation_fraction.get()), random_state=22)
                                    cbr.fit(X=check_tr_X, y=check_tr_y, eval_set=(check_val_X, check_val_y),
                                        early_stopping_rounds=int(self.cbr_n_iter_no_change.get()),
                                        use_best_model=True)
                                    score = sklearn.metrics.mean_squared_error(cbr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    cbr_scores = np.append(cbr_scores, score)
                                    self.pb1.step(1)
                                    print('CB Done')
                    elif self.x_st_var.get() == 'Yes' and self.cbr_early_stopping.get()==False:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    cbr.fit(prev.X_St[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(cbr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    cbr_scores = np.append(cbr_scores, score)
                                    self.pb1.step(1)
                                    print('CB Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    cbr.fit(prev.X.iloc[train_index], prev.y[train_index])
                                    score = sklearn.metrics.mean_squared_error(cbr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    cbr_scores = np.append(cbr_scores, score)
                                    self.pb1.step(1)
                                    print('CB Done')
                    self.scores['cbr'] = cbr_scores.mean()
                # LightGBM
                if self.lgbmr_include_comp.get() and self.continue_flag:
                    from lightgbm import LGBMRegressor
                    lgbmr = LGBMRegressor(
                        boosting_type=self.lgbmr_boosting_type.get(),
                        num_leaves=int(self.lgbmr_num_leaves.get()),
                        max_depth=int(self.lgbmr_max_depth.get()),
                        learning_rate=float(self.lgbmr_learning_rate.get()),
                        n_estimators=int(self.lgbmr_n_estimators.get()),
                        subsample_for_bin=int(self.lgbmr_subsample_for_bin.get()),
                        min_split_gain=float(self.lgbmr_min_split_gain.get()),
                        min_child_weight=float(self.lgbmr_min_child_weight.get()),
                        min_child_samples=int(self.lgbmr_min_child_samples.get()),
                        subsample=float(self.lgbmr_subsample.get()),
                        subsample_freq=int(self.lgbmr_subsample_freq.get()),
                        colsample_bytree=float(self.lgbmr_colsample_bytree.get()),
                        reg_alpha=float(self.lgbmr_reg_alpha.get()),
                        reg_lambda=float(self.lgbmr_reg_lambda.get()),
                        n_jobs=int(self.lgbmr_n_jobs.get()),
                        silent=self.lgbmr_silent.get(),
                    )
                    lgbmr_scores = np.array([])
                    if self.x_st_var.get() == 'Yes' and self.lgbmr_early_stopping.get()==True:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    check_tr_X_St, check_val_X_St, check_tr_y, check_val_y = \
                                        train_test_split(prev.X_St[train_index], prev.y[train_index], 
                                            test_size=float(self.cbr_validation_fraction.get()), random_state=22)
                                    lgbmr.fit(X=check_tr_X_St, y=check_tr_y, eval_set=(check_val_X_St, check_val_y),
                                        early_stopping_rounds=int(self.cbr_n_iter_no_change.get()),
                                        categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                            else eval(self.lgbmr_categorical_feature.get()))
                                    )
                                    score = sklearn.metrics.mean_squared_error(lgbmr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    lgbmr_scores = np.append(lgbmr_scores, score)
                                    self.pb1.step(1)
                                    print('LGBM Done')
                    elif self.lgbmr_early_stopping.get()==True:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    check_tr_X, check_val_X, check_tr_y, check_val_y = \
                                        train_test_split(prev.X.iloc[train_index], prev.y[train_index], 
                                            test_size=float(self.cbr_validation_fraction.get()), random_state=22)
                                    lgbmr.fit(X=check_tr_X, y=check_tr_y, eval_set=(check_val_X, check_val_y),
                                        early_stopping_rounds=int(self.cbr_n_iter_no_change.get()),
                                        categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                            else eval(self.lgbmr_categorical_feature.get()))
                                    )
                                    score = sklearn.metrics.mean_squared_error(lgbmr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    lgbmr_scores = np.append(lgbmr_scores, score)
                                    self.pb1.step(1)
                                    print('LGBM Done')
                    elif self.x_st_var.get() == 'Yes' and self.lgbmr_early_stopping.get()==False:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X_St):
                                if self.continue_flag:
                                    lgbmr.fit(
                                        prev.X_St[train_index], prev.y[train_index],
                                        categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                            else eval(self.lgbmr_categorical_feature.get()))
                                    )
                                    score = sklearn.metrics.mean_squared_error(lgbmr.predict(prev.X_St[test_index]), 
                                        prev.y[test_index])
                                    lgbmr_scores = np.append(lgbmr_scores, score)
                                    self.pb1.step(1)
                                    print('LGBM Done')
                    else:
                        for fold in folds_list:
                            for train_index, test_index in fold.split(prev.X):
                                if self.continue_flag:
                                    lgbmr.fit(
                                        prev.X.iloc[train_index], prev.y[train_index],
                                        categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                            else eval(self.lgbmr_categorical_feature.get()))
                                    )
                                    score = sklearn.metrics.mean_squared_error(lgbmr.predict(prev.X.iloc[test_index]), 
                                        prev.y[test_index])
                                    lgbmr_scores = np.append(lgbmr_scores, score)
                                    self.pb1.step(1)
                                    print('LGBM Done')
                    self.scores['lgbmr'] = lgbmr_scores.mean()
            except ValueError as e:
                messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))
            self.continue_flag = True
            self.pb1.destroy()
            self.stop_button.destroy()
            self.show_res_button = ttk.Button(self.frame, text='Show Results', 
                command=lambda: self.Comp_results(self, parent))
            self.show_res_button.place(x=400, y=165)

        # function to stop comparison
        def stop_thread(thread):
            self.continue_flag = False

        # function to initialize comparison in a particular thread
        def try_compare_methods(prev, main):
            try:
                self.show_res_button.destroy()
                
                thread1 = Thread(target=compare_methods, args=(prev, main))
                thread1.daemon = True
                thread1.start()

                l = 1
                for x in [self.lr_include_comp.get(), self.rr_include_comp.get(), 
                          self.lassor_include_comp.get(), self.dtr_include_comp.get(),
                          self.rfr_include_comp.get(), self.svr_include_comp.get(),
                          self.sgdr_include_comp.get(), self.knr_include_comp.get(),
                          self.gpr_include_comp.get(), self.mlpr_include_comp.get(),
                          self.xgbr_include_comp.get(), self.cbr_include_comp.get(),
                          self.lgbmr_include_comp.get()]:
                    if x==True:
                        l+=1*int(rep_entry.get())*int(e2.get())

                self.pb1 = ttk.Progressbar(self.frame, mode='determinate', maximum=l, length=100)
                self.pb1.place(x=400, y=170)

                self.stop_button = ttk.Button(self.frame, text='Stop', width=5, 
                    command=lambda: stop_thread(thread1))
                self.stop_button.place(x=510, y=168)
            except ValueError as e:
                self.pb1.destroy()
                self.stop_button.destroy()
                self.show_res_button = ttk.Button(self.frame, text='Show Results', 
                    command=lambda: self.Comp_results(self, parent))
                self.show_res_button.place(x=400, y=165)
                messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))

        # function to predict values
        def make_regression(method):
            try:
                try:
                    tr_x_from = self.training.data.columns.get_loc(self.training.x_from_var.get())
                    tr_x_to = self.training.data.columns.get_loc(self.training.x_to_var.get()) + 1
                    pr_x_from = self.prediction.data.columns.get_loc(self.prediction.x_from_var.get())
                    pr_x_to = self.prediction.data.columns.get_loc(self.prediction.x_to_var.get()) + 1
                except:
                    tr_x_from = int(self.training.x_from_var.get())
                    tr_x_to = int(self.training.x_to_var.get()) + 1
                    pr_x_from = int(self.prediction.x_from_var.get())
                    pr_x_to = int(self.prediction.x_to_var.get()) + 1

                if self.handle_cat_var.get()=='No':
                    try:
                        training_X = self.training.data.iloc[:,tr_x_from : tr_x_to]
                        training_y = self.training.data[self.y_var.get()]
                    except:
                        training_X = self.training.data.iloc[:,tr_x_from : tr_x_to]
                        training_y = self.training.data[int(self.y_var.get())]
                    X = self.prediction.data.iloc[:,pr_x_from : pr_x_to]

                    X[X.select_dtypes(['object']).columns] = \
                        X.select_dtypes(['object']).apply(lambda x: x.astype('category'))
                    training_X[training_X.select_dtypes(['object']).columns] = \
                        training_X.select_dtypes(['object']).apply(lambda x: x.astype('category'))
                elif self.handle_cat_var.get()=='Dummies':
                    from sklearn.preprocessing import OneHotEncoder
                    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    try:
                        training_X = self.training.data.iloc[:,tr_x_from : tr_x_to]
                        onehot_columns = training_X.select_dtypes(include=['object']).columns
                        onehot_df = training_X[onehot_columns]
                        onehot_df = pd.DataFrame(enc.fit_transform(onehot_df))
                        df_onehot_drop = training_X.drop(onehot_columns, axis = 1)
                        training_X = pd.concat([df_onehot_drop, onehot_df], axis = 1)

                        training_y = self.training.data[self.y_var.get()]

                        X = self.prediction.data.iloc[:,pr_x_from : pr_x_to]
                        onehot_columns = X.select_dtypes(include=['object']).columns
                        onehot_df = X[onehot_columns]
                        onehot_df = pd.DataFrame(enc.transform(onehot_df))
                        df_onehot_drop = X.drop(onehot_columns, axis = 1)
                        X = pd.concat([df_onehot_drop, onehot_df], axis = 1)
                    except:
                        training_X = self.training.data.iloc[:,tr_x_from : tr_x_to]
                        onehot_columns = training_X.select_dtypes(include=['object']).columns
                        onehot_df = training_X[onehot_columns]
                        onehot_df = pd.DataFrame(enc.fit_transform(onehot_df))
                        df_onehot_drop = training_X.drop(onehot_columns, axis = 1)
                        training_X = pd.concat([df_onehot_drop, onehot_df], axis = 1)
                        
                        training_y = self.training.data[int(self.y_var.get())]

                        X = self.prediction.data.iloc[:,pr_x_from : pr_x_to]
                        onehot_columns = X.select_dtypes(include=['object']).columns
                        onehot_df = X[onehot_columns]
                        onehot_df = pd.DataFrame(enc.transform(onehot_df))
                        df_onehot_drop = X.drop(onehot_columns, axis = 1)
                        X = pd.concat([df_onehot_drop, onehot_df], axis = 1)
                elif self.handle_cat_var.get()=='One Hot Encoding':
                    from sklearn.preprocessing import OneHotEncoder
                    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    try:
                        training_X = pd.DataFrame(enc.fit_transform(self.training.data.iloc[:,tr_x_from : tr_x_to]))
                        training_y = self.training.data[self.y_var.get()]

                        X = pd.DataFrame(enc.transform(self.prediction.data.iloc[:,pr_x_from : pr_x_to]))
                    except:
                        training_X = pd.DataFrame(enc.fit_transform(self.training.data.iloc[:,tr_x_from : tr_x_to]))
                        training_y = self.training.data[int(self.y_var.get())]
                        X = pd.DataFrame(enc.transform(self.prediction.data.iloc[:,pr_x_from : pr_x_to]))
                elif self.handle_cat_var.get()=='Ordinal Encoding':
                    from sklearn.preprocessing import OrdinalEncoder
                    enc = OrdinalEncoder(handle_unknown='ignore')
                    try:
                        training_X = pd.DataFrame(enc.fit_transform(self.training.data.iloc[:,tr_x_from : tr_x_to]))
                        training_y = self.training.data[self.y_var.get()]

                        X = pd.DataFrame(enc.transform(self.prediction.data.iloc[:,pr_x_from : pr_x_to]))
                    except:
                        training_X = pd.DataFrame(enc.fit_transform(self.training.data.iloc[:,tr_x_from : tr_x_to]))
                        training_y = self.training.data[int(self.y_var.get())]
                        X = pd.DataFrame(enc.transform(self.prediction.data.iloc[:,pr_x_from : pr_x_to]))

                from sklearn import preprocessing
                scaler = preprocessing.StandardScaler()
                try:
                    X_St = scaler.fit_transform(X)
                    training_X_St = scaler.fit_transform(training_X)
                except:
                    self.x_st_var.set('No')
                from sklearn.model_selection import train_test_split, KFold
                check_tr_y, check_val_y = train_test_split(training_y, 
                        test_size=float(self.cbr_validation_fraction.get()), random_state=22)
                try:
                    check_tr_X_St, check_val_X_St = train_test_split(training_X_St, 
                            test_size=float(self.cbr_validation_fraction.get()), random_state=22)
                except:
                    pass
                check_tr_X, check_val_X = train_test_split(training_X, 
                        test_size=float(self.cbr_validation_fraction.get()), random_state=22)

                # Linear regression
                if method == 'Least squares':
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression(
                        fit_intercept=self.lr_fit_intercept.get(), normalize=self.lr_normalize.get(),
                        copy_X=self.lr_copy_X.get(), 
                        n_jobs=(int(self.lr_n_jobs.get()) 
                            if (self.lr_n_jobs.get() != 'None') else None))
                    if self.x_st_var.get() == 'Yes':
                        lr.fit(training_X_St, training_y)
                        pr_values = lr.predict(X_St)
                    else:
                        lr.fit(training_X, training_y)
                        pr_values = lr.predict(X)
                # Ridge regression
                if method == 'Ridge':
                    from sklearn.linear_model import Ridge
                    rr = Ridge(
                        alpha=float(self.rr_alpha.get()), fit_intercept=self.rr_fit_intercept.get(), 
                        normalize=self.rr_normalize.get(),
                        copy_X=self.rr_copy_X.get(), 
                        max_iter=(int(self.rr_max_iter.get()) 
                            if (self.rr_max_iter.get() != 'None') else None),
                        tol=float(self.rr_tol.get()), solver=self.rr_solver.get(),
                        random_state=(int(self.rr_random_state.get()) 
                            if (self.rr_random_state.get() != 'None') else None))
                    if self.x_st_var.get() == 'Yes':
                        rr.fit(training_X_St, training_y)
                        pr_values = rr.predict(X_St)
                    else:
                        rr.fit(training_X, training_y)
                        pr_values = rr.predict(X)
                # Lasso regression
                if method == 'Lasso':
                    from sklearn.linear_model import Lasso
                    lassor = Lasso(
                        alpha=float(self.lassor_alpha.get()), 
                        fit_intercept=self.lassor_fit_intercept.get(),
                        normalize=self.lassor_normalize.get(), 
                        precompute=self.lassor_precompute.get(), 
                        copy_X=self.lassor_copy_X.get(), 
                        max_iter=(int(self.lassor_max_iter.get()) 
                            if (self.lassor_max_iter.get() != 'None') else None),
                        tol=float(self.lassor_tol.get()), warm_start=self.lassor_warm_start.get(),
                        positive=self.lassor_positive.get(),
                        random_state=(int(self.lassor_random_state.get()) 
                            if (self.lassor_random_state.get() != 'None') else None),
                        selection=self.lassor_selection.get())
                    if self.x_st_var.get() == 'No':
                        lassor.fit(training_X, training_y)
                        pr_values = lassor.predict(X)
                    else:
                        lassor.fit(training_X_St, training_y)
                        pr_values = lassor.predict(X_St)
                # Random forest
                if method == 'Random Forest':
                    from sklearn.ensemble import RandomForestRegressor
                    rfr = RandomForestRegressor(
                        n_estimators=int(self.rfr_n_estimators.get()), 
                        criterion=self.rfr_criterion.get(),
                        max_depth=(int(self.rfr_max_depth.get()) 
                            if (self.rfr_max_depth.get() != 'None') else None), 
                        min_samples_split=(float(self.rfr_min_samples_split.get()) 
                            if '.' in self.rfr_min_samples_split.get()
                            else int(self.rfr_min_samples_split.get())),
                        min_samples_leaf=(float(self.rfr_min_samples_leaf.get()) 
                            if '.' in self.rfr_min_samples_leaf.get()
                            else int(self.rfr_min_samples_leaf.get())),
                        min_weight_fraction_leaf=float(self.rfr_min_weight_fraction_leaf.get()),
                        max_features=(float(self.rfr_max_features.get()) 
                            if '.' in self.rfr_max_features.get() 
                            else int(self.rfr_max_features.get()) 
                            if len(self.rfr_max_features.get()) < 4 
                            else self.rfr_max_features.get() 
                            if (self.rfr_max_features.get() != 'None') 
                            else None),
                        max_leaf_nodes=(int(self.rfr_max_leaf_nodes.get()) 
                            if (self.rfr_max_leaf_nodes.get() != 'None') else None),
                        min_impurity_decrease=float(self.rfr_min_impurity_decrease.get()),
                        bootstrap=self.rfr_bootstrap.get(), oob_score=self.rfr_oob_score.get(),
                        n_jobs=(int(self.rfr_n_jobs.get()) 
                            if (self.rfr_n_jobs.get() != 'None') else None),
                        random_state=(int(self.rfr_random_state.get()) 
                            if (self.rfr_random_state.get() != 'None') else None),
                        verbose=int(self.rfr_verbose.get()), warm_start=self.rfr_warm_start.get(),
                        ccp_alpha=float(self.rfr_ccp_alpha.get()),
                        max_samples=(float(self.rfr_max_samples.get()) 
                            if '.' in self.rfr_max_samples.get() 
                            else int(self.rfr_max_samples.get()) 
                            if (self.rfr_max_samples.get() != 'None') 
                            else None))
                    if self.x_st_var.get() == 'Yes':
                        rfr.fit(training_X_St, training_y)
                        pr_values = rfr.predict(X_St)
                    else:
                        rfr.fit(training_X, training_y)
                        pr_values = rfr.predict(X)
                # Support Vector
                if method == 'Support Vector':
                    from sklearn.svm import SVR
                    svr = SVR(
                        kernel=self.svr_kernel.get(), degree=int(self.svr_degree.get()), 
                        gamma=(float(self.svr_gamma.get()) 
                            if '.' in self.svr_gamma.get() else self.svr_gamma.get()),
                        coef0=float(self.svr_coef0.get()), tol=float(self.svr_tol.get()), 
                        C=float(self.svr_C.get()), epsilon=float(self.svr_epsilon.get()), 
                        shrinking=self.svr_shrinking.get(), 
                        cache_size=float(self.svr_cache_size.get()), 
                        verbose=self.svr_verbose.get(), max_iter=int(self.svr_max_iter.get()))
                    if self.x_st_var.get() == 'No':
                        svr.fit(training_X, training_y)
                        pr_values = svr.predict(X)
                    else:
                        svr.fit(training_X_St, training_y)
                        pr_values = svr.predict(X_St)
                # Support Vector
                if method == 'SGD':
                    from sklearn.linear_model import SGDRegressor
                    sgdr = SGDRegressor(
                        loss=self.sgdr_loss.get(), penalty=self.sgdr_penalty.get(),
                        alpha=float(self.sgdr_alpha.get()), l1_ratio=float(self.sgdr_l1_ratio.get()),
                        fit_intercept=self.sgdr_fit_intercept.get(), 
                        max_iter=int(self.sgdr_max_iter.get()),
                        tol=float(self.sgdr_tol.get()), shuffle=self.sgdr_shuffle.get(), 
                        verbose=int(self.sgdr_verbose.get()), 
                        epsilon=float(self.sgdr_epsilon.get()),
                        random_state=(int(self.sgdr_random_state.get()) 
                            if (self.sgdr_random_state.get() != 'None') else None),
                        learning_rate=self.sgdr_learning_rate.get(), eta0=float(self.sgdr_eta0.get()),
                        power_t=float(self.sgdr_power_t.get()), 
                        early_stopping=self.sgdr_early_stopping.get(),
                        validation_fraction=float(self.sgdr_validation_fraction.get()),
                        n_iter_no_change=int(self.sgdr_n_iter_no_change.get()), 
                        warm_start=self.sgdr_warm_start.get(),
                        average=(True if self.sgdr_average.get()=='True' else False 
                            if self.sgdr_average.get()=='False' else int(self.sgdr_average.get())))
                    if self.x_st_var.get() == 'No':
                        sgdr.fit(training_X, training_y)
                        pr_values = sgdr.predict(X)
                    else:
                        sgdr.fit(training_X_St, training_y)
                        pr_values = sgdr.predict(X_St)
                # Nearest Neighbor
                if method == 'Nearest Neighbor':
                    from sklearn.neighbors import KNeighborsRegressor
                    knr = KNeighborsRegressor(
                        n_neighbors=int(self.knr_n_neighbors.get()), 
                        weights=self.knr_weights.get(), algorithm=self.knr_algorithm.get(),
                        leaf_size=int(self.knr_leaf_size.get()), p=int(self.knr_p.get()),
                        metric=self.knr_metric.get(), 
                        n_jobs=(int(self.knr_n_jobs.get()) 
                            if (self.knr_n_jobs.get() != 'None') else None))
                    if self.x_st_var.get() == 'No':
                        knr.fit(training_X, training_y)
                        pr_values = knr.predict(X)
                    else:
                        knr.fit(training_X_St, training_y)
                        pr_values = knr.predict(X_St)
                # Gaussian Process
                if method == 'Gaussian Process':
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    gpr = GaussianProcessRegressor(
                        alpha=float(self.gpr_alpha.get()),
                        n_restarts_optimizer=int(self.gpr_n_restarts_optimizer.get()),
                        normalize_y=self.gpr_normalize_y.get(), 
                        copy_X_train=self.gpr_copy_X_train.get(),
                        random_state=(int(self.gpr_random_state.get()) 
                            if (self.gpr_random_state.get() != 'None') else None))
                    if self.x_st_var.get() == 'No':
                        gpr.fit(training_X, training_y)
                        pr_values = gpr.predict(X)
                    else:
                        gpr.fit(training_X_St, training_y)
                        pr_values = gpr.predict(X_St)
                # Decision Tree
                if method == 'Decision Tree':
                    from sklearn.tree import DecisionTreeRegressor
                    dtr = DecisionTreeRegressor(
                        criterion=self.dtr_criterion.get(), splitter=self.dtr_splitter.get(), 
                        max_depth=(int(self.dtr_max_depth.get()) 
                            if (self.dtr_max_depth.get() != 'None') else None), 
                        min_samples_split=(float(self.dtr_min_samples_split.get()) 
                            if '.' in self.dtr_min_samples_split.get()
                            else int(self.dtr_min_samples_split.get())), 
                        min_samples_leaf=(float(self.dtr_min_samples_leaf.get()) 
                            if '.' in self.dtr_min_samples_leaf.get()
                            else int(self.dtr_min_samples_leaf.get())),
                        min_weight_fraction_leaf=float(self.dtr_min_weight_fraction_leaf.get()),
                        max_features=(float(self.dtr_max_features.get()) 
                            if '.' in self.dtr_max_features.get() 
                            else int(self.dtr_max_features.get()) 
                            if len(self.dtr_max_features.get()) < 4 
                            else self.dtr_max_features.get() 
                            if (self.dtr_max_features.get() != 'None') 
                            else None), 
                        random_state=(int(self.dtr_random_state.get()) 
                            if (self.dtr_random_state.get() != 'None') else None),
                        max_leaf_nodes=(int(self.dtr_max_leaf_nodes.get()) 
                            if (self.dtr_max_leaf_nodes.get() != 'None') else None),
                        min_impurity_decrease=float(self.dtr_min_impurity_decrease.get()),
                        ccp_alpha=float(self.dtr_ccp_alpha.get()))
                    if self.x_st_var.get() == 'Yes':
                        dtr.fit(training_X_St, training_y)
                        pr_values = dtr.predict(X_St)
                    else:
                        dtr.fit(training_X, training_y)
                        pr_values = dtr.predict(X)
                # MLP
                if method == 'Multi-layer Perceptron':
                    from sklearn.neural_network import MLPRegressor
                    mlpr = MLPRegressor(
                        hidden_layer_sizes=eval(self.mlpr_hidden_layer_sizes.get()),
                        activation=self.mlpr_activation.get(),solver=self.mlpr_solver.get(),
                        alpha=float(self.mlpr_alpha.get()), 
                        batch_size=(int(self.mlpr_batch_size.get()) 
                            if (self.mlpr_batch_size.get() != 'auto') else 'auto'),
                        learning_rate=self.mlpr_learning_rate.get(), 
                        learning_rate_init=float(self.mlpr_learning_rate_init.get()),
                        power_t=float(self.mlpr_power_t.get()), 
                        max_iter=int(self.mlpr_max_iter.get()),
                        shuffle=self.mlpr_shuffle.get(),
                        random_state=(int(self.mlpr_random_state.get()) 
                            if (self.mlpr_random_state.get() != 'None') else None),
                        tol=float(self.mlpr_tol.get()), verbose=self.mlpr_verbose.get(),
                        warm_start=self.mlpr_warm_start.get(), 
                        momentum=float(self.mlpr_momentum.get()),
                        nesterovs_momentum=self.mlpr_nesterovs_momentum.get(),
                        early_stopping=self.mlpr_early_stopping.get(), 
                        validation_fraction=float(self.mlpr_validation_fraction.get()),
                        beta_1=float(self.mlpr_beta_1.get()), beta_2=float(self.mlpr_beta_2.get()),
                        epsilon=float(self.mlpr_epsilon.get()), 
                        n_iter_no_change=int(self.mlpr_n_iter_no_change.get()),
                        max_fun=int(self.mlpr_max_fun.get()))
                    if self.x_st_var.get() == 'No':
                        mlpr.fit(training_X, training_y)
                        pr_values = mlpr.predict(X)
                    else:
                        mlpr.fit(training_X_St, training_y)
                        pr_values = mlpr.predict(X_St)
                # XGBoost
                if method == 'XGBoost':
                    from xgboost import XGBRegressor
                    if self.xgbr_use_gpu.get() == False:
                        xgbr = XGBRegressor(
                            learning_rate=float(self.xgbr_eta.get()), 
                            n_estimators=int(self.xgbr_n_estimators.get()), 
                            max_depth=int(self.xgbr_max_depth.get()),
                            min_child_weight=int(self.xgbr_min_child_weight.get()),
                            gamma=float(self.xgbr_gamma.get()), 
                            subsample=float(self.xgbr_subsample.get()),
                            colsample_bytree=float(self.xgbr_colsample_bytree.get()),
                            reg_lambda=float(self.xgbr_lambda.get()), 
                            reg_alpha=float(self.xgbr_alpha.get()),
                            verbosity=2)
                    else:
                        xgbr = XGBRegressor(
                            learning_rate=float(self.xgbr_eta.get()), 
                            n_estimators=int(self.xgbr_n_estimators.get()), 
                            max_depth=int(self.xgbr_max_depth.get()),
                            min_child_weight=int(self.xgbr_min_child_weight.get()),
                            gamma=float(self.xgbr_gamma.get()), 
                            subsample=float(self.xgbr_subsample.get()),
                            colsample_bytree=float(self.xgbr_colsample_bytree.get()),
                            reg_lambda=float(self.xgbr_lambda.get()), 
                            reg_alpha=float(self.xgbr_alpha.get()),
                            verbosity=2,
                            tree_method='gpu_hist', gpu_id=0)

                    #cv-voting mode
                    if self.x_st_var.get() == 'Yes' and self.xgbr_cv_voting_mode.get()==True:
                        first_col = True
                        cross_fold = KFold(n_splits = int(self.xgbr_cv_voting_folds.get()), 
                            shuffle=True)
                        azaza = []
                        for train_index, test_index in cross_fold.split(training_X_St):
                            xgbr.fit(X=training_X_St.iloc[train_index], y=training_y[train_index], 
                                eval_set=[(training_X_St.iloc[test_index], training_y[test_index])], 
                                early_stopping_rounds=int(self.xgbr_n_iter_no_change.get()),
                                eval_metric=self.xgbr_eval_metric.get()
                                )
                            best_iteration = xgbr.get_booster().best_ntree_limit
                            azaza.append(xgbr.get_booster().best_score)
                            print(azaza)
                            predict = xgbr.predict(X_St, ntree_limit=best_iteration)
                            if first_col:
                                pr_values = np.array(predict, ndmin=2)
                                pr_values = np.transpose(pr_values)
                                first_col = False
                            else:
                                pr_values = np.insert(pr_values, -1, predict, axis=1)
                        pr_values= np.mean(pr_values, axis=1)
                        print(np.mean(azaza))
                    elif self.xgbr_cv_voting_mode.get()==True:
                        first_col = True
                        cross_fold = KFold(n_splits = int(self.xgbr_cv_voting_folds.get()), 
                            shuffle=True)
                        azaza = []
                        for train_index, test_index in cross_fold.split(training_X):
                            xgbr.fit(X=training_X.iloc[train_index], y=training_y[train_index], 
                                eval_set=[(training_X.iloc[test_index], training_y[test_index])],
                                early_stopping_rounds=int(self.xgbr_n_iter_no_change.get()),
                                eval_metric=self.xgbr_eval_metric.get()
                                )
                            best_iteration = xgbr.get_booster().best_ntree_limit
                            azaza.append(xgbr.get_booster().best_score)
                            print(azaza)
                            predict = xgbr.predict(X, ntree_limit=best_iteration)
                            if first_col:
                                pr_values = np.array(predict, ndmin=2)
                                pr_values = np.transpose(pr_values)
                                first_col = False
                            else:
                                pr_values = np.insert(pr_values, -1, predict, axis=1)
                        pr_values= np.mean(pr_values, axis=1)
                        print(np.mean(azaza))

                    # early-stopping mode
                    elif self.x_st_var.get() == 'Yes' and self.xgbr_early_stopping.get()==True:
                        xgbr.fit(X=check_tr_X_St, y=check_tr_y, eval_set=[(check_val_X_St, check_val_y)], 
                            early_stopping_rounds=int(self.xgbr_n_iter_no_change.get()),
                            eval_metric=self.xgbr_eval_metric.get()
                            )
                        best_iteration = xgbr.get_booster().best_ntree_limit
                        pr_values = xgbr.predict(X_St, ntree_limit=best_iteration)
                    elif self.xgbr_early_stopping.get()==True:
                        xgbr.fit(X=check_tr_X, y=check_tr_y, eval_set=[(check_val_X, check_val_y)], 
                            early_stopping_rounds=int(self.xgbr_n_iter_no_change.get()),
                            eval_metric=self.xgbr_eval_metric.get()
                            )
                        best_iteration = xgbr.get_booster().best_ntree_limit
                        pr_values = xgbr.predict(X, ntree_limit=best_iteration)

                    # standard mode
                    elif self.x_st_var.get() == 'Yes':
                        xgbr.fit(training_X_St, training_y)
                        pr_values = xgbr.predict(X_St)
                    else:
                        xgbr.fit(training_X, training_y)
                        pr_values = xgbr.predict(X)
                # CatBoost
                if method == 'CatBoost':
                    from catboost import CatBoostRegressor
                    if self.cbr_use_gpu.get() == False:
                        cbr = CatBoostRegressor(
                            eval_metric=self.cbr_eval_metric.get(),
                            cat_features=eval(self.cbr_cf_list.get()),
                            iterations=int(self.cbr_iterations.get()), 
                            learning_rate=(None if self.cbr_learning_rate.get()=='None' 
                                else float(self.cbr_learning_rate.get())),
                            depth=int(self.cbr_depth.get()), 
                            reg_lambda=(None if self.cbr_reg_lambda.get()=='None' 
                                    else float(self.cbr_reg_lambda.get())),
                            subsample=(None if self.cbr_subsample.get()=='None' 
                                else float(self.cbr_subsample.get())), 
                            colsample_bylevel=float(self.cbr_colsample_bylevel.get()),
                            random_strength=float(self.cbr_random_strength.get()))
                    else:
                        cbr = CatBoostRegressor(
                            eval_metric=self.cbr_eval_metric.get(),
                            cat_features=eval(self.cbr_cf_list.get()),
                            iterations=int(self.cbr_iterations.get()), 
                            task_type="GPU", devices='0:1',
                            learning_rate=(None if self.cbr_learning_rate.get()=='None' 
                                else float(self.cbr_learning_rate.get())),
                            depth=int(self.cbr_depth.get()), 
                            reg_lambda=(None if self.cbr_reg_lambda.get()=='None' 
                                    else float(self.cbr_reg_lambda.get())),
                            subsample=(None if self.cbr_subsample.get()=='None' 
                                else float(self.cbr_subsample.get())), 
                            colsample_bylevel=float(self.cbr_colsample_bylevel.get()),
                            random_strength=float(self.cbr_random_strength.get()))

                    #cv-voting mode
                    if self.x_st_var.get() == 'Yes' and self.cbr_cv_voting_mode.get()==True:
                        first_col = True
                        cross_fold = KFold(n_splits = int(self.cbr_cv_voting_folds.get()), 
                            shuffle=True)
                        for train_index, test_index in cross_fold.split(training_X_St):
                            cbr.fit(X=training_X_St.iloc[train_index], y=training_y[train_index], 
                                eval_set=(training_X_St.iloc[test_index], training_y[test_index]), 
                                early_stopping_rounds=int(self.cbr_n_iter_no_change.get()),
                                use_best_model=True)
                            predict = cbr.predict(X_St)
                            if first_col:
                                pr_values = np.array(predict, ndmin=2)
                                pr_values = np.transpose(pr_values)
                                first_col = False
                            else:
                                pr_values = np.insert(pr_values, -1, predict, axis=1)
                        pr_values= np.mean(pr_values, axis=1)
                    elif self.cbr_cv_voting_mode.get()==True:
                        first_col = True
                        cross_fold = KFold(n_splits = int(self.cbr_cv_voting_folds.get()), 
                            shuffle=True)
                        for train_index, test_index in cross_fold.split(training_X):
                            cbr.fit(X=training_X.iloc[train_index], y=training_y[train_index], 
                                eval_set=(training_X.iloc[test_index], training_y[test_index]), 
                                early_stopping_rounds=int(self.cbr_n_iter_no_change.get()),
                                use_best_model=True)
                            predict = cbr.predict(X)
                            if first_col:
                                pr_values = np.array(predict, ndmin=2)
                                pr_values = np.transpose(pr_values)
                                first_col = False
                            else:
                                pr_values = np.insert(pr_values, -1, predict, axis=1)
                        pr_values= np.mean(pr_values, axis=1)

                    # early-stopping mode
                    elif self.x_st_var.get() == 'Yes' and self.cbr_early_stopping.get()==True:
                        cbr.fit(X=check_tr_X_St, y=check_tr_y, eval_set=(check_val_X_St, check_val_y), 
                            early_stopping_rounds=int(self.cbr_n_iter_no_change.get()),
                            use_best_model=True)
                        pr_values = cbr.predict(X_St)
                    elif self.cbr_early_stopping.get()==True:
                        cbr.fit(X=check_tr_X, y=check_tr_y, eval_set=(check_val_X, check_val_y), 
                            early_stopping_rounds=int(self.cbr_n_iter_no_change.get()),
                            use_best_model=True)
                        pr_values = cbr.predict(X)

                    # standard mode
                    elif self.x_st_var.get() == 'Yes':
                        cbr.fit(training_X_St, training_y)
                        pr_values = cbr.predict(X_St)
                    else:
                        cbr.fit(training_X, training_y)
                        pr_values = cbr.predict(X)

                # LightGBM
                if method == 'LightGBM':
                    from lightgbm import LGBMRegressor
                    lgbmr = LGBMRegressor(
                        boosting_type=self.lgbmr_boosting_type.get(),
                        num_leaves=int(self.lgbmr_num_leaves.get()),
                        max_depth=int(self.lgbmr_max_depth.get()),
                        learning_rate=float(self.lgbmr_learning_rate.get()),
                        n_estimators=int(self.lgbmr_n_estimators.get()),
                        subsample_for_bin=int(self.lgbmr_subsample_for_bin.get()),
                        min_split_gain=float(self.lgbmr_min_split_gain.get()),
                        min_child_weight=float(self.lgbmr_min_child_weight.get()),
                        min_child_samples=int(self.lgbmr_min_child_samples.get()),
                        subsample=float(self.lgbmr_subsample.get()),
                        subsample_freq=int(self.lgbmr_subsample_freq.get()),
                        colsample_bytree=float(self.lgbmr_colsample_bytree.get()),
                        reg_alpha=float(self.lgbmr_reg_alpha.get()),
                        reg_lambda=float(self.lgbmr_reg_lambda.get()),
                        n_jobs=int(self.lgbmr_n_jobs.get()),
                        silent=self.lgbmr_silent.get(),
                        # device_type='gpu',
                    )
                    #cv-voting mode
                    if self.x_st_var.get() == 'Yes' and self.lgbmr_cv_voting_mode.get()==True:
                        first_col = True
                        cross_fold = KFold(n_splits = int(self.lgbmr_cv_voting_folds.get()), 
                            shuffle=True)
                        azaza = []
                        for train_index, test_index in cross_fold.split(training_X_St):
                            lgbmr.fit(X=training_X_St.iloc[train_index], y=training_y[train_index], 
                                eval_set=[(training_X_St.iloc[test_index], training_y[test_index])], 
                                early_stopping_rounds=int(self.lgbmr_n_iter_no_change.get()),
                                categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                    else eval(self.lgbmr_categorical_feature.get()))
                            )
                            # best_iteration = lgbmr.get_booster().best_ntree_limit
                            azaza.append(lgbmr.best_score_['valid_0']['l2'])
                            print(azaza)
                            predict = lgbmr.predict(X_St)
                            if first_col:
                                pr_values = np.array(predict, ndmin=2)
                                pr_values = np.transpose(pr_values)
                                first_col = False
                            else:
                                pr_values = np.insert(pr_values, -1, predict, axis=1)
                        pr_values= np.mean(pr_values, axis=1)
                        print(np.mean(azaza))
                    elif self.lgbmr_cv_voting_mode.get()==True:
                        first_col = True
                        cross_fold = KFold(n_splits = int(self.lgbmr_cv_voting_folds.get()), 
                            shuffle=True)
                        azaza = []
                        for train_index, test_index in cross_fold.split(training_X):
                            lgbmr.fit(X=training_X.iloc[train_index], y=training_y[train_index], 
                                eval_set=[(training_X.iloc[test_index], training_y[test_index])],
                                early_stopping_rounds=int(self.lgbmr_n_iter_no_change.get()),
                                categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                    else eval(self.lgbmr_categorical_feature.get()))
                            )
                            # best_iteration = lgbmr.get_booster().best_ntree_limit
                            azaza.append(lgbmr.best_score_['valid_0']['l2'])
                            print(azaza)
                            predict = lgbmr.predict(X)
                            if first_col:
                                pr_values = np.array(predict, ndmin=2)
                                pr_values = np.transpose(pr_values)
                                first_col = False
                            else:
                                pr_values = np.insert(pr_values, -1, predict, axis=1)
                        pr_values= np.mean(pr_values, axis=1)
                        print(np.mean(azaza))

                    # early-stopping mode
                    elif self.x_st_var.get() == 'Yes' and self.lgbmr_early_stopping.get()==True:
                        lgbmr.fit(
                            X=check_tr_X_St, y=check_tr_y, eval_set=(check_val_X_St, check_val_y), 
                            early_stopping_rounds=int(self.lgbmr_n_iter_no_change.get()),
                            categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                else eval(self.lgbmr_categorical_feature.get()))
                        )
                        pr_values = lgbmr.predict(X_St)
                    elif self.lgbmr_early_stopping.get()==True:
                        lgbmr.fit(
                            X=check_tr_X, y=check_tr_y, eval_set=(check_val_X, check_val_y), 
                            early_stopping_rounds=int(self.lgbmr_n_iter_no_change.get()),
                            categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                else eval(self.lgbmr_categorical_feature.get()))
                        )
                        pr_values = lgbmr.predict(X)
                    # standard mode
                    elif self.x_st_var.get() == 'Yes':
                        lgbmr.fit(
                            training_X_St, training_y,
                            categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                else eval(self.lgbmr_categorical_feature.get()))
                        )
                        pr_values = lgbmr.predict(X_St)
                    else:
                        lgbmr.fit(
                            training_X, training_y,
                            categorical_feature=('auto' if self.lgbmr_categorical_feature.get()=='auto'
                                else eval(self.lgbmr_categorical_feature.get()))
                        )
                        pr_values = lgbmr.predict(X)

                self.prediction.data = self.prediction.data.drop('Y', axis = 1, errors='ignore')
                if self.place_result_var.get() == 'Start':
                    self.prediction.data.insert(0, 'Y', pr_values)
                elif self.place_result_var.get() == 'End':
                    self.prediction.data['Y'] = pr_values

                if self.prediction.Viewed.get() == True:
                    self.prediction.pt = Table(self.prediction.view_frame, 
                        dataframe=self.prediction.data, showtoolbar=True, showstatusbar=True, 
                        height=350, notebook=Data_Preview.notebook.nb, dp_main=self.prediction)
                    self.prediction.pt.show()
                    self.prediction.pt.redraw()
                    Data_Preview.notebook.nb.select(self.prediction.view_frame)
                    Data_Preview.root.lift()
            except ValueError as e:
                messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))

            self.pb2.destroy()

        # function to run prediction in a particular thread
        def try_make_regression(method):
            x_are_same = tk.BooleanVar(value=False)

            if (self.training.x_from_var.get()!=self.prediction.x_from_var.get()
                or self.training.x_to_var.get()!=self.prediction.x_to_var.get()):
                if messagebox.askyesno("Warning",
                    "X from and x to for training and prediction are different. Continue?"):
                    x_are_same.set(True)
            if ((self.training.x_from_var.get()==self.prediction.x_from_var.get()
                and self.training.x_to_var.get()==self.prediction.x_to_var.get()) 
                or x_are_same.get()==True):
                try:
                    if hasattr(self, 'pb2'):
                        self.pb2.destroy()
                    self.pb2 = ttk.Progressbar(self.frame, mode='indeterminate', length=100)
                    self.pb2.place(x=425, y=360)
                    self.pb2.start(10)

                    thread1 = Thread(target=make_regression, args=(self.pr_method.get(),))
                    thread1.daemon = True
                    thread1.start()
                    # make_regression(method)
                except ValueError as e:
                    self.pb2.destroy()
                    messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))

        # Application interface

        self.bg_frame = ttk.Frame(rgr_app.root, width=10, height=h)
        self.bg_frame.place(x=0, y=0)

        self.frame = ttk.Frame(rgr_app.root, width=w, height=h)
        self.frame.place(x=10, y=0)
        
        ttk.Label(self.frame, text='Train Data file:', font=myfont1).place(x=10, y=10)
        e1 = ttk.Entry(self.frame, font=myfont1, width=40)
        e1.place(x=120, y=10)
        
        ttk.Button(self.frame, text='Choose file', 
            command=lambda: open_file(self, e1)).place(x=490, y=10)
        
        ttk.Label(self.frame, text='List number:').place(x=120,y=50)
        training_sheet_entry = ttk.Entry(self.frame, 
            textvariable=self.training.sheet, font=myfont1, width=3)
        training_sheet_entry.place(x=215,y=52)
                
        ttk.Button(self.frame, text='Load data ', 
            command=lambda: load_data(self, self.training, e1, 
                'rgr training')).place(x=490, y=50)
        
        cb1 = ttk.Checkbutton(self.frame, text="header", 
            variable=self.training.header_var, takefocus=False)
        cb1.place(x=10, y=50)
        
        ttk.Label(self.frame, text='Data status:').place(x=10, y=95)
        self.training.data_status = ttk.Label(self.frame, text='Not Loaded')
        self.training.data_status.place(x=120, y=95)

        ttk.Button(self.frame, text='View/Change', 
            command=lambda: Data_Preview(self, self.training, 
                'rgr training', parent)).place(x=230, y=95)

        ttk.Button(self.frame, text="Methods' specifications", 
                  command=lambda: rgr_mtds_specification(self, parent)).place(x=400, y=95)
        ttk.Button(self.frame, text='Perform comparison', 
            command=lambda: try_compare_methods(self, self.training)).place(x=400, y=130)
        self.show_res_button = ttk.Button(self.frame, text='Show Results', 
            command=lambda: self.Comp_results(self, parent))

        ttk.Button(self.frame, text='Grid Search', 
            command=lambda: self.Grid_search(self, parent)).place(x=400, y=200)
        
        ttk.Label(self.frame, text='Choose y', font=myfont1).place(x=30, y=140)
        self.y_var = tk.StringVar()
        self.combobox1 = ttk.Combobox(self.frame, 
            textvariable=self.y_var, width=14, values=[])
        self.combobox1.place(x=105,y=142)
        
        ttk.Label(self.frame, text='X from', font=myfont1).place(x=225, y=130)
        self.tr_x_from_combobox = ttk.Combobox(self.frame, 
            textvariable=self.training.x_from_var, width=14, values=[])
        self.tr_x_from_combobox.place(x=275, y=132)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=225, y=155)
        self.tr_x_to_combobox = ttk.Combobox(self.frame, 
            textvariable=self.training.x_to_var, width=14, values=[])
        self.tr_x_to_combobox.place(x=275, y=157)

        ttk.Label(self.frame, text='X Standartization', font=myfont1).place(x=30, y=175)

        self.x_st_var = tk.StringVar(value='If needed')
        self.combobox2 = ttk.Combobox(self.frame, textvariable=self.x_st_var, width=12,
                                        values=['No', 'If needed', 'Yes'])
        self.combobox2.place(x=160,y=180)

        ttk.Label(self.frame, text='Handle categorical', font=myfont1).place(x=30, y=205)
        self.handle_cat_var = tk.StringVar(value='No')
        self.handle_cat_combobox = ttk.Combobox(self.frame, textvariable=self.handle_cat_var, width=12,
                                        values=['No', 'Dummies', 'One Hot Encoding', 'Ordinal Encoding'])
        self.handle_cat_combobox.place(x=160,y=208)

        ttk.Label(self.frame, text='Number of folds:', font=myfont1).place(x=30, y=235)
        e2 = ttk.Entry(self.frame, font=myfont1, width=4)
        e2.place(x=150, y=237)
        e2.insert(0, "5")

        ttk.Label(self.frame, text='Number of repeats:', font=myfont1).place(x=200, y=235)
        rep_entry = ttk.Entry(self.frame, font=myfont1, width=4)
        rep_entry.place(x=330, y=238)
        rep_entry.insert(0, "1")

        ttk.Label(self.frame, text='Predict Data file:', font=myfont1).place(x=10, y=280)
        pr_data_entry = ttk.Entry(self.frame, font=myfont1, width=38)
        pr_data_entry.place(x=140, y=280)
        
        ttk.Button(self.frame, text='Choose file', 
            command=lambda: open_file(self, pr_data_entry)).place(x=490, y=280)
        
        ttk.Label(self.frame, text='List number:', font=myfont1).place(x=120,y=325)
        pr_sheet_entry = ttk.Entry(self.frame, 
            textvariable=self.prediction.sheet, font=myfont1, width=3)
        pr_sheet_entry.place(x=215,y=327)
        
        ttk.Button(self.frame, text='Load data ', 
            command=lambda: load_data(self, self.prediction, pr_data_entry, 
                'rgr prediction')).place(x=490, y=320)
        
        cb4 = ttk.Checkbutton(self.frame, text="header", 
            variable=self.prediction.header_var, takefocus=False)
        cb4.place(x=10, y=320)
        
        ttk.Label(self.frame, text='Data status:', font=myfont).place(x=10, y=385)
        self.prediction.data_status = ttk.Label(self.frame, text='Not Loaded', font=myfont)
        self.prediction.data_status.place(x=120, y=385)

        ttk.Button(self.frame, text='View/Change', command=lambda: 
            Data_Preview(self, self.prediction, 
                'rgr prediction', parent)).place(x=230, y=375)
        
        self.pr_method = tk.StringVar(value='Least squares')
        
        self.combobox9 = ttk.Combobox(self.frame, textvariable=self.pr_method, width=15, 
            values=['Least squares', 'Ridge', 'Lasso', 'Random Forest', 'Support Vector',
                'SGD', 'Nearest Neighbor', 'Gaussian Process', 'Decision Tree',
                'Multi-layer Perceptron', 'XGBoost', 'CatBoost', 'LightGBM'])
        self.combobox9.place(x=105,y=422)
        ttk.Label(self.frame, text='Method', font=myfont1).place(x=30, y=420)

        ttk.Label(self.frame, text='Place result', font=myfont1).place(x=30, y=445)
        self.place_result_var = tk.StringVar(value='End')
        self.combobox9 = ttk.Combobox(self.frame, 
            textvariable=self.place_result_var, width=10, values=['Start', 'End'])
        self.combobox9.place(x=120,y=447)
        
        ttk.Label(self.frame, text='X from', font=myfont1).place(x=225, y=420)
        self.pr_x_from_combobox = ttk.Combobox(self.frame, 
            textvariable=self.prediction.x_from_var, width=14, values=[])
        self.pr_x_from_combobox.place(x=275, y=422)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=225, y=445)
        self.pr_x_to_combobox = ttk.Combobox(self.frame, 
            textvariable=self.prediction.x_to_var, width=14, values=[])
        self.pr_x_to_combobox.place(x=275, y=447)

        ttk.Button(self.frame, text='Predict', width=7,
            command=lambda: 
                try_make_regression(method=self.pr_method.get())).place(x=400, y=395)
        ttk.Button(self.frame, text='Quit', width=7,
            command=lambda: quit_back(rgr_app.root, parent)).place(x=400, y=430)
        
        ttk.Button(self.frame, text='Save to file', width=9,
            command=lambda: save_results(self, self.prediction, 'rgr result')).place(x=500, y=395)
        ttk.Button(self.frame, text='Save to sql', width=9,
            command=lambda: save_to_sql(self, self.prediction, 'rgr_result')).place(x=500, y=430)

# sub-app for methods' specifications  
class rgr_mtds_specification:
    def __init__(self, prev, parent):
        self.root = tk.Toplevel(parent)

        #setting main window's parameters       
        w = 640
        h = 500
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.focus_force()
        self.root.resizable(False, False)
        self.root.title('Regression methods specification')

        self.canvas = tk.Canvas(self.root)
        self.frame = ttk.Frame(self.canvas, width=w, height=h)
        # self.scrollbar = ttk.Scrollbar(self.canvas, orient="horizontal", 
        #     command=self.canvas.xview)
        # self.canvas.configure(xscrollcommand=self.scrollbar.set)
        # self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # self.frame.bind(
        #     "<Configure>",
        #     lambda e: self.canvas.configure(
        #         scrollregion=self.canvas.bbox("all")
        #     )
        # )
        # def mouse_scroll(event):
        #     if event.delta:
        #         self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        #     else:
        #         if event.num == 5:
        #             move = 1
        #         else:
        #             move = -1
        #         self.canvas.xview_scroll(move, "units")
        # self.canvas.bind_all("<MouseWheel>", mouse_scroll)
        self.canvas_frame = self.canvas.create_window(0, 0, window=self.frame, anchor="nw")

        main_menu = tk.Menu(self.root)
        self.root.config(menu=main_menu)
        settings_menu = tk.Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label='Restore Defaults', 
            command=lambda: self.restore_defaults(prev, parent))
        settings_menu.add_command(label='Save settings', 
            command=lambda: self.save_settings(prev, parent))
        settings_menu.add_command(label='Load settings', 
            command=lambda: self.load_settings(prev, parent))

        self.nb = ttk.Notebook(self.frame)
        self.nb.place(x=0, y=0)
        self.general_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.general_frame, text='General')
        self.ls_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.ls_frame, text='Linear')
        self.sgd_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.sgd_frame, text='SGD')
        self.sv_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.sv_frame, text='SVM')
        self.gp_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.gp_frame, text='GP-KN')
        self.dt_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.dt_frame, text='DT')
        self.rf_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.rf_frame, text='RF')
        self.mlp_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.mlp_frame, text='MLP')
        self.xgb_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.xgb_frame, text='XGB')
        self.cb_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.cb_frame, text='CB')
        self.lgb_frame = ttk.Frame(self.nb, width=w, height=h)
        self.nb.add(self.lgb_frame, text='LGBM')
        # self.nb.select(self.general_frame)
        
        ttk.Label(self.general_frame, text=' Include in Comparison', font=myfont_b).place(x=80, y=20)
        ttk.Label(self.general_frame, text='Least Squares', font=myfont1).place(x=20, y=60)
        inc_cb1 = ttk.Checkbutton(self.general_frame, variable=prev.lr_include_comp, takefocus=False)
        inc_cb1.place(x=140, y=60)
        ttk.Label(self.general_frame, text='Ridge', font=myfont1).place(x=20, y=85)
        inc_cb2 = ttk.Checkbutton(self.general_frame, variable=prev.rr_include_comp, takefocus=False)
        inc_cb2.place(x=140, y=85)
        ttk.Label(self.general_frame, text='Lasso', font=myfont1).place(x=20, y=110)
        inc_cb3 = ttk.Checkbutton(self.general_frame, variable=prev.lassor_include_comp, takefocus=False)
        inc_cb3.place(x=140, y=110)
        ttk.Label(self.general_frame, text='Decision Tree', font=myfont1).place(x=20, y=135)
        inc_cb4 = ttk.Checkbutton(self.general_frame, variable=prev.dtr_include_comp, takefocus=False)
        inc_cb4.place(x=140, y=135)
        ttk.Label(self.general_frame, text='Random Forest', font=myfont1).place(x=20, y=160)
        inc_cb5 = ttk.Checkbutton(self.general_frame, variable=prev.rfr_include_comp, takefocus=False)
        inc_cb5.place(x=140, y=160)
        ttk.Label(self.general_frame, text='Support Vector', font=myfont1).place(x=20, y=185)
        inc_cb6 = ttk.Checkbutton(self.general_frame, variable=prev.svr_include_comp, takefocus=False)
        inc_cb6.place(x=140, y=185)
        ttk.Label(self.general_frame, text='SGD', font=myfont1).place(x=20, y=210)
        inc_cb7 = ttk.Checkbutton(self.general_frame, variable=prev.sgdr_include_comp, takefocus=False)
        inc_cb7.place(x=140, y=210)
        ttk.Label(self.general_frame, text='Nearest Neighbor', font=myfont1).place(x=210, y=60)
        inc_cb8 = ttk.Checkbutton(self.general_frame, variable=prev.knr_include_comp, takefocus=False)
        inc_cb8.place(x=340, y=60)
        ttk.Label(self.general_frame, text='Gaussian Process', font=myfont1).place(x=210, y=85)
        inc_cb9 = ttk.Checkbutton(self.general_frame, variable=prev.gpr_include_comp, takefocus=False)
        inc_cb9.place(x=340, y=85)
        ttk.Label(self.general_frame, text='Multi-layer\nPerceptron', font=myfont1).place(x=210, y=110)
        inc_cb10 = ttk.Checkbutton(self.general_frame, variable=prev.mlpr_include_comp, takefocus=False)
        inc_cb10.place(x=340, y=120)
        ttk.Label(self.general_frame, text='XGBoost', font=myfont1).place(x=210, y=155)
        inc_cb11 = ttk.Checkbutton(self.general_frame, variable=prev.xgbr_include_comp, takefocus=False)
        inc_cb11.place(x=340, y=155)
        ttk.Label(self.general_frame, text='CatBoost', font=myfont1).place(x=210, y=180)
        inc_cb12 = ttk.Checkbutton(self.general_frame, variable=prev.cbr_include_comp, takefocus=False)
        inc_cb12.place(x=340, y=180)
        ttk.Label(self.general_frame, text='LightGBM', font=myfont1).place(x=210, y=205)
        inc_cb13 = ttk.Checkbutton(self.general_frame, variable=prev.lgbmr_include_comp, takefocus=False)
        inc_cb13.place(x=340, y=205)
        # ttk.Label(self.frame, text='Scoring', font=myfont1).place(x=5, y=320)
        # inc_combobox1 = ttk.Combobox(self.frame, textvariable=prev.inc_scoring, width=15, 
        #     values=['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 
        #         'neg_mean_squared_log_error', 'neg_mean_absolute_error'])
        # inc_combobox1.place(x=60,y=323)

        ttk.Label(self.ls_frame, text='Least Squares', font=myfont_b).place(x=40, y=10)
        ttk.Label(self.ls_frame, text='Equation type', font=myfont1).place(x=20, y=40)
        ls_combobox1 = ttk.Combobox(self.ls_frame, textvariable=prev.lr_function_type, width=6, 
            values=['Linear', 'Polynomial', 'Exponential', 'Power', 'Logarithmic'])
        ls_combobox1.place(x=140,y=40)
        ttk.Label(self.ls_frame, text='Fit intercept', font=myfont1).place(x=20, y=65)
        ls_cb2 = ttk.Checkbutton(self.ls_frame, variable=prev.lr_fit_intercept, takefocus=False)
        ls_cb2.place(x=140, y=65)
        ttk.Label(self.ls_frame, text='Normalize', font=myfont1).place(x=20, y=90)
        ls_cb3 = ttk.Checkbutton(self.ls_frame, variable=prev.lr_normalize, takefocus=False)
        ls_cb3.place(x=140, y=90)
        ttk.Label(self.ls_frame, text='Copy X', font=myfont1).place(x=20, y=115)
        ls_cb4 = ttk.Checkbutton(self.ls_frame, variable=prev.lr_copy_X, takefocus=False)
        ls_cb4.place(x=140, y=115)
        ttk.Label(self.ls_frame, text='n jobs', font=myfont1).place(x=20, y=140)
        ls_e1 = ttk.Entry(self.ls_frame, textvariable=prev.lr_n_jobs, font=myfont2, width=7)
        ls_e1.place(x=140, y=140)
        ttk.Label(self.ls_frame, text='Positive', font=myfont1).place(x=20, y=165)
        ls_cb5 = ttk.Checkbutton(self.ls_frame, variable=prev.lr_positive, takefocus=False)
        ls_cb5.place(x=140, y=165)

        ttk.Label(self.ls_frame, text='Lasso', font=myfont_b).place(x=270, y=10)
        ttk.Label(self.ls_frame, text='Alpha', font=myfont1).place(x=220, y=40)
        lasso_e1 = ttk.Entry(self.ls_frame, textvariable=prev.lassor_alpha, font=myfont2, width=7)
        lasso_e1.place(x=340, y=40)
        ttk.Label(self.ls_frame, text='Fit intercept', font=myfont1).place(x=220, y=65)
        lasso_cb1 = ttk.Checkbutton(self.ls_frame, variable=prev.lassor_fit_intercept, takefocus=False)
        lasso_cb1.place(x=340, y=65)
        ttk.Label(self.ls_frame, text='Normalize', font=myfont1).place(x=220, y=90)
        lasso_cb2 = ttk.Checkbutton(self.ls_frame, variable=prev.lassor_normalize, takefocus=False)
        lasso_cb2.place(x=340, y=90)
        ttk.Label(self.ls_frame, text='Precompute', font=myfont1).place(x=220, y=115)
        lasso_cb6 = ttk.Checkbutton(self.ls_frame, variable=prev.lassor_precompute, takefocus=False)
        lasso_cb6.place(x=340, y=115)
        ttk.Label(self.ls_frame, text='Copy X', font=myfont1).place(x=220, y=140)
        lasso_cb3 = ttk.Checkbutton(self.ls_frame, variable=prev.lassor_copy_X, takefocus=False)
        lasso_cb3.place(x=340, y=140)
        ttk.Label(self.ls_frame, text='Max iter', font=myfont1).place(x=220, y=165)
        lasso_e2 = ttk.Entry(self.ls_frame, textvariable=prev.lassor_max_iter, font=myfont2, width=7)
        lasso_e2.place(x=340, y=165)
        ttk.Label(self.ls_frame, text='tol', font=myfont1).place(x=220, y=190)
        lasso_e3 = ttk.Entry(self.ls_frame, textvariable=prev.lassor_tol, font=myfont2, width=7)
        lasso_e3.place(x=340, y=190)
        ttk.Label(self.ls_frame, text='Warm start', font=myfont1).place(x=220, y=215)
        lasso_cb4 = ttk.Checkbutton(self.ls_frame, variable=prev.lassor_warm_start, takefocus=False)
        lasso_cb4.place(x=340, y=215)
        ttk.Label(self.ls_frame, text='Positive', font=myfont1).place(x=220, y=240)
        lasso_cb5 = ttk.Checkbutton(self.ls_frame, variable=prev.lassor_positive, takefocus=False)
        lasso_cb5.place(x=340, y=240)
        ttk.Label(self.ls_frame, text='Random state', font=myfont1).place(x=220, y=265)
        lasso_e4 = ttk.Entry(self.ls_frame, 
            textvariable=prev.lassor_random_state, font=myfont2, width=7)
        lasso_e4.place(x=340, y=265)
        ttk.Label(self.ls_frame, text='Selection', font=myfont1).place(x=220, y=290)
        lasso_combobox2 = ttk.Combobox(self.ls_frame, textvariable=prev.lassor_selection, width=6, 
            values=['cyclic', 'random'])
        lasso_combobox2.place(x=340,y=290)

        ttk.Label(self.ls_frame, text='Ridge', font=myfont_b).place(x=460, y=10)
        ttk.Label(self.ls_frame, text='Alpha', font=myfont1).place(x=420, y=40)
        rr_e1 = ttk.Entry(self.ls_frame, textvariable=prev.rr_alpha, font=myfont2, width=7)
        rr_e1.place(x=540, y=40)
        ttk.Label(self.ls_frame, text='Fit intercept', font=myfont1).place(x=420, y=65)
        rr_cb2 = ttk.Checkbutton(self.ls_frame, variable=prev.rr_fit_intercept, takefocus=False)
        rr_cb2.place(x=540, y=65)
        ttk.Label(self.ls_frame, text='Normalize', font=myfont1).place(x=420, y=90)
        rr_cb3 = ttk.Checkbutton(self.ls_frame, variable=prev.rr_normalize, takefocus=False)
        rr_cb3.place(x=540, y=90)
        ttk.Label(self.ls_frame, text='Copy X', font=myfont1).place(x=420, y=115)
        rr_cb4 = ttk.Checkbutton(self.ls_frame, variable=prev.rr_copy_X, takefocus=False)
        rr_cb4.place(x=540, y=115)
        ttk.Label(self.ls_frame, text='Max iter', font=myfont1).place(x=420, y=140)
        rr_e2 = ttk.Entry(self.ls_frame, textvariable=prev.rr_max_iter, font=myfont2, width=7)
        rr_e2.place(x=540, y=140)
        ttk.Label(self.ls_frame, text='tol', font=myfont1).place(x=420, y=165)
        rr_e3 = ttk.Entry(self.ls_frame, textvariable=prev.rr_tol, font=myfont2, width=7)
        rr_e3.place(x=540, y=165)
        ttk.Label(self.ls_frame, text='Solver', font=myfont1).place(x=420, y=190)
        rr_combobox1 = ttk.Combobox(self.ls_frame, textvariable=prev.rr_solver, width=6, 
            values=['auto', 'svd', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        rr_combobox1.place(x=540,y=190)
        ttk.Label(self.ls_frame, text='Random state', font=myfont1).place(x=420, y=215)
        rr_e4 = ttk.Entry(self.ls_frame, textvariable=prev.rr_random_state, font=myfont2, width=7)
        rr_e4.place(x=540, y=215)

        ttk.Label(self.sgd_frame, text='Stochastic Gradient Descent', font=myfont_b).place(x=100, y=10)
        ttk.Label(self.sgd_frame, text='Loss', font=myfont1).place(x=20, y=40)
        sgd_combobox1 = ttk.Combobox(self.sgd_frame, textvariable=prev.sgdr_loss, width=11, 
            values=['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])
        sgd_combobox1.place(x=130,y=40)
        ttk.Label(self.sgd_frame, text='Penalty', font=myfont1).place(x=20, y=65)
        sgd_combobox2 = ttk.Combobox(self.sgd_frame, textvariable=prev.sgdr_penalty, width=7, 
            values=['l2', 'l1', 'elasticnet'])
        sgd_combobox2.place(x=140,y=65)
        ttk.Label(self.sgd_frame, text='Alpha', font=myfont1).place(x=20, y=90)
        sgd_e1 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_alpha, font=myfont2, width=7)
        sgd_e1.place(x=140,y=90)
        ttk.Label(self.sgd_frame, text='l1 ratio', font=myfont1).place(x=20, y=115)
        sgd_e2 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_l1_ratio, font=myfont2, width=7)
        sgd_e2.place(x=140,y=115)
        ttk.Label(self.sgd_frame, text='fit intercept', font=myfont1).place(x=20, y=140)
        sgd_cb1 = ttk.Checkbutton(self.sgd_frame, variable=prev.sgdr_fit_intercept, takefocus=False)
        sgd_cb1.place(x=140,y=140)
        ttk.Label(self.sgd_frame, text='Max iter', font=myfont1).place(x=20, y=165)
        sgd_e3 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_max_iter, font=myfont2, width=7)
        sgd_e3.place(x=140,y=165)
        ttk.Label(self.sgd_frame, text='tol', font=myfont1).place(x=20, y=190)
        sgd_e4 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_tol, font=myfont2, width=7)
        sgd_e4.place(x=140,y=190)
        ttk.Label(self.sgd_frame, text='Shuffle', font=myfont1).place(x=20, y=215)
        sgd_cb2 = ttk.Checkbutton(self.sgd_frame, variable=prev.sgdr_shuffle, takefocus=False)
        sgd_cb2.place(x=140,y=215)
        ttk.Label(self.sgd_frame, text='Verbose', font=myfont1).place(x=20, y=240)
        sgd_e5 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_verbose, font=myfont2, width=7)
        sgd_e5.place(x=140,y=240)
        ttk.Label(self.sgd_frame, text='Epsilon', font=myfont1).place(x=220, y=40)
        sgd_e6 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_epsilon, font=myfont2, width=7)
        sgd_e6.place(x=320,y=40)
        ttk.Label(self.sgd_frame, text='Random state', font=myfont1).place(x=220, y=65)
        sgd_e8 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_random_state, font=myfont2, width=7)
        sgd_e8.place(x=340,y=65)
        ttk.Label(self.sgd_frame, text='Learning rate', font=myfont1).place(x=220, y=90)
        sgd_combobox3 = ttk.Combobox(self.sgd_frame, textvariable=prev.sgdr_learning_rate, width=8, 
            values=['constant', 'optimal', 'invscaling', 'adaptive'])
        sgd_combobox3.place(x=340,y=90)
        ttk.Label(self.sgd_frame, text='eta0', font=myfont1).place(x=220, y=115)
        sgd_e9 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_eta0, font=myfont2, width=7)
        sgd_e9.place(x=340,y=115)
        ttk.Label(self.sgd_frame, text='power t', font=myfont1).place(x=220, y=140)
        sgd_e10 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_power_t, font=myfont2, width=7)
        sgd_e10.place(x=340,y=140)
        ttk.Label(self.sgd_frame, text='Early stopping', font=myfont1).place(x=220, y=165)
        sgd_cb3 = ttk.Checkbutton(self.sgd_frame, variable=prev.sgdr_early_stopping, takefocus=False)
        sgd_cb3.place(x=340,y=165)
        ttk.Label(self.sgd_frame, text='Validation fraction', font=myfont1).place(x=220, y=190)
        sgd_e11 = ttk.Entry(self.sgd_frame, 
            textvariable=prev.sgdr_validation_fraction, font=myfont2, width=7)
        sgd_e11.place(x=340,y=190)
        ttk.Label(self.sgd_frame, text='n iter no change', font=myfont1).place(x=220, y=215)
        sgd_e12 = ttk.Entry(self.sgd_frame, 
            textvariable=prev.sgdr_n_iter_no_change, font=myfont2, width=7)
        sgd_e12.place(x=340,y=215)
        ttk.Label(self.sgd_frame, text='Warm start', font=myfont1).place(x=220, y=240)
        sgd_cb4 = ttk.Checkbutton(self.sgd_frame, variable=prev.sgdr_warm_start, takefocus=False)
        sgd_cb4.place(x=340,y=240)
        ttk.Label(self.sgd_frame, text='Average', font=myfont1).place(x=220, y=265)
        sgd_e13 = ttk.Entry(self.sgd_frame, textvariable=prev.sgdr_average, font=myfont2, width=7)
        sgd_e13.place(x=340,y=265)

        ttk.Label(self.sv_frame, text='Support Vector', font=myfont_b).place(x=170, y=10)
        ttk.Label(self.sv_frame, text='Kernel', font=myfont1).place(x=20, y=40)
        sv_combobox1 = ttk.Combobox(self.sv_frame, textvariable=prev.svr_kernel, width=8, 
            values=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
        sv_combobox1.place(x=135,y=40)
        ttk.Label(self.sv_frame, text='Degree', font=myfont1).place(x=20, y=65)
        sv_e2 = ttk.Entry(self.sv_frame, textvariable=prev.svr_degree, font=myfont2, width=7)
        sv_e2.place(x=140,y=65)
        ttk.Label(self.sv_frame, text='Gamma', font=myfont1).place(x=20, y=90)
        sv_e3 = ttk.Entry(self.sv_frame, textvariable=prev.svr_gamma, font=myfont2, width=7)
        sv_e3.place(x=140,y=90)
        ttk.Label(self.sv_frame, text='coef0', font=myfont1).place(x=20, y=115)
        sv_e4 = ttk.Entry(self.sv_frame, textvariable=prev.svr_coef0, font=myfont2, width=7)
        sv_e4.place(x=140,y=115)
        ttk.Label(self.sv_frame, text='tol', font=myfont1).place(x=20, y=140)
        sv_e5 = ttk.Entry(self.sv_frame, textvariable=prev.svr_tol, font=myfont2, width=7)
        sv_e5.place(x=140,y=140)
        ttk.Label(self.sv_frame, text='C', font=myfont1).place(x=20, y=165)
        sv_e1 = ttk.Entry(self.sv_frame, textvariable=prev.svr_C, font=myfont2, width=7)
        sv_e1.place(x=140,y=165)
        ttk.Label(self.sv_frame, text='epsilon', font=myfont1).place(x=220, y=40)
        sv_e8 = ttk.Entry(self.sv_frame, textvariable=prev.svr_epsilon, font=myfont2, width=7)
        sv_e8.place(x=340,y=40)
        ttk.Label(self.sv_frame, text='shrinking', font=myfont1).place(x=220, y=65)
        sv_cb1 = ttk.Checkbutton(self.sv_frame, variable=prev.svr_shrinking, takefocus=False)
        sv_cb1.place(x=340,y=65)
        ttk.Label(self.sv_frame, text='Cache size', font=myfont1).place(x=220, y=90)
        sv_e6 = ttk.Entry(self.sv_frame, textvariable=prev.svr_cache_size, font=myfont2, width=7)
        sv_e6.place(x=340,y=90)
        ttk.Label(self.sv_frame, text='Verbose', font=myfont1).place(x=220, y=115)
        sv_cb3 = ttk.Checkbutton(self.sv_frame, variable=prev.svr_verbose, takefocus=False)
        sv_cb3.place(x=340,y=115)
        ttk.Label(self.sv_frame, text='Max iter', font=myfont1).place(x=220, y=140)
        sv_e7 = ttk.Entry(self.sv_frame, textvariable=prev.svr_max_iter, font=myfont2, width=7)
        sv_e7.place(x=340,y=140)

        ttk.Label(self.gp_frame, text='Gaussian Process', font=myfont_b).place(x=30, y=10)
        ttk.Label(self.gp_frame, text='Alpha', font=myfont1).place(x=20, y=40)
        gp_e2 = ttk.Entry(self.gp_frame, textvariable=prev.gpr_alpha, font=myfont2, width=7)
        gp_e2.place(x=140,y=40)
        ttk.Label(self.gp_frame, text='n restarts\noptimizer', font=myfont1).place(x=20, y=65)
        gp_e1 = ttk.Entry(self.gp_frame, textvariable=prev.gpr_n_restarts_optimizer, font=myfont2, 
            width=7)
        gp_e1.place(x=140,y=75)
        ttk.Label(self.gp_frame, text='Normalize y', font=myfont1).place(x=20, y=100)
        gp_cb1 = ttk.Checkbutton(self.gp_frame, variable=prev.gpr_normalize_y, takefocus=False)
        gp_cb1.place(x=140,y=100)
        ttk.Label(self.gp_frame, text='Copy X train', font=myfont1).place(x=20, y=125)
        gp_cb2 = ttk.Checkbutton(self.gp_frame, variable=prev.gpr_copy_X_train, takefocus=False)
        gp_cb2.place(x=140,y=125)
        ttk.Label(self.gp_frame, text='Random state', font=myfont1).place(x=20, y=150)
        gp_e3 = ttk.Entry(self.gp_frame, textvariable=prev.gpr_random_state, font=myfont2, width=7)
        gp_e3.place(x=140,y=150)
        
        ttk.Label(self.gp_frame, text='Nearest Neighbor', font=myfont_b).place(x=230, y=10)
        ttk.Label(self.gp_frame, text='n neighbors', font=myfont1).place(x=220, y=40)
        kn_e1 = ttk.Entry(self.gp_frame, textvariable=prev.knr_n_neighbors, font=myfont2, width=7)
        kn_e1.place(x=340,y=40)
        ttk.Label(self.gp_frame, text='Weights', font=myfont1).place(x=220, y=65)
        gp_combobox1 = ttk.Combobox(self.gp_frame, textvariable=prev.knr_weights, width=7, 
            values=['uniform', 'distance'])
        gp_combobox1.place(x=340,y=65)
        ttk.Label(self.gp_frame, text='Algorithm', font=myfont1).place(x=220, y=90)
        gp_combobox2 = ttk.Combobox(self.gp_frame, textvariable=prev.knr_algorithm, width=7, 
            values=['auto', 'ball_tree', 'kd_tree', 'brute'])
        gp_combobox2.place(x=340,y=90)
        ttk.Label(self.gp_frame, text='Leaf size', font=myfont1).place(x=220, y=115)
        kn_e2 = ttk.Entry(self.gp_frame, textvariable=prev.knr_leaf_size, font=myfont2, width=7)
        kn_e2.place(x=340,y=115)
        ttk.Label(self.gp_frame, text='p', font=myfont1).place(x=220, y=140)
        kn_e3 = ttk.Entry(self.gp_frame, textvariable=prev.knr_p, font=myfont2, width=7)
        kn_e3.place(x=340,y=140)
        ttk.Label(self.gp_frame, text='Metric', font=myfont1).place(x=220, y=165)
        gp_combobox3 = ttk.Combobox(self.gp_frame, textvariable=prev.knr_metric, width=9, 
            values=['euclidean', 'manhattan', 'chebyshev', 'minkowski', 
                'wminkowski', 'seuclidean', 'mahalanobis'])
        gp_combobox3.place(x=330,y=165)
        ttk.Label(self.gp_frame, text='n jobs', font=myfont1).place(x=220, y=190)
        kn_e4 = ttk.Entry(self.gp_frame, textvariable=prev.knr_n_jobs, font=myfont2, width=7)
        kn_e4.place(x=340,y=190)

        ttk.Label(self.dt_frame, text='Decision Tree', font=myfont_b).place(x=170, y=10)
        ttk.Label(self.dt_frame, text='Criterion', font=myfont1).place(x=20, y=40)
        dt_combobox3 = ttk.Combobox(self.dt_frame, textvariable=prev.dtr_criterion, width=8, 
            values=['mse', 'friedman_mse', 'mae', 'poisson'])
        dt_combobox3.place(x=140,y=40)
        ttk.Label(self.dt_frame, text='Splitter', font=myfont1).place(x=20, y=65)
        dt_combobox4 = ttk.Combobox(self.dt_frame, textvariable=prev.dtr_splitter, width=6, 
            values=['best', 'random'])
        dt_combobox4.place(x=140,y=65)
        ttk.Label(self.dt_frame, text='Max Depth', font=myfont1).place(x=20, y=90)
        dt_e1 = ttk.Entry(self.dt_frame, textvariable=prev.dtr_max_depth, font=myfont2, width=7)
        dt_e1.place(x=140,y=90)
        ttk.Label(self.dt_frame, text='Min samples split', font=myfont1).place(x=20, y=115)
        dt_e2 = ttk.Entry(self.dt_frame, 
            textvariable=prev.dtr_min_samples_split, font=myfont2, width=7)
        dt_e2.place(x=140,y=115)
        ttk.Label(self.dt_frame, text='Min samples leaf', font=myfont1).place(x=20, y=140)
        dt_e3 = ttk.Entry(self.dt_frame, 
            textvariable=prev.dtr_min_samples_leaf, font=myfont2, width=7)
        dt_e3.place(x=140,y=140)
        ttk.Label(self.dt_frame, text="Min weight\nfraction leaf", font=myfont1).place(x=20, y=165)
        dt_e4 = ttk.Entry(self.dt_frame, 
            textvariable=prev.dtr_min_weight_fraction_leaf, font=myfont2, width=7)
        dt_e4.place(x=140,y=175)
        ttk.Label(self.dt_frame, text='Max features', font=myfont1).place(x=220, y=40)
        dt_e5 = ttk.Entry(self.dt_frame, textvariable=prev.dtr_max_features, font=myfont2, width=7)
        dt_e5.place(x=340,y=40)
        ttk.Label(self.dt_frame, text='Random state', font=myfont1).place(x=220, y=65)
        dt_e6 = ttk.Entry(self.dt_frame, textvariable=prev.dtr_random_state, font=myfont2, width=7)
        dt_e6.place(x=340,y=65)
        ttk.Label(self.dt_frame, text='Max leaf nodes', font=myfont1).place(x=220, y=90)
        dt_e7 = ttk.Entry(self.dt_frame, textvariable=prev.dtr_max_leaf_nodes, font=myfont2, width=7)
        dt_e7.place(x=340,y=90)
        ttk.Label(self.dt_frame, text="Min impurity\ndecrease", font=myfont1).place(x=220, y=115)
        dt_e8 = ttk.Entry(self.dt_frame, 
            textvariable=prev.dtr_min_impurity_decrease, font=myfont2, width=7)
        dt_e8.place(x=340,y=125)
        ttk.Label(self.dt_frame, text="CCP alpha", font=myfont1).place(x=220, y=160)
        dt_e9 = ttk.Entry(self.dt_frame, textvariable=prev.dtr_ccp_alpha, font=myfont2, width=7)
        dt_e9.place(x=340,y=160)

        ttk.Label(self.rf_frame, text='Random Forest', font=myfont_b).place(x=170, y=10)
        ttk.Label(self.rf_frame, text='Trees number', font=myfont1).place(x=20, y=40)
        rf_e1 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_n_estimators, font=myfont2, width=7)
        rf_e1.place(x=140,y=40)
        ttk.Label(self.rf_frame, text='Criterion', font=myfont1).place(x=20, y=65)
        rf_combobox5 = ttk.Combobox(self.rf_frame, textvariable=prev.rfr_criterion, width=7, 
            values=['mse', 'mae'])
        rf_combobox5.place(x=140,y=65)
        ttk.Label(self.rf_frame, text='Max Depth', font=myfont1).place(x=20, y=90)
        rf_e2 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_max_depth, font=myfont2, width=7)
        rf_e2.place(x=140,y=90)
        ttk.Label(self.rf_frame, text='Min samples split', font=myfont1).place(x=20, y=115)
        rf_e3 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_min_samples_split, font=myfont2, width=7)
        rf_e3.place(x=140,y=115)
        ttk.Label(self.rf_frame, text='Min samples leaf', font=myfont1).place(x=20, y=140)
        rf_e4 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_min_samples_leaf, font=myfont2, width=7)
        rf_e4.place(x=140,y=140)
        ttk.Label(self.rf_frame, text='Min weight\nfraction leaf', font=myfont1).place(x=20, y=165)
        rf_e5 = ttk.Entry(self.rf_frame, 
            textvariable=prev.rfr_min_weight_fraction_leaf, font=myfont2, width=7)
        rf_e5.place(x=140,y=175)
        ttk.Label(self.rf_frame, text='Max features', font=myfont1).place(x=20, y=210)
        rf_e6 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_max_features, font=myfont2, width=7)
        rf_e6.place(x=140,y=210)
        ttk.Label(self.rf_frame, text='Max leaf nodes', font=myfont1).place(x=20, y=235)
        rf_e7 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_max_leaf_nodes, font=myfont2, width=7)
        rf_e7.place(x=140,y=235)
        ttk.Label(self.rf_frame, text='Min impurity\ndecrease', font=myfont1).place(x=220, y=40)
        rf_e8 = ttk.Entry(self.rf_frame, 
            textvariable=prev.rfr_min_impurity_decrease, font=myfont2, width=7)
        rf_e8.place(x=340,y=50)
        ttk.Label(self.rf_frame, text='Bootstrap', font=myfont1).place(x=220, y=85)
        rf_cb1 = ttk.Checkbutton(self.rf_frame, variable=prev.rfr_bootstrap, takefocus=False)
        rf_cb1.place(x=340,y=85)
        ttk.Label(self.rf_frame, text='oob score', font=myfont1).place(x=220, y=110)
        rf_cb2 = ttk.Checkbutton(self.rf_frame, variable=prev.rfr_oob_score, takefocus=False)
        rf_cb2.place(x=340,y=110)
        ttk.Label(self.rf_frame, text='n jobs', font=myfont1).place(x=220, y=135)
        rf_e9 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_n_jobs, font=myfont2, width=7)
        rf_e9.place(x=340,y=135)
        ttk.Label(self.rf_frame, text='Random state', font=myfont1).place(x=220, y=160)
        rf_e10 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_random_state, font=myfont2, width=7)
        rf_e10.place(x=340,y=160)
        ttk.Label(self.rf_frame, text='Verbose', font=myfont1).place(x=220, y=185)
        rf_e11 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_verbose, font=myfont2, width=7)
        rf_e11.place(x=340,y=185)
        ttk.Label(self.rf_frame, text='Warm start', font=myfont1).place(x=220, y=210)
        rf_cb3 = ttk.Checkbutton(self.rf_frame, variable=prev.rfr_warm_start, takefocus=False)
        rf_cb3.place(x=340,y=210)
        ttk.Label(self.rf_frame, text='CCP alpha', font=myfont1).place(x=220, y=235)
        rf_e12 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_ccp_alpha, font=myfont2, width=7)
        rf_e12.place(x=340,y=235)
        ttk.Label(self.rf_frame, text='Max samples', font=myfont1).place(x=220, y=260)
        rf_e13 = ttk.Entry(self.rf_frame, textvariable=prev.rfr_max_samples, font=myfont2, width=7)
        rf_e13.place(x=340,y=260)

        ttk.Label(self.mlp_frame, text='Multi-layer Perceptron', font=myfont_b).place(x=220, y=10)
        ttk.Label(self.mlp_frame, text='hidden layer\nsizes', font=myfont1).place(x=20, y=40)
        mlp_e1 = ttk.Entry(self.mlp_frame, 
            textvariable=prev.mlpr_hidden_layer_sizes, font=myfont2, width=7)
        mlp_e1.place(x=140,y=50)
        ttk.Label(self.mlp_frame, text='Activation', font=myfont1).place(x=20, y=85)
        mlp_combobox1 = ttk.Combobox(self.mlp_frame, textvariable=prev.mlpr_activation, width=7, 
            values=['identity', 'logistic', 'tanh', 'relu'])
        mlp_combobox1.place(x=140,y=85)
        ttk.Label(self.mlp_frame, text='Solver', font=myfont1).place(x=20, y=110)
        mlp_combobox2 = ttk.Combobox(self.mlp_frame, textvariable=prev.mlpr_solver, width=7, 
            values=['lbfgs', 'sgd', 'adam'])
        mlp_combobox2.place(x=140,y=110)
        ttk.Label(self.mlp_frame, text='Alpha', font=myfont1).place(x=20, y=135)
        mlp_e2 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_alpha, font=myfont2, width=7)
        mlp_e2.place(x=140,y=135)
        ttk.Label(self.mlp_frame, text='Batch size', font=myfont1).place(x=20, y=160)
        mlp_e3 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_batch_size, font=myfont2, width=7)
        mlp_e3.place(x=140,y=160)
        ttk.Label(self.mlp_frame, text='Learning rate', font=myfont1).place(x=20, y=185)
        mlp_combobox3 = ttk.Combobox(self.mlp_frame, textvariable=prev.mlpr_learning_rate, width=7, 
            values=['constant', 'invscaling', 'adaptive'])
        mlp_combobox3.place(x=140,y=185)
        ttk.Label(self.mlp_frame, text='Learning\nrate init', font=myfont1).place(x=20, y=210)
        mlp_e4 = ttk.Entry(self.mlp_frame, 
            textvariable=prev.mlpr_learning_rate_init, font=myfont2, width=7)
        mlp_e4.place(x=140,y=220)
        ttk.Label(self.mlp_frame, text='Power t', font=myfont1).place(x=220, y=40)
        mlp_e5 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_power_t, font=myfont2, width=7)
        mlp_e5.place(x=340,y=40)
        ttk.Label(self.mlp_frame, text='Max iter', font=myfont1).place(x=220, y=65)
        mlp_e6 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_max_iter, font=myfont2, width=7)
        mlp_e6.place(x=340,y=65)
        ttk.Label(self.mlp_frame, text='Shuffle', font=myfont1).place(x=220, y=90)
        mlp_cb2 = ttk.Checkbutton(self.mlp_frame, variable=prev.mlpr_shuffle, takefocus=False)
        mlp_cb2.place(x=340, y=90)
        ttk.Label(self.mlp_frame, text='Random state', font=myfont1).place(x=220, y=115)
        mlp_e7 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_random_state, font=myfont2, width=7)
        mlp_e7.place(x=340,y=115)
        ttk.Label(self.mlp_frame, text='tol', font=myfont1).place(x=220, y=140)
        mlp_e8 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_tol, font=myfont2, width=7)
        mlp_e8.place(x=340,y=140)
        ttk.Label(self.mlp_frame, text='Verbose', font=myfont1).place(x=220, y=165)
        mlp_cb3 = ttk.Checkbutton(self.mlp_frame, variable=prev.mlpr_verbose, takefocus=False)
        mlp_cb3.place(x=340, y=165)
        ttk.Label(self.mlp_frame, text='Warm start', font=myfont1).place(x=220, y=190)
        mlp_cb4 = ttk.Checkbutton(self.mlp_frame, variable=prev.mlpr_warm_start, takefocus=False)
        mlp_cb4.place(x=340, y=190)
        ttk.Label(self.mlp_frame, text='Momentum', font=myfont1).place(x=220, y=215)
        mlp_e9 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_momentum, font=myfont2, width=7)
        mlp_e9.place(x=340,y=215)
        ttk.Label(self.mlp_frame, text='Nesterovs\nmomentum', font=myfont1).place(x=420, y=40)
        mlp_cb5 = ttk.Checkbutton(self.mlp_frame, 
            variable=prev.mlpr_nesterovs_momentum, takefocus=False)
        mlp_cb5.place(x=540, y=50)
        ttk.Label(self.mlp_frame, text='Early stopping', font=myfont1).place(x=420, y=85)
        mlp_cb6 = ttk.Checkbutton(self.mlp_frame, variable=prev.mlpr_early_stopping, takefocus=False)
        mlp_cb6.place(x=540, y=85)
        ttk.Label(self.mlp_frame, text='Validation fraction', font=myfont1).place(x=420, y=110)
        mlp_e10 = ttk.Entry(self.mlp_frame, 
            textvariable=prev.mlpr_validation_fraction, font=myfont2, width=7)
        mlp_e10.place(x=540,y=110)
        ttk.Label(self.mlp_frame, text='Beta 1', font=myfont1).place(x=420, y=135)
        mlp_e11 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_beta_1, font=myfont2, width=7)
        mlp_e11.place(x=540,y=135)
        ttk.Label(self.mlp_frame, text='Beta 2', font=myfont1).place(x=420, y=160)
        mlp_e12 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_beta_2, font=myfont2, width=7)
        mlp_e12.place(x=540,y=160)
        ttk.Label(self.mlp_frame, text='Epsilon', font=myfont1).place(x=420, y=185)
        mlp_e13 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_epsilon, font=myfont2, width=7)
        mlp_e13.place(x=540,y=185)
        ttk.Label(self.mlp_frame, text='n iter no change', font=myfont1).place(x=420, y=210)
        mlp_e14 = ttk.Entry(self.mlp_frame, 
            textvariable=prev.mlpr_n_iter_no_change, font=myfont2, width=7)
        mlp_e14.place(x=540,y=210)
        ttk.Label(self.mlp_frame, text='Max fun', font=myfont1).place(x=420, y=235)
        mlp_e15 = ttk.Entry(self.mlp_frame, textvariable=prev.mlpr_max_fun, font=myfont2, width=7)
        mlp_e15.place(x=540,y=235)

        ttk.Label(self.xgb_frame, text='XGBoost', font=myfont_b).place(x=180, y=10)
        ttk.Label(self.xgb_frame, text='Learning rate', font=myfont1).place(x=20, y=40)
        xgb_e1 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_eta, font=myfont2, width=7)
        xgb_e1.place(x=140,y=40)
        ttk.Label(self.xgb_frame, text='Min Child Weight', font=myfont1).place(x=20, y=65)
        xgb_e2 = ttk.Entry(self.xgb_frame, 
            textvariable=prev.xgbr_min_child_weight, font=myfont2, width=7)
        xgb_e2.place(x=140,y=65)
        ttk.Label(self.xgb_frame, text='Max Depth', font=myfont1).place(x=20, y=90)
        xgb_e3 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_max_depth, font=myfont2, width=7)
        xgb_e3.place(x=140,y=90)
        ttk.Label(self.xgb_frame, text='Gamma', font=myfont1).place(x=20, y=115)
        xgb_e4 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_gamma, font=myfont2, width=7)
        xgb_e4.place(x=140,y=115)
        ttk.Label(self.xgb_frame, text='Subsample', font=myfont1).place(x=20, y=140)
        xgb_e5 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_subsample, font=myfont2, width=7)
        xgb_e5.place(x=140,y=140)
        ttk.Label(self.xgb_frame, text='Colsample bytree', font=myfont1).place(x=20, y=165)
        xgb_e6 = ttk.Entry(self.xgb_frame, 
            textvariable=prev.xgbr_colsample_bytree, font=myfont2, width=7)
        xgb_e6.place(x=140,y=165)
        ttk.Label(self.xgb_frame, text='Reg Lambda', font=myfont1).place(x=20, y=190)
        xgb_e7 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_lambda, font=myfont2, width=7)
        xgb_e7.place(x=140,y=190)
        ttk.Label(self.xgb_frame, text='Reg Alpha', font=myfont1).place(x=20, y=215)
        xgb_e8 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_alpha, font=myfont2, width=7)
        xgb_e8.place(x=140,y=215)
        ttk.Label(self.xgb_frame, text='n estimators', font=myfont1).place(x=220, y=40)
        xgb_e9 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_n_estimators, font=myfont2, width=7)
        xgb_e9.place(x=340,y=40)
        ttk.Label(self.xgb_frame, text='Use GPU', font=myfont1).place(x=220, y=65)
        xgb_cb2 = ttk.Checkbutton(self.xgb_frame, variable=prev.xgbr_use_gpu, takefocus=False)
        xgb_cb2.place(x=340, y=65)
        ttk.Label(self.xgb_frame, text='Eval metric', font=myfont1).place(x=220, y=90)
        xgb_combobox1 = ttk.Combobox(self.xgb_frame, textvariable=prev.xgbr_eval_metric, width=7, 
            values=['rmse', 'rmsle', 'mae', 'mape'])
        xgb_combobox1.place(x=340,y=90)
        ttk.Label(self.xgb_frame, text='Early Stopping', font=myfont1).place(x=220, y=115)
        xgb_cb3 = ttk.Checkbutton(self.xgb_frame, variable=prev.xgbr_early_stopping, takefocus=False)
        xgb_cb3.place(x=340, y=115)
        ttk.Label(self.xgb_frame, text='n iter no change', font=myfont1).place(x=220, y=140)
        xgb_e10 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_n_iter_no_change, font=myfont2, width=7)
        xgb_e10.place(x=340,y=140)
        ttk.Label(self.xgb_frame, text='Validation fraction', font=myfont1).place(x=220, y=165)
        xgb_e11 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_validation_fraction, font=myfont2, width=7)
        xgb_e11.place(x=340,y=165)
        ttk.Label(self.xgb_frame, text='CV-avg mode', font=myfont1).place(x=220, y=190)
        xgb_cb4 = ttk.Checkbutton(self.xgb_frame, variable=prev.xgbr_cv_voting_mode, takefocus=False)
        xgb_cb4.place(x=340, y=190)
        ttk.Label(self.xgb_frame, text='Number of folds', font=myfont1).place(x=220, y=215)
        xgb_e12 = ttk.Entry(self.xgb_frame, textvariable=prev.xgbr_cv_voting_folds, font=myfont2, width=7)
        xgb_e12.place(x=340,y=215)

        ttk.Label(self.cb_frame, text='CatBoost', font=myfont_b).place(x=180, y=10)
        ttk.Label(self.cb_frame, text='Eval metric', font=myfont1).place(x=20, y=40)
        cb_combobox1 = ttk.Combobox(self.cb_frame, textvariable=prev.cbr_eval_metric, width=7, 
            values=['RMSE', 'MAE', 'MAPE', 'SMAPE', 'R2', 'MSLE'])
        cb_combobox1.place(x=140,y=40)
        ttk.Label(self.cb_frame, text='Iterations', font=myfont1).place(x=20, y=65)
        cb_e1 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_iterations, font=myfont2, width=7)
        cb_e1.place(x=140,y=65)
        ttk.Label(self.cb_frame, text='Learning rate', font=myfont1).place(x=20, y=90)
        cb_e2 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_learning_rate, font=myfont2, width=7)
        cb_e2.place(x=140,y=90)
        ttk.Label(self.cb_frame, text='Depth', font=myfont1).place(x=20, y=115)
        cb_e3 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_depth, font=myfont2, width=7)
        cb_e3.place(x=140,y=115)
        ttk.Label(self.cb_frame, text='Reg Lambda', font=myfont1).place(x=20, y=140)
        cb_e4 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_reg_lambda, font=myfont2, width=7)
        cb_e4.place(x=140,y=140)
        ttk.Label(self.cb_frame, text='Subsample', font=myfont1).place(x=20, y=165)
        cb_e5 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_subsample, font=myfont2, width=7)
        cb_e5.place(x=140,y=165)
        ttk.Label(self.cb_frame, text='Colsample bylevel', font=myfont1).place(x=20, y=190)
        cb_e6 = ttk.Entry(self.cb_frame, 
            textvariable=prev.cbr_colsample_bylevel, font=myfont2, width=7)
        cb_e6.place(x=140,y=190)
        ttk.Label(self.cb_frame, text='Random strength', font=myfont1).place(x=20, y=215)
        cb_e7 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_random_strength, font=myfont2, width=7)
        cb_e7.place(x=140,y=215)
        ttk.Label(self.cb_frame, text='Use GPU', font=myfont1).place(x=220, y=40)
        cb_cb2 = ttk.Checkbutton(self.cb_frame, variable=prev.cbr_use_gpu, takefocus=False)
        cb_cb2.place(x=340, y=40)
        ttk.Label(self.cb_frame, text='cat_features list', font=myfont1).place(x=220, y=65)
        cb_e8 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_cf_list, font=myfont2, width=9)
        cb_e8.place(x=340,y=65)
        ttk.Label(self.cb_frame, text='Early stopping', font=myfont1).place(x=220, y=90)
        cb_cb3 = ttk.Checkbutton(self.cb_frame, variable=prev.cbr_early_stopping, takefocus=False)
        cb_cb3.place(x=340, y=90)
        ttk.Label(self.cb_frame, text='n iter no change', font=myfont1).place(x=220, y=115)
        cb_e9 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_n_iter_no_change, font=myfont2, width=7)
        cb_e9.place(x=340,y=115)
        ttk.Label(self.cb_frame, text='Validation fraction', font=myfont1).place(x=220, y=140)
        cb_e10 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_validation_fraction, font=myfont2, width=7)
        cb_e10.place(x=340,y=140)
        ttk.Label(self.cb_frame, text='CV-avg mode', font=myfont1).place(x=220, y=165)
        cb_cb4 = ttk.Checkbutton(self.cb_frame, variable=prev.cbr_cv_voting_mode, takefocus=False)
        cb_cb4.place(x=340, y=165)
        ttk.Label(self.cb_frame, text='Number of folds', font=myfont1).place(x=220, y=190)
        cb_e11 = ttk.Entry(self.cb_frame, textvariable=prev.cbr_cv_voting_folds, font=myfont2, width=7)
        cb_e11.place(x=340,y=190)

        ttk.Label(self.lgb_frame, text='LightGBM', font=myfont_b).place(x=220, y=10)
        ttk.Label(self.lgb_frame, text='Boosting type', font=myfont1).place(x=20, y=40)
        lgb_combobox1 = ttk.Combobox(self.lgb_frame, textvariable=prev.lgbmr_boosting_type, width=7, 
            values=['gbdt', 'dart', 'goss', 'rf'])
        lgb_combobox1.place(x=140,y=40)
        ttk.Label(self.lgb_frame, text='Num leaves', font=myfont1).place(x=20, y=65)
        lgb_e1 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_num_leaves, font=myfont2, width=7)
        lgb_e1.place(x=140,y=65)
        ttk.Label(self.lgb_frame, text='Max depth', font=myfont1).place(x=20, y=90)
        lgb_e2 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_max_depth, font=myfont2, width=7)
        lgb_e2.place(x=140,y=90)
        ttk.Label(self.lgb_frame, text='Learning rate', font=myfont1).place(x=20, y=115)
        lgb_e3 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_learning_rate, font=myfont2, width=7)
        lgb_e3.place(x=140,y=115)
        ttk.Label(self.lgb_frame, text='Iterations', font=myfont1).place(x=20, y=140)
        lgb_e4 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_n_estimators, font=myfont2, width=7)
        lgb_e4.place(x=140,y=140)
        ttk.Label(self.lgb_frame, text='Subsample for bin', font=myfont1).place(x=20, y=165)
        lgb_e5 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_subsample_for_bin, font=myfont2, width=7)
        lgb_e5.place(x=140,y=165)
        ttk.Label(self.lgb_frame, text='Min split gain', font=myfont1).place(x=20, y=190)
        lgb_e6 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_min_split_gain, font=myfont2, width=7)
        lgb_e6.place(x=140,y=190)
        ttk.Label(self.lgb_frame, text='Min child weight', font=myfont1).place(x=20, y=215)
        lgb_e7 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_min_child_weight, font=myfont2, width=7)
        lgb_e7.place(x=140,y=215)
        ttk.Label(self.lgb_frame, text='Min child samples', font=myfont1).place(x=220, y=40)
        lgb_e8 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_min_child_samples, font=myfont2, width=7)
        lgb_e8.place(x=340,y=40)
        ttk.Label(self.lgb_frame, text='Subsample', font=myfont1).place(x=220, y=65)
        lgb_e9 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_subsample, font=myfont2, width=7)
        lgb_e9.place(x=340,y=65)
        ttk.Label(self.lgb_frame, text='Subsample freq', font=myfont1).place(x=220, y=90)
        lgb_e10 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_subsample_freq, font=myfont2, width=7)
        lgb_e10.place(x=340,y=90)
        ttk.Label(self.lgb_frame, text='Colsample bytree', font=myfont1).place(x=220, y=115)
        lgb_e11 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_colsample_bytree, font=myfont2, width=7)
        lgb_e11.place(x=340,y=115)
        ttk.Label(self.lgb_frame, text='Reg alpha', font=myfont1).place(x=220, y=140)
        lgb_e12 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_reg_alpha, font=myfont2, width=7)
        lgb_e12.place(x=340,y=140)
        ttk.Label(self.lgb_frame, text='Reg lambda', font=myfont1).place(x=220, y=165)
        lgb_e13 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_reg_lambda, font=myfont2, width=7)
        lgb_e13.place(x=340,y=165)
        ttk.Label(self.lgb_frame, text='n jobs', font=myfont1).place(x=220, y=190)
        lgb_e14 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_n_jobs, font=myfont2, width=7)
        lgb_e14.place(x=340,y=190)
        ttk.Label(self.lgb_frame, text='silent', font=myfont1).place(x=420, y=40)
        lgb_cb1 = ttk.Checkbutton(self.lgb_frame, variable=prev.lgbmr_silent, takefocus=False)
        lgb_cb1.place(x=540, y=40)
        ttk.Label(self.lgb_frame, text='cat_features list', font=myfont1).place(x=420, y=65)
        lgb_e15 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_categorical_feature, font=myfont2, width=9)
        lgb_e15.place(x=540,y=65)
        ttk.Label(self.lgb_frame, text='Early stopping', font=myfont1).place(x=420, y=90)
        lgb_cb2 = ttk.Checkbutton(self.lgb_frame, variable=prev.lgbmr_early_stopping, takefocus=False)
        lgb_cb2.place(x=540, y=90)
        ttk.Label(self.lgb_frame, text='n iter no change', font=myfont1).place(x=420, y=115)
        lgb_e16 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_n_iter_no_change, font=myfont2, width=7)
        lgb_e16.place(x=540,y=115)
        ttk.Label(self.lgb_frame, text='Validation fraction', font=myfont1).place(x=420, y=140)
        lgb_e17 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_validation_fraction, font=myfont2, width=7)
        lgb_e17.place(x=540,y=140)
        ttk.Label(self.lgb_frame, text='CV-avg mode', font=myfont1).place(x=420, y=165)
        lgb_cb3 = ttk.Checkbutton(self.lgb_frame, variable=prev.lgbmr_cv_voting_mode, takefocus=False)
        lgb_cb3.place(x=540, y=165)
        ttk.Label(self.lgb_frame, text='Number of folds', font=myfont1).place(x=420, y=190)
        lgb_e18 = ttk.Entry(self.lgb_frame, textvariable=prev.lgbmr_cv_voting_folds, font=myfont2, width=7)
        lgb_e18.place(x=540,y=190)
        
        ttk.Button(self.root, text='OK', width=5,
            command=lambda: quit_back(self.root, rgr_app.root)).place(relx=0.82, rely=0.92)

    # function to restore default parameters
    def restore_defaults(self, prev, parent):
        if tk.messagebox.askyesno("Restore", "Restore default settings?"):
            prev.inc_scoring = tk.StringVar(value='r2')
            prev.lr_include_comp = tk.BooleanVar(value=True)
            prev.lr_function_type = tk.StringVar(value='Linear')
            prev.lr_fit_intercept = tk.BooleanVar(value=True)
            prev.lr_normalize = tk.BooleanVar(value=False)
            prev.lr_copy_X = tk.BooleanVar(value=True)
            prev.lr_n_jobs = tk.StringVar(value='None')
            prev.lr_positive = tk.BooleanVar(value=False)
            
            prev.rr_include_comp = tk.BooleanVar(value=True)
            prev.rr_alpha = tk.StringVar(value='1.0')
            prev.rr_fit_intercept = tk.BooleanVar(value=True)
            prev.rr_normalize = tk.BooleanVar(value=False)
            prev.rr_copy_X = tk.BooleanVar(value=True)
            prev.rr_max_iter = tk.StringVar(value='None')
            prev.rr_tol = tk.StringVar(value='1e-3')
            prev.rr_solver = tk.StringVar(value='auto')
            prev.rr_random_state = tk.StringVar(value='None')
            
            prev.lassor_include_comp = tk.BooleanVar(value=True)
            prev.lassor_alpha = tk.StringVar(value='1.0')
            prev.lassor_fit_intercept = tk.BooleanVar(value=True)
            prev.lassor_normalize = tk.BooleanVar(value=False)
            prev.lassor_precompute = tk.BooleanVar(value=False)
            prev.lassor_copy_X = tk.BooleanVar(value=True)
            prev.lassor_max_iter = tk.StringVar(value='1000')
            prev.lassor_tol = tk.StringVar(value='1e-4')
            prev.lassor_warm_start = tk.BooleanVar(value=False)
            prev.lassor_positive = tk.BooleanVar(value=False)
            prev.lassor_random_state = tk.StringVar(value='None')
            prev.lassor_selection = tk.StringVar(value='cyclic')
            
            prev.rfr_include_comp = tk.BooleanVar(value=True)
            prev.rfr_n_estimators = tk.StringVar(value='100')
            prev.rfr_criterion = tk.StringVar(value='mse')
            prev.rfr_max_depth = tk.StringVar(value='None')
            prev.rfr_min_samples_split = tk.StringVar(value='2')
            prev.rfr_min_samples_leaf = tk.StringVar(value='1')
            prev.rfr_min_weight_fraction_leaf = tk.StringVar(value='0.0')
            prev.rfr_max_features = tk.StringVar(value='auto')
            prev.rfr_max_leaf_nodes = tk.StringVar(value='None')
            prev.rfr_min_impurity_decrease = tk.StringVar(value='0.0')
            prev.rfr_bootstrap = tk.BooleanVar(value=True)
            prev.rfr_oob_score = tk.BooleanVar(value=False)
            prev.rfr_n_jobs = tk.StringVar(value='3')
            prev.rfr_random_state = tk.StringVar(value='None')
            prev.rfr_verbose = tk.StringVar(value='0')
            prev.rfr_warm_start = tk.BooleanVar(value=False)
            prev.rfr_ccp_alpha = tk.StringVar(value='0.0')
            prev.rfr_max_samples = tk.StringVar(value='None')
            
            prev.svr_include_comp = tk.BooleanVar(value=True)
            prev.svr_kernel = tk.StringVar(value='rbf')
            prev.svr_degree = tk.StringVar(value='3')
            prev.svr_gamma = tk.StringVar(value='scale')
            prev.svr_coef0 = tk.StringVar(value='0.0')
            prev.svr_tol = tk.StringVar(value='1e-3')
            prev.svr_C = tk.StringVar(value='1.0')
            prev.svr_epsilon = tk.StringVar(value='0.1')
            prev.svr_shrinking = tk.BooleanVar(value=True)
            prev.svr_cache_size = tk.StringVar(value='200')
            prev.svr_verbose = tk.BooleanVar(value=False)
            prev.svr_max_iter = tk.StringVar(value='-1')
            
            prev.sgdr_include_comp = tk.BooleanVar(value=True)
            prev.sgdr_loss = tk.StringVar(value='squared_loss')
            prev.sgdr_penalty = tk.StringVar(value='l2')
            prev.sgdr_alpha = tk.StringVar(value='0.0001')
            prev.sgdr_l1_ratio = tk.StringVar(value='0.15')
            prev.sgdr_fit_intercept = tk.BooleanVar(value=True)
            prev.sgdr_max_iter = tk.StringVar(value='1000')
            prev.sgdr_tol = tk.StringVar(value='1e-3')
            prev.sgdr_shuffle = tk.BooleanVar(value=True)
            prev.sgdr_verbose = tk.StringVar(value='0')
            prev.sgdr_epsilon = tk.StringVar(value='0.1')
            prev.sgdr_random_state = tk.StringVar(value='None')
            prev.sgdr_learning_rate = tk.StringVar(value='invscaling')
            prev.sgdr_eta0 = tk.StringVar(value='0.01')
            prev.sgdr_power_t = tk.StringVar(value='0.25')
            prev.sgdr_early_stopping = tk.BooleanVar(value=False)
            prev.sgdr_validation_fraction = tk.StringVar(value='0.1')
            prev.sgdr_n_iter_no_change = tk.StringVar(value='5')
            prev.sgdr_warm_start = tk.BooleanVar(value=False)
            prev.sgdr_average = tk.StringVar(value='False')
            
            prev.knr_include_comp = tk.BooleanVar(value=True)
            prev.knr_n_neighbors = tk.StringVar(value='5')
            prev.knr_weights = tk.StringVar(value='uniform')
            prev.knr_algorithm = tk.StringVar(value='auto')
            prev.knr_leaf_size = tk.StringVar(value='30')
            prev.knr_p = tk.StringVar(value='2')
            prev.knr_metric = tk.StringVar(value='minkowski')
            prev.knr_n_jobs = tk.StringVar(value='None')
            
            prev.gpr_include_comp = tk.BooleanVar(value=True)
            prev.gpr_alpha = tk.StringVar(value='1e-10')
            prev.gpr_n_restarts_optimizer = tk.StringVar(value='0')
            prev.gpr_normalize_y = tk.BooleanVar(value=True)
            prev.gpr_copy_X_train = tk.BooleanVar(value=True)
            prev.gpr_random_state = tk.StringVar(value='None')
            
            prev.dtr_include_comp = tk.BooleanVar(value=True)
            prev.dtr_criterion = tk.StringVar(value='mse')
            prev.dtr_splitter = tk.StringVar(value='best')
            prev.dtr_max_depth = tk.StringVar(value='None')
            prev.dtr_min_samples_split = tk.StringVar(value='2')
            prev.dtr_min_samples_leaf = tk.StringVar(value='1')
            prev.dtr_min_weight_fraction_leaf = tk.StringVar(value='0.0')
            prev.dtr_max_features = tk.StringVar(value='None')
            prev.dtr_random_state = tk.StringVar(value='None')
            prev.dtr_max_leaf_nodes = tk.StringVar(value='None')
            prev.dtr_min_impurity_decrease = tk.StringVar(value='0.0')
            prev.dtr_ccp_alpha = tk.StringVar(value='0.0')
            
            prev.mlpr_include_comp = tk.BooleanVar(value=True)
            prev.mlpr_hidden_layer_sizes = tk.StringVar(value='(100,)')
            prev.mlpr_activation = tk.StringVar(value='relu')
            prev.mlpr_solver = tk.StringVar(value='adam')
            prev.mlpr_alpha = tk.StringVar(value='0.0001')
            prev.mlpr_batch_size = tk.StringVar(value='auto')
            prev.mlpr_learning_rate = tk.StringVar(value='constant')
            prev.mlpr_learning_rate_init = tk.StringVar(value='0.001')
            prev.mlpr_power_t = tk.StringVar(value='0.5')
            prev.mlpr_max_iter = tk.StringVar(value='200')
            prev.mlpr_shuffle = tk.BooleanVar(value=True)
            prev.mlpr_random_state = tk.StringVar(value='None')
            prev.mlpr_tol = tk.StringVar(value='1e-4')
            prev.mlpr_verbose = tk.BooleanVar(value=False)
            prev.mlpr_warm_start = tk.BooleanVar(value=False)
            prev.mlpr_momentum = tk.StringVar(value='0.9')
            prev.mlpr_nesterovs_momentum = tk.BooleanVar(value=True)
            prev.mlpr_early_stopping = tk.BooleanVar(value=False)
            prev.mlpr_validation_fraction = tk.StringVar(value='0.1')
            prev.mlpr_beta_1 = tk.StringVar(value='0.9')
            prev.mlpr_beta_2 = tk.StringVar(value='0.999')
            prev.mlpr_epsilon = tk.StringVar(value='1e-8')
            prev.mlpr_n_iter_no_change = tk.StringVar(value='10')
            prev.mlpr_max_fun = tk.StringVar(value='15000')

            prev.xgbr_include_comp = tk.BooleanVar(value=True)
            prev.xgbr_n_estimators = tk.StringVar(value='1000')
            prev.xgbr_eta = tk.StringVar(value='0.1')
            prev.xgbr_min_child_weight = tk.StringVar(value='1')
            prev.xgbr_max_depth = tk.StringVar(value='6')
            prev.xgbr_gamma = tk.StringVar(value='1')
            prev.xgbr_subsample = tk.StringVar(value='1.0')
            prev.xgbr_colsample_bytree = tk.StringVar(value='1.0')
            prev.xgbr_lambda = tk.StringVar(value='1.0')
            prev.xgbr_alpha = tk.StringVar(value='0.0')
            prev.xgbr_use_gpu = tk.BooleanVar(value=False)
            prev.xgbr_eval_metric = tk.StringVar(value='rmse')
            prev.xgbr_early_stopping = tk.BooleanVar(value=False)
            prev.xgbr_n_iter_no_change = tk.StringVar(value='100')
            prev.xgbr_validation_fraction = tk.StringVar(value='0.2')
            prev.xgbr_cv_voting_mode = tk.BooleanVar(value=False)
            prev.xgbr_cv_voting_folds = tk.StringVar(value='5')

            prev.cbr_include_comp = tk.BooleanVar(value=True)
            prev.cbr_eval_metric = tk.StringVar(value='RMSE')
            prev.cbr_iterations = tk.StringVar(value='1000')
            prev.cbr_learning_rate = tk.StringVar(value='None')
            prev.cbr_depth = tk.StringVar(value='6')
            prev.cbr_reg_lambda = tk.StringVar(value='None')
            prev.cbr_subsample = tk.StringVar(value='None')
            prev.cbr_colsample_bylevel = tk.StringVar(value='1.0')
            prev.cbr_random_strength = tk.StringVar(value='1.0')
            prev.cbr_use_gpu = tk.BooleanVar(value=False)
            prev.cbr_cf_list = tk.StringVar(value='[]')
            prev.cbr_early_stopping = tk.BooleanVar(value=False)
            prev.cbr_n_iter_no_change = tk.StringVar(value='100')
            prev.cbr_validation_fraction = tk.StringVar(value='0.2')
            prev.cbr_cv_voting_mode = tk.BooleanVar(value=False)
            prev.cbr_cv_voting_folds = tk.StringVar(value='5')

            prev.lgbmr_include_comp = tk.BooleanVar(value=True)
            prev.lgbmr_boosting_type = tk.StringVar(value='gbdt')
            prev.lgbmr_num_leaves = tk.StringVar(value='31')
            prev.lgbmr_max_depth = tk.StringVar(value='-1')
            prev.lgbmr_learning_rate = tk.StringVar(value='0.1')
            prev.lgbmr_n_estimators = tk.StringVar(value='100')
            prev.lgbmr_subsample_for_bin = tk.StringVar(value='200000')
            prev.lgbmr_min_split_gain = tk.StringVar(value='0.0')
            prev.lgbmr_min_child_weight = tk.StringVar(value='1e-3')
            prev.lgbmr_min_child_samples = tk.StringVar(value='20')
            prev.lgbmr_subsample = tk.StringVar(value='1.0')
            prev.lgbmr_subsample_freq = tk.StringVar(value='0')
            prev.lgbmr_colsample_bytree = tk.StringVar(value='1.0')
            prev.lgbmr_reg_alpha = tk.StringVar(value='0.0')
            prev.lgbmr_reg_lambda = tk.StringVar(value='0.0')
            prev.lgbmr_n_jobs = tk.StringVar(value='-1')
            prev.lgbmr_silent = tk.BooleanVar(value=True)
            prev.lgbmr_categorical_feature = tk.StringVar(value='auto')
            prev.lgbmr_eval_metric = tk.StringVar(value='rmse')
            prev.lgbmr_early_stopping = tk.BooleanVar(value=False)
            prev.lgbmr_n_iter_no_change = tk.StringVar(value='100')
            prev.lgbmr_validation_fraction = tk.StringVar(value='0.2')
            prev.lgbmr_cv_voting_mode = tk.BooleanVar(value=False)
            prev.lgbmr_cv_voting_folds = tk.StringVar(value='5')

            quit_back(self.root, prev.root)
            rgr_mtds_specification(prev, parent)

    def save_settings(self, prev, parent):
        save_file = open(asksaveasfilename(parent=self.root, defaultextension=".txt",
            filetypes = (("Text files","*.txt"),)
            ), 'w')
        save_file.write(
            str({
                'inc_scoring' : prev.inc_scoring.get(),
                'lr_include_comp' : prev.lr_include_comp.get(),
                'lr_function_type' : prev.lr_function_type.get(),
                'lr_fit_intercept' : prev.lr_fit_intercept.get(),
                'lr_normalize' : prev.lr_normalize.get(),
                'lr_copy_X' : prev.lr_copy_X.get(),
                'lr_n_jobs' : prev.lr_n_jobs.get(),
                'lr_positive' : prev.lr_positive.get(),
                
                'rr_include_comp' : prev.rr_include_comp.get(),
                'rr_alpha' : prev.rr_alpha.get(),
                'rr_fit_intercept' : prev.rr_fit_intercept.get(),
                'rr_normalize' : prev.rr_normalize.get(),
                'rr_copy_X' : prev.rr_copy_X.get(),
                'rr_max_iter' : prev.rr_max_iter.get(),
                'rr_tol' : prev.rr_tol.get(),
                'rr_solver' : prev.rr_solver.get(),
                'rr_random_state' : prev.rr_random_state.get(),
                
                'lassor_include_comp' : prev.lassor_include_comp.get(),
                'lassor_alpha' : prev.lassor_alpha.get(),
                'lassor_fit_intercept' : prev.lassor_fit_intercept.get(),
                'lassor_normalize' : prev.lassor_normalize.get(),
                'lassor_precompute' : prev.lassor_precompute.get(),
                'lassor_copy_X' : prev.lassor_copy_X.get(),
                'lassor_max_iter' : prev.lassor_max_iter.get(),
                'lassor_tol' : prev.lassor_tol.get(),
                'lassor_warm_start' : prev.lassor_warm_start.get(),
                'lassor_positive' : prev.lassor_positive.get(),
                'lassor_random_state' : prev.lassor_random_state.get(),
                'lassor_selection' : prev.lassor_selection.get(),
                
                'rfr_include_comp' : prev.rfr_include_comp.get(),
                'rfr_n_estimators' : prev.rfr_n_estimators.get(),
                'rfr_criterion' : prev.rfr_criterion.get(),
                'rfr_max_depth' : prev.rfr_max_depth.get(),
                'rfr_min_samples_split' : prev.rfr_min_samples_split.get(),
                'rfr_min_samples_leaf' : prev.rfr_min_samples_leaf.get(),
                'rfr_min_weight_fraction_leaf' : prev.rfr_min_weight_fraction_leaf.get(),
                'rfr_max_features' : prev.rfr_max_features.get(),
                'rfr_max_leaf_nodes' : prev.rfr_max_leaf_nodes.get(),
                'rfr_min_impurity_decrease' : prev.rfr_min_impurity_decrease.get(),
                'rfr_bootstrap' : prev.rfr_bootstrap.get(),
                'rfr_oob_score' : prev.rfr_oob_score.get(),
                'rfr_n_jobs' : prev.rfr_n_jobs.get(),
                'rfr_random_state' : prev.rfr_random_state.get(),
                'rfr_verbose' : prev.rfr_verbose.get(),
                'rfr_warm_start' : prev.rfr_warm_start.get(),
                'rfr_ccp_alpha' : prev.rfr_ccp_alpha.get(),
                'rfr_max_samples' : prev.rfr_max_samples.get(),
                
                'svr_include_comp' : prev.svr_include_comp.get(),
                'svr_kernel' : prev.svr_kernel.get(),
                'svr_degree' : prev.svr_degree.get(),
                'svr_gamma' : prev.svr_gamma.get(),
                'svr_coef0' : prev.svr_coef0.get(),
                'svr_tol' : prev.svr_tol.get(),
                'svr_C' : prev.svr_C.get(),
                'svr_epsilon' : prev.svr_epsilon.get(),
                'svr_shrinking' : prev.svr_shrinking.get(),
                'svr_cache_size' : prev.svr_cache_size.get(),
                'svr_verbose' : prev.svr_verbose.get(),
                'svr_max_iter' : prev.svr_max_iter.get(),
                
                'sgdr_include_comp' : prev.sgdr_include_comp.get(),
                'sgdr_loss' : prev.sgdr_loss.get(),
                'sgdr_penalty' : prev.sgdr_penalty.get(),
                'sgdr_alpha' : prev.sgdr_alpha.get(),
                'sgdr_l1_ratio' : prev.sgdr_l1_ratio.get(),
                'sgdr_fit_intercept' : prev.sgdr_fit_intercept.get(),
                'sgdr_max_iter' : prev.sgdr_max_iter.get(),
                'sgdr_tol' : prev.sgdr_tol.get(),
                'sgdr_shuffle' : prev.sgdr_shuffle.get(),
                'sgdr_verbose' : prev.sgdr_verbose.get(),
                'sgdr_epsilon' : prev.sgdr_epsilon.get(),
                'sgdr_random_state' : prev.sgdr_random_state.get(),
                'sgdr_learning_rate' : prev.sgdr_learning_rate.get(),
                'sgdr_eta0' : prev.sgdr_eta0.get(),
                'sgdr_power_t' : prev.sgdr_power_t.get(),
                'sgdr_early_stopping' : prev.sgdr_early_stopping.get(),
                'sgdr_validation_fraction' : prev.sgdr_validation_fraction.get(),
                'sgdr_n_iter_no_change' : prev.sgdr_n_iter_no_change.get(),
                'sgdr_warm_start' : prev.sgdr_warm_start.get(),
                'sgdr_average' : prev.sgdr_average.get(),
                
                'knr_include_comp' : prev.knr_include_comp.get(),
                'knr_n_neighbors' : prev.knr_n_neighbors.get(),
                'knr_weights' : prev.knr_weights.get(),
                'knr_algorithm' : prev.knr_algorithm.get(),
                'knr_leaf_size' : prev.knr_leaf_size.get(),
                'knr_p' : prev.knr_p.get(),
                'knr_metric' : prev.knr_metric.get(),
                'knr_n_jobs' : prev.knr_n_jobs.get(),
                
                'gpr_include_comp' : prev.gpr_include_comp.get(),
                'gpr_alpha' : prev.gpr_alpha.get(),
                'gpr_n_restarts_optimizer' : prev.gpr_n_restarts_optimizer.get(),
                'gpr_normalize_y' : prev.gpr_normalize_y.get(),
                'gpr_copy_X_train' : prev.gpr_copy_X_train.get(),
                'gpr_random_state' : prev.gpr_random_state.get(),
                
                'dtr_include_comp' : prev.dtr_include_comp.get(),
                'dtr_criterion' : prev.dtr_criterion.get(),
                'dtr_splitter' : prev.dtr_splitter.get(),
                'dtr_max_depth' : prev.dtr_max_depth.get(),
                'dtr_min_samples_split' : prev.dtr_min_samples_split.get(),
                'dtr_min_samples_leaf' : prev.dtr_min_samples_leaf.get(),
                'dtr_min_weight_fraction_leaf' : prev.dtr_min_weight_fraction_leaf.get(),
                'dtr_max_features' : prev.dtr_max_features.get(),
                'dtr_random_state' : prev.dtr_random_state.get(),
                'dtr_max_leaf_nodes' : prev.dtr_max_leaf_nodes.get(),
                'dtr_min_impurity_decrease' : prev.dtr_min_impurity_decrease.get(),
                'dtr_ccp_alpha' : prev.dtr_ccp_alpha.get(),
                
                'mlpr_include_comp' : prev.mlpr_include_comp.get(),
                'mlpr_hidden_layer_sizes' : prev.mlpr_hidden_layer_sizes.get(),
                'mlpr_activation' : prev.mlpr_activation.get(),
                'mlpr_solver' : prev.mlpr_solver.get(),
                'mlpr_alpha' : prev.mlpr_alpha.get(),
                'mlpr_batch_size' : prev.mlpr_batch_size.get(),
                'mlpr_learning_rate' : prev.mlpr_learning_rate.get(),
                'mlpr_learning_rate_init' : prev.mlpr_learning_rate_init.get(),
                'mlpr_power_t' : prev.mlpr_power_t.get(),
                'mlpr_max_iter' : prev.mlpr_max_iter.get(),
                'mlpr_shuffle' : prev.mlpr_shuffle.get(),
                'mlpr_random_state' : prev.mlpr_random_state.get(),
                'mlpr_tol' : prev.mlpr_tol.get(),
                'mlpr_verbose' : prev.mlpr_verbose.get(),
                'mlpr_warm_start' : prev.mlpr_warm_start.get(),
                'mlpr_momentum' : prev.mlpr_momentum.get(),
                'mlpr_nesterovs_momentum' : prev.mlpr_nesterovs_momentum.get(),
                'mlpr_early_stopping' : prev.mlpr_early_stopping.get(),
                'mlpr_validation_fraction' : prev.mlpr_validation_fraction.get(),
                'mlpr_beta_1' : prev.mlpr_beta_1.get(),
                'mlpr_beta_2' : prev.mlpr_beta_2.get(),
                'mlpr_epsilon' : prev.mlpr_epsilon.get(),
                'mlpr_n_iter_no_change' : prev.mlpr_n_iter_no_change.get(),
                'mlpr_max_fun' : prev.mlpr_max_fun.get(),

                'xgbr_include_comp' : prev.xgbr_include_comp.get(),
                'xgbr_n_estimators' : prev.xgbr_n_estimators.get(),
                'xgbr_eta' : prev.xgbr_eta.get(),
                'xgbr_min_child_weight' : prev.xgbr_min_child_weight.get(),
                'xgbr_max_depth' : prev.xgbr_max_depth.get(),
                'xgbr_gamma' : prev.xgbr_gamma.get(),
                'xgbr_subsample' : prev.xgbr_subsample.get(),
                'xgbr_colsample_bytree' : prev.xgbr_colsample_bytree.get(),
                'xgbr_lambda' : prev.xgbr_lambda.get(),
                'xgbr_alpha' : prev.xgbr_alpha.get(),
                'xgbr_use_gpu' : prev.xgbr_use_gpu.get(),
                'xgbr_eval_metric' : prev.xgbr_eval_metric.get(),
                'xgbr_early_stopping' : prev.xgbr_early_stopping.get(),
                'xgbr_n_iter_no_change' : prev.xgbr_n_iter_no_change.get(),
                'xgbr_validation_fraction' : prev.xgbr_validation_fraction.get(),
                'xgbr_cv_voting_mode' : prev.xgbr_cv_voting_mode.get(),
                'xgbr_cv_voting_folds' : prev.xgbr_cv_voting_folds.get(),

                'cbr_include_comp' : prev.cbr_include_comp.get(),
                'cbr_eval_metric' : prev.cbr_eval_metric.get(),
                'cbr_iterations' : prev.cbr_iterations.get(),
                'cbr_learning_rate' : prev.cbr_learning_rate.get(),
                'cbr_depth' : prev.cbr_depth.get(),
                'cbr_reg_lambda' : prev.cbr_reg_lambda.get(),
                'cbr_subsample' : prev.cbr_subsample.get(),
                'cbr_colsample_bylevel' : prev.cbr_colsample_bylevel.get(),
                'cbr_random_strength' : prev.cbr_random_strength.get(),
                'cbr_use_gpu' : prev.cbr_use_gpu.get(),
                'cbr_cf_list' : prev.cbr_cf_list.get(),
                'cbr_early_stopping' : prev.cbr_early_stopping.get(),
                'cbr_n_iter_no_change' : prev.cbr_n_iter_no_change.get(),
                'cbr_validation_fraction' : prev.cbr_validation_fraction.get(),
                'cbr_cv_voting_mode' : prev.cbr_cv_voting_mode.get(),
                'cbr_cv_voting_folds' : prev.cbr_cv_voting_folds.get(),

                'lgbmr_include_comp' : prev.lgbmr_include_comp.get(),
                'lgbmr_boosting_type' : prev.lgbmr_boosting_type.get(),
                'lgbmr_num_leaves' : prev.lgbmr_num_leaves.get(),
                'lgbmr_max_depth' : prev.lgbmr_max_depth.get(),
                'lgbmr_learning_rate' : prev.lgbmr_learning_rate.get(),
                'lgbmr_n_estimators' : prev.lgbmr_n_estimators.get(),
                'lgbmr_subsample_for_bin' : prev.lgbmr_subsample_for_bin.get(),
                'lgbmr_min_split_gain' : prev.lgbmr_min_split_gain.get(),
                'lgbmr_min_child_weight' : prev.lgbmr_min_child_weight.get(),
                'lgbmr_min_child_samples' : prev.lgbmr_min_child_samples.get(),
                'lgbmr_subsample' : prev.lgbmr_subsample.get(),
                'lgbmr_subsample_freq' : prev.lgbmr_subsample_freq.get(),
                'lgbmr_colsample_bytree' : prev.lgbmr_colsample_bytree.get(),
                'lgbmr_reg_alpha' : prev.lgbmr_reg_alpha.get(),
                'lgbmr_reg_lambda' : prev.lgbmr_reg_lambda.get(),
                'lgbmr_n_jobs' : prev.lgbmr_n_jobs.get(),
                'lgbmr_silent' : prev.lgbmr_silent.get(),
                'lgbmr_categorical_feature' : prev.lgbmr_categorical_feature.get(),
                'lgbmr_eval_metric' : prev.lgbmr_eval_metric.get(),
                'lgbmr_early_stopping' : prev.lgbmr_early_stopping.get(),
                'lgbmr_n_iter_no_change' : prev.lgbmr_n_iter_no_change.get(),
                'lgbmr_validation_fraction' : prev.lgbmr_validation_fraction.get(),
                'lgbmr_cv_voting_mode' : prev.lgbmr_cv_voting_mode.get(),
                'lgbmr_cv_voting_folds' : prev.lgbmr_cv_voting_folds.get(),
            })
        )
        save_file.close()

    def load_settings(self, prev, parent):
        load_file = open(askopenfilename(parent=self.root,
            filetypes = (("Text files","*.txt"),)
            ), 'r')
        settings_dict = eval(load_file.read())
        for key in settings_dict:
            getattr(prev, key).set(settings_dict[key])
            # prev.eval(key).set(settings_dict[key])
            # setattr(prev, key, tk.StringVar(value=settings_dict[key]))

        quit_back(self.root, prev.root)
        rgr_mtds_specification(prev, parent)