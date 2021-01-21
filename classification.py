# Imports
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, scrolledtext
from threading import Thread
import os
import sys
import numpy as np
import webbrowser
import pandas as pd
import sklearn
from monkey_pt import Table
from utils import Data_Preview, quit_back, open_file, load_data, save_results

# fonts
myfont = (None, 13)
myfont_b = (None, 13, 'bold')
myfont1 = (None, 11)
myfont1_b = (None, 11, 'bold')
myfont2 = (None, 10)
myfont2_b = (None, 10, 'bold')

# App to do classification
class clf_app:
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
            h = 430    
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

            ttk.Label(self.frame, text='The results of classification\n methods comparison', 
                font=myfont).place(x=10,y=10)
            ttk.Label(self.frame, text='Decision Tree:', font=myfont1).place(x=10,y=60)
            ttk.Label(self.frame, text='Ridge:', font=myfont1).place(x=10,y=90)
            ttk.Label(self.frame, text='Random Forest:', font=myfont1).place(x=10,y=120)
            ttk.Label(self.frame, text='Support Vector:', font=myfont1).place(x=10,y=150)
            ttk.Label(self.frame, text='SGD:', font=myfont1).place(x=10,y=180)
            ttk.Label(self.frame, text='Nearest Neighbor:', font=myfont1).place(x=10,y=210)
            ttk.Label(self.frame, text='Gaussian process:', font=myfont1).place(x=10,y=240)
            ttk.Label(self.frame, text='Multi-layer Perceptron:', font=myfont1).place(x=10,y=270)
            ttk.Label(self.frame, text='XGBoost:', font=myfont1).place(x=10,y=300)
            ttk.Label(self.frame, text='CatBoost:', font=myfont1).place(x=10,y=330)

            if 'dtc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['dtc']), 
                    font=myfont1).place(x=200,y=60)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=60)
            if 'rc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['rc']), 
                    font=myfont1).place(x=200,y=90)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=90)
            if 'rfc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['rfc']), 
                    font=myfont1).place(x=200,y=120)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=120)
            if 'svc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['svc']), 
                    font=myfont1).place(x=200,y=150)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=150)
            if 'sgdc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['sgdc']), 
                    font=myfont1).place(x=200,y=180)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=180)
            if 'knc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['knc']), 
                    font=myfont1).place(x=200,y=210)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=210)
            if 'gpc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['gpc']), 
                    font=myfont1).place(x=200,y=240)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=240)
            if 'mlpc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['mlpc']), 
                    font=myfont1).place(x=200,y=270)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=270)
            if 'xgbc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['xgbc']), 
                    font=myfont1).place(x=200,y=300)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=300)
            if 'cbc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.4f}". format(prev.scores['cbc']), 
                    font=myfont1).place(x=200,y=330)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=330)

            ttk.Button(self.frame, text='OK', 
                command=lambda: quit_back(self.root, prev.root)).place(x=110, y=380)
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
                values=['Decision Tree', 'Ridge', 'Random Forest', 'Support Vector', 'SGD',  
                        'Nearest Neighbor', 'Gaussian Process', 'Multi-layer Perceptron',
                        'XGBoost', 'CatBoost'])
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
                    if prev.dummies_var.get()==0:
                        try:
                            prev.X = main.data.iloc[:,x_from : x_to]
                            prev.y = main.data[prev.y_var.get()]
                        except:
                            prev.X = main.data.iloc[:,x_from : x_to]
                            prev.y = main.data[int(prev.y_var.get())]
                    elif prev.dummies_var.get()==1:
                        try:
                            prev.X = pd.get_dummies(main.data.iloc[:,x_from : x_to])
                            prev.y = main.data[prev.y_var.get()]
                        except:
                            prev.X = pd.get_dummies(main.data.iloc[:,x_from : x_to])
                            prev.y = main.data[int(prev.y_var.get())]
                    from sklearn import preprocessing
                    scaler = preprocessing.StandardScaler()
                    prev.X_St = scaler.fit_transform(prev.X)
                    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
                    folds = RepeatedStratifiedKFold(n_splits = int(e2.get()), 
                        n_repeats=int(rep_entry.get()), random_state = None)
                    if method == 'Decision Tree':
                        from sklearn.tree import DecisionTreeClassifier
                        gc = GridSearchCV(estimator=DecisionTreeClassifier(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'Ridge':
                        from sklearn.linear_model import RidgeClassifier
                        gc = GridSearchCV(estimator=RidgeClassifier(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'Random Forest':
                        from sklearn.ensemble import RandomForestClassifier
                        gc = GridSearchCV(estimator=RandomForestClassifier(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'Support Vector':
                        from sklearn.svm import SVC
                        gc = GridSearchCV(estimator=SVC(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'SGD':
                        from sklearn.linear_model import SGDClassifier
                        gc = GridSearchCV(estimator=SGDClassifier(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'Nearest Neighbor':
                        from sklearn.neighbors import KNeighborsClassifier
                        gc = GridSearchCV(estimator=KNeighborsClassifier(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'Gaussian Process':
                        from sklearn.gaussian_process import GaussianProcessClassifier
                        gc = GridSearchCV(estimator=GaussianProcessClassifier(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'Multi-layer Perceptron':
                        from sklearn.neural_network import MLPClassifier
                        gc = GridSearchCV(estimator=MLPClassifier(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
                        if prev.x_st_var.get() == 'No':
                            gc.fit(prev.X, prev.y)
                        else:
                            gc.fit(prev.X_St, prev.y)
                    elif method == 'XGBoost':
                        from xgboost import XGBClassifier
                        gc = GridSearchCV(estimator=XGBClassifier(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
                        if prev.x_st_var.get() == 'Yes':
                            gc.fit(prev.X_St, prev.y)
                        else:
                            gc.fit(prev.X, prev.y)
                    elif method == 'CatBoost':
                        from catboost import CatBoostClassifier
                        gc = GridSearchCV(estimator=CatBoostClassifier(), 
                            param_grid=eval(param_grid_entry.get("1.0",'end')), 
                            cv=folds, n_jobs=int(n_jobs_entry.get()))
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

                    # do_grid_search(method, main)

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
    # Running classification app itself
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
        w = 600
        h = 500   
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        clf_app.root = tk.Toplevel(parent)
        clf_app.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        clf_app.root.title('Monkey Classification')
        clf_app.root.lift()
        clf_app.root.tkraise()
        clf_app.root.focus_force()
        clf_app.root.resizable(False, False)

        clf_app.root.protocol("WM_DELETE_WINDOW", lambda: quit_back(clf_app.root, parent))

        parent.iconify()
        
        self.frame = ttk.Frame(clf_app.root, width=w, height=h)
        self.frame.place(x=0, y=0)
        
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
            command=lambda: load_data(self, self.training, e1, 'clf training')).place(x=490, y=50)
        
        cb1 = ttk.Checkbutton(self.frame, text="header", 
            variable=self.training.header_var, takefocus=False)
        cb1.place(x=10, y=50)
        
        ttk.Label(self.frame, text='Data status:').place(x=10, y=95)
        self.training.data_status = ttk.Label(self.frame, text='Not Loaded')
        self.training.data_status.place(x=120, y=95)

        ttk.Button(self.frame, text='View/Change', 
            command=lambda: Data_Preview(self, self.training, 
                'clf training', parent)).place(x=230, y=95)
        #methods parameters
        self.rc_include_comp = tk.BooleanVar(value=True)
        self.rc_alpha = tk.StringVar(value='1.0')
        self.rc_fit_intercept = tk.BooleanVar(value=True)
        self.rc_normalize = tk.BooleanVar(value=False)
        self.rc_copy_X = tk.BooleanVar(value=True)
        self.rc_max_iter = tk.StringVar(value='None')
        self.rc_tol = tk.StringVar(value='1e-3')
        self.rc_solver = tk.StringVar(value='auto')
        self.rc_random_state = tk.StringVar(value='None')
        
        self.dtc_include_comp = tk.BooleanVar(value=True)
        self.dtc_criterion = tk.StringVar(value='gini')
        self.dtc_splitter = tk.StringVar(value='best')
        self.dtc_max_depth = tk.StringVar(value='None')
        self.dtc_min_samples_split = tk.StringVar(value='2')
        self.dtc_min_samples_leaf = tk.StringVar(value='1')
        self.dtc_min_weight_fraction_leaf = tk.StringVar(value='0.0')
        self.dtc_max_features = tk.StringVar(value='None')
        self.dtc_random_state = tk.StringVar(value='None')
        self.dtc_max_leaf_nodes = tk.StringVar(value='None')
        self.dtc_min_impurity_decrease = tk.StringVar(value='0.0')
        self.dtc_ccp_alpha = tk.StringVar(value='0.0')
        
        self.rfc_include_comp = tk.BooleanVar(value=True)
        self.rfc_n_estimators = tk.StringVar(value='100')
        self.rfc_criterion = tk.StringVar(value='gini')
        self.rfc_max_depth = tk.StringVar(value='None')
        self.rfc_min_samples_split = tk.StringVar(value='2')
        self.rfc_min_samples_leaf = tk.StringVar(value='1')
        self.rfc_min_weight_fraction_leaf = tk.StringVar(value='0.0')
        self.rfc_max_features = tk.StringVar(value='auto')
        self.rfc_max_leaf_nodes = tk.StringVar(value='None')
        self.rfc_min_impurity_decrease = tk.StringVar(value='0.0')
        self.rfc_bootstrap = tk.BooleanVar(value=True)
        self.rfc_oob_score = tk.BooleanVar(value=False)
        self.rfc_n_jobs = tk.StringVar(value='3')
        self.rfc_random_state = tk.StringVar(value='None')
        self.rfc_verbose = tk.StringVar(value='0')
        self.rfc_warm_start = tk.BooleanVar(value=False)
        self.rfc_ccp_alpha = tk.StringVar(value='0.0')
        self.rfc_max_samples = tk.StringVar(value='None')
        
        self.svc_include_comp = tk.BooleanVar(value=True)
        self.svc_C = tk.StringVar(value='1.0')
        self.svc_kernel = tk.StringVar(value='rbf')
        self.svc_degree = tk.StringVar(value='3')
        self.svc_gamma = tk.StringVar(value='scale')
        self.svc_coef0 = tk.StringVar(value='0.0')
        self.svc_shrinking = tk.BooleanVar(value=True)
        self.svc_probability = tk.BooleanVar(value=False)
        self.svc_tol = tk.StringVar(value='1e-3')
        self.svc_cache_size = tk.StringVar(value='200')
        self.svc_verbose = tk.BooleanVar(value=False)
        self.svc_max_iter = tk.StringVar(value='-1')
        self.svc_decision_function_shape = tk.StringVar(value='ovr')
        self.svc_break_ties = tk.BooleanVar(value=False)
        self.svc_random_state = tk.StringVar(value='None')
        
        self.sgdc_include_comp = tk.BooleanVar(value=True)
        self.sgdc_loss = tk.StringVar(value='hinge')
        self.sgdc_penalty = tk.StringVar(value='l2')
        self.sgdc_alpha = tk.StringVar(value='0.0001')
        self.sgdc_l1_ratio = tk.StringVar(value='0.15')
        self.sgdc_fit_intercept = tk.BooleanVar(value=True)
        self.sgdc_max_iter = tk.StringVar(value='1000')
        self.sgdc_tol = tk.StringVar(value='1e-3')
        self.sgdc_shuffle = tk.BooleanVar(value=True)
        self.sgdc_verbose = tk.StringVar(value='0')
        self.sgdc_epsilon = tk.StringVar(value='0.1')
        self.sgdc_n_jobs = tk.StringVar(value='None')
        self.sgdc_random_state = tk.StringVar(value='None')
        self.sgdc_learning_rate = tk.StringVar(value='optimal')
        self.sgdc_eta0 = tk.StringVar(value='0.0')
        self.sgdc_power_t = tk.StringVar(value='0.5')
        self.sgdc_early_stopping = tk.BooleanVar(value=False)
        self.sgdc_validation_fraction = tk.StringVar(value='0.1')
        self.sgdc_n_iter_no_change = tk.StringVar(value='5')
        self.sgdc_warm_start = tk.BooleanVar(value=False)
        self.sgdc_average = tk.StringVar(value='False')
        
        self.gpc_include_comp = tk.BooleanVar(value=True)
        self.gpc_n_restarts_optimizer = tk.StringVar(value='0')
        self.gpc_max_iter_predict = tk.StringVar(value='100')
        self.gpc_warm_start = tk.BooleanVar(value=False)
        self.gpc_copy_X_train = tk.BooleanVar(value=True)
        self.gpc_random_state = tk.StringVar(value='None')
        self.gpc_multi_class = tk.StringVar(value='one_vs_rest')
        self.gpc_n_jobs = tk.StringVar(value='None')
        
        self.knc_include_comp = tk.BooleanVar(value=True)
        self.knc_n_neighbors = tk.StringVar(value='5')
        self.knc_weights = tk.StringVar(value='uniform')
        self.knc_algorithm = tk.StringVar(value='auto')
        self.knc_leaf_size = tk.StringVar(value='30')
        self.knc_p = tk.StringVar(value='2')
        self.knc_metric = tk.StringVar(value='minkowski')
        self.knc_n_jobs = tk.StringVar(value='None')
        
        self.mlpc_include_comp = tk.BooleanVar(value=True)
        self.mlpc_hidden_layer_sizes = tk.StringVar(value='(100,)')
        self.mlpc_activation = tk.StringVar(value='relu')
        self.mlpc_solver = tk.StringVar(value='adam')
        self.mlpc_alpha = tk.StringVar(value='0.0001')
        self.mlpc_batch_size = tk.StringVar(value='auto')
        self.mlpc_learning_rate = tk.StringVar(value='constant')
        self.mlpc_learning_rate_init = tk.StringVar(value='0.001')
        self.mlpc_power_t = tk.StringVar(value='0.5')
        self.mlpc_max_iter = tk.StringVar(value='200')
        self.mlpc_shuffle = tk.BooleanVar(value=True)
        self.mlpc_random_state = tk.StringVar(value='None')
        self.mlpc_tol = tk.StringVar(value='1e-4')
        self.mlpc_verbose = tk.BooleanVar(value=False)
        self.mlpc_warm_start = tk.BooleanVar(value=False)
        self.mlpc_momentum = tk.StringVar(value='0.9')
        self.mlpc_nesterovs_momentum = tk.BooleanVar(value=True)
        self.mlpc_early_stopping = tk.BooleanVar(value=False)
        self.mlpc_validation_fraction = tk.StringVar(value='0.1')
        self.mlpc_beta_1 = tk.StringVar(value='0.9')
        self.mlpc_beta_2 = tk.StringVar(value='0.999')
        self.mlpc_epsilon = tk.StringVar(value='1e-8')
        self.mlpc_n_iter_no_change = tk.StringVar(value='10')
        self.mlpc_max_fun = tk.StringVar(value='15000')

        self.xgbc_include_comp = tk.BooleanVar(value=True)
        self.xgbc_n_estimators = tk.StringVar(value='1000')
        self.xgbc_eta = tk.StringVar(value='0.1')
        self.xgbc_min_child_weight = tk.StringVar(value='1')
        self.xgbc_max_depth = tk.StringVar(value='6')
        self.xgbc_gamma = tk.StringVar(value='1')
        self.xgbc_subsample = tk.StringVar(value='1.0')
        self.xgbc_colsample_bytree = tk.StringVar(value='1.0')
        self.xgbc_lambda = tk.StringVar(value='1.0')
        self.xgbc_alpha = tk.StringVar(value='0.0')
        self.xgbc_use_gpu = tk.BooleanVar(value=False)

        self.cbc_include_comp = tk.BooleanVar(value=True)
        self.cbc_iterations = tk.StringVar(value='1000')
        self.cbc_learning_rate = tk.StringVar(value='None')
        self.cbc_depth = tk.StringVar(value='6')
        self.cbc_reg_lambda = tk.StringVar(value='None')
        self.cbc_subsample = tk.StringVar(value='None')
        self.cbc_colsample_bylevel = tk.StringVar(value='1.0')
        self.cbc_random_strength = tk.StringVar(value='1.0')
        self.cbc_use_gpu = tk.BooleanVar(value=False)
        self.cbc_cf_list = tk.StringVar(value='[]')

        # flag for stopping comparison
        self.continue_flag = True

        # function to compare classification methods
        def compare_methods(prev, main):
            try:
                try:
                    x_from = main.data.columns.get_loc(main.x_from_var.get())
                    x_to = main.data.columns.get_loc(main.x_to_var.get()) + 1
                except:
                    x_from = int(main.x_from_var.get())
                    x_to = int(main.x_to_var.get()) + 1
                if self.dummies_var.get()==0:
                    try:
                        prev.X = main.data.iloc[:,x_from : x_to]
                        prev.y = main.data[self.y_var.get()]
                    except:
                        prev.X = main.data.iloc[:,x_from : x_to]
                        prev.y = main.data[int(self.y_var.get())]
                elif self.dummies_var.get()==1:
                    try:
                        prev.X = pd.get_dummies(main.data.iloc[:,x_from : x_to])
                        prev.y = main.data[self.y_var.get()]
                    except:
                        prev.X = pd.get_dummies(main.data.iloc[:,x_from : x_to])
                        prev.y = main.data[int(self.y_var.get())]
                from sklearn import preprocessing
                scaler = preprocessing.StandardScaler()
                try:
                    prev.X_St = scaler.fit_transform(prev.X)
                except:
                    self.x_st_var.set('No')
                self.scores = {}
                from sklearn.model_selection import cross_val_score
                from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
                folds_list = []
                for i in range(int(rep_entry.get())):
                    folds_list.append(StratifiedKFold(n_splits = int(e2.get()), shuffle=True))
                self.pb1.step(1)
                # Decision Tree Classifier
                if self.dtc_include_comp.get() and self.continue_flag:
                    from sklearn.tree import DecisionTreeClassifier
                    dtc = DecisionTreeClassifier(
                        criterion=self.dtc_criterion.get(), splitter=self.dtc_splitter.get(), 
                        max_depth=(int(self.dtc_max_depth.get()) 
                            if (self.dtc_max_depth.get() != 'None') else None), 
                        min_samples_split=(float(self.dtc_min_samples_split.get()) 
                            if '.' in self.dtc_min_samples_split.get()
                            else int(self.dtc_min_samples_split.get())), 
                        min_samples_leaf=(float(self.dtc_min_samples_leaf.get()) 
                            if '.' in self.dtc_min_samples_leaf.get()
                            else int(self.dtc_min_samples_leaf.get())),
                        min_weight_fraction_leaf=float(self.dtc_min_weight_fraction_leaf.get()),
                        max_features=(float(self.dtc_max_features.get()) 
                            if '.' in self.dtc_max_features.get() 
                            else int(self.dtc_max_features.get()) 
                            if len(self.dtc_max_features.get()) < 4 
                            else self.dtc_max_features.get() 
                            if (self.dtc_max_features.get() != 'None') 
                            else None), 
                        random_state=(int(self.dtc_random_state.get()) 
                            if (self.dtc_random_state.get() != 'None') else None),
                        max_leaf_nodes=(int(self.dtc_max_leaf_nodes.get()) 
                            if (self.dtc_max_leaf_nodes.get() != 'None') else None),
                        min_impurity_decrease=float(self.dtc_min_impurity_decrease.get()),
                        ccp_alpha=float(self.dtc_ccp_alpha.get()))
                    dtc_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        for fold in folds_list:
                            if self.continue_flag:
                                dtc_scores = np.append(dtc_scores, 
                                    cross_val_score(dtc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                        # dtc_scores = cross_val_score(dtc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                dtc_scores = np.append(dtc_scores, 
                                    cross_val_score(dtc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['dtc'] = dtc_scores.mean()
                # Ridge Classifier
                if self.rc_include_comp.get() and self.continue_flag:
                    from sklearn.linear_model import RidgeClassifier
                    rc = RidgeClassifier(
                        alpha=float(self.rc_alpha.get()), fit_intercept=self.rc_fit_intercept.get(),
                        normalize=self.rc_normalize.get(), copy_X=self.rc_copy_X.get(),
                        max_iter=(int(self.rc_max_iter.get()) if 
                                                     (self.rc_max_iter.get() != 'None') else None),
                        tol=float(self.rc_tol.get()), solver=self.rc_solver.get(),
                        random_state=(int(self.rc_random_state.get()) if 
                                      (self.rc_random_state.get() != 'None') else None))
                    rc_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        for fold in folds_list:
                            if self.continue_flag:
                                rc_scores = np.append(rc_scores, 
                                    cross_val_score(rc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                        # rc_scores = cross_val_score(rc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                rc_scores = np.append(rc_scores, 
                                    cross_val_score(rc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['rc'] = rc_scores.mean()
                # Random forest Classifier
                if self.rfc_include_comp.get() and self.continue_flag:
                    from sklearn.ensemble import RandomForestClassifier
                    rfc = RandomForestClassifier(
                        n_estimators=int(self.rfc_n_estimators.get()), 
                        criterion=self.rfc_criterion.get(),
                        max_depth=(int(self.rfc_max_depth.get()) 
                            if (self.rfc_max_depth.get() != 'None') else None), 
                        min_samples_split=(float(self.rfc_min_samples_split.get()) 
                            if '.' in self.rfc_min_samples_split.get()
                            else int(self.rfc_min_samples_split.get())),
                        min_samples_leaf=(float(self.rfc_min_samples_leaf.get()) 
                            if '.' in self.rfc_min_samples_leaf.get()
                            else int(self.rfc_min_samples_leaf.get())),
                        min_weight_fraction_leaf=float(self.rfc_min_weight_fraction_leaf.get()),
                        max_features=(float(self.rfc_max_features.get()) 
                            if '.' in self.rfc_max_features.get() 
                            else int(self.rfc_max_features.get()) 
                            if len(self.rfc_max_features.get()) < 4 
                            else self.rfc_max_features.get() 
                            if (self.rfc_max_features.get() != 'None') 
                            else None),
                        max_leaf_nodes=(int(self.rfc_max_leaf_nodes.get()) 
                            if (self.rfc_max_leaf_nodes.get() != 'None') else None),
                        min_impurity_decrease=float(self.rfc_min_impurity_decrease.get()),
                        bootstrap=self.rfc_bootstrap.get(), oob_score=self.rfc_oob_score.get(),
                        n_jobs=(int(self.rfc_n_jobs.get()) 
                            if (self.rfc_n_jobs.get() != 'None') else None),
                        random_state=(int(self.rfc_random_state.get()) 
                            if (self.rfc_random_state.get() != 'None') else None),
                        verbose=int(self.rfc_verbose.get()), warm_start=self.rfc_warm_start.get(),
                        ccp_alpha=float(self.rfc_ccp_alpha.get()),
                        max_samples=(float(self.rfc_max_samples.get()) 
                            if '.' in self.rfc_max_samples.get() 
                            else int(self.rfc_max_samples.get()) 
                            if (self.rfc_max_samples.get() != 'None') 
                            else None))
                    rfc_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        for fold in folds_list:
                            if self.continue_flag:
                                rfc_scores = np.append(rfc_scores, 
                                    cross_val_score(rfc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                        # rfc_scores = cross_val_score(rfc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                rfc_scores = np.append(rfc_scores, 
                                    cross_val_score(rfc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['rfc'] = rfc_scores.mean()
                # Support Vector Classifier
                if self.svc_include_comp.get() and self.continue_flag:
                    from sklearn.svm import SVC
                    svc = SVC(
                        C=float(self.svc_C.get()), kernel=self.svc_kernel.get(), 
                        degree=int(self.svc_degree.get()), 
                        gamma=(float(self.svc_gamma.get()) 
                            if '.' in self.svc_gamma.get() else self.svc_gamma.get()),
                        coef0=float(self.svc_coef0.get()), shrinking=self.svc_shrinking.get(), 
                        probability=self.svc_probability.get(), tol=float(self.svc_tol.get()),
                        cache_size=float(self.svc_cache_size.get()), 
                        verbose=self.svc_verbose.get(),
                        max_iter=int(self.svc_max_iter.get()),
                        decision_function_shape=self.svc_decision_function_shape.get(),
                        break_ties=self.svc_break_ties.get(), 
                        random_state=(int(self.svc_random_state.get()) 
                            if (self.svc_random_state.get() != 'None') else None))
                    svc_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            if self.continue_flag:
                                svc_scores = np.append(svc_scores, 
                                    cross_val_score(svc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                        # svc_scores = cross_val_score(svc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                svc_scores = np.append(svc_scores, 
                                    cross_val_score(svc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['svc'] = svc_scores.mean()
                # Stochastic Gradient Descent Classifier
                if self.sgdc_include_comp.get() and self.continue_flag:
                    from sklearn.linear_model import SGDClassifier
                    sgdc = SGDClassifier(
                        loss=self.sgdc_loss.get(), penalty=self.sgdc_penalty.get(),
                        alpha=float(self.sgdc_alpha.get()), 
                        l1_ratio=float(self.sgdc_l1_ratio.get()),
                        fit_intercept=self.sgdc_fit_intercept.get(), 
                        max_iter=int(self.sgdc_max_iter.get()),
                        tol=float(self.sgdc_tol.get()), shuffle=self.sgdc_shuffle.get(), 
                        verbose=int(self.sgdc_verbose.get()), 
                        epsilon=float(self.sgdc_epsilon.get()),
                        n_jobs=(int(self.sgdc_n_jobs.get()) 
                            if (self.sgdc_n_jobs.get() != 'None') else None), 
                        random_state=(int(self.sgdc_random_state.get()) 
                            if (self.sgdc_random_state.get() != 'None') else None),
                        learning_rate=self.sgdc_learning_rate.get(), 
                        eta0=float(self.sgdc_eta0.get()),
                        power_t=float(self.sgdc_power_t.get()), 
                        early_stopping=self.sgdc_early_stopping.get(),
                        validation_fraction=float(self.sgdc_validation_fraction.get()),
                        n_iter_no_change=int(self.sgdc_n_iter_no_change.get()), 
                        warm_start=self.sgdc_warm_start.get(),
                        average=(True if self.sgdc_average.get()=='True' else False 
                            if self.sgdc_average.get()=='False' else int(self.sgdc_average.get())))
                    sgdc_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            if self.continue_flag:
                                sgdc_scores = np.append(sgdc_scores, 
                                    cross_val_score(sgdc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                        # sgdc_scores = cross_val_score(sgdc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                sgdc_scores = np.append(sgdc_scores, 
                                    cross_val_score(sgdc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['sgdc'] = sgdc_scores.mean()
                # Nearest Neighbor Classifier
                if self.knc_include_comp.get() and self.continue_flag:
                    from sklearn.neighbors import KNeighborsClassifier
                    knc = KNeighborsClassifier(
                        n_neighbors=int(self.knc_n_neighbors.get()), 
                        weights=self.knc_weights.get(), algorithm=self.knc_algorithm.get(),
                        leaf_size=int(self.knc_leaf_size.get()), p=int(self.knc_p.get()),
                        metric=self.knc_metric.get(), 
                        n_jobs=(int(self.knc_n_jobs.get()) 
                            if (self.knc_n_jobs.get() != 'None') else None))
                    knc_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            if self.continue_flag:
                                knc_scores = np.append(knc_scores, 
                                    cross_val_score(knc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                        # knc_scores = cross_val_score(knc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                knc_scores = np.append(knc_scores, 
                                    cross_val_score(knc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['knc'] = knc_scores.mean()
                # Gaussian process Classifier
                if self.gpc_include_comp.get() and self.continue_flag:
                    from sklearn.gaussian_process import GaussianProcessClassifier
                    gpc = GaussianProcessClassifier(
                        n_restarts_optimizer=int(self.gpc_n_restarts_optimizer.get()),
                        max_iter_predict=int(self.gpc_max_iter_predict.get()),
                        warm_start=self.gpc_warm_start.get(), 
                        copy_X_train=self.gpc_copy_X_train.get(),
                        random_state=(int(self.gpc_random_state.get()) if 
                                      (self.gpc_random_state.get() != 'None') else None),
                        multi_class=self.gpc_multi_class.get(), 
                        n_jobs=(int(self.gpc_n_jobs.get()) if 
                                      (self.gpc_n_jobs.get() != 'None') else None))
                    gpc_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            if self.continue_flag:
                                gpc_scores = np.append(gpc_scores, 
                                    cross_val_score(gpc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                        # gpc_scores = cross_val_score(gpc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                gpc_scores = np.append(gpc_scores, 
                                    cross_val_score(gpc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['gpc'] = gpc_scores.mean()
                # Multi-layer Perceptron Classifier
                if self.mlpc_include_comp.get() and self.continue_flag:
                    from sklearn.neural_network import MLPClassifier
                    mlpc = MLPClassifier(
                        hidden_layer_sizes=eval(self.mlpc_hidden_layer_sizes.get()),
                        activation=self.mlpc_activation.get(),solver=self.mlpc_solver.get(),
                        alpha=float(self.mlpc_alpha.get()), 
                        batch_size=(int(self.mlpc_batch_size.get()) 
                            if (self.mlpc_batch_size.get() != 'auto') else 'auto'),
                        learning_rate=self.mlpc_learning_rate.get(), 
                        learning_rate_init=float(self.mlpc_learning_rate_init.get()),
                        power_t=float(self.mlpc_power_t.get()), 
                        max_iter=int(self.mlpc_max_iter.get()),
                        shuffle=self.mlpc_shuffle.get(),
                        random_state=(int(self.mlpc_random_state.get()) 
                            if (self.mlpc_random_state.get() != 'None') else None),
                        tol=float(self.mlpc_tol.get()), verbose=self.mlpc_verbose.get(),
                        warm_start=self.mlpc_warm_start.get(), 
                        momentum=float(self.mlpc_momentum.get()),
                        nesterovs_momentum=self.mlpc_nesterovs_momentum.get(),
                        early_stopping=self.mlpc_early_stopping.get(), 
                        validation_fraction=float(self.mlpc_validation_fraction.get()),
                        beta_1=float(self.mlpc_beta_1.get()), 
                        beta_2=float(self.mlpc_beta_2.get()),
                        epsilon=float(self.mlpc_epsilon.get()), 
                        n_iter_no_change=int(self.mlpc_n_iter_no_change.get()),
                        max_fun=int(self.mlpc_max_fun.get()))
                    mlpc_scores = np.array([])
                    if self.x_st_var.get() == 'No':
                        for fold in folds_list:
                            if self.continue_flag:
                                mlpc_scores = np.append(mlpc_scores, 
                                    cross_val_score(mlpc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                        # mlpc_scores = cross_val_score(mlpc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                mlpc_scores = np.append(mlpc_scores, 
                                    cross_val_score(mlpc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['mlpc'] = mlpc_scores.mean()
                # XGB Classifier
                if self.xgbc_include_comp.get() and self.continue_flag:
                    from xgboost import XGBClassifier
                    if self.xgbc_use_gpu.get() == False:
                        xgbc = XGBClassifier(
                            learning_rate=float(self.xgbc_eta.get()), 
                            n_estimators=int(self.xgbc_n_estimators.get()), 
                            max_depth=int(self.xgbc_max_depth.get()),
                            min_child_weight=int(self.xgbc_min_child_weight.get()),
                            gamma=float(self.xgbc_gamma.get()), 
                            subsample=float(self.xgbc_subsample.get()),
                            colsample_bytree=float(self.xgbc_colsample_bytree.get()),
                            reg_lambda=float(self.xgbc_lambda.get()), 
                            reg_alpha=float(self.xgbc_alpha.get()))
                    else:
                        xgbc = XGBClassifier(
                            learning_rate=float(self.xgbc_eta.get()), 
                            n_estimators=int(self.xgbc_n_estimators.get()), 
                            max_depth=int(self.xgbc_max_depth.get()),
                            min_child_weight=int(self.xgbc_min_child_weight.get()),
                            gamma=float(self.xgbc_gamma.get()), 
                            subsample=float(self.xgbc_subsample.get()),
                            colsample_bytree=float(self.xgbc_colsample_bytree.get()),
                            reg_lambda=float(self.xgbc_lambda.get()), 
                            reg_alpha=float(self.xgbc_alpha.get()),
                            tree_method='gpu_hist', gpu_id=0)
                    xgbc_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        for fold in folds_list:
                            if self.continue_flag:
                                xgbc_scores = np.append(xgbc_scores, 
                                    cross_val_score(xgbc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                        # xgbc_scores = cross_val_score(xgbc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                xgbc_scores = np.append(xgbc_scores, 
                                    cross_val_score(xgbc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['xgbc'] = xgbc_scores.mean()
                # CatBoost
                if self.cbc_include_comp.get() and self.continue_flag:
                    from catboost import CatBoostClassifier
                    if self.cbc_use_gpu.get() == False:
                        cbc = CatBoostClassifier(
                            cat_features=eval(self.cbc_cf_list.get()),
                            iterations=int(self.cbc_iterations.get()),
                            learning_rate=(None if self.cbc_learning_rate.get()=='None' 
                                else float(self.cbc_learning_rate.get())),
                            depth=int(self.cbc_depth.get()), 
                            reg_lambda=(None if self.cbc_reg_lambda.get()=='None' 
                                else float(self.cbc_reg_lambda.get())),
                            subsample=(None if self.cbc_subsample.get()=='None' 
                                else float(self.cbc_subsample.get())), 
                            colsample_bylevel=float(self.cbc_colsample_bylevel.get()),
                            random_strength=float(self.cbc_random_strength.get()))
                    else:
                        cbc = CatBoostClassifier(
                            cat_features=eval(self.cbc_cf_list.get()),
                            iterations=int(self.cbc_iterations.get()), 
                            task_type="GPU", devices='0:1',
                            learning_rate=(None if self.cbc_learning_rate.get()=='None' 
                                else float(self.cbc_learning_rate.get())),
                            depth=int(self.cbc_depth.get()), 
                            reg_lambda=(None if self.cbc_reg_lambda.get()=='None' 
                                else float(self.cbc_reg_lambda.get())),
                            subsample=(None if self.cbc_subsample.get()=='None' 
                                else float(self.cbc_subsample.get())), 
                            colsample_bylevel=float(self.cbc_colsample_bylevel.get()),
                            random_strength=float(self.cbc_random_strength.get()))
                    cbc_scores = np.array([])
                    if self.x_st_var.get() == 'Yes':
                        for fold in folds_list:
                            if self.continue_flag:
                                cbc_scores = np.append(cbc_scores, 
                                    cross_val_score(cbc, prev.X_St, prev.y, cv=fold))
                                self.pb1.step(1)
                        # cbc_scores = cross_val_score(cbc, prev.X_St, prev.y, cv=folds)
                    else:
                        for fold in folds_list:
                            if self.continue_flag:
                                cbc_scores = np.append(cbc_scores, 
                                    cross_val_score(cbc, prev.X, prev.y, cv=fold))
                                self.pb1.step(1)
                    self.scores['cbc'] = cbc_scores.mean()
            except ValueError as e:
                messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))

            self.continue_flag = True
            self.pb1.destroy()
            self.stop_button.destroy()
            self.show_res_button = ttk.Button(self.frame, 
                text='Show Results', command=lambda: self.Comp_results(self, parent))
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
                for x in [self.dtc_include_comp.get(), self.rc_include_comp.get(),
                          self.rfc_include_comp.get(), self.svc_include_comp.get(),
                          self.sgdc_include_comp.get(), self.knc_include_comp.get(),
                          self.gpc_include_comp.get(), self.mlpc_include_comp.get(),
                          self.xgbc_include_comp.get(), self.cbc_include_comp.get(),]:
                    if x==True:
                        l+=1*int(rep_entry.get())

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

        ttk.Button(self.frame, text="Methods' specifications", 
                  command=lambda: clf_mtds_specification(self, parent)).place(x=400, y=95)
        ttk.Button(self.frame, text='Perform comparison', 
            command=lambda: try_compare_methods(self, self.training)).place(x=400, y=130)
        self.show_res_button = ttk.Button(self.frame, text='Show Results', 
            command=lambda: self.Comp_results(self, parent))
        
        ttk.Button(self.frame, text='Grid Search', 
            command=lambda: self.Grid_search(self, parent)).place(x=400, y=200)
        
        ttk.Label(self.frame, text='Choose y', font=myfont1).place(x=30, y=140)
        self.y_var = tk.StringVar()
        self.combobox1 = ttk.Combobox(self.frame, textvariable=self.y_var, width=14, values=[])
        self.combobox1.place(x=105,y=142)
        
        ttk.Label(self.frame, text='X from', font=myfont1).place(x=225, y=130)
        self.tr_x_from_combobox = ttk.Combobox(self.frame, 
            textvariable=self.training.x_from_var, width=14, values=[])
        self.tr_x_from_combobox.place(x=275, y=132)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=225, y=155)
        self.tr_x_to_combobox = ttk.Combobox(self.frame, 
            textvariable=self.training.x_to_var, width=14, values=[])
        self.tr_x_to_combobox.place(x=275, y=157)

        self.x_st_var = tk.StringVar(value='If needed')
        ttk.Label(self.frame, text='X Standartization', font=myfont1).place(x=30, y=175)
        self.combobox2 = ttk.Combobox(self.frame, textvariable=self.x_st_var, width=10,
                                        values=['No', 'If needed', 'Yes'])
        self.combobox2.place(x=150,y=180)

        ttk.Label(self.frame, text='Number of folds:', font=myfont1).place(x=30, y=205)
        e2 = ttk.Entry(self.frame, font=myfont1, width=4)
        e2.place(x=150, y=207)
        e2.insert(0, "5")

        self.dummies_var = tk.IntVar(value=0)
        cb2 = ttk.Checkbutton(self.frame, text="Dummies", 
            variable=self.dummies_var, takefocus=False)
        cb2.place(x=270, y=182)

        ttk.Label(self.frame, text='Number of repeats:', font=myfont1).place(x=200, y=205)
        rep_entry = ttk.Entry(self.frame, font=myfont1, width=4)
        rep_entry.place(x=330, y=208)
        rep_entry.insert(0, "1")
        
        ttk.Label(self.frame, text='Predict Data file:', font=myfont1).place(x=10, y=250)
        pr_data_entry = ttk.Entry(self.frame, font=myfont1, width=38)
        pr_data_entry.place(x=140, y=250)
        
        ttk.Button(self.frame, text='Choose file', 
            command=lambda: open_file(self, pr_data_entry)).place(x=490, y=250)
        
        ttk.Label(self.frame, text='List number:', font=myfont1).place(x=120,y=295)
        pr_sheet_entry = ttk.Entry(self.frame, 
            textvariable=self.prediction.sheet, font=myfont1, width=3)
        pr_sheet_entry.place(x=215,y=297)
        
        ttk.Button(self.frame, text='Load data ', 
            command=lambda: load_data(self, self.prediction, 
                pr_data_entry, 'clf prediction')).place(x=490, y=290)
        
        cb4 = ttk.Checkbutton(self.frame, text="header", 
            variable=self.prediction.header_var, takefocus=False)
        cb4.place(x=10, y=290)
        
        ttk.Label(self.frame, text='Data status:', font=myfont).place(x=10, y=345)
        self.prediction.data_status = ttk.Label(self.frame, text='Not Loaded')
        self.prediction.data_status.place(x=120, y=345)

        ttk.Button(self.frame, text='View/Change', command=lambda: 
            Data_Preview(self, self.prediction, 
                'clf prediction', parent)).place(x=230, y=345)
        
        self.pr_method = tk.StringVar(value='Decision Tree')
        self.combobox9 = ttk.Combobox(self.frame, textvariable=self.pr_method, width=15, 
            values=['Decision Tree', 'Ridge', 'Random Forest', 'Support Vector', 'SGD',  
                    'Nearest Neighbor', 'Gaussian Process', 'Multi-layer Perceptron',
                    'XGBoost', 'CatBoost'])
        self.combobox9.place(x=105,y=402)
        ttk.Label(self.frame, text='Method', font=myfont1).place(x=30, y=400)

        ttk.Label(self.frame, text='Place result', font=myfont1).place(x=30, y=425)
        self.place_result_var = tk.StringVar(value='End')
        self.combobox9 = ttk.Combobox(self.frame, textvariable=self.place_result_var, 
            width=10, values=['Start', 'End'])
        self.combobox9.place(x=120,y=427)
        
        ttk.Label(self.frame, text='X from', font=myfont1).place(x=225, y=400)
        self.pr_x_from_combobox = ttk.Combobox(self.frame, 
            textvariable=self.prediction.x_from_var, width=14, values=[])
        self.pr_x_from_combobox.place(x=275, y=402)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=225, y=425)
        self.pr_x_to_combobox = ttk.Combobox(self.frame, 
            textvariable=self.prediction.x_to_var, width=14, values=[])
        self.pr_x_to_combobox.place(x=275, y=427)

        # function to predict class        
        def clf_predict_class(method):
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
                if self.dummies_var.get()==0:
                    try:
                        training_X = self.training.data.iloc[:,tr_x_from : tr_x_to]
                        training_y = self.training.data[self.y_var.get()]
                    except:
                        training_X = self.training.data.iloc[:,tr_x_from : tr_x_to]
                        training_y = self.training.data[int(self.y_var.get())]
                    X = self.prediction.data.iloc[:,pr_x_from : pr_x_to]
                elif self.dummies_var.get()==1:
                    X = pd.get_dummies(self.prediction.data.iloc[:,pr_x_from : pr_x_to])
                    try:
                        training_X = pd.get_dummies(self.training.data.iloc[:,tr_x_from : tr_x_to])
                        training_y = self.training.data[self.y_var.get()]
                    except:
                        training_X = pd.get_dummies(self.training.data.iloc[:,tr_x_from : tr_x_to])
                        training_y = self.training.data[int(self.y_var.get())]
                from sklearn import preprocessing
                scaler = preprocessing.StandardScaler()
                try:
                    X_St = scaler.fit_transform(X)
                    training_X_St = scaler.fit_transform(training_X)
                except:
                    self.x_st_var.set('No')
                training_X_St = scaler.fit_transform(training_X)
                
                if method == 'Decision Tree':
                    from sklearn.tree import DecisionTreeClassifier
                    dtc = DecisionTreeClassifier(
                        criterion=self.dtc_criterion.get(), splitter=self.dtc_splitter.get(), 
                        max_depth=(int(self.dtc_max_depth.get()) 
                            if (self.dtc_max_depth.get() != 'None') else None), 
                        min_samples_split=(float(self.dtc_min_samples_split.get()) 
                            if '.' in self.dtc_min_samples_split.get()
                            else int(self.dtc_min_samples_split.get())), 
                        min_samples_leaf=(float(self.dtc_min_samples_leaf.get()) 
                            if '.' in self.dtc_min_samples_leaf.get()
                            else int(self.dtc_min_samples_leaf.get())),
                        min_weight_fraction_leaf=float(self.dtc_min_weight_fraction_leaf.get()),
                        max_features=(float(self.dtc_max_features.get()) 
                            if '.' in self.dtc_max_features.get() 
                            else int(self.dtc_max_features.get()) 
                            if len(self.dtc_max_features.get()) < 4 
                            else self.dtc_max_features.get() 
                            if (self.dtc_max_features.get() != 'None') 
                            else None), 
                        random_state=(int(self.dtc_random_state.get()) 
                            if (self.dtc_random_state.get() != 'None') else None),
                        max_leaf_nodes=(int(self.dtc_max_leaf_nodes.get()) 
                            if (self.dtc_max_leaf_nodes.get() != 'None') else None),
                        min_impurity_decrease=float(self.dtc_min_impurity_decrease.get()),
                        ccp_alpha=float(self.dtc_ccp_alpha.get()))
                    if self.x_st_var.get() == 'Yes':
                        dtc.fit(training_X_st, training_y)
                        pr_values = dtc.predict(X_St)
                    else:
                        dtc.fit(training_X, training_y)
                        pr_values = dtc.predict(X)
                elif method == 'Ridge':
                    from sklearn.linear_model import RidgeClassifier
                    rc = RidgeClassifier(
                        alpha=float(self.rc_alpha.get()), fit_intercept=self.rc_fit_intercept.get(),
                        normalize=self.rc_normalize.get(), copy_X=self.rc_copy_X.get(),
                        max_iter=(int(self.rc_max_iter.get()) if 
                                                     (self.rc_max_iter.get() != 'None') else None),
                        tol=float(self.rc_tol.get()), solver=self.rc_solver.get(),
                        random_state=(int(self.rc_random_state.get()) if 
                                      (self.rc_random_state.get() != 'None') else None))
                    if self.x_st_var.get() == 'Yes':
                        rc.fit(training_X_st, training_y)
                        pr_values = rc.predict(X_St)
                    else:
                        rc.fit(training_X, training_y)
                        pr_values = rc.predict(X)
                elif method == 'Random Forest':
                    from sklearn.ensemble import RandomForestClassifier
                    rfc = RandomForestClassifier(
                        n_estimators=int(self.rfc_n_estimators.get()), 
                        criterion=self.rfc_criterion.get(),
                        max_depth=(int(self.rfc_max_depth.get()) 
                            if (self.rfc_max_depth.get() != 'None') else None), 
                        min_samples_split=(float(self.rfc_min_samples_split.get()) 
                            if '.' in self.rfc_min_samples_split.get()
                            else int(self.rfc_min_samples_split.get())),
                        min_samples_leaf=(float(self.rfc_min_samples_leaf.get()) 
                            if '.' in self.rfc_min_samples_leaf.get()
                            else int(self.rfc_min_samples_leaf.get())),
                        min_weight_fraction_leaf=float(self.rfc_min_weight_fraction_leaf.get()),
                        max_features=(float(self.rfc_max_features.get()) 
                            if '.' in self.rfc_max_features.get() 
                            else int(self.rfc_max_features.get()) 
                            if len(self.rfc_max_features.get()) < 4 
                            else self.rfc_max_features.get() 
                            if (self.rfc_max_features.get() != 'None') else None),
                        max_leaf_nodes=(int(self.rfc_max_leaf_nodes.get()) 
                            if (self.rfc_max_leaf_nodes.get() != 'None') else None),
                        min_impurity_decrease=float(self.rfc_min_impurity_decrease.get()),
                        bootstrap=self.rfc_bootstrap.get(), oob_score=self.rfc_oob_score.get(),
                        n_jobs=(int(self.rfc_n_jobs.get()) 
                            if (self.rfc_n_jobs.get() != 'None') else None),
                        random_state=(int(self.rfc_random_state.get()) 
                            if (self.rfc_random_state.get() != 'None') else None),
                        verbose=int(self.rfc_verbose.get()), warm_start=self.rfc_warm_start.get(),
                        ccp_alpha=float(self.rfc_ccp_alpha.get()),
                        max_samples=(float(self.rfc_max_samples.get()) 
                            if '.' in self.rfc_max_samples.get() 
                            else int(self.rfc_max_samples.get()) 
                            if (self.rfc_max_samples.get() != 'None') 
                            else None))
                    if self.x_st_var.get() == 'Yes':
                        rfc.fit(training_X_St, training_y)
                        pr_values = rfc.predict(X_St)
                    else:
                        rfc.fit(training_X, training_y)
                        pr_values = rfc.predict(X)
                elif method == 'Support Vector':
                    from sklearn.svm import SVC
                    svc = SVC(
                        C=float(self.svc_C.get()), kernel=self.svc_kernel.get(), 
                        degree=int(self.svc_degree.get()), 
                        gamma=(float(self.svc_gamma.get()) 
                            if '.' in self.svc_gamma.get() else self.svc_gamma.get()),
                        coef0=float(self.svc_coef0.get()), shrinking=self.svc_shrinking.get(), 
                        probability=self.svc_probability.get(), tol=float(self.svc_tol.get()),
                        cache_size=float(self.svc_cache_size.get()), verbose=self.svc_verbose.get(),
                        max_iter=int(self.svc_max_iter.get()),
                        decision_function_shape=self.svc_decision_function_shape.get(),
                        break_ties=self.svc_break_ties.get(), 
                        random_state=(int(self.svc_random_state.get()) 
                            if (self.svc_random_state.get() != 'None') else None))
                    if self.x_st_var.get() == 'No':
                        svc.fit(training_X, training_y)
                        pr_values = svc.predict(X)
                    else:
                        svc.fit(training_X_St, training_y)
                        pr_values = svc.predict(X_St)
                elif method == 'SGD':
                    from sklearn.linear_model import SGDClassifier
                    sgdc = SGDClassifier(
                        loss=self.sgdc_loss.get(), penalty=self.sgdc_penalty.get(),
                        alpha=float(self.sgdc_alpha.get()), l1_ratio=float(self.sgdc_l1_ratio.get()),
                        fit_intercept=self.sgdc_fit_intercept.get(), 
                        max_iter=int(self.sgdc_max_iter.get()),
                        tol=float(self.sgdc_tol.get()), shuffle=self.sgdc_shuffle.get(), 
                        verbose=int(self.sgdc_verbose.get()), epsilon=float(self.sgdc_epsilon.get()),
                        n_jobs=(int(self.sgdc_n_jobs.get()) 
                            if (self.sgdc_n_jobs.get() != 'None') else None), 
                        random_state=(int(self.sgdc_random_state.get()) 
                            if (self.sgdc_random_state.get() != 'None') else None),
                        learning_rate=self.sgdc_learning_rate.get(), eta0=float(self.sgdc_eta0.get()),
                        power_t=float(self.sgdc_power_t.get()), 
                        early_stopping=self.sgdc_early_stopping.get(),
                        validation_fraction=float(self.sgdc_validation_fraction.get()),
                        n_iter_no_change=int(self.sgdc_n_iter_no_change.get()), 
                        warm_start=self.sgdc_warm_start.get(),
                        average=(True if self.sgdc_average.get()=='True' else False 
                            if self.sgdc_average.get()=='False' else int(self.sgdc_average.get())))
                    if self.x_st_var.get() == 'No':
                        sgdc.fit(training_X, training_y)
                        pr_values = sgdc.predict(X)
                    else:
                        sgdc.fit(training_X_St, training_y)
                        pr_values = sgdc.predict(X_St)
                elif method == 'Nearest Neighbor':
                    from sklearn.neighbors import KNeighborsClassifier
                    knc = KNeighborsClassifier(
                        n_neighbors=int(self.knc_n_neighbors.get()), 
                        weights=self.knc_weights.get(), algorithm=self.knc_algorithm.get(),
                        leaf_size=int(self.knc_leaf_size.get()), p=int(self.knc_p.get()),
                        metric=self.knc_metric.get(), 
                        n_jobs=(int(self.knc_n_jobs.get()) 
                            if (self.knc_n_jobs.get() != 'None') else None))
                    if self.x_st_var.get() == 'No':
                        knc.fit(training_X, training_y)
                        pr_values = knc.predict(X)
                    else:
                        knc.fit(training_X_St, training_y)
                        pr_values = knc.predict(X_St)
                elif method == 'Gaussian Process':
                    from sklearn.gaussian_process import GaussianProcessClassifier
                    gpc = GaussianProcessClassifier(
                        n_restarts_optimizer=int(self.gpc_n_restarts_optimizer.get()),
                        max_iter_predict=int(self.gpc_max_iter_predict.get()),
                        warm_start=self.gpc_warm_start.get(), 
                        copy_X_train=self.gpc_copy_X_train.get(),
                        random_state=(int(self.gpc_random_state.get()) 
                            if (self.gpc_random_state.get() != 'None') else None),
                        multi_class=self.gpc_multi_class.get(), 
                        n_jobs=(int(self.gpc_n_jobs.get()) 
                            if (self.gpc_n_jobs.get() != 'None') else None))
                    if self.x_st_var.get() == 'No':
                        gpc.fit(training_X, training_y)
                        pr_values = gpc.predict(X)
                    else:
                        gpc.fit(training_X_St, training_y)
                        pr_values = gpc.predict(X_St)
                elif method == 'Multi-layer Perceptron':
                    from sklearn.neural_network import MLPClassifier
                    mlpc = MLPClassifier(
                        hidden_layer_sizes=eval(self.mlpc_hidden_layer_sizes.get()),
                        activation=self.mlpc_activation.get(),solver=self.mlpc_solver.get(),
                        alpha=float(self.mlpc_alpha.get()), 
                        batch_size=(int(self.mlpc_batch_size.get()) 
                            if (self.mlpc_batch_size.get() != 'auto') else 'auto'),
                        learning_rate=self.mlpc_learning_rate.get(), 
                        learning_rate_init=float(self.mlpc_learning_rate_init.get()),
                        power_t=float(self.mlpc_power_t.get()), 
                        max_iter=int(self.mlpc_max_iter.get()),
                        shuffle=self.mlpc_shuffle.get(),
                        random_state=(int(self.mlpc_random_state.get()) 
                            if (self.mlpc_random_state.get() != 'None') else None),
                        tol=float(self.mlpc_tol.get()), verbose=self.mlpc_verbose.get(),
                        warm_start=self.mlpc_warm_start.get(), 
                        momentum=float(self.mlpc_momentum.get()),
                        nesterovs_momentum=self.mlpc_nesterovs_momentum.get(),
                        early_stopping=self.mlpc_early_stopping.get(), 
                        validation_fraction=float(self.mlpc_validation_fraction.get()),
                        beta_1=float(self.mlpc_beta_1.get()), beta_2=float(self.mlpc_beta_2.get()),
                        epsilon=float(self.mlpc_epsilon.get()), 
                        n_iter_no_change=int(self.mlpc_n_iter_no_change.get()),
                        max_fun=int(self.mlpc_max_fun.get()))
                    if self.x_st_var.get() == 'No':
                        mlpc.fit(training_X, training_y)
                        pr_values = mlpc.predict(X)
                    else:
                        mlpc.fit(training_X_St, training_y)
                        pr_values = mlpc.predict(X_St)
                elif method == 'XGBoost':
                    from xgboost import XGBClassifier
                    if self.xgbc_use_gpu.get() == False:
                        xgbc = XGBClassifier(
                            learning_rate=float(self.xgbc_eta.get()), 
                            n_estimators=int(self.xgbc_n_estimators.get()), 
                            max_depth=int(self.xgbc_max_depth.get()),
                            min_child_weight=int(self.xgbc_min_child_weight.get()),
                            gamma=float(self.xgbc_gamma.get()), 
                            subsample=float(self.xgbc_subsample.get()),
                            colsample_bytree=float(self.xgbc_colsample_bytree.get()),
                            reg_lambda=float(self.xgbc_lambda.get()), 
                            reg_alpha=float(self.xgbc_alpha.get()),
                            verbosity=2)
                    else:
                        xgbc = XGBClassifier(
                            learning_rate=float(self.xgbc_eta.get()), 
                            n_estimators=int(self.xgbc_n_estimators.get()), 
                            max_depth=int(self.xgbc_max_depth.get()),
                            min_child_weight=int(self.xgbc_min_child_weight.get()),
                            gamma=float(self.xgbc_gamma.get()), 
                            subsample=float(self.xgbc_subsample.get()),
                            colsample_bytree=float(self.xgbc_colsample_bytree.get()),
                            reg_lambda=float(self.xgbc_lambda.get()), 
                            reg_alpha=float(self.xgbc_alpha.get()),
                            verbosity=2,
                            tree_method='gpu_hist', gpu_id=0)
                    if self.x_st_var.get() == 'Yes':
                        xgbc.fit(training_X_St, training_y)
                        pr_values = xgbc.predict(X_St)
                    else:
                        xgbc.fit(training_X, training_y)
                        pr_values = xgbc.predict(X)
                elif method == 'CatBoost':
                    from catboost import CatBoostClassifier
                    if self.cbc_use_gpu.get() == False:
                        cbc = CatBoostClassifier(
                            cat_features=eval(self.cbc_cf_list.get()),
                            iterations=int(self.cbc_iterations.get()),
                            learning_rate=(None if self.cbc_learning_rate.get()=='None' 
                                else float(self.cbc_learning_rate.get())),
                            depth=int(self.cbc_depth.get()), 
                            reg_lambda=(None if self.cbc_reg_lambda.get()=='None' 
                                    else float(self.cbc_reg_lambda.get())),
                            subsample=(None if self.cbc_subsample.get()=='None' 
                                else float(self.cbc_subsample.get())), 
                            colsample_bylevel=float(self.cbc_colsample_bylevel.get()),
                            random_strength=float(self.cbc_random_strength.get()))
                    else:
                        cbc = CatBoostClassifier(
                            cat_features=eval(self.cbc_cf_list.get()),
                            iterations=int(self.cbc_iterations.get()), 
                            task_type="GPU", devices='0:1',
                            learning_rate=(None if self.cbc_learning_rate.get()=='None' 
                                else float(self.cbc_learning_rate.get())),
                            depth=int(self.cbc_depth.get()), 
                            reg_lambda=(None if self.cbc_reg_lambda.get()=='None' 
                                    else float(self.cbc_reg_lambda.get())),
                            subsample=(None if self.cbc_subsample.get()=='None' 
                                else float(self.cbc_subsample.get())), 
                            colsample_bylevel=float(self.cbc_colsample_bylevel.get()),
                            random_strength=float(self.cbc_random_strength.get()))
                    if self.x_st_var.get() == 'Yes':
                        cbc.fit(training_X_St, training_y)
                        pr_values = cbc.predict(X_St)
                    else:
                        cbc.fit(training_X, training_y)
                        pr_values = cbc.predict(X)

                if self.place_result_var.get() == 'Start':
                    self.prediction.data.insert(0, 'Class', pr_values)
                elif self.place_result_var.get() == 'End':
                    self.prediction.data['Class'] = pr_values

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
        def try_clf_predict_class(method):
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
                    self.pb2.place(x=425, y=330)
                    self.pb2.start(10)

                    thread1 = Thread(target=clf_predict_class, args=(self.pr_method.get(),))
                    thread1.daemon = True
                    thread1.start()

                except ValueError as e:
                    self.pb2.destroy()
                    messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))
        
        ttk.Button(self.frame, text='Predict classes', 
            command=lambda: 
                try_clf_predict_class(method=self.pr_method.get())).place(x=420, y=360)
        
        ttk.Button(self.frame, text='Save results', 
            command=lambda: save_results(self, self.prediction, 'clf result')).place(x=420, y=400)
        ttk.Button(self.frame, text='Quit', 
            command=lambda: quit_back(clf_app.root, parent)).place(x=420, y=440)

# sub-app for methods' specifications    
class clf_mtds_specification:
    def __init__(self, prev, parent):
        self.root = tk.Toplevel(parent)

        w = 690
        h = 640
        #setting main window's parameters       
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.focus_force()
        self.root.resizable(False, False)
        self.root.title('Classification methods specification')

        self.canvas = tk.Canvas(self.root)
        self.frame = ttk.Frame(self.canvas, width=1280, height=640)
        self.scrollbar = ttk.Scrollbar(self.canvas, orient="horizontal", 
            command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        def mouse_scroll(event):
            if event.delta:
                self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
            else:
                if event.num == 5:
                    move = 1
                else:
                    move = -1
                self.canvas.xview_scroll(move, "units")
        self.canvas.bind_all("<MouseWheel>", mouse_scroll)
        self.canvas_frame = self.canvas.create_window(0, 0, window=self.frame, anchor="nw")

        main_menu = tk.Menu(self.root)
        self.root.config(menu=main_menu)
        settings_menu = tk.Menu(main_menu, tearoff=False)
        main_menu.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label='Restore Defaults', 
            command=lambda: self.restore_defaults(prev))

        ttk.Label(self.frame, text=' Include in\nComparison', font=myfont_b).place(x=30, y=10)
        ttk.Label(self.frame, text='Decision Tree', font=myfont2).place(x=5, y=60)
        inc_cb1 = ttk.Checkbutton(self.frame, variable=prev.dtc_include_comp, takefocus=False)
        inc_cb1.place(x=120, y=60)
        ttk.Label(self.frame, text='Ridge', font=myfont2).place(x=5, y=80)
        inc_cb2 = ttk.Checkbutton(self.frame, variable=prev.rc_include_comp, takefocus=False)
        inc_cb2.place(x=120, y=80)
        ttk.Label(self.frame, text='Random Forest', font=myfont2).place(x=5, y=100)
        inc_cb3 = ttk.Checkbutton(self.frame, variable=prev.rfc_include_comp, takefocus=False)
        inc_cb3.place(x=120, y=100)
        ttk.Label(self.frame, text='Support Vector', font=myfont2).place(x=5, y=120)
        inc_cb4 = ttk.Checkbutton(self.frame, variable=prev.svc_include_comp, takefocus=False)
        inc_cb4.place(x=120, y=120)
        ttk.Label(self.frame, text='SGD', font=myfont2).place(x=5, y=140)
        inc_cb5 = ttk.Checkbutton(self.frame, variable=prev.sgdc_include_comp, takefocus=False)
        inc_cb5.place(x=120, y=140)
        ttk.Label(self.frame, text='Nearest Neighbor', font=myfont2).place(x=5, y=160)
        inc_cb6 = ttk.Checkbutton(self.frame, variable=prev.knc_include_comp, takefocus=False)
        inc_cb6.place(x=120, y=160)
        ttk.Label(self.frame, text='Gaussian Process', font=myfont2).place(x=5, y=180)
        inc_cb7 = ttk.Checkbutton(self.frame, variable=prev.gpc_include_comp, takefocus=False)
        inc_cb7.place(x=120, y=180)
        ttk.Label(self.frame, text='Multi-layer\nPerceptron', font=myfont2).place(x=5, y=200)
        inc_cb8 = ttk.Checkbutton(self.frame, variable=prev.mlpc_include_comp, takefocus=False)
        inc_cb8.place(x=120, y=210)
        ttk.Label(self.frame, text='XGBoost', font=myfont2).place(x=5, y=240)
        inc_cb9 = ttk.Checkbutton(self.frame, variable=prev.xgbc_include_comp, takefocus=False)
        inc_cb9.place(x=120, y=240)
        ttk.Label(self.frame, text='CatBoost', font=myfont2).place(x=5, y=260)
        inc_cb10 = ttk.Checkbutton(self.frame, variable=prev.cbc_include_comp, takefocus=False)
        inc_cb10.place(x=120, y=260)

        ttk.Label(self.frame, text='Decision Tree', font=myfont_b).place(x=30, y=290)
        ttk.Label(self.frame, text='Criterion', font=myfont2).place(x=5, y=320)
        dt_combobox3 = ttk.Combobox(self.frame, textvariable=prev.dtc_criterion, 
            width=6, values=['gini', 'entropy'])
        dt_combobox3.place(x=110,y=320)
        ttk.Label(self.frame, text='Splitter', font=myfont2).place(x=5, y=340)
        dt_combobox4 = ttk.Combobox(self.frame, textvariable=prev.dtc_splitter, 
            width=6, values=['best', 'random'])
        dt_combobox4.place(x=110,y=340)
        ttk.Label(self.frame, text='Max Depth', font=myfont2).place(x=5, y=360)
        dt_e1 = ttk.Entry(self.frame, textvariable=prev.dtc_max_depth, font=myfont2, width=5)
        dt_e1.place(x=115,y=360)
        ttk.Label(self.frame, text='Min samples split', font=myfont2).place(x=5, y=380)
        dt_e2 = ttk.Entry(self.frame, textvariable=prev.dtc_min_samples_split, 
            font=myfont2, width=5)
        dt_e2.place(x=115,y=380)
        ttk.Label(self.frame, text='Min samples leaf', font=myfont2).place(x=5, y=400)
        dt_e3 = ttk.Entry(self.frame, textvariable=prev.dtc_min_samples_leaf, 
            font=myfont2, width=5)
        dt_e3.place(x=115,y=400)
        ttk.Label(self.frame, text="Min weight\nfraction leaf", font=myfont2).place(x=5, y=420)
        dt_e4 = ttk.Entry(self.frame, textvariable=prev.dtc_min_weight_fraction_leaf, 
            font=myfont2, width=5)
        dt_e4.place(x=115,y=430)
        ttk.Label(self.frame, text='Max features', font=myfont2).place(x=5, y=460)
        dt_e5 = ttk.Entry(self.frame, textvariable=prev.dtc_max_features, font=myfont2, width=5)
        dt_e5.place(x=115,y=460)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=5, y=480)
        dt_e6 = ttk.Entry(self.frame, textvariable=prev.dtc_random_state, font=myfont2, width=5)
        dt_e6.place(x=115,y=480)
        ttk.Label(self.frame, text='Max leaf nodes', font=myfont2).place(x=5, y=500)
        dt_e7 = ttk.Entry(self.frame, textvariable=prev.dtc_max_leaf_nodes, font=myfont2, width=5)
        dt_e7.place(x=115,y=500)
        ttk.Label(self.frame, text="Min impurity\ndecrease", font=myfont2).place(x=5, y=520)
        dt_e8 = ttk.Entry(self.frame, textvariable=prev.dtc_min_impurity_decrease, 
            font=myfont2, width=5)
        dt_e8.place(x=115,y=530)
        ttk.Label(self.frame, text="CCP alpha", font=myfont2).place(x=5, y=560)
        dt_e9 = ttk.Entry(self.frame, textvariable=prev.dtc_ccp_alpha, font=myfont2, width=5)
        dt_e9.place(x=115,y=560)

        ttk.Label(self.frame, text='Random Forest', font=myfont_b).place(x=190, y=10)
        ttk.Label(self.frame, text='Trees number', font=myfont2).place(x=175, y=40)
        rf_e1 = ttk.Entry(self.frame, textvariable=prev.rfc_n_estimators, font=myfont2, width=5)
        rf_e1.place(x=285,y=40)
        ttk.Label(self.frame, text='Criterion', font=myfont2).place(x=175, y=60)
        rf_combobox5 = ttk.Combobox(self.frame, textvariable=prev.rfc_criterion, width=6, 
            values=['gini', 'entropy'])
        rf_combobox5.place(x=275,y=60)
        ttk.Label(self.frame, text='Max Depth', font=myfont2).place(x=175, y=80)
        rf_e2 = ttk.Entry(self.frame, textvariable=prev.rfc_max_depth, font=myfont2, width=5)
        rf_e2.place(x=285,y=80)
        ttk.Label(self.frame, text='Min samples split', font=myfont2).place(x=175, y=100)
        rf_e3 = ttk.Entry(self.frame, 
            textvariable=prev.rfc_min_samples_split, font=myfont2, width=5)
        rf_e3.place(x=285,y=100)
        ttk.Label(self.frame, text='Min samples leaf', font=myfont2).place(x=175, y=120)
        rf_e4 = ttk.Entry(self.frame, 
            textvariable=prev.rfc_min_samples_leaf, font=myfont2, width=5)
        rf_e4.place(x=285,y=120)
        ttk.Label(self.frame, text='Min weight\nfraction leaf', font=myfont2).place(x=175, y=140)
        rf_e5 = ttk.Entry(self.frame, 
            textvariable=prev.rfc_min_weight_fraction_leaf, font=myfont2, width=5)
        rf_e5.place(x=285,y=150)
        ttk.Label(self.frame, text='Max features', font=myfont2).place(x=175, y=180)
        rf_e6 = ttk.Entry(self.frame, textvariable=prev.rfc_max_features, font=myfont2, width=5)
        rf_e6.place(x=285,y=180)
        ttk.Label(self.frame, text='Max leaf nodes', font=myfont2).place(x=175, y=200)
        rf_e7 = ttk.Entry(self.frame, textvariable=prev.rfc_max_leaf_nodes, font=myfont2, width=5)
        rf_e7.place(x=285,y=200)
        ttk.Label(self.frame, text='Min impurity\ndecrease', font=myfont2).place(x=175, y=220)
        rf_e8 = ttk.Entry(self.frame, 
            textvariable=prev.rfc_min_impurity_decrease, font=myfont2, width=5)
        rf_e8.place(x=285,y=230)
        ttk.Label(self.frame, text='Bootstrap', font=myfont2).place(x=175, y=260)
        rf_cb1 = ttk.Checkbutton(self.frame, variable=prev.rfc_bootstrap, takefocus=False)
        rf_cb1.place(x=285,y=260)
        ttk.Label(self.frame, text='oob score', font=myfont2).place(x=175, y=280)
        rf_cb2 = ttk.Checkbutton(self.frame, variable=prev.rfc_oob_score, takefocus=False)
        rf_cb2.place(x=285,y=280)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=175, y=300)
        rf_e9 = ttk.Entry(self.frame, textvariable=prev.rfc_n_jobs, font=myfont2, width=5)
        rf_e9.place(x=285,y=300)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=175, y=320)
        rf_e10 = ttk.Entry(self.frame, textvariable=prev.rfc_random_state, font=myfont2, width=5)
        rf_e10.place(x=285,y=320)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=175, y=340)
        rf_e11 = ttk.Entry(self.frame, textvariable=prev.rfc_verbose, font=myfont2, width=5)
        rf_e11.place(x=285,y=340)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=175, y=360)
        rf_cb3 = ttk.Checkbutton(self.frame, variable=prev.rfc_warm_start, takefocus=False)
        rf_cb3.place(x=285,y=360)
        ttk.Label(self.frame, text='CCP alpha', font=myfont2).place(x=175, y=380)
        rf_e12 = ttk.Entry(self.frame, textvariable=prev.rfc_ccp_alpha, font=myfont2, width=5)
        rf_e12.place(x=285,y=380)
        ttk.Label(self.frame, text='Max samples', font=myfont2).place(x=175, y=400)
        rf_e13 = ttk.Entry(self.frame, textvariable=prev.rfc_max_samples, font=myfont2, width=5)
        rf_e13.place(x=285,y=400)

        ttk.Label(self.frame, text='Support Vector', font=myfont_b).place(x=355, y=10)
        ttk.Label(self.frame, text='C', font=myfont2).place(x=340, y=40)
        sv_e1 = ttk.Entry(self.frame, textvariable=prev.svc_C, font=myfont2, width=5)
        sv_e1.place(x=445,y=40)
        ttk.Label(self.frame, text='Kernel', font=myfont2).place(x=340, y=60)
        sv_combobox1 = ttk.Combobox(self.frame, textvariable=prev.svc_kernel, width=6, 
            values=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
        sv_combobox1.place(x=435,y=60)
        ttk.Label(self.frame, text='Degree', font=myfont2).place(x=340, y=80)
        sv_e2 = ttk.Entry(self.frame, textvariable=prev.svc_degree, font=myfont2, width=5)
        sv_e2.place(x=445,y=80)
        ttk.Label(self.frame, text='Gamma', font=myfont2).place(x=340, y=100)
        sv_e3 = ttk.Entry(self.frame, textvariable=prev.svc_gamma, font=myfont2, width=5)
        sv_e3.place(x=445,y=100)
        ttk.Label(self.frame, text='coef0', font=myfont2).place(x=340, y=120)
        sv_e4 = ttk.Entry(self.frame, textvariable=prev.svc_coef0, font=myfont2, width=5)
        sv_e4.place(x=445,y=120)
        ttk.Label(self.frame, text='shrinking', font=myfont2).place(x=340, y=140)
        sv_cb1 = ttk.Checkbutton(self.frame, variable=prev.svc_shrinking, takefocus=False)
        sv_cb1.place(x=445,y=140)
        ttk.Label(self.frame, text='probability', font=myfont2).place(x=340, y=160)
        sv_cb2 = ttk.Checkbutton(self.frame, variable=prev.svc_probability, takefocus=False)
        sv_cb2.place(x=445,y=160)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=340, y=180)
        sv_e5 = ttk.Entry(self.frame, textvariable=prev.svc_tol, font=myfont2, width=5)
        sv_e5.place(x=445,y=180)
        ttk.Label(self.frame, text='Cache size', font=myfont2).place(x=340, y=200)
        sv_e6 = ttk.Entry(self.frame, textvariable=prev.svc_cache_size, font=myfont2, width=5)
        sv_e6.place(x=445,y=200)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=340, y=220)
        sv_cb3 = ttk.Checkbutton(self.frame, variable=prev.svc_verbose, takefocus=False)
        sv_cb3.place(x=445,y=220)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=340, y=240)
        sv_e7 = ttk.Entry(self.frame, textvariable=prev.svc_max_iter, font=myfont2, width=5)
        sv_e7.place(x=445,y=240)
        ttk.Label(self.frame, text='Decision\nfunction shape', font=myfont2).place(x=340, y=260)
        sv_combobox2 = ttk.Combobox(self.frame, 
            textvariable=prev.svc_decision_function_shape, width=5, values=['ovo', 'ovr'])
        sv_combobox2.place(x=435,y=270)
        ttk.Label(self.frame, text='Break ties', font=myfont2).place(x=340, y=300)
        sv_cb4 = ttk.Checkbutton(self.frame, variable=prev.svc_break_ties, takefocus=False)
        sv_cb4.place(x=445,y=300)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=340, y=320)
        sv_e8 = ttk.Entry(self.frame, textvariable=prev.svc_random_state, font=myfont2, width=5)
        sv_e8.place(x=445,y=320)

        ttk.Label(self.frame, text='Ridge', font=myfont_b).place(x=380, y=350)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=345, y=380)
        r_e1 = ttk.Entry(self.frame, textvariable=prev.rc_alpha, font=myfont2, width=5)
        r_e1.place(x=445,y=380)
        ttk.Label(self.frame, text='fit intercept', font=myfont2).place(x=345, y=400)
        r_cb2 = ttk.Checkbutton(self.frame, variable=prev.rc_fit_intercept, takefocus=False)
        r_cb2.place(x=445, y=400)
        ttk.Label(self.frame, text='Normalize', font=myfont2).place(x=345, y=420)
        r_cb3 = ttk.Checkbutton(self.frame, variable=prev.rc_normalize, takefocus=False)
        r_cb3.place(x=445, y=420)
        ttk.Label(self.frame, text='Copy X', font=myfont2).place(x=345, y=440)
        r_cb4 = ttk.Checkbutton(self.frame, variable=prev.rc_copy_X, takefocus=False)
        r_cb4.place(x=445, y=440)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=345, y=460)
        r_e2 = ttk.Entry(self.frame, textvariable=prev.rc_max_iter, font=myfont2, width=5)
        r_e2.place(x=445,y=460)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=345, y=480)
        r_e3 = ttk.Entry(self.frame, textvariable=prev.rc_tol, font=myfont2, width=5)
        r_e3.place(x=445,y=480)
        ttk.Label(self.frame, text='Solver', font=myfont2).place(x=345, y=500)
        r_combobox1 = ttk.Combobox(self.frame, textvariable=prev.rc_solver, width=6, 
            values=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        r_combobox1.place(x=440,y=500)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=345, y=520)
        r_e4 = ttk.Entry(self.frame, textvariable=prev.rc_random_state, font=myfont2, width=5)
        r_e4.place(x=445,y=520)

        ttk.Label(self.frame, text='SGD', font=myfont_b).place(x=535, y=10)
        ttk.Label(self.frame, text='Loss', font=myfont2).place(x=505, y=40)
        sgd_combobox1 = ttk.Combobox(self.frame, textvariable=prev.sgdc_loss, width=10, 
            values=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
                'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])
        sgd_combobox1.place(x=602,y=40)
        ttk.Label(self.frame, text='Penalty', font=myfont2).place(x=505, y=60)
        sgd_combobox2 = ttk.Combobox(self.frame, textvariable=prev.sgdc_penalty, 
            width=7, values=['l2', 'l1', 'elasticnet'])
        sgd_combobox2.place(x=610,y=60)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=505, y=80)
        sgd_e1 = ttk.Entry(self.frame, textvariable=prev.sgdc_alpha, font=myfont2, width=7)
        sgd_e1.place(x=615,y=80)
        ttk.Label(self.frame, text='l1 ratio', font=myfont2).place(x=505, y=100)
        sgd_e2 = ttk.Entry(self.frame, textvariable=prev.sgdc_l1_ratio, font=myfont2, width=7)
        sgd_e2.place(x=615,y=100)
        ttk.Label(self.frame, text='fit intercept', font=myfont2).place(x=505, y=120)
        sgd_cb1 = ttk.Checkbutton(self.frame, variable=prev.sgdc_fit_intercept, takefocus=False)
        sgd_cb1.place(x=615,y=120)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=505, y=140)
        sgd_e3 = ttk.Entry(self.frame, textvariable=prev.sgdc_max_iter, font=myfont2, width=7)
        sgd_e3.place(x=615,y=140)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=505, y=160)
        sgd_e4 = ttk.Entry(self.frame, textvariable=prev.sgdc_tol, font=myfont2, width=7)
        sgd_e4.place(x=615,y=160)
        ttk.Label(self.frame, text='Shuffle', font=myfont2).place(x=505, y=180)
        sgd_cb2 = ttk.Checkbutton(self.frame, variable=prev.sgdc_shuffle, takefocus=False)
        sgd_cb2.place(x=615,y=180)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=505, y=200)
        sgd_e5 = ttk.Entry(self.frame, textvariable=prev.sgdc_verbose, font=myfont2, width=7)
        sgd_e5.place(x=615,y=200)
        ttk.Label(self.frame, text='Epsilon', font=myfont2).place(x=505, y=220)
        sgd_e6 = ttk.Entry(self.frame, textvariable=prev.sgdc_epsilon, font=myfont2, width=7)
        sgd_e6.place(x=615,y=220)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=505, y=240)
        sgd_e7 = ttk.Entry(self.frame, textvariable=prev.sgdc_n_jobs, font=myfont2, width=7)
        sgd_e7.place(x=615,y=240)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=505, y=260)
        sgd_e8 = ttk.Entry(self.frame, textvariable=prev.sgdc_random_state, font=myfont2, width=7)
        sgd_e8.place(x=615,y=260)
        ttk.Label(self.frame, text='Learning rate', font=myfont2).place(x=505, y=280)
        sgd_combobox3 = ttk.Combobox(self.frame, textvariable=prev.sgdc_learning_rate, 
            width=7, values=['constant', 'optimal', 'invscaling', 'adaptive'])
        sgd_combobox3.place(x=610,y=280)
        ttk.Label(self.frame, text='eta0', font=myfont2).place(x=505, y=300)
        sgd_e9 = ttk.Entry(self.frame, textvariable=prev.sgdc_eta0, font=myfont2, width=7)
        sgd_e9.place(x=615,y=300)
        ttk.Label(self.frame, text='power t', font=myfont2).place(x=505, y=320)
        sgd_e10 = ttk.Entry(self.frame, textvariable=prev.sgdc_power_t, font=myfont2, width=7)
        sgd_e10.place(x=615,y=320)
        ttk.Label(self.frame, text='Early stopping', font=myfont2).place(x=505, y=340)
        sgd_cb3 = ttk.Checkbutton(self.frame, variable=prev.sgdc_early_stopping, takefocus=False)
        sgd_cb3.place(x=615,y=340)
        ttk.Label(self.frame, text='Validation fraction', font=myfont2).place(x=505, y=360)
        sgd_e11 = ttk.Entry(self.frame, 
            textvariable=prev.sgdc_validation_fraction, font=myfont2, width=7)
        sgd_e11.place(x=615,y=360)
        ttk.Label(self.frame, text='n iter no change', font=myfont2).place(x=505, y=380)
        sgd_e12 = ttk.Entry(self.frame, textvariable=prev.sgdc_n_iter_no_change, 
            font=myfont2, width=7)
        sgd_e12.place(x=615,y=380)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=505, y=400)
        sgd_cb4 = ttk.Checkbutton(self.frame, variable=prev.sgdc_warm_start, takefocus=False)
        sgd_cb4.place(x=615,y=400)
        ttk.Label(self.frame, text='Average', font=myfont2).place(x=505, y=420)
        sgd_e13 = ttk.Entry(self.frame, textvariable=prev.sgdc_average, font=myfont2, width=7)
        sgd_e13.place(x=615,y=420)

        ttk.Label(self.frame, text='Gaussian Process', font=myfont_b).place(x=705, y=10)
        ttk.Label(self.frame, text='n restarts\noptimizer', font=myfont2).place(x=695, y=40)
        gp_e1 = ttk.Entry(self.frame, textvariable=prev.gpc_n_restarts_optimizer, 
            font=myfont2, width=7)
        gp_e1.place(x=795,y=50)
        ttk.Label(self.frame, text='max iter predict', font=myfont2).place(x=695, y=80)
        gp_e2 = ttk.Entry(self.frame, 
            textvariable=prev.gpc_max_iter_predict, font=myfont2, width=7)
        gp_e2.place(x=795,y=80)
        ttk.Label(self.frame, text='Warm predict', font=myfont2).place(x=695, y=100)
        gp_cb1 = ttk.Checkbutton(self.frame, variable=prev.gpc_warm_start, takefocus=False)
        gp_cb1.place(x=795,y=100)
        ttk.Label(self.frame, text='Copy X train', font=myfont2).place(x=695, y=120)
        gp_cb2 = ttk.Checkbutton(self.frame, variable=prev.gpc_copy_X_train, takefocus=False)
        gp_cb2.place(x=795,y=120)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=695, y=140)
        gp_e3 = ttk.Entry(self.frame, textvariable=prev.gpc_random_state, font=myfont2, width=7)
        gp_e3.place(x=795,y=140)
        ttk.Label(self.frame, text='Multi class', font=myfont2).place(x=695, y=160)
        gp_combobox1 = ttk.Combobox(self.frame, textvariable=prev.gpc_multi_class, width=9, 
            values=['one_vs_rest', 'one_vs_one'])
        gp_combobox1.place(x=785,y=160)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=695, y=180)
        gp_e4 = ttk.Entry(self.frame, textvariable=prev.gpc_n_jobs, font=myfont2, width=7)
        gp_e4.place(x=795,y=180)

        ttk.Label(self.frame, text='Nearest Neighbor', font=myfont_b).place(x=705, y=210)
        ttk.Label(self.frame, text='n neighbors', font=myfont2).place(x=695, y=240)
        kn_e1 = ttk.Entry(self.frame, textvariable=prev.knc_n_neighbors, font=myfont2, width=7)
        kn_e1.place(x=795,y=240)
        ttk.Label(self.frame, text='Weights', font=myfont2).place(x=695, y=260)
        gp_combobox1 = ttk.Combobox(self.frame, textvariable=prev.knc_weights, width=7, 
            values=['uniform', 'distance'])
        gp_combobox1.place(x=795,y=260)
        ttk.Label(self.frame, text='Algorithm', font=myfont2).place(x=695, y=280)
        gp_combobox2 = ttk.Combobox(self.frame, textvariable=prev.knc_algorithm, width=7, 
            values=['auto', 'ball_tree', 'kd_tree', 'brute'])
        gp_combobox2.place(x=795,y=280)
        ttk.Label(self.frame, text='Leaf size', font=myfont2).place(x=695, y=300)
        kn_e2 = ttk.Entry(self.frame, textvariable=prev.knc_leaf_size, font=myfont2, width=7)
        kn_e2.place(x=795,y=300)
        ttk.Label(self.frame, text='p', font=myfont2).place(x=695, y=320)
        kn_e3 = ttk.Entry(self.frame, textvariable=prev.knc_p, font=myfont2, width=7)
        kn_e3.place(x=795,y=320)
        ttk.Label(self.frame, text='Metric', font=myfont2).place(x=695, y=340)
        gp_combobox3 = ttk.Combobox(self.frame, textvariable=prev.knc_metric, width=9, 
            values=['euclidean', 'manhattan', 'chebyshev', 'minkowski',
                'wminkowski', 'seuclidean', 'mahalanobis'])
        gp_combobox3.place(x=785,y=340)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=695, y=360)
        kn_e4 = ttk.Entry(self.frame, textvariable=prev.knc_n_jobs, font=myfont2, width=7)
        kn_e4.place(x=795,y=360)

        ttk.Label(self.frame, text='Multi-layer Perceptron', font=myfont_b).place(x=875, y=10)
        ttk.Label(self.frame, text='hidden layer\nsizes', font=myfont2).place(x=875, y=40)
        mlp_e1 = ttk.Entry(self.frame, 
            textvariable=prev.mlpc_hidden_layer_sizes, font=myfont2, width=7)
        mlp_e1.place(x=985,y=50)
        ttk.Label(self.frame, text='Activation', font=myfont2).place(x=875, y=80)
        mlp_combobox1 = ttk.Combobox(self.frame, textvariable=prev.mlpc_activation, width=7, 
            values=['identity', 'logistic', 'tanh', 'relu'])
        mlp_combobox1.place(x=985,y=80)
        ttk.Label(self.frame, text='Solver', font=myfont2).place(x=875, y=100)
        mlp_combobox2 = ttk.Combobox(self.frame, textvariable=prev.mlpc_solver, width=7, 
            values=['lbfgs', 'sgd', 'adam'])
        mlp_combobox2.place(x=985,y=100)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=875, y=120)
        mlp_e2 = ttk.Entry(self.frame, textvariable=prev.mlpc_alpha, font=myfont2, width=7)
        mlp_e2.place(x=985,y=120)
        ttk.Label(self.frame, text='Batch size', font=myfont2).place(x=875, y=140)
        mlp_e3 = ttk.Entry(self.frame, textvariable=prev.mlpc_batch_size, font=myfont2, width=7)
        mlp_e3.place(x=985,y=140)
        ttk.Label(self.frame, text='Learning rate', font=myfont2).place(x=875, y=160)
        mlp_combobox3 = ttk.Combobox(self.frame, textvariable=prev.mlpc_learning_rate, width=7, 
            values=['constant', 'invscaling', 'adaptive'])
        mlp_combobox3.place(x=985,y=160)
        ttk.Label(self.frame, text='Learning\nrate init', font=myfont2).place(x=875, y=180)
        mlp_e4 = ttk.Entry(self.frame, 
            textvariable=prev.mlpc_learning_rate_init, font=myfont2, width=7)
        mlp_e4.place(x=985,y=190)
        ttk.Label(self.frame, text='Power t', font=myfont2).place(x=875, y=220)
        mlp_e5 = ttk.Entry(self.frame, textvariable=prev.mlpc_power_t, font=myfont2, width=7)
        mlp_e5.place(x=985,y=220)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=875, y=240)
        mlp_e6 = ttk.Entry(self.frame, textvariable=prev.mlpc_max_iter, font=myfont2, width=7)
        mlp_e6.place(x=985,y=240)
        ttk.Label(self.frame, text='Shuffle', font=myfont2).place(x=875, y=260)
        mlp_cb2 = ttk.Checkbutton(self.frame, variable=prev.mlpc_shuffle, takefocus=False)
        mlp_cb2.place(x=985, y=260)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=875, y=280)
        mlp_e7 = ttk.Entry(self.frame, textvariable=prev.mlpc_random_state, font=myfont2, width=7)
        mlp_e7.place(x=985,y=280)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=875, y=300)
        mlp_e8 = ttk.Entry(self.frame, textvariable=prev.mlpc_tol, font=myfont2, width=7)
        mlp_e8.place(x=985,y=300)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=875, y=320)
        mlp_cb3 = ttk.Checkbutton(self.frame, variable=prev.mlpc_verbose, takefocus=False)
        mlp_cb3.place(x=985, y=320)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=875, y=340)
        mlp_cb4 = ttk.Checkbutton(self.frame, variable=prev.mlpc_warm_start, takefocus=False)
        mlp_cb4.place(x=985, y=340)
        ttk.Label(self.frame, text='Momentum', font=myfont2).place(x=875, y=360)
        mlp_e9 = ttk.Entry(self.frame, textvariable=prev.mlpc_momentum, font=myfont2, width=7)
        mlp_e9.place(x=985,y=360)
        ttk.Label(self.frame, text='Nesterovs\nmomentum', font=myfont2).place(x=875, y=380)
        mlp_cb5 = ttk.Checkbutton(self.frame, 
            variable=prev.mlpc_nesterovs_momentum, takefocus=False)
        mlp_cb5.place(x=985, y=390)
        ttk.Label(self.frame, text='Early stopping', font=myfont2).place(x=875, y=420)
        mlp_cb6 = ttk.Checkbutton(self.frame, variable=prev.mlpc_early_stopping, takefocus=False)
        mlp_cb6.place(x=985, y=420)
        ttk.Label(self.frame, text='Validation fraction', font=myfont2).place(x=875, y=440)
        mlp_e10 = ttk.Entry(self.frame, 
            textvariable=prev.mlpc_validation_fraction, font=myfont2, width=7)
        mlp_e10.place(x=985,y=440)
        ttk.Label(self.frame, text='Beta 1', font=myfont2).place(x=875, y=460)
        mlp_e11 = ttk.Entry(self.frame, textvariable=prev.mlpc_beta_1, font=myfont2, width=7)
        mlp_e11.place(x=985,y=460)
        ttk.Label(self.frame, text='Beta 2', font=myfont2).place(x=875, y=480)
        mlp_e12 = ttk.Entry(self.frame, textvariable=prev.mlpc_beta_2, font=myfont2, width=7)
        mlp_e12.place(x=985,y=480)
        ttk.Label(self.frame, text='Epsilon', font=myfont2).place(x=875, y=500)
        mlp_e13 = ttk.Entry(self.frame, textvariable=prev.mlpc_epsilon, font=myfont2, width=7)
        mlp_e13.place(x=985,y=500)
        ttk.Label(self.frame, text='n iter no change', font=myfont2).place(x=875, y=520)
        mlp_e14 = ttk.Entry(self.frame, 
            textvariable=prev.mlpc_n_iter_no_change, font=myfont2, width=7)
        mlp_e14.place(x=985,y=520)
        ttk.Label(self.frame, text='Max fun', font=myfont2).place(x=875, y=540)
        mlp_e15 = ttk.Entry(self.frame, textvariable=prev.mlpc_max_fun, font=myfont2, width=7)
        mlp_e15.place(x=985,y=540)

        ttk.Label(self.frame, text='XGBoost', font=myfont_b).place(x=1085, y=10)
        ttk.Label(self.frame, text='Learning rate', font=myfont2).place(x=1055, y=40)
        xgb_e1 = ttk.Entry(self.frame, textvariable=prev.xgbc_eta, font=myfont2, width=7)
        xgb_e1.place(x=1165,y=40)
        ttk.Label(self.frame, text='Min Child Weight', font=myfont2).place(x=1055, y=60)
        xgb_e2 = ttk.Entry(self.frame, 
            textvariable=prev.xgbc_min_child_weight, font=myfont2, width=7)
        xgb_e2.place(x=1165,y=60)
        ttk.Label(self.frame, text='Max Depth', font=myfont2).place(x=1055, y=80)
        xgb_e3 = ttk.Entry(self.frame, textvariable=prev.xgbc_max_depth, font=myfont2, width=7)
        xgb_e3.place(x=1165,y=80)
        ttk.Label(self.frame, text='Gamma', font=myfont2).place(x=1055, y=100)
        xgb_e4 = ttk.Entry(self.frame, textvariable=prev.xgbc_gamma, font=myfont2, width=7)
        xgb_e4.place(x=1165,y=100)
        ttk.Label(self.frame, text='Subsample', font=myfont2).place(x=1055, y=120)
        xgb_e5 = ttk.Entry(self.frame, textvariable=prev.xgbc_subsample, font=myfont2, width=7)
        xgb_e5.place(x=1165,y=120)
        ttk.Label(self.frame, text='Colsample bytree', font=myfont2).place(x=1055, y=140)
        xgb_e6 = ttk.Entry(self.frame, 
            textvariable=prev.xgbc_colsample_bytree, font=myfont2, width=7)
        xgb_e6.place(x=1165,y=140)
        ttk.Label(self.frame, text='Lambda', font=myfont2).place(x=1055, y=160)
        xgb_e7 = ttk.Entry(self.frame, textvariable=prev.xgbc_lambda, font=myfont2, width=7)
        xgb_e7.place(x=1165,y=160)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=1055, y=180)
        xgb_e8 = ttk.Entry(self.frame, textvariable=prev.xgbc_alpha, font=myfont2, width=7)
        xgb_e8.place(x=1165,y=180)
        ttk.Label(self.frame, text='n estimators', font=myfont2).place(x=1055, y=200)
        xgb_e9 = ttk.Entry(self.frame, textvariable=prev.xgbc_n_estimators, font=myfont2, width=7)
        xgb_e9.place(x=1165,y=200)
        ttk.Label(self.frame, text='Use GPU', font=myfont2).place(x=1055, y=220)
        xgb_cb2 = ttk.Checkbutton(self.frame, variable=prev.xgbc_use_gpu, takefocus=False)
        xgb_cb2.place(x=1165, y=220)

        ttk.Label(self.frame, text='CatBoost', font=myfont_b).place(x=1085, y=250)
        ttk.Label(self.frame, text='Iterations', font=myfont2).place(x=1055, y=280)
        cb_e1 = ttk.Entry(self.frame, textvariable=prev.cbc_iterations, font=myfont2, width=7)
        cb_e1.place(x=1165,y=280)
        ttk.Label(self.frame, text='Learning rate', font=myfont2).place(x=1055, y=300)
        cb_e2 = ttk.Entry(self.frame, textvariable=prev.cbc_learning_rate, font=myfont2, width=7)
        cb_e2.place(x=1165,y=300)
        ttk.Label(self.frame, text='Depth', font=myfont2).place(x=1055, y=320)
        cb_e3 = ttk.Entry(self.frame, textvariable=prev.cbc_depth, font=myfont2, width=7)
        cb_e3.place(x=1165,y=320)
        ttk.Label(self.frame, text='Lambda', font=myfont2).place(x=1055, y=340)
        cb_e4 = ttk.Entry(self.frame, textvariable=prev.cbc_reg_lambda, font=myfont2, width=7)
        cb_e4.place(x=1165,y=340)
        ttk.Label(self.frame, text='Subsample', font=myfont2).place(x=1055, y=360)
        cb_e5 = ttk.Entry(self.frame, textvariable=prev.cbc_subsample, font=myfont2, width=7)
        cb_e5.place(x=1165,y=360)
        ttk.Label(self.frame, text='Colsample bylevel', font=myfont2).place(x=1055, y=380)
        cb_e6 = ttk.Entry(self.frame, 
            textvariable=prev.cbc_colsample_bylevel, font=myfont2, width=7)
        cb_e6.place(x=1165,y=380)
        ttk.Label(self.frame, text='Random strength', font=myfont2).place(x=1055, y=400)
        cb_e7 = ttk.Entry(self.frame, 
            textvariable=prev.cbc_random_strength, font=myfont2, width=7)
        cb_e7.place(x=1165,y=400)
        ttk.Label(self.frame, text='Use GPU', font=myfont2).place(x=1055, y=420)
        cb_cb2 = ttk.Checkbutton(self.frame, variable=prev.cbc_use_gpu, takefocus=False)
        cb_cb2.place(x=1165, y=420)
        ttk.Label(self.frame, text='cat_features list', font=myfont2).place(x=1055, y=440)
        cb_e8 = ttk.Entry(self.frame, textvariable=prev.cbc_cf_list, font=myfont2, width=10)
        cb_e8.place(x=1160,y=440)

        ttk.Button(self.root, text='OK', 
            command=lambda: quit_back(self.root, clf_app.root)).place(relx=0.85, rely=0.92)

        self.root.lift()

    # function to restore default parameters
    def restore_defaults(self, prev):
        #methods parameters
        prev.rc_include_comp = tk.BooleanVar(value=True)
        prev.rc_alpha = tk.StringVar(value='1.0')
        prev.rc_fit_intercept = tk.BooleanVar(value=True)
        prev.rc_normalize = tk.BooleanVar(value=False)
        prev.rc_copy_X = tk.BooleanVar(value=True)
        prev.rc_max_iter = tk.StringVar(value='None')
        prev.rc_tol = tk.StringVar(value='1e-3')
        prev.rc_solver = tk.StringVar(value='auto')
        prev.rc_random_state = tk.StringVar(value='None')
        
        prev.dtc_include_comp = tk.BooleanVar(value=True)
        prev.dtc_criterion = tk.StringVar(value='gini')
        prev.dtc_splitter = tk.StringVar(value='best')
        prev.dtc_max_depth = tk.StringVar(value='None')
        prev.dtc_min_samples_split = tk.StringVar(value='2')
        prev.dtc_min_samples_leaf = tk.StringVar(value='1')
        prev.dtc_min_weight_fraction_leaf = tk.StringVar(value='0.0')
        prev.dtc_max_features = tk.StringVar(value='None')
        prev.dtc_random_state = tk.StringVar(value='None')
        prev.dtc_max_leaf_nodes = tk.StringVar(value='None')
        prev.dtc_min_impurity_decrease = tk.StringVar(value='0.0')
        prev.dtc_ccp_alpha = tk.StringVar(value='0.0')
        
        prev.rfc_include_comp = tk.BooleanVar(value=True)
        prev.rfc_n_estimators = tk.StringVar(value='100')
        prev.rfc_criterion = tk.StringVar(value='gini')
        prev.rfc_max_depth = tk.StringVar(value='None')
        prev.rfc_min_samples_split = tk.StringVar(value='2')
        prev.rfc_min_samples_leaf = tk.StringVar(value='1')
        prev.rfc_min_weight_fraction_leaf = tk.StringVar(value='0.0')
        prev.rfc_max_features = tk.StringVar(value='auto')
        prev.rfc_max_leaf_nodes = tk.StringVar(value='None')
        prev.rfc_min_impurity_decrease = tk.StringVar(value='0.0')
        prev.rfc_bootstrap = tk.BooleanVar(value=True)
        prev.rfc_oob_score = tk.BooleanVar(value=False)
        prev.rfc_n_jobs = tk.StringVar(value='None')
        prev.rfc_random_state = tk.StringVar(value='None')
        prev.rfc_verbose = tk.StringVar(value='0')
        prev.rfc_warm_start = tk.BooleanVar(value=False)
        prev.rfc_ccp_alpha = tk.StringVar(value='0.0')
        prev.rfc_max_samples = tk.StringVar(value='None')
        
        prev.svc_include_comp = tk.BooleanVar(value=True)
        prev.svc_C = tk.StringVar(value='1.0')
        prev.svc_kernel = tk.StringVar(value='rbf')
        prev.svc_degree = tk.StringVar(value='3')
        prev.svc_gamma = tk.StringVar(value='scale')
        prev.svc_coef0 = tk.StringVar(value='0.0')
        prev.svc_shrinking = tk.BooleanVar(value=True)
        prev.svc_probability = tk.BooleanVar(value=False)
        prev.svc_tol = tk.StringVar(value='1e-3')
        prev.svc_cache_size = tk.StringVar(value='200')
        prev.svc_verbose = tk.BooleanVar(value=False)
        prev.svc_max_iter = tk.StringVar(value='-1')
        prev.svc_decision_function_shape = tk.StringVar(value='ovr')
        prev.svc_break_ties = tk.BooleanVar(value=False)
        prev.svc_random_state = tk.StringVar(value='None')
        
        prev.sgdc_include_comp = tk.BooleanVar(value=True)
        prev.sgdc_loss = tk.StringVar(value='hinge')
        prev.sgdc_penalty = tk.StringVar(value='l2')
        prev.sgdc_alpha = tk.StringVar(value='0.0001')
        prev.sgdc_l1_ratio = tk.StringVar(value='0.15')
        prev.sgdc_fit_intercept = tk.BooleanVar(value=True)
        prev.sgdc_max_iter = tk.StringVar(value='1000')
        prev.sgdc_tol = tk.StringVar(value='1e-3')
        prev.sgdc_shuffle = tk.BooleanVar(value=True)
        prev.sgdc_verbose = tk.StringVar(value='0')
        prev.sgdc_epsilon = tk.StringVar(value='0.1')
        prev.sgdc_n_jobs = tk.StringVar(value='None')
        prev.sgdc_random_state = tk.StringVar(value='None')
        prev.sgdc_learning_rate = tk.StringVar(value='optimal')
        prev.sgdc_eta0 = tk.StringVar(value='0.0')
        prev.sgdc_power_t = tk.StringVar(value='0.5')
        prev.sgdc_early_stopping = tk.BooleanVar(value=False)
        prev.sgdc_validation_fraction = tk.StringVar(value='0.1')
        prev.sgdc_n_iter_no_change = tk.StringVar(value='5')
        prev.sgdc_warm_start = tk.BooleanVar(value=False)
        prev.sgdc_average = tk.StringVar(value='False')
        
        prev.gpc_include_comp = tk.BooleanVar(value=True)
        prev.gpc_n_restarts_optimizer = tk.StringVar(value='0')
        prev.gpc_max_iter_predict = tk.StringVar(value='100')
        prev.gpc_warm_start = tk.BooleanVar(value=False)
        prev.gpc_copy_X_train = tk.BooleanVar(value=True)
        prev.gpc_random_state = tk.StringVar(value='None')
        prev.gpc_multi_class = tk.StringVar(value='one_vs_rest')
        prev.gpc_n_jobs = tk.StringVar(value='None')
        
        prev.knc_include_comp = tk.BooleanVar(value=True)
        prev.knc_n_neighbors = tk.StringVar(value='5')
        prev.knc_weights = tk.StringVar(value='uniform')
        prev.knc_algorithm = tk.StringVar(value='auto')
        prev.knc_leaf_size = tk.StringVar(value='30')
        prev.knc_p = tk.StringVar(value='2')
        prev.knc_metric = tk.StringVar(value='minkowski')
        prev.knc_n_jobs = tk.StringVar(value='None')
        
        prev.mlpc_include_comp = tk.BooleanVar(value=True)
        prev.mlpc_hidden_layer_sizes = tk.StringVar(value='100')
        prev.mlpc_activation = tk.StringVar(value='relu')
        prev.mlpc_solver = tk.StringVar(value='adam')
        prev.mlpc_alpha = tk.StringVar(value='0.0001')
        prev.mlpc_batch_size = tk.StringVar(value='auto')
        prev.mlpc_learning_rate = tk.StringVar(value='constant')
        prev.mlpc_learning_rate_init = tk.StringVar(value='0.001')
        prev.mlpc_power_t = tk.StringVar(value='0.5')
        prev.mlpc_max_iter = tk.StringVar(value='200')
        prev.mlpc_shuffle = tk.BooleanVar(value=True)
        prev.mlpc_random_state = tk.StringVar(value='None')
        prev.mlpc_tol = tk.StringVar(value='1e-4')
        prev.mlpc_verbose = tk.BooleanVar(value=False)
        prev.mlpc_warm_start = tk.BooleanVar(value=False)
        prev.mlpc_momentum = tk.StringVar(value='0.9')
        prev.mlpc_nesterovs_momentum = tk.BooleanVar(value=True)
        prev.mlpc_early_stopping = tk.BooleanVar(value=False)
        prev.mlpc_validation_fraction = tk.StringVar(value='0.1')
        prev.mlpc_beta_1 = tk.StringVar(value='0.9')
        prev.mlpc_beta_2 = tk.StringVar(value='0.999')
        prev.mlpc_epsilon = tk.StringVar(value='1e-8')
        prev.mlpc_n_iter_no_change = tk.StringVar(value='10')
        prev.mlpc_max_fun = tk.StringVar(value='15000')

        prev.xgbc_include_comp = tk.BooleanVar(value=True)
        prev.xgbc_n_estimators = tk.StringVar(value='1000')
        prev.xgbc_eta = tk.StringVar(value='0.3')
        prev.xgbc_min_child_weight = tk.StringVar(value='1')
        prev.xgbc_max_depth = tk.StringVar(value='6')
        prev.xgbc_gamma = tk.StringVar(value='0')
        prev.xgbc_subsample = tk.StringVar(value='1.0')
        prev.xgbc_colsample_bytree = tk.StringVar(value='1.0')
        prev.xgbc_lambda = tk.StringVar(value='1.0')
        prev.xgbc_alpha = tk.StringVar(value='0.0')
        prev.xgbc_use_gpu = tk.BooleanVar(value=False)

        prev.cbc_include_comp = tk.BooleanVar(value=False)
        prev.cbc_iterations = tk.StringVar(value='1000')
        prev.cbc_learning_rate = tk.StringVar(value='None')
        prev.cbc_depth = tk.StringVar(value='6')
        prev.cbc_reg_lambda = tk.StringVar(value='3.0')
        prev.cbc_subsample = tk.StringVar(value='None')
        prev.cbc_colsample_bylevel = tk.StringVar(value='1.0')
        prev.cbc_random_strength = tk.StringVar(value='1.0')
        prev.cbc_use_gpu = tk.BooleanVar(value=False)
        prev.cbc_cf_list = tk.StringVar(value='[]')

        quit_back(self.root, prev.root)