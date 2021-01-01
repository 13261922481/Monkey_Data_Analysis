# Imports
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
import sys
import numpy as np
import webbrowser
import pandas as pd
import sklearn
from monkey_pt import Table

# getting the right path to images for frozen version
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.abspath('')

master = ThemedTk(theme=open('settings\style.txt', 'r').read(), fonts=True)

myfont = (None, 13)
myfont_b = (None, 13, 'bold')
myfont1 = (None, 11)
myfont1_b = (None, 11, 'bold')
myfont2 = (None, 10)
myfont2_b = (None, 10, 'bold')

ws = master.winfo_screenwidth() # width of the screen
hs = master.winfo_screenheight() # height of the screen

style = ttk.Style()
#style.theme_use('xpnative')
style.configure('.', font=myfont)

big_themes = ['blue', 'clam', 'kroc', 'radiance', 'smog', 'ubuntu']

#start window class
class start_window:
    def __init__(self):
        # Setting geometrical things
        w = 500
        h = 400
        
        sw_frame =ttk.Frame(master, width=w, height=h)
        sw_frame.place(x=0, y=0)
        
        # placing funny monkey
        img_name1 = 'imgs\ML\monkey1.png'
        img_path1 = os.path.join(application_path, img_name1)
        img1 = tk.PhotoImage(file=img_path1)
        panel = tk.Label(sw_frame, image = img1)
        panel.place(relx=0.04, rely=0.15)
        
#         ttk.Label(sw_frame, text="Choose ML problem").place(relx=0.55, rely=0.15)
        ttk.Button(sw_frame, text='Classification', command=lambda: cl_app(master)).place(relx=0.6, rely=0.25)
        ttk.Button(sw_frame, text='  Regression ', command=lambda: rgr_app(master)).place(relx=0.6, rely=0.4)
        ttk.Button(sw_frame, text='   Clustering  ', command=None).place(relx=0.6, rely=0.55)
        #setting main window's parameters       
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)  - 30
        master.geometry('%dx%d+%d+%d' % (w, h, x, y))
        master.title('Monkey Data Analysis')
        master.lift()
        #master.grab_set()
        master.focus_force()
        master.resizable(False, False)
        
        #setting application icon
        start_window.img_ico = 'imgs\ML\logo1.png'
        start_window.img_ico_path = os.path.join(application_path, start_window.img_ico)
        start_window.ico_photo = tk.PhotoImage(file=start_window.img_ico_path)
        master.iconphoto(True, start_window.ico_photo)
        #Creating menus
        main_menu = tk.Menu(master)
        master.config(menu=main_menu)
        settings_menu = tk.Menu(main_menu, tearoff=False)
        def set_theme(theme):
            master.set_theme(theme)
            style.configure('.', font=myfont)
            master.update()
            open('settings\style.txt', 'w').write(theme)
        style_menu = tk.Menu(settings_menu, tearoff=False)
        style_menu.add_command(label='Alt', command=lambda: set_theme('alt'))
        style_menu.add_command(label='Aquativo', command=lambda: set_theme('aquativo'))
        style_menu.add_command(label='Black', command=lambda: set_theme('black'))
        style_menu.add_command(label='Blue', command=lambda: set_theme('blue')) #big
        style_menu.add_command(label='Clam', command=lambda: set_theme('clam'))
        style_menu.add_command(label='Clearlooks', command=lambda: set_theme('clearlooks'))
        style_menu.add_command(label='Default', command=lambda: set_theme('default'))
        style_menu.add_command(label='Kroc', command=lambda: set_theme('kroc')) #big
        style_menu.add_command(label='Plastik', command=lambda: set_theme('plastik'))
        style_menu.add_command(label='Radiance', command=lambda: set_theme('radiance')) #big
        scid_menu = tk.Menu(style_menu, tearoff=False)
        scid_menu.add_command(label='Mint', command=lambda: set_theme('scidmint'))
        scid_menu.add_command(label='Sand', command=lambda: set_theme('scidsand'))
        scid_menu.add_command(label='Green', command=lambda: set_theme('scidgreen'))
        scid_menu.add_command(label='Blue', command=lambda: set_theme('scidblue'))
        scid_menu.add_command(label='Purple', command=lambda: set_theme('scidpurple'))
        scid_menu.add_command(label='Grey', command=lambda: set_theme('scidgrey'))
        scid_menu.add_command(label='Pink', command=lambda: set_theme('scidpink'))
        style_menu.add_cascade(label="Scid", menu=scid_menu)
        style_menu.add_command(label='Smog', command=lambda: set_theme('smog'))
        style_menu.add_command(label='Ubuntu', command=lambda: set_theme('ubuntu')) #big
        style_menu.add_command(label='Vista', command=lambda: set_theme('vista'))
        style_menu.add_command(label='winnative', command=lambda: set_theme('winnative'))
        style_menu.add_command(label='winxpblue', command=lambda: set_theme('winxpblue'))
        style_menu.add_command(label='xpnative', command=lambda: set_theme('xpnative'))
        settings_menu.add_cascade(label="Style", menu=style_menu)
        help_menu = tk.Menu(main_menu, tearoff=False)
        def open_sklearn():
            webbrowser.open('https://scikit-learn.org')
        def open_about_window():
            about_message = 'Machine Learning tool 0.7'
            about_detail = "by Monkey22\na.k.a Osipov A."
            tk.messagebox.showinfo(title='About', message=about_message, detail=about_detail)
        help_menu.add_command(label='Scikit-learn site', command=open_sklearn)
        help_menu.add_command(label='About', command=open_about_window)
        main_menu.add_cascade(label="Settings", menu=settings_menu)
        main_menu.add_cascade(label="Help", menu=help_menu)
        
        master.mainloop()
class cl_app:
    class training:
        data = None
        sheet = tk.StringVar()
        sheet.set('1')
        header_var = tk.IntVar()
        header_var.set(1)
        x_from_var = tk.StringVar()
        x_to_var = tk.StringVar()
    class prediction:
        data = None
        sheet = tk.StringVar()
        sheet.set('1')
        header_var = tk.IntVar()
        header_var.set(1)
        x_from_var = tk.StringVar()
        x_to_var = tk.StringVar()
    class Comp_results:
        def __init__(self, prev, parent):
            self.root = tk.Toplevel(parent)
            w = 250
            h = 360
            #setting main window's parameters       
            x = (ws/2) - (w/2)
            y = (hs/2) - (h/2) - 30
            self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
            self.root.lift()
            self.root.grab_set()
            self.root.focus_force()
            self.root.resizable(False, False)
            self.root.title('Comparison results')

            self.frame = ttk.Frame(self.root, width=w, height=h)
            self.frame.place(x=0, y=0)

            ttk.Label(self.frame, text='The results of classification\n methods comparison', font=myfont).place(x=10,y=10)
            ttk.Label(self.frame, text='Decision Tree:', font=myfont1).place(x=10,y=60)
            ttk.Label(self.frame, text='Ridge:', font=myfont1).place(x=10,y=90)
            ttk.Label(self.frame, text='Random Forest:', font=myfont1).place(x=10,y=120)
            ttk.Label(self.frame, text='Support Vector:', font=myfont1).place(x=10,y=150)
            ttk.Label(self.frame, text='SGD:', font=myfont1).place(x=10,y=180)
            ttk.Label(self.frame, text='Nearest Neighbor:', font=myfont1).place(x=10,y=210)
            ttk.Label(self.frame, text='Gaussian process:', font=myfont1).place(x=10,y=240)
            ttk.Label(self.frame, text='Multi-layer Perceptron:', font=myfont1).place(x=10,y=270)

            if 'dtc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['dtc']), font=myfont1).place(x=200,y=60)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=60)
            if 'rc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['rc']), font=myfont1).place(x=200,y=90)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=90)
            if 'rfc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['rfc']), font=myfont1).place(x=200,y=120)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=120)
            if 'svc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['svc']), font=myfont1).place(x=200,y=150)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=150)
            if 'sgdc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['sgdc']), font=myfont1).place(x=200,y=180)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=180)
            if 'knc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['knc']), font=myfont1).place(x=200,y=210)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=210)
            if 'gpc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['gpc']), font=myfont1).place(x=200,y=240)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=240)
            if 'mlpc' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['mlpc']), font=myfont1).place(x=200,y=270)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=270)

            ttk.Button(self.frame, text='OK', command=lambda: quit_back(self.root, prev.root)).place(x=110, y=310)
    def __init__(self, parent):
        w = 600
        h = 500
        
        #setting main window's parameters       
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2) - 30
        self.root = tk.Toplevel(parent)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.title('Monkey Classification')
        self.root.lift()
        self.root.tkraise()
        self.root.focus_force()
        self.root.resizable(False, False)
        
        self.frame = ttk.Frame(self.root, width=w, height=h)
        self.frame.place(x=0, y=0)
        
        ttk.Label(self.frame, text='Train Data file:', font=myfont1).place(x=10, y=10)
        e1 = ttk.Entry(self.frame, font=myfont1, width=40)
        e1.place(x=120, y=13)
        e1.insert(0, "C:/Users/csded/Documents/Python/Anaconda/Machine Learning/data/Pokemon.xlsx")
        e2 = ttk.Entry(self.frame, font=myfont1, width=4)
        e2.place(x=150, y=203)
        e2.insert(0, "5")

        ttk.Label(self.frame, text='Number of repeats:', font=myfont1).place(x=200, y=200)
        rep_entry = ttk.Entry(self.frame, font=myfont1, width=4)
        rep_entry.place(x=330, y=203)
        rep_entry.insert(0, "1")
        
        ttk.Button(self.frame, text='Choose file', command=lambda: open_file(self, e1)).place(x=490, y=10)
        
        parent.x_st_var = tk.StringVar()
        
        parent.dummies_var = tk.IntVar()
        parent.dummies_var.set(0)
        
        ttk.Label(self.frame, text='List number').place(x=120,y=50)
        training_sheet_entry = ttk.Entry(self.frame, textvariable=self.training.sheet, font=myfont1, width=3)
        training_sheet_entry.place(x=210,y=52)
                
        ttk.Button(self.frame, text='Load data ', command=lambda: load_data(self, self.training, e1, 'training')).place(x=490, y=50)
        
        cb2 = ttk.Checkbutton(self.frame, text="Dummies", variable=parent.dummies_var, takefocus=False)
        cb1 = ttk.Checkbutton(self.frame, text="header", variable=self.training.header_var, takefocus=False)
        cb1.place(x=10, y=50)
        
        ttk.Label(self.frame, text='Data status:').place(x=10, y=100)
        self.tr_data_status = ttk.Label(self.frame, text='Not Loaded')
        self.tr_data_status.place(x=120, y=100)

        ttk.Button(self.frame, text='View/Change', 
            command=lambda: Data_Preview(self, self.training, 'training', parent)).place(x=230, y=100)
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
        self.rfc_n_jobs = tk.StringVar(value='None')
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
        self.mlpc_hidden_layer_sizes = tk.StringVar(value='100')
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
        
        def compare_methods(prev, main):
            if master.dummies_var.get()==0:
                try:
                    prev.X = main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())]
                    prev.y = main.data[parent.y_var.get()]
                except:
                    prev.X = main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())]
                    prev.y = main.data[int(parent.y_var.get())]
            elif master.dummies_var.get()==1:
                try:
                    X=main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())]
                except:
                    X=main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())]
                try:
                    prev.X = pd.get_dummies(X)
                    prev.y = main.data[parent.y_var.get()]
                except:
                    prev.X = pd.get_dummies(X)
                    prev.y = main.data[int(parent.y_var.get())]
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            prev.X_St = scaler.fit_transform(prev.X)
            self.scores = {}
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import RepeatedStratifiedKFold
            folds = RepeatedStratifiedKFold(n_splits = int(e2.get()), n_repeats=int(rep_entry.get()), random_state = 100)
            # Decision Tree Classifier
            if self.dtc_include_comp.get():
                from sklearn.tree import DecisionTreeClassifier
                dtc = DecisionTreeClassifier(criterion=self.dtc_criterion.get(), splitter=self.dtc_splitter.get(), 
                                             max_depth=(int(self.dtc_max_depth.get()) if (self.dtc_max_depth.get() != 'None') else None), 
                                             min_samples_split=(float(self.dtc_min_samples_split.get()) if '.' in self.dtc_min_samples_split.get()
                                                               else int(self.dtc_min_samples_split.get())), 
                                             min_samples_leaf=(float(self.dtc_min_samples_leaf.get()) if '.' in self.dtc_min_samples_leaf.get()
                                                              else int(self.dtc_min_samples_leaf.get())),
                                             min_weight_fraction_leaf=float(self.dtc_min_weight_fraction_leaf.get()),
                                             max_features=(float(self.dtc_max_features.get()) if '.' in self.dtc_max_features.get() 
                                                           else int(self.dtc_max_features.get()) if len(self.dtc_max_features.get()) < 4 
                                                           else self.dtc_max_features.get() if (self.dtc_max_features.get() != 'None') 
                                                           else None), 
                                             random_state=(int(self.dtc_random_state.get()) if (self.dtc_random_state.get() != 'None') else None),
                                             max_leaf_nodes=(int(self.dtc_max_leaf_nodes.get()) if (self.dtc_max_leaf_nodes.get() != 'None') else None),
                                             min_impurity_decrease=float(self.dtc_min_impurity_decrease.get()),
                                             ccp_alpha=float(self.dtc_ccp_alpha.get()))
                if master.x_st_var.get() == 'everywhere':
                    dtc_scores = cross_val_score(dtc, prev.X_St, prev.y, cv=folds)
                else:
                    dtc_scores = cross_val_score(dtc, prev.X, prev.y, cv=folds)
                self.scores['dtc'] = dtc_scores.mean()
            # Ridge Classifier
            if self.rc_include_comp.get():
                from sklearn.linear_model import RidgeClassifier
                rc = RidgeClassifier(alpha=float(self.rc_alpha.get()), fit_intercept=self.rc_fit_intercept.get(),
                                     normalize=self.rc_normalize.get(), copy_X=self.rc_copy_X.get(),
                                     max_iter=(int(self.rc_max_iter.get()) if 
                                                                  (self.rc_max_iter.get() != 'None') else None),
                                     tol=float(self.rc_tol.get()), solver=self.rc_solver.get(),
                                     random_state=(int(self.rc_random_state.get()) if 
                                                   (self.rc_random_state.get() != 'None') else None))
                if master.x_st_var.get() == 'everywhere':
                    rc_scores = cross_val_score(rc, prev.X_St, prev.y, cv=folds)
                else:
                    rc_scores = cross_val_score(rc, prev.X, prev.y, cv=folds)
                self.scores['rc'] = rc_scores.mean()
            # Random forest Classifier
            if self.rfc_include_comp.get():
                from sklearn.ensemble import RandomForestClassifier
                rfc = RandomForestClassifier(n_estimators=int(self.rfc_n_estimators.get()), criterion=self.rfc_criterion.get(),
                                            max_depth=(int(self.rfc_max_depth.get()) if (self.rfc_max_depth.get() != 'None') else None), 
                                            min_samples_split=(float(self.rfc_min_samples_split.get()) if '.' in self.rfc_min_samples_split.get()
                                                               else int(self.rfc_min_samples_split.get())),
                                            min_samples_leaf=(float(self.rfc_min_samples_leaf.get()) if '.' in self.rfc_min_samples_leaf.get()
                                                               else int(self.rfc_min_samples_leaf.get())),
                                            min_weight_fraction_leaf=float(self.rfc_min_weight_fraction_leaf.get()),
                                            max_features=(float(self.rfc_max_features.get()) if '.' in self.rfc_max_features.get() 
                                                           else int(self.rfc_max_features.get()) if len(self.rfc_max_features.get()) < 4 
                                                           else self.rfc_max_features.get() if (self.rfc_max_features.get() != 'None') 
                                                           else None),
                                            max_leaf_nodes=(int(self.rfc_max_leaf_nodes.get()) if (self.rfc_max_leaf_nodes.get() != 'None') else None),
                                            min_impurity_decrease=float(self.rfc_min_impurity_decrease.get()),
                                            bootstrap=self.rfc_bootstrap.get(), oob_score=self.rfc_oob_score.get(),
                                            n_jobs=(int(self.rfc_n_jobs.get()) if (self.rfc_n_jobs.get() != 'None') else None),
                                            random_state=(int(self.rfc_random_state.get()) if
                                                          (self.rfc_random_state.get() != 'None') else None),
                                            verbose=int(self.rfc_verbose.get()), warm_start=self.rfc_warm_start.get(),
                                            ccp_alpha=float(self.rfc_ccp_alpha.get()),
                                            max_samples=(float(self.rfc_max_samples.get()) if '.' in self.rfc_max_samples.get() 
                                                           else int(self.rfc_max_samples.get()) if (self.rfc_max_samples.get() != 'None') 
                                                           else None))
                if master.x_st_var.get() == 'everywhere':
                    rfc_scores = cross_val_score(rfc, prev.X_St, prev.y, cv=folds)
                else:
                    rfc_scores = cross_val_score(rfc, prev.X, prev.y, cv=folds)
                self.scores['rfc'] = rfc_scores.mean()
            # Support Vector Classifier
            if self.svc_include_comp.get():
                from sklearn.svm import SVC
                svc = SVC(C=float(self.svc_C.get()), kernel=self.svc_kernel.get(), degree=int(self.svc_degree.get()), 
                          gamma=(float(self.svc_gamma.get()) if '.' in self.svc_gamma.get() else self.svc_gamma.get()),
                          coef0=float(self.svc_coef0.get()), shrinking=self.svc_shrinking.get(), 
                          probability=self.svc_probability.get(), tol=float(self.svc_tol.get()),
                          cache_size=float(self.svc_cache_size.get()), verbose=self.svc_verbose.get(),
                          max_iter=int(self.svc_max_iter.get()),
                          decision_function_shape=self.svc_decision_function_shape.get(),
                          break_ties=self.svc_break_ties.get(), 
                          random_state=(int(self.svc_random_state.get()) if (self.svc_random_state.get() != 'None') else None))
                if master.x_st_var.get() == 'None':
                    svc_scores = cross_val_score(svc, prev.X, prev.y, cv=folds)
                else:
                    svc_scores = cross_val_score(svc, prev.X_St, prev.y, cv=folds)
                self.scores['svc'] = svc_scores.mean()
            # Stochastic Gradient Descent Classifier
            if self.sgdc_include_comp.get():
                from sklearn.linear_model import SGDClassifier
                sgdc = SGDClassifier(loss=self.sgdc_loss.get(), penalty=self.sgdc_penalty.get(),
                                     alpha=float(self.sgdc_alpha.get()), l1_ratio=float(self.sgdc_l1_ratio.get()),
                                     fit_intercept=self.sgdc_fit_intercept.get(), max_iter=int(self.sgdc_max_iter.get()),
                                     tol=float(self.sgdc_tol.get()), shuffle=self.sgdc_shuffle.get(), 
                                     verbose=int(self.sgdc_verbose.get()), epsilon=float(self.sgdc_epsilon.get()),
                                     n_jobs=(int(self.sgdc_n_jobs.get()) if (self.sgdc_n_jobs.get() 
                                                                                            != 'None') else None), 
                                     random_state=(int(self.sgdc_random_state.get()) if (self.sgdc_random_state.get() 
                                                                                           != 'None') else None),
                                     learning_rate=self.sgdc_learning_rate.get(), eta0=float(self.sgdc_eta0.get()),
                                     power_t=float(self.sgdc_power_t.get()), early_stopping=self.sgdc_early_stopping.get(),
                                     validation_fraction=float(self.sgdc_validation_fraction.get()),
                                     n_iter_no_change=int(self.sgdc_n_iter_no_change.get()), warm_start=self.sgdc_warm_start.get(),
                                     average=(True if self.sgdc_average.get()=='True' else False 
                                              if self.sgdc_average.get()=='False' else int(self.sgdc_average.get())))
                if master.x_st_var.get() == 'None':
                    sgdc_scores = cross_val_score(sgdc, prev.X, prev.y, cv=folds)
                else:
                    sgdc_scores = cross_val_score(sgdc, prev.X_St, prev.y, cv=folds)
                self.scores['sgdc'] = sgdc_scores.mean()
            # Nearest Neighbor Classifier
            if self.knc_include_comp.get():
                from sklearn.neighbors import KNeighborsClassifier
                knc = KNeighborsClassifier(n_neighbors=int(self.knc_n_neighbors.get()), 
                                           weights=self.knc_weights.get(), algorithm=self.knc_algorithm.get(),
                                           leaf_size=int(self.knc_leaf_size.get()), p=int(self.knc_p.get()),
                                           metric=self.knc_metric.get(), 
                                           n_jobs=(int(self.knc_n_jobs.get()) if (self.knc_n_jobs.get() 
                                                                                                != 'None') else None))
                if master.x_st_var.get() == 'None':
                    knc_scores = cross_val_score(knc, prev.X, prev.y, cv=folds)
                else:
                    knc_scores = cross_val_score(knc, prev.X_St, prev.y, cv=folds)
                self.scores['knc'] = knc_scores.mean()
            # Gaussian process Classifier
            if self.gpc_include_comp.get():
                from sklearn.gaussian_process import GaussianProcessClassifier
                gpc = GaussianProcessClassifier(n_restarts_optimizer=int(self.gpc_n_restarts_optimizer.get()),
                                                max_iter_predict=int(self.gpc_max_iter_predict.get()),
                                                warm_start=self.gpc_warm_start.get(), copy_X_train=self.gpc_copy_X_train.get(),
                                                random_state=(int(self.gpc_random_state.get()) if 
                                                              (self.gpc_random_state.get() != 'None') else None),
                                                multi_class=self.gpc_multi_class.get(), 
                                                n_jobs=(int(self.gpc_n_jobs.get()) if 
                                                              (self.gpc_n_jobs.get() != 'None') else None))
                if master.x_st_var.get() == 'None':
                    gpc_scores = cross_val_score(gpc, prev.X, prev.y, cv=folds)
                else:
                    gpc_scores = cross_val_score(gpc, prev.X_St, prev.y, cv=folds)
                self.scores['gpc'] = gpc_scores.mean()
            # Multi-layer Perceptron Classifier
            if self.mlpc_include_comp.get():
                from sklearn.neural_network import MLPClassifier
                mlpc = MLPClassifier(hidden_layer_sizes=tuple([int(self.mlpc_hidden_layer_sizes.get())]),
                                     activation=self.mlpc_activation.get(),solver=self.mlpc_solver.get(),
                                     alpha=float(self.mlpc_alpha.get()), batch_size=(int(self.mlpc_batch_size.get()) if 
                                                                  (self.mlpc_batch_size.get() != 'auto') else 'auto'),
                                     learning_rate=self.mlpc_learning_rate.get(), 
                                     learning_rate_init=float(self.mlpc_learning_rate_init.get()),
                                     power_t=float(self.mlpc_power_t.get()), max_iter=int(self.mlpc_max_iter.get()),
                                     shuffle=self.mlpc_shuffle.get(),
                                     random_state=(int(self.mlpc_random_state.get()) if 
                                                                  (self.mlpc_random_state.get() != 'None') else None),
                                     tol=float(self.mlpc_tol.get()), verbose=self.mlpc_verbose.get(),
                                     warm_start=self.mlpc_warm_start.get(), momentum=float(self.mlpc_momentum.get()),
                                     nesterovs_momentum=self.mlpc_nesterovs_momentum.get(),
                                     early_stopping=self.mlpc_early_stopping.get(), 
                                     validation_fraction=float(self.mlpc_validation_fraction.get()),
                                     beta_1=float(self.mlpc_beta_1.get()), beta_2=float(self.mlpc_beta_2.get()),
                                     epsilon=float(self.mlpc_epsilon.get()), 
                                     n_iter_no_change=int(self.mlpc_n_iter_no_change.get()),
                                     max_fun=int(self.mlpc_max_fun.get()))
                if master.x_st_var.get() == 'everywhere':
                    mlpc_scores = cross_val_score(mlpc, prev.X_St, prev.y, cv=folds)
                else:
                    mlpc_scores = cross_val_score(mlpc, prev.X, prev.y, cv=folds)
                self.scores['mlpc'] = mlpc_scores.mean()
            show_rel_button.place(x=400, y=180)
        ttk.Button(self.frame, text="Methods' specifications", 
                  command=lambda: cl_mtds_specification(self, parent)).place(x=400, y=100)
        ttk.Button(self.frame, text='Perform comparison', command=lambda: compare_methods(self, self.training)).place(x=400, y=140)
        show_rel_button = ttk.Button(self.frame, text='Show Results', command=lambda: self.Comp_results(self, parent))
        
        ttk.Label(self.frame, text='Choose y', font=myfont1).place(x=30, y=140)
        parent.y_var = tk.StringVar()
        self.combobox1 = ttk.Combobox(self.frame, textvariable=parent.y_var, width=13, values=[])
        self.combobox1.place(x=100,y=140)
        
        ttk.Label(self.frame, text='X from', font=myfont1).place(x=205, y=140)
        self.tr_x_from_combobox = ttk.Combobox(self.frame, textvariable=self.training.x_from_var, width=4, values=[])
        self.tr_x_from_combobox.place(x=255, y=140)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=305, y=140)
        self.tr_x_to_combobox = ttk.Combobox(self.frame, textvariable=self.training.x_to_var, width=4, values=[])
        self.tr_x_to_combobox.place(x=325, y=140)
        self.combobox2 = ttk.Combobox(self.frame, textvariable=parent.x_st_var, width=13,
                                        values=['None', 'Where needed', 'everywhere'])
        self.combobox2.current(1)
        self.combobox2.place(x=150,y=175)
        ttk.Label(self.frame, text='X Standartization', font=myfont1).place(x=30, y=170)
        ttk.Label(self.frame, text='Number of folds:', font=myfont1).place(x=30, y=200)
        cb2.place(x=270, y=175)
        
        ttk.Label(self.frame, text='Predict Data file:', font=myfont1).place(x=10, y=250)
        pr_data_entry = ttk.Entry(self.frame, font=myfont1, width=38)
        pr_data_entry.place(x=140, y=250)
        pr_data_entry.insert(0, "C:/Users/csded/Documents/Python/Anaconda/Machine Learning/data/Pokemon.xlsx")
        
        ttk.Button(self.frame, text='Choose file', command=lambda: open_file(self, pr_data_entry)).place(x=490, y=245)
        
        parent.x2_st_var = tk.StringVar()
        
        ttk.Label(self.frame, text='List number', font=myfont1).place(x=120,y=295)
        pr_sheet_entry = ttk.Entry(self.frame, textvariable=self.prediction.sheet, font=myfont1, width=3)
        pr_sheet_entry.place(x=210,y=297)
        
        ttk.Button(self.frame, text='Load data ', command=lambda: load_data(self, self.prediction, pr_data_entry, 'prediction')).place(x=490, y=295)
        
        cb4 = ttk.Checkbutton(self.frame, text="header", variable=self.prediction.header_var, takefocus=False)
        cb4.place(x=10, y=290)
        
        ttk.Label(self.frame, text='Data status:', font=myfont).place(x=10, y=350)
        self.pr_data_status = ttk.Label(self.frame, text='Not Loaded', font=myfont)
        self.pr_data_status.place(x=120, y=350)

        ttk.Button(self.frame, text='View/Change', command=lambda: Data_Preview(self, self.prediction, 'prediction', parent)).place(x=230, y=350)
        
        parent.pr_method = tk.StringVar()
        parent.pr_method.set('Decision Tree')
        self.combobox9 = ttk.Combobox(self.frame, textvariable=parent.pr_method, values=['Decision Tree', 'Ridge',
                                                                                         'Random Forest',
                                                                                        'Support Vector', 'SGD', 
                                                                                        'Nearest Neighbor', 'Gaussian Process', 
                                                                                        'Multi-layer Perceptron'])
        self.combobox9.place(x=200,y=425)
        ttk.Label(self.frame, text='Choose method', font=myfont1).place(x=30, y=420)
        
        ttk.Label(self.frame, text='X from', font=myfont1).place(x=205, y=390)
        self.pr_x_from_combobox = ttk.Combobox(self.frame, textvariable=self.prediction.x_from_var, width=4, values=[])
        self.pr_x_from_combobox.place(x=255, y=390)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=305, y=390)
        self.pr_x_to_combobox = ttk.Combobox(self.frame, textvariable=self.prediction.x_to_var, width=4, values=[])
        self.pr_x_to_combobox.place(x=325, y=390)
        
        self.combobox10 = ttk.Combobox(self.frame, textvariable=parent.x2_st_var, 
                                        values=['No', 'If needed', 'Yes'])
        self.combobox10.current(1)
        self.combobox10.place(x=200,y=455)
        ttk.Label(self.frame, text='X Standartization', font=myfont1).place(x=30, y=450)
        
        def cl_predict_class(method):
            if master.dummies_var.get()==0:
                try:
                    training_X = self.training.data.iloc[:,(int(self.training.x_from_var.get())-1): int(self.training.x_to_var.get())]
                    training_y = self.training.data[parent.y_var.get()]
                except:
                    training_X = self.training.data.iloc[:,(int(self.training.x_from_var.get())-1): int(self.training.x_to_var.get())]
                    training_y = self.training.data[int(parent.y_var.get())]
                X = self.prediction.data.iloc[:,(int(self.prediction.x_from_var.get())-1): int(self.prediction.x_to_var.get())]
            elif master.dummies_var.get()==1:
                X = pd.get_dummies(self.prediction.data.iloc[:,(int(self.prediction.x_from_var.get())-1): int(self.prediction.x_to_var.get())])
                try:
                    training_X=self.training.data.iloc[:,(int(self.training.x_from_var.get())-1): int(self.training.x_to_var.get())]
                except:
                    training_X=self.training.data.iloc[:,(int(self.training.x_from_var.get())-1): int(self.training.x_to_var.get())]
                try:
                    training_X = pd.get_dummies(training_X)
                    training_y = self.training.data[parent.y_var.get()]
                except:
                    training_X = pd.get_dummies(training_X)
                    training_y = self.training.data[int(parent.y_var.get())]
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            X_St = scaler.fit_transform(X)
            training_X_St = scaler.fit_transform(training_X)
            
            if method == 'Decision Tree':
                from sklearn.tree import DecisionTreeClassifier
                dtc = DecisionTreeClassifier(criterion=self.dtc_criterion.get(), splitter=self.dtc_splitter.get(), 
                                         max_depth=(int(self.dtc_max_depth.get()) if (self.dtc_max_depth.get() != 'None') else None), 
                                         min_samples_split=(float(self.dtc_min_samples_split.get()) if '.' in self.dtc_min_samples_split.get()
                                                           else int(self.dtc_min_samples_split.get())), 
                                         min_samples_leaf=(float(self.dtc_min_samples_leaf.get()) if '.' in self.dtc_min_samples_leaf.get()
                                                          else int(self.dtc_min_samples_leaf.get())),
                                         min_weight_fraction_leaf=float(self.dtc_min_weight_fraction_leaf.get()),
                                         max_features=(float(self.dtc_max_features.get()) if '.' in self.dtc_max_features.get() 
                                                       else int(self.dtc_max_features.get()) if len(self.dtc_max_features.get()) < 4 
                                                       else self.dtc_max_features.get() if (self.dtc_max_features.get() != 'None') 
                                                       else None), 
                                         random_state=(int(self.dtc_random_state.get()) if (self.dtc_random_state.get() != 'None') else None),
                                         max_leaf_nodes=(int(self.dtc_max_leaf_nodes.get()) if (self.dtc_max_leaf_nodes.get() != 'None') else None),
                                         min_impurity_decrease=float(self.dtc_min_impurity_decrease.get()),
                                         ccp_alpha=float(self.dtc_ccp_alpha.get()))
                if master.x2_st_var.get() == 'Yes':
                    dtc.fit(training_X_st, training_y)
                    pr_values = dtc.predict(X_St)
                else:
                    dtc.fit(training_X, training_y)
                    pr_values = dtc.predict(X)
            elif method == 'Ridge':
                from sklearn.linear_model import RidgeClassifier
                rc = RidgeClassifier(alpha=float(self.rc_alpha.get()), fit_intercept=self.rc_fit_intercept.get(),
                                     normalize=self.rc_normalize.get(), copy_X=self.rc_copy_X.get(),
                                     max_iter=(int(self.rc_max_iter.get()) if 
                                                                  (self.rc_max_iter.get() != 'None') else None),
                                     tol=float(self.rc_tol.get()), solver=self.rc_solver.get(),
                                     random_state=(int(self.rc_random_state.get()) if 
                                                   (self.rc_random_state.get() != 'None') else None))
                if master.x2_st_var.get() == 'Yes':
                    rc.fit(training_X_st, training_y)
                    pr_values = rc.predict(X_St)
                else:
                    rc.fit(training_X, training_y)
                    pr_values = rc.predict(X)
            elif method == 'Random Forest':
                from sklearn.ensemble import RandomForestClassifier
                rfc = RandomForestClassifier(n_estimators=int(self.rfc_n_estimators.get()), criterion=self.rfc_criterion.get(),
                                            max_depth=(int(self.rfc_max_depth.get()) if (self.rfc_max_depth.get() != 'None') else None), 
                                            min_samples_split=(float(self.rfc_min_samples_split.get()) if '.' in self.rfc_min_samples_split.get()
                                                               else int(self.rfc_min_samples_split.get())),
                                            min_samples_leaf=(float(self.rfc_min_samples_leaf.get()) if '.' in self.rfc_min_samples_leaf.get()
                                                               else int(self.rfc_min_samples_leaf.get())),
                                            min_weight_fraction_leaf=float(self.rfc_min_weight_fraction_leaf.get()),
                                            max_features=(float(self.rfc_max_features.get()) if '.' in self.rfc_max_features.get() 
                                                           else int(self.rfc_max_features.get()) if len(self.rfc_max_features.get()) < 4 
                                                           else self.rfc_max_features.get() if (self.rfc_max_features.get() != 'None') 
                                                           else None),
                                            max_leaf_nodes=(int(self.rfc_max_leaf_nodes.get()) if (self.rfc_max_leaf_nodes.get() != 'None') else None),
                                            min_impurity_decrease=float(self.rfc_min_impurity_decrease.get()),
                                            bootstrap=self.rfc_bootstrap.get(), oob_score=self.rfc_oob_score.get(),
                                            n_jobs=(int(self.rfc_n_jobs.get()) if (self.rfc_n_jobs.get() != 'None') else None),
                                            random_state=(int(self.rfc_random_state.get()) if
                                                          (self.rfc_random_state.get() != 'None') else None),
                                            verbose=int(self.rfc_verbose.get()), warm_start=self.rfc_warm_start.get(),
                                            ccp_alpha=float(self.rfc_ccp_alpha.get()),
                                            max_samples=(float(self.rfc_max_samples.get()) if '.' in self.rfc_max_samples.get() 
                                                           else int(self.rfc_max_samples.get()) if (self.rfc_max_samples.get() != 'None') 
                                                           else None))
                if master.x2_st_var.get() == 'Yes':
                    rfc.fit(training_X_St, training_y)
                    pr_values = rfc.predict(X_St)
                else:
                    rfc.fit(training_X, training_y)
                    pr_values = rfc.predict(X)
            elif method == 'Support Vector':
                from sklearn.svm import SVC
                svc = SVC(C=float(self.svc_C.get()), kernel=self.svc_kernel.get(), degree=int(self.svc_degree.get()), 
                          gamma=(float(self.svc_gamma.get()) if '.' in self.svc_gamma.get() else self.svc_gamma.get()),
                          coef0=float(self.svc_coef0.get()), shrinking=self.svc_shrinking.get(), 
                          probability=self.svc_probability.get(), tol=float(self.svc_tol.get()),
                          cache_size=float(self.svc_cache_size.get()), verbose=self.svc_verbose.get(),
                          max_iter=int(self.svc_max_iter.get()),
                          decision_function_shape=self.svc_decision_function_shape.get(),
                          break_ties=self.svc_break_ties.get(), 
                          random_state=(int(self.svc_random_state.get()) if (self.svc_random_state.get() != 'None') else None))
                if master.x2_st_var.get() == 'No':
                    svc.fit(training_X, training_y)
                    pr_values = svc.predict(X)
                else:
                    svc.fit(training_X_St, training_y)
                    pr_values = svc.predict(X_St)
            elif method == 'SGD':
                from sklearn.linear_model import SGDClassifier
                sgdc = SGDClassifier(loss=self.sgdc_loss.get(), penalty=self.sgdc_penalty.get(),
                                     alpha=float(self.sgdc_alpha.get()), l1_ratio=float(self.sgdc_l1_ratio.get()),
                                     fit_intercept=self.sgdc_fit_intercept.get(), max_iter=int(self.sgdc_max_iter.get()),
                                     tol=float(self.sgdc_tol.get()), shuffle=self.sgdc_shuffle.get(), 
                                     verbose=int(self.sgdc_verbose.get()), epsilon=float(self.sgdc_epsilon.get()),
                                     n_jobs=(int(self.sgdc_n_jobs.get()) if (self.sgdc_n_jobs.get() 
                                                                                            != 'None') else None), 
                                     random_state=(int(self.sgdc_random_state.get()) if (self.sgdc_random_state.get() 
                                                                                           != 'None') else None),
                                     learning_rate=self.sgdc_learning_rate.get(), eta0=float(self.sgdc_eta0.get()),
                                     power_t=float(self.sgdc_power_t.get()), early_stopping=self.sgdc_early_stopping.get(),
                                     validation_fraction=float(self.sgdc_validation_fraction.get()),
                                     n_iter_no_change=int(self.sgdc_n_iter_no_change.get()), warm_start=self.sgdc_warm_start.get(),
                                     average=(True if self.sgdc_average.get()=='True' else False 
                                              if self.sgdc_average.get()=='False' else int(self.sgdc_average.get())))
                if master.x2_st_var.get() == 'No':
                    sgdc.fit(training_X, training_y)
                    pr_values = sgdc.predict(X)
                else:
                    sgdc.fit(training_X_St, training_y)
                    pr_values = sgdc.predict(X_St)
            elif method == 'Nearest Neighbor':
                from sklearn.neighbors import KNeighborsClassifier
                knc = KNeighborsClassifier(n_neighbors=int(self.knc_n_neighbors.get()), 
                                           weights=self.knc_weights.get(), algorithm=self.knc_algorithm.get(),
                                           leaf_size=int(self.knc_leaf_size.get()), p=int(self.knc_p.get()),
                                           metric=self.knc_metric.get(), 
                                           n_jobs=(int(self.knc_n_jobs.get()) if (self.knc_n_jobs.get() 
                                                                                                != 'None') else None))
                if master.x2_st_var.get() == 'No':
                    knc.fit(training_X, training_y)
                    pr_values = knc.predict(X)
                else:
                    knc.fit(training_X_St, training_y)
                    pr_values = knc.predict(X_St)
            elif method == 'Gaussian Process':
                from sklearn.gaussian_process import GaussianProcessClassifier
                gpc = GaussianProcessClassifier(n_restarts_optimizer=int(self.gpc_n_restarts_optimizer.get()),
                                                max_iter_predict=int(self.gpc_max_iter_predict.get()),
                                                warm_start=self.gpc_warm_start.get(), 
                                                copy_X_train=self.gpc_copy_X_train.get(),
                                                random_state=(int(self.gpc_random_state.get()) if 
                                                              (self.gpc_random_state.get() != 'None') else None),
                                                multi_class=self.gpc_multi_class.get(), 
                                                n_jobs=(int(self.gpc_n_jobs.get()) if 
                                                              (self.gpc_n_jobs.get() != 'None') else None))
                if master.x2_st_var.get() == 'No':
                    gpc.fit(training_X, training_y)
                    pr_values = gpc.predict(X)
                else:
                    gpc.fit(training_X_St, training_y)
                    pr_values = gpc.predict(X_St)
            elif method == 'Multi-layer Perceptron':
                from sklearn.neural_network import MLPClassifier
                mlpc = MLPClassifier(hidden_layer_sizes=tuple([int(self.mlpc_hidden_layer_sizes.get())]),
                                     activation=self.mlpc_activation.get(),solver=self.mlpc_solver.get(),
                                     alpha=float(self.mlpc_alpha.get()), batch_size=(int(self.mlpc_batch_size.get()) if 
                                                                  (self.mlpc_batch_size.get() != 'auto') else 'auto'),
                                     learning_rate=self.mlpc_learning_rate.get(), 
                                     learning_rate_init=float(self.mlpc_learning_rate_init.get()),
                                     power_t=float(self.mlpc_power_t.get()), max_iter=int(self.mlpc_max_iter.get()),
                                     shuffle=self.mlpc_shuffle.get(),
                                     random_state=(int(self.mlpc_random_state.get()) if 
                                                                  (self.mlpc_random_state.get() != 'None') else None),
                                     tol=float(self.mlpc_tol.get()), verbose=self.mlpc_verbose.get(),
                                     warm_start=self.mlpc_warm_start.get(), momentum=float(self.mlpc_momentum.get()),
                                     nesterovs_momentum=self.mlpc_nesterovs_momentum.get(),
                                     early_stopping=self.mlpc_early_stopping.get(), 
                                     validation_fraction=float(self.mlpc_validation_fraction.get()),
                                     beta_1=float(self.mlpc_beta_1.get()), beta_2=float(self.mlpc_beta_2.get()),
                                     epsilon=float(self.mlpc_epsilon.get()), 
                                     n_iter_no_change=int(self.mlpc_n_iter_no_change.get()),
                                     max_fun=int(self.mlpc_max_fun.get()))
                if master.x2_st_var.get() == 'Yes':
                    mlpc.fit(training_X_St, training_y)
                    pr_values = mlpc.predict(X_St)
                else:
                    mlpc.fit(training_X, training_y)
                    pr_values = mlpc.predict(X)
            self.prediction.data['Class'] = pr_values
        
        ttk.Button(self.frame, text='Predict values', 
                  command=lambda: cl_predict_class(method=parent.pr_method.get())).place(x=420, y=360)
        
        def save_results():
            files = [('Excel', '*.xlsx'),
                     ('All Files', '*.*'), 
                     ('Text Document', '*.txt')] 
            file = asksaveasfilename(parent=self.frame, filetypes = files, defaultextension = files)
            with pd.ExcelWriter(file, engine="openpyxl", mode='a') as writer: 
                pd.DataFrame(self.prediction.data).to_excel(writer, sheet_name='Result', 
                                                      index=False, header=(False if self.prediction.header_var.get()==0 else True))
        
        ttk.Button(self.frame, text='Save results', 
                  command=lambda: save_results()).place(x=420, y=400)
        ttk.Button(self.frame, text='Quit', 
                  command=lambda: quit_back(self.root, parent)).place(x=420, y=440)
    
class cl_mtds_specification:
    def __init__(self, prev, parent):
        self.root = tk.Toplevel(parent)

        w = 690
        h = 640
        #setting main window's parameters       
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2) - 30
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.grab_set()
        self.root.focus_force()
        self.root.resizable(False, False)
        self.root.title('Classification methods specification')

        self.canvas = tk.Canvas(self.root)
        self.frame = ttk.Frame(self.canvas, width=1180, height=640)
        self.scrollbar = ttk.Scrollbar(self.canvas, orient="horizontal", command=self.canvas.xview)
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

        ttk.Label(self.frame, text='Decision Tree', font=myfont_b).place(x=30, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=20, y=40)
        dt_cb1 = ttk.Checkbutton(self.frame, variable=prev.dtc_include_comp, takefocus=False)
        dt_cb1.place(x=110, y=50)
        ttk.Label(self.frame, text='Criterion', font=myfont2).place(x=5, y=80)
        dt_combobox3 = ttk.Combobox(self.frame, textvariable=prev.dtc_criterion, width=6, values=['gini', 'entropy'])
        dt_combobox3.place(x=110,y=80)
        ttk.Label(self.frame, text='Splitter', font=myfont2).place(x=5, y=100)
        dt_combobox4 = ttk.Combobox(self.frame, textvariable=prev.dtc_splitter, width=6, values=['best', 'random'])
        dt_combobox4.place(x=110,y=100)
        ttk.Label(self.frame, text='Max Depth', font=myfont2).place(x=5, y=120)
        dt_e1 = ttk.Entry(self.frame, textvariable=prev.dtc_max_depth, font=myfont2, width=5)
        dt_e1.place(x=115,y=120)
        ttk.Label(self.frame, text='Min samples split', font=myfont2).place(x=5, y=140)
        dt_e2 = ttk.Entry(self.frame, textvariable=prev.dtc_min_samples_split, font=myfont2, width=5)
        dt_e2.place(x=115,y=140)
        ttk.Label(self.frame, text='Min samples leaf', font=myfont2).place(x=5, y=160)
        dt_e3 = ttk.Entry(self.frame, textvariable=prev.dtc_min_samples_leaf, font=myfont2, width=5)
        dt_e3.place(x=115,y=160)
        ttk.Label(self.frame, text="Min weight\nfraction leaf", font=myfont2).place(x=5, y=180)
        dt_e4 = ttk.Entry(self.frame, textvariable=prev.dtc_min_weight_fraction_leaf, font=myfont2, width=5)
        dt_e4.place(x=115,y=190)
        ttk.Label(self.frame, text='Max features', font=myfont2).place(x=5, y=220)
        dt_e5 = ttk.Entry(self.frame, textvariable=prev.dtc_max_features, font=myfont2, width=5)
        dt_e5.place(x=115,y=220)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=5, y=240)
        dt_e6 = ttk.Entry(self.frame, textvariable=prev.dtc_random_state, font=myfont2, width=5)
        dt_e6.place(x=115,y=240)
        ttk.Label(self.frame, text='Max leaf nodes', font=myfont2).place(x=5, y=260)
        dt_e7 = ttk.Entry(self.frame, textvariable=prev.dtc_max_leaf_nodes, font=myfont2, width=5)
        dt_e7.place(x=115,y=260)
        ttk.Label(self.frame, text="Min impurity\ndecrease", font=myfont2).place(x=5, y=280)
        dt_e8 = ttk.Entry(self.frame, textvariable=prev.dtc_min_impurity_decrease, font=myfont2, width=5)
        dt_e8.place(x=115,y=290)
        ttk.Label(self.frame, text="CCP alpha", font=myfont2).place(x=5, y=320)
        dt_e9 = ttk.Entry(self.frame, textvariable=prev.dtc_ccp_alpha, font=myfont2, width=5)
        dt_e9.place(x=115,y=320)
        
        ttk.Label(self.frame, text='Ridge', font=myfont_b).place(x=40, y=340)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=20, y=360)
        r_cb1 = ttk.Checkbutton(self.frame, variable=prev.rc_include_comp, takefocus=False)
        r_cb1.place(x=115, y=370)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=5, y=400)
        r_e1 = ttk.Entry(self.frame, textvariable=prev.rc_alpha, font=myfont2, width=5)
        r_e1.place(x=115,y=400)
        ttk.Label(self.frame, text='fit intercept', font=myfont2).place(x=5, y=420)
        r_cb2 = ttk.Checkbutton(self.frame, variable=prev.rc_fit_intercept, takefocus=False)
        r_cb2.place(x=115, y=420)
        ttk.Label(self.frame, text='Normalize', font=myfont2).place(x=5, y=440)
        r_cb3 = ttk.Checkbutton(self.frame, variable=prev.rc_normalize, takefocus=False)
        r_cb3.place(x=115, y=440)
        ttk.Label(self.frame, text='Copy X', font=myfont2).place(x=5, y=460)
        r_cb4 = ttk.Checkbutton(self.frame, variable=prev.rc_copy_X, takefocus=False)
        r_cb4.place(x=115, y=460)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=5, y=480)
        r_e2 = ttk.Entry(self.frame, textvariable=prev.rc_max_iter, font=myfont2, width=5)
        r_e2.place(x=115,y=480)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=5, y=500)
        r_e3 = ttk.Entry(self.frame, textvariable=prev.rc_tol, font=myfont2, width=5)
        r_e3.place(x=115,y=500)
        ttk.Label(self.frame, text='Solver', font=myfont2).place(x=5, y=520)
        r_combobox1 = ttk.Combobox(self.frame, textvariable=prev.rc_solver, width=6, values=['auto', 'svd', 'cholesky',
                                                                                             'lsqr', 'sparse_cg', 
                                                                                             'sag', 'saga'])
        r_combobox1.place(x=110,y=520)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=5, y=540)
        r_e4 = ttk.Entry(self.frame, textvariable=prev.rc_random_state, font=myfont2, width=5)
        r_e4.place(x=115,y=540)

        ttk.Label(self.frame, text='Random Forest', font=myfont_b).place(x=190, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=185, y=40)
        rf_cb4 = ttk.Checkbutton(self.frame, variable=prev.rfc_include_comp, takefocus=False)
        rf_cb4.place(x=285, y=50)
        ttk.Label(self.frame, text='Trees number', font=myfont2).place(x=175, y=80)
        rf_e1 = ttk.Entry(self.frame, textvariable=prev.rfc_n_estimators, font=myfont2, width=5)
        rf_e1.place(x=285,y=80)
        ttk.Label(self.frame, text='Criterion', font=myfont2).place(x=175, y=100)
        rf_combobox5 = ttk.Combobox(self.frame, textvariable=prev.rfc_criterion, width=6, values=['gini', 'entropy'])
        rf_combobox5.place(x=275,y=100)
        ttk.Label(self.frame, text='Max Depth', font=myfont2).place(x=175, y=120)
        rf_e2 = ttk.Entry(self.frame, textvariable=prev.rfc_max_depth, font=myfont2, width=5)
        rf_e2.place(x=285,y=120)
        ttk.Label(self.frame, text='Min samples split', font=myfont2).place(x=175, y=140)
        rf_e3 = ttk.Entry(self.frame, textvariable=prev.rfc_min_samples_split, font=myfont2, width=5)
        rf_e3.place(x=285,y=140)
        ttk.Label(self.frame, text='Min samples leaf', font=myfont2).place(x=175, y=160)
        rf_e4 = ttk.Entry(self.frame, textvariable=prev.rfc_min_samples_leaf, font=myfont2, width=5)
        rf_e4.place(x=285,y=160)
        ttk.Label(self.frame, text='Min weight\nfraction leaf', font=myfont2).place(x=175, y=180)
        rf_e5 = ttk.Entry(self.frame, textvariable=prev.rfc_min_weight_fraction_leaf, font=myfont2, width=5)
        rf_e5.place(x=285,y=190)
        ttk.Label(self.frame, text='Max features', font=myfont2).place(x=175, y=220)
        rf_e6 = ttk.Entry(self.frame, textvariable=prev.rfc_max_features, font=myfont2, width=5)
        rf_e6.place(x=285,y=220)
        ttk.Label(self.frame, text='Max leaf nodes', font=myfont2).place(x=175, y=240)
        rf_e7 = ttk.Entry(self.frame, textvariable=prev.rfc_max_leaf_nodes, font=myfont2, width=5)
        rf_e7.place(x=285,y=240)
        ttk.Label(self.frame, text='Min impurity\ndecrease', font=myfont2).place(x=175, y=260)
        rf_e8 = ttk.Entry(self.frame, textvariable=prev.rfc_min_impurity_decrease, font=myfont2, width=5)
        rf_e8.place(x=285,y=270)
        ttk.Label(self.frame, text='Bootstrap', font=myfont2).place(x=175, y=300)
        rf_cb1 = ttk.Checkbutton(self.frame, variable=prev.rfc_bootstrap, takefocus=False)
        rf_cb1.place(x=285,y=300)
        ttk.Label(self.frame, text='oob score', font=myfont2).place(x=175, y=320)
        rf_cb2 = ttk.Checkbutton(self.frame, variable=prev.rfc_oob_score, takefocus=False)
        rf_cb2.place(x=285,y=320)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=175, y=340)
        rf_e9 = ttk.Entry(self.frame, textvariable=prev.rfc_n_jobs, font=myfont2, width=5)
        rf_e9.place(x=285,y=340)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=175, y=360)
        rf_e10 = ttk.Entry(self.frame, textvariable=prev.rfc_random_state, font=myfont2, width=5)
        rf_e10.place(x=285,y=360)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=175, y=380)
        rf_e11 = ttk.Entry(self.frame, textvariable=prev.rfc_verbose, font=myfont2, width=5)
        rf_e11.place(x=285,y=380)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=175, y=400)
        rf_cb3 = ttk.Checkbutton(self.frame, variable=prev.rfc_warm_start, takefocus=False)
        rf_cb3.place(x=285,y=400)
        ttk.Label(self.frame, text='CCP alpha', font=myfont2).place(x=175, y=420)
        rf_e12 = ttk.Entry(self.frame, textvariable=prev.rfc_ccp_alpha, font=myfont2, width=5)
        rf_e12.place(x=285,y=420)
        ttk.Label(self.frame, text='Max samples', font=myfont2).place(x=175, y=440)
        rf_e13 = ttk.Entry(self.frame, textvariable=prev.rfc_max_samples, font=myfont2, width=5)
        rf_e13.place(x=285,y=440)

        ttk.Label(self.frame, text='Support Vector', font=myfont_b).place(x=355, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=350, y=40)
        sv_cb5 = ttk.Checkbutton(self.frame, variable=prev.svc_include_comp, takefocus=False)
        sv_cb5.place(x=445, y=50)
        ttk.Label(self.frame, text='C', font=myfont2).place(x=340, y=80)
        sv_e1 = ttk.Entry(self.frame, textvariable=prev.svc_C, font=myfont2, width=5)
        sv_e1.place(x=445,y=80)
        ttk.Label(self.frame, text='Kernel', font=myfont2).place(x=340, y=100)
        sv_combobox1 = ttk.Combobox(self.frame, textvariable=prev.svc_kernel, width=6, values=['linear', 'poly', 
                                                                                                 'rbf', 'sigmoid', 'precomputed'])
        sv_combobox1.place(x=435,y=100)
        ttk.Label(self.frame, text='Degree', font=myfont2).place(x=340, y=120)
        sv_e2 = ttk.Entry(self.frame, textvariable=prev.svc_degree, font=myfont2, width=5)
        sv_e2.place(x=445,y=120)
        ttk.Label(self.frame, text='Gamma', font=myfont2).place(x=340, y=140)
        sv_e3 = ttk.Entry(self.frame, textvariable=prev.svc_gamma, font=myfont2, width=5)
        sv_e3.place(x=445,y=140)
        ttk.Label(self.frame, text='coef0', font=myfont2).place(x=340, y=160)
        sv_e4 = ttk.Entry(self.frame, textvariable=prev.svc_coef0, font=myfont2, width=5)
        sv_e4.place(x=445,y=160)
        ttk.Label(self.frame, text='shrinking', font=myfont2).place(x=340, y=180)
        sv_cb1 = ttk.Checkbutton(self.frame, variable=prev.svc_shrinking, takefocus=False)
        sv_cb1.place(x=445,y=180)
        ttk.Label(self.frame, text='probability', font=myfont2).place(x=340, y=200)
        sv_cb2 = ttk.Checkbutton(self.frame, variable=prev.svc_probability, takefocus=False)
        sv_cb2.place(x=445,y=200)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=340, y=220)
        sv_e5 = ttk.Entry(self.frame, textvariable=prev.svc_tol, font=myfont2, width=5)
        sv_e5.place(x=445,y=220)
        ttk.Label(self.frame, text='Cache size', font=myfont2).place(x=340, y=240)
        sv_e6 = ttk.Entry(self.frame, textvariable=prev.svc_cache_size, font=myfont2, width=5)
        sv_e6.place(x=445,y=240)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=340, y=260)
        sv_cb3 = ttk.Checkbutton(self.frame, variable=prev.svc_verbose, takefocus=False)
        sv_cb3.place(x=445,y=260)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=340, y=280)
        sv_e7 = ttk.Entry(self.frame, textvariable=prev.svc_max_iter, font=myfont2, width=5)
        sv_e7.place(x=445,y=280)
        ttk.Label(self.frame, text='Decision\nfunction shape', font=myfont2).place(x=340, y=300)
        sv_combobox2 = ttk.Combobox(self.frame, textvariable=prev.svc_decision_function_shape, width=5, values=['ovo', 'ovr'])
        sv_combobox2.place(x=435,y=310)
        ttk.Label(self.frame, text='Break ties', font=myfont2).place(x=340, y=340)
        sv_cb4 = ttk.Checkbutton(self.frame, variable=prev.svc_break_ties, takefocus=False)
        sv_cb4.place(x=445,y=340)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=340, y=360)
        sv_e8 = ttk.Entry(self.frame, textvariable=prev.svc_random_state, font=myfont2, width=5)
        sv_e8.place(x=445,y=360)

        ttk.Label(self.frame, text='SGD', font=myfont_b).place(x=535, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=515, y=40)
        sgd_cb5 = ttk.Checkbutton(self.frame, variable=prev.sgdc_include_comp, takefocus=False)
        sgd_cb5.place(x=610, y=50)
        ttk.Label(self.frame, text='Loss', font=myfont2).place(x=505, y=80)
        sgd_combobox1 = ttk.Combobox(self.frame, textvariable=prev.sgdc_loss, width=10, values=['hinge', 'log', 'modified_huber', 
                                                                                             'squared_hinge', 'perceptron',
                                                                                             'squared_loss', 'huber', 
                                                                                             'epsilon_insensitive', 
                                                                                             'squared_epsilon_insensitive'])
        sgd_combobox1.place(x=602,y=80)
        ttk.Label(self.frame, text='Penalty', font=myfont2).place(x=505, y=100)
        sgd_combobox2 = ttk.Combobox(self.frame, textvariable=prev.sgdc_penalty, width=7, values=['l2', 'l1', 'elasticnet'])
        sgd_combobox2.place(x=610,y=100)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=505, y=120)
        sgd_e1 = ttk.Entry(self.frame, textvariable=prev.sgdc_alpha, font=myfont2, width=7)
        sgd_e1.place(x=615,y=120)
        ttk.Label(self.frame, text='l1 ratio', font=myfont2).place(x=505, y=140)
        sgd_e2 = ttk.Entry(self.frame, textvariable=prev.sgdc_l1_ratio, font=myfont2, width=7)
        sgd_e2.place(x=615,y=140)
        ttk.Label(self.frame, text='fit intercept', font=myfont2).place(x=505, y=160)
        sgd_cb1 = ttk.Checkbutton(self.frame, variable=prev.sgdc_fit_intercept, takefocus=False)
        sgd_cb1.place(x=615,y=160)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=505, y=180)
        sgd_e3 = ttk.Entry(self.frame, textvariable=prev.sgdc_max_iter, font=myfont2, width=7)
        sgd_e3.place(x=615,y=180)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=505, y=200)
        sgd_e4 = ttk.Entry(self.frame, textvariable=prev.sgdc_tol, font=myfont2, width=7)
        sgd_e4.place(x=615,y=200)
        ttk.Label(self.frame, text='Shuffle', font=myfont2).place(x=505, y=220)
        sgd_cb2 = ttk.Checkbutton(self.frame, variable=prev.sgdc_shuffle, takefocus=False)
        sgd_cb2.place(x=615,y=220)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=505, y=240)
        sgd_e5 = ttk.Entry(self.frame, textvariable=prev.sgdc_verbose, font=myfont2, width=7)
        sgd_e5.place(x=615,y=240)
        ttk.Label(self.frame, text='Epsilon', font=myfont2).place(x=505, y=260)
        sgd_e6 = ttk.Entry(self.frame, textvariable=prev.sgdc_epsilon, font=myfont2, width=7)
        sgd_e6.place(x=615,y=260)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=505, y=280)
        sgd_e7 = ttk.Entry(self.frame, textvariable=prev.sgdc_n_jobs, font=myfont2, width=7)
        sgd_e7.place(x=615,y=280)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=505, y=300)
        sgd_e8 = ttk.Entry(self.frame, textvariable=prev.sgdc_random_state, font=myfont2, width=7)
        sgd_e8.place(x=615,y=300)
        ttk.Label(self.frame, text='Learning rate', font=myfont2).place(x=505, y=320)
        sgd_combobox3 = ttk.Combobox(self.frame, textvariable=prev.sgdc_learning_rate, width=7, values=['constant', 'optimal',
                                                                                                      'invscaling', 'adaptive'])
        sgd_combobox3.place(x=610,y=320)
        ttk.Label(self.frame, text='eta0', font=myfont2).place(x=505, y=340)
        sgd_e9 = ttk.Entry(self.frame, textvariable=prev.sgdc_eta0, font=myfont2, width=7)
        sgd_e9.place(x=615,y=340)
        ttk.Label(self.frame, text='power t', font=myfont2).place(x=505, y=360)
        sgd_e10 = ttk.Entry(self.frame, textvariable=prev.sgdc_power_t, font=myfont2, width=7)
        sgd_e10.place(x=615,y=360)
        ttk.Label(self.frame, text='Early stopping', font=myfont2).place(x=505, y=380)
        sgd_cb3 = ttk.Checkbutton(self.frame, variable=prev.sgdc_early_stopping, takefocus=False)
        sgd_cb3.place(x=615,y=380)
        ttk.Label(self.frame, text='Validation fraction', font=myfont2).place(x=505, y=400)
        sgd_e11 = ttk.Entry(self.frame, textvariable=prev.sgdc_validation_fraction, font=myfont2, width=7)
        sgd_e11.place(x=615,y=400)
        ttk.Label(self.frame, text='n iter no change', font=myfont2).place(x=505, y=420)
        sgd_e12 = ttk.Entry(self.frame, textvariable=prev.sgdc_n_iter_no_change, font=myfont2, width=7)
        sgd_e12.place(x=615,y=420)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=505, y=440)
        sgd_cb4 = ttk.Checkbutton(self.frame, variable=prev.sgdc_warm_start, takefocus=False)
        sgd_cb4.place(x=615,y=440)
        ttk.Label(self.frame, text='Average', font=myfont2).place(x=505, y=460)
        sgd_e13 = ttk.Entry(self.frame, textvariable=prev.sgdc_average, font=myfont2, width=7)
        sgd_e13.place(x=615,y=460)

        ttk.Label(self.frame, text='Gaussian Process', font=myfont_b).place(x=705, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=705, y=40)
        gp_cb3 = ttk.Checkbutton(self.frame, variable=prev.gpc_include_comp, takefocus=False)
        gp_cb3.place(x=795, y=50)
        ttk.Label(self.frame, text='n restarts\noptimizer', font=myfont2).place(x=695, y=80)
        gp_e1 = ttk.Entry(self.frame, textvariable=prev.gpc_n_restarts_optimizer, font=myfont2, width=7)
        gp_e1.place(x=795,y=90)
        ttk.Label(self.frame, text='max iter predict', font=myfont2).place(x=695, y=120)
        gp_e2 = ttk.Entry(self.frame, textvariable=prev.gpc_max_iter_predict, font=myfont2, width=7)
        gp_e2.place(x=795,y=120)
        ttk.Label(self.frame, text='Warm predict', font=myfont2).place(x=695, y=140)
        gp_cb1 = ttk.Checkbutton(self.frame, variable=prev.gpc_warm_start, takefocus=False)
        gp_cb1.place(x=795,y=140)
        ttk.Label(self.frame, text='Copy X train', font=myfont2).place(x=695, y=160)
        gp_cb2 = ttk.Checkbutton(self.frame, variable=prev.gpc_copy_X_train, takefocus=False)
        gp_cb2.place(x=795,y=160)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=695, y=180)
        gp_e3 = ttk.Entry(self.frame, textvariable=prev.gpc_random_state, font=myfont2, width=7)
        gp_e3.place(x=795,y=180)
        ttk.Label(self.frame, text='Multi class', font=myfont2).place(x=695, y=200)
        gp_combobox1 = ttk.Combobox(self.frame, textvariable=prev.gpc_multi_class, width=9, values=['one_vs_rest', 'one_vs_one'])
        gp_combobox1.place(x=785,y=200)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=695, y=220)
        gp_e4 = ttk.Entry(self.frame, textvariable=prev.gpc_n_jobs, font=myfont2, width=7)
        gp_e4.place(x=795,y=220)

        ttk.Label(self.frame, text='Nearest Neighbor', font=myfont_b).place(x=705, y=250)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=705, y=280)
        kn_cb1 = ttk.Checkbutton(self.frame, variable=prev.knc_include_comp, takefocus=False)
        kn_cb1.place(x=795, y=280)
        ttk.Label(self.frame, text='n neighbors', font=myfont2).place(x=695, y=320)
        kn_e1 = ttk.Entry(self.frame, textvariable=prev.knc_n_neighbors, font=myfont2, width=7)
        kn_e1.place(x=795,y=320)
        ttk.Label(self.frame, text='Weights', font=myfont2).place(x=695, y=340)
        gp_combobox1 = ttk.Combobox(self.frame, textvariable=prev.knc_weights, width=7, values=['uniform', 'distance'])
        gp_combobox1.place(x=795,y=340)
        ttk.Label(self.frame, text='Algorithm', font=myfont2).place(x=695, y=360)
        gp_combobox2 = ttk.Combobox(self.frame, textvariable=prev.knc_algorithm, width=7, values=['auto', 'ball_tree',
                                                                                                'kd_tree', 'brute'])
        gp_combobox2.place(x=795,y=360)
        ttk.Label(self.frame, text='Leaf size', font=myfont2).place(x=695, y=380)
        kn_e2 = ttk.Entry(self.frame, textvariable=prev.knc_leaf_size, font=myfont2, width=7)
        kn_e2.place(x=795,y=380)
        ttk.Label(self.frame, text='p', font=myfont2).place(x=695, y=400)
        kn_e3 = ttk.Entry(self.frame, textvariable=prev.knc_p, font=myfont2, width=7)
        kn_e3.place(x=795,y=400)
        ttk.Label(self.frame, text='Metric', font=myfont2).place(x=695, y=420)
        gp_combobox3 = ttk.Combobox(self.frame, textvariable=prev.knc_metric, width=9, values=['euclidean', 'manhattan', 
                                                                                             'chebyshev', 'minkowski',
                                                                                             'wminkowski', 'seuclidean', 
                                                                                             'mahalanobis'])
        gp_combobox3.place(x=785,y=420)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=695, y=440)
        kn_e4 = ttk.Entry(self.frame, textvariable=prev.knc_n_jobs, font=myfont2, width=7)
        kn_e4.place(x=795,y=440)

        ttk.Label(self.frame, text='Multi-layer Perceptron', font=myfont_b).place(x=875, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=895, y=40)
        mlp_cb1 = ttk.Checkbutton(self.frame, variable=prev.mlpc_include_comp, takefocus=False)
        mlp_cb1.place(x=985, y=50)
        ttk.Label(self.frame, text='hidden layer\nsizes', font=myfont2).place(x=875, y=80)
        mlp_e1 = ttk.Entry(self.frame, textvariable=prev.mlpc_hidden_layer_sizes, font=myfont2, width=7)
        mlp_e1.place(x=985,y=90)
        ttk.Label(self.frame, text='Activation', font=myfont2).place(x=875, y=120)
        mlp_combobox1 = ttk.Combobox(self.frame, textvariable=prev.mlpc_activation, width=7, values=['identity', 'logistic',
                                                                                                   'tanh', 'relu'])
        mlp_combobox1.place(x=985,y=120)
        ttk.Label(self.frame, text='Solver', font=myfont2).place(x=875, y=140)
        mlp_combobox2 = ttk.Combobox(self.frame, textvariable=prev.mlpc_solver, width=7, values=['lbfgs', 'sgd', 'adam'])
        mlp_combobox2.place(x=985,y=140)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=875, y=160)
        mlp_e2 = ttk.Entry(self.frame, textvariable=prev.mlpc_alpha, font=myfont2, width=7)
        mlp_e2.place(x=985,y=160)
        ttk.Label(self.frame, text='Batch size', font=myfont2).place(x=875, y=180)
        mlp_e3 = ttk.Entry(self.frame, textvariable=prev.mlpc_batch_size, font=myfont2, width=7)
        mlp_e3.place(x=985,y=180)
        ttk.Label(self.frame, text='Learning rate', font=myfont2).place(x=875, y=200)
        mlp_combobox3 = ttk.Combobox(self.frame, textvariable=prev.mlpc_learning_rate, width=7, values=['constant', 'invscaling',
                                                                                                      'adaptive'])
        mlp_combobox3.place(x=985,y=200)
        ttk.Label(self.frame, text='Learning\nrate init', font=myfont2).place(x=875, y=220)
        mlp_e4 = ttk.Entry(self.frame, textvariable=prev.mlpc_learning_rate_init, font=myfont2, width=7)
        mlp_e4.place(x=985,y=230)
        ttk.Label(self.frame, text='Power t', font=myfont2).place(x=875, y=260)
        mlp_e5 = ttk.Entry(self.frame, textvariable=prev.mlpc_power_t, font=myfont2, width=7)
        mlp_e5.place(x=985,y=260)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=875, y=280)
        mlp_e6 = ttk.Entry(self.frame, textvariable=prev.mlpc_max_iter, font=myfont2, width=7)
        mlp_e6.place(x=985,y=280)
        ttk.Label(self.frame, text='Shuffle', font=myfont2).place(x=875, y=300)
        mlp_cb2 = ttk.Checkbutton(self.frame, variable=prev.mlpc_shuffle, takefocus=False)
        mlp_cb2.place(x=985, y=300)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=875, y=320)
        mlp_e7 = ttk.Entry(self.frame, textvariable=prev.mlpc_random_state, font=myfont2, width=7)
        mlp_e7.place(x=985,y=320)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=875, y=340)
        mlp_e8 = ttk.Entry(self.frame, textvariable=prev.mlpc_tol, font=myfont2, width=7)
        mlp_e8.place(x=985,y=340)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=875, y=360)
        mlp_cb3 = ttk.Checkbutton(self.frame, variable=prev.mlpc_verbose, takefocus=False)
        mlp_cb3.place(x=985, y=360)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=875, y=380)
        mlp_cb4 = ttk.Checkbutton(self.frame, variable=prev.mlpc_warm_start, takefocus=False)
        mlp_cb4.place(x=985, y=380)
        ttk.Label(self.frame, text='Momentum', font=myfont2).place(x=875, y=400)
        mlp_e9 = ttk.Entry(self.frame, textvariable=prev.mlpc_momentum, font=myfont2, width=7)
        mlp_e9.place(x=985,y=400)
        ttk.Label(self.frame, text='Nesterovs\nmomentum', font=myfont2).place(x=875, y=420)
        mlp_cb5 = ttk.Checkbutton(self.frame, variable=prev.mlpc_nesterovs_momentum, takefocus=False)
        mlp_cb5.place(x=985, y=430)
        ttk.Label(self.frame, text='Early stopping', font=myfont2).place(x=875, y=460)
        mlp_cb6 = ttk.Checkbutton(self.frame, variable=prev.mlpc_early_stopping, takefocus=False)
        mlp_cb6.place(x=985, y=460)
        ttk.Label(self.frame, text='Validation fraction', font=myfont2).place(x=875, y=480)
        mlp_e10 = ttk.Entry(self.frame, textvariable=prev.mlpc_validation_fraction, font=myfont2, width=7)
        mlp_e10.place(x=985,y=480)
        ttk.Label(self.frame, text='Beta 1', font=myfont2).place(x=875, y=500)
        mlp_e11 = ttk.Entry(self.frame, textvariable=prev.mlpc_beta_1, font=myfont2, width=7)
        mlp_e11.place(x=985,y=500)
        ttk.Label(self.frame, text='Beta 2', font=myfont2).place(x=875, y=520)
        mlp_e12 = ttk.Entry(self.frame, textvariable=prev.mlpc_beta_2, font=myfont2, width=7)
        mlp_e12.place(x=985,y=520)
        ttk.Label(self.frame, text='Epsilon', font=myfont2).place(x=875, y=540)
        mlp_e13 = ttk.Entry(self.frame, textvariable=prev.mlpc_epsilon, font=myfont2, width=7)
        mlp_e13.place(x=985,y=540)
        ttk.Label(self.frame, text='n iter no change', font=myfont2).place(x=875, y=560)
        mlp_e14 = ttk.Entry(self.frame, textvariable=prev.mlpc_n_iter_no_change, font=myfont2, width=7)
        mlp_e14.place(x=985,y=560)
        ttk.Label(self.frame, text='Max fun', font=myfont2).place(x=875, y=580)
        mlp_e15 = ttk.Entry(self.frame, textvariable=prev.mlpc_max_fun, font=myfont2, width=7)
        mlp_e15.place(x=985,y=580)

        ttk.Button(self.root, text='OK', command=lambda: quit_back(self.root, prev.root)).place(relx=0.85, rely=0.92)

        self.root.lift()
        
class rgr_app:
    class training:
        data = None
        sheet = tk.StringVar()
        sheet.set('1')
        header_var = tk.IntVar()
        header_var.set(1)
        x_from_var = tk.StringVar()
        x_to_var = tk.StringVar()
    class prediction:
        data = None
        sheet = tk.StringVar()
        sheet.set('1')
        header_var = tk.IntVar()
        header_var.set(1)
        x_from_var = tk.StringVar()
        x_to_var = tk.StringVar()
    class Comp_results:
        def __init__(self, prev, parent):
            self.root = tk.Toplevel(parent)
            w = 250
            h = 450
            #setting main window's parameters       
            x = (ws/2) - (w/2)
            y = (hs/2) - (h/2) - 30
            self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
            self.root.lift()
            self.root.grab_set()
            self.root.focus_force()
            self.root.resizable(False, False)
            self.root.title('Comparison results')

            self.frame = ttk.Frame(self.root, width=w, height=h)
            self.frame.place(x=0, y=0)

            ttk.Label(self.frame, text='The results of regression\n methods comparison', font=myfont).place(x=10,y=10)
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

            if 'lr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['lr']), font=myfont1).place(x=200,y=60)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=60)
            if 'rr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['rr']), font=myfont1).place(x=200,y=90)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=90)
            if 'lassor' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['lassor']), font=myfont1).place(x=200,y=120)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=120)
            if 'rfr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['rfr']), font=myfont1).place(x=200,y=150)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=150)
            if 'svr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['svr']), font=myfont1).place(x=200,y=180)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=180)
            if 'sgdr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['sgdr']), font=myfont1).place(x=200,y=210)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=210)
            if 'knr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['knr']), font=myfont1).place(x=200,y=240)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=240)
            if 'gpr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['gpr']), font=myfont1).place(x=200,y=270)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=270)
            if 'dtr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['dtr']), font=myfont1).place(x=200,y=300)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=300)
            if 'mlpr' in prev.scores.keys():
                ttk.Label(self.frame, text="{:.3f}". format(prev.scores['mlpr']), font=myfont1).place(x=200,y=330)
            else:
                ttk.Label(self.frame, text="nan", font=myfont1).place(x=200,y=330)

            ttk.Button(self.frame, text='OK', command=lambda: quit_back(self.root, prev.root)).place(x=110, y=410)
    def __init__(self, parent):
        w = 600
        h = 500
        
        #setting main window's parameters       
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2) - 30
        self.root = tk.Toplevel(parent)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.title('Monkey Regression')
        self.root.lift()
        self.root.tkraise()
        self.root.focus_force()
        self.root.resizable(False, False)
        
        self.frame = ttk.Frame(self.root, width=w, height=h)
        self.frame.place(x=0, y=0)
        
        ttk.Label(self.frame, text='Train Data file:', font=myfont1).place(x=10, y=10)
        e1 = ttk.Entry(self.frame, font=myfont1, width=40)
        e1.place(x=120, y=13)
#         e1.insert(0, "C:/Users/csded/Documents/Python/Anaconda/Machine Learning/data/Pokemon.xlsx")
        e2 = ttk.Entry(self.frame, font=myfont1, width=4)
        e2.place(x=150, y=203)
        e2.insert(0, "5")

        ttk.Label(self.frame, text='Number of repeats:', font=myfont1).place(x=200, y=200)
        rep_entry = ttk.Entry(self.frame, font=myfont1, width=4)
        rep_entry.place(x=330, y=203)
        rep_entry.insert(0, "1")
        
        ttk.Button(self.frame, text='Choose file', command=lambda: open_file(self, e1)).place(x=490, y=10)
        
        parent.y_var = tk.StringVar()
        parent.x_st_var = tk.StringVar()
        
        parent.dummies_var = tk.IntVar()
        parent.dummies_var.set(0)
        
        ttk.Label(self.frame, text='List number').place(x=120,y=50)
        training_sheet_entry = ttk.Entry(self.frame, textvariable=self.training.sheet, font=myfont1, width=3)
        training_sheet_entry.place(x=210,y=52)
                
        ttk.Button(self.frame, text='Load data ', command=lambda: load_data(self, self.training, e1, 'training')).place(x=490, y=50)
        
        cb2 = ttk.Checkbutton(self.frame, text="Dummies", variable=parent.dummies_var, takefocus=False)
        cb1 = ttk.Checkbutton(self.frame, text="header", variable=self.training.header_var, takefocus=False)
        cb1.place(x=10, y=50)
        
        ttk.Label(self.frame, text='Data status:').place(x=10, y=100)
        self.tr_data_status = ttk.Label(self.frame, text='Not Loaded')
        self.tr_data_status.place(x=120, y=100)

        ttk.Button(self.frame, text='View/Change', 
            command=lambda: Data_Preview(self, self.training, 'training', parent)).place(x=230, y=100)

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
        self.rfr_n_jobs = tk.StringVar(value='None')
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
        self.mlpr_hidden_layer_sizes = tk.StringVar(value='100')
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

        def compare_methods(prev, main):
            if master.dummies_var.get()==0:
                try:
                    prev.X = main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())]
                    prev.y = main.data[parent.y_var.get()]
                except:
                    prev.X = main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())]
                    prev.y = main.data[int(parent.y_var.get())]
            elif master.dummies_var.get()==1:
                try:
                    X=main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())]
                except:
                    X=main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())]
                try:
                    prev.X = pd.get_dummies(X)
                    prev.y = main.data[parent.y_var.get()]
                except:
                    prev.X = pd.get_dummies(X)
                    prev.y = main.data[int(parent.y_var.get())]
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            prev.X_St = scaler.fit_transform(prev.X)
            self.scores = {}
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import RepeatedKFold
            folds = RepeatedKFold(n_splits = int(e2.get()), n_repeats=int(rep_entry.get()), random_state = 100)
            # Linear regression
            if self.lr_include_comp.get():
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression(fit_intercept=self.lr_fit_intercept.get(), normalize=self.lr_normalize.get(),
                                      copy_X=self.lr_copy_X.get(), 
                                      n_jobs=(int(self.lr_n_jobs.get()) if (self.lr_n_jobs.get() != 'None') else None))
                if master.x_st_var.get() == 'everywhere':
                    if self.lr_function_type.get() == 'Linear':
                        lr.fit(prev.X_St, prev.y)
                        lr_scores = cross_val_score(lr, prev.X_St, prev.y, scoring='r2', cv=folds)
                    elif self.lr_function_type.get() == 'Polynomial':
                        from sklearn.preprocessing import PolynomialFeatures
                        lr.fit(PolynomialFeatures(degree=2).fit_transform(prev.X_St), prev.y)
                        lr_scores = cross_val_score(lr, PolynomialFeatures(degree=2).fit_transform(prev.X_St), 
                                                    prev.y, scoring='r2', cv=folds)
                    elif self.lr_function_type.get() == 'Exponential':
                        lr.fit(prev.X_St, np.log(prev.y))
                        lr_scores = cross_val_score(lr, prev.X_St, np.log(prev.y), scoring='r2', cv=folds)
                    elif self.lr_function_type.get() == 'Power':
                        lr.fit(np.log(prev.X_St), np.log(prev.y))
                        lr_scores = cross_val_score(lr, np.log(prev.X_St), np.log(prev.y), scoring='r2', cv=folds)
                    elif self.lr_function_type.get() == 'Logarithmic':
                        lr.fit(np.log(prev.X_St), prev.y)
                        lr_scores = cross_val_score(lr, np.log(prev.X_St), prev.y, scoring='r2', cv=folds)
                else:
                    if self.lr_function_type.get() == 'Linear':
                        lr.fit(prev.X, prev.y)
                        lr_scores = cross_val_score(lr, prev.X, prev.y, scoring='r2', cv=folds)
                    elif self.lr_function_type.get() == 'Polynomial':
                        from sklearn.preprocessing import PolynomialFeatures
                        lr.fit(PolynomialFeatures(degree=2).fit_transform(prev.X), prev.y)
                        lr_scores = cross_val_score(lr, PolynomialFeatures(degree=2).fit_transform(prev.X), 
                                                    prev.y, scoring='r2', cv=folds)
                    elif self.lr_function_type.get() == 'Exponential':
                        lr.fit(prev.X, np.log(prev.y))
                        lr_scores = cross_val_score(lr, prev.X, np.log(prev.y), scoring='r2', cv=folds)
                    elif self.lr_function_type.get() == 'Power':
                        lr.fit(np.log(prev.X), np.log(prev.y))
                        lr_scores = cross_val_score(lr, np.log(prev.X), np.log(prev.y), scoring='r2', cv=folds)
                    elif self.lr_function_type.get() == 'Logarithmic':
                        lr.fit(np.log(prev.X), prev.y)
                        lr_scores = cross_val_score(lr, np.log(prev.X), prev.y, scoring='r2', cv=folds)
                self.scores['lr'] = lr_scores.mean()
            # Ridge regression
            if self.rr_include_comp.get():
                from sklearn.linear_model import Ridge
                rr = Ridge(alpha=float(self.rr_alpha.get()), fit_intercept=self.rr_fit_intercept.get(), normalize=self.rr_normalize.get(),
                           copy_X=self.rr_copy_X.get(), 
                           max_iter=(int(self.rr_max_iter.get()) if (self.rr_max_iter.get() != 'None') else None),
                           tol=float(self.rr_tol.get()), solver=self.rr_solver.get(),
                           random_state=(int(self.rr_random_state.get()) if (self.rr_random_state.get() != 'None') else None))
                if master.x_st_var.get() == 'everywhere':
                    rr.fit(prev.X_St, prev.y)
                    rr_scores = cross_val_score(rr, prev.X_St, prev.y, scoring='r2', cv=folds)
                else:
                    rr.fit(prev.X, prev.y)
                    rr_scores = cross_val_score(rr, prev.X, prev.y, scoring='r2', cv=folds)
                self.scores['rr'] = rr_scores.mean()
            # Lasso regression
            if self.lassor_include_comp.get():
                from sklearn.linear_model import Lasso
                lassor = Lasso(alpha=float(self.lassor_alpha.get()), fit_intercept=self.lassor_fit_intercept.get(),
                               normalize=self.lassor_normalize.get(), precompute=self.lassor_precompute.get(), 
                               copy_X=self.lassor_copy_X.get(), max_iter=(int(self.lassor_max_iter.get()) 
                                                                          if (self.lassor_max_iter.get() != 'None') else None),
                               tol=float(self.lassor_tol.get()), warm_start=self.lassor_warm_start.get(),
                               positive=self.lassor_positive.get(),
                               random_state=(int(self.lassor_random_state.get()) if (self.lassor_random_state.get() != 'None') else None),
                               selection=self.lassor_selection.get())
                if master.x_st_var.get() == 'None':
                    lassor.fit(prev.X, prev.y)
                    lassor_scores = cross_val_score(lassor, prev.X, prev.y, scoring='r2', cv=folds)
                else:
                    lassor.fit(prev.X_St, prev.y)
                    lassor_scores = cross_val_score(lassor, prev.X_St, prev.y, scoring='r2', cv=folds)
                self.scores['lassor'] = lassor_scores.mean()
            # Random forest regression
            if self.rfr_include_comp.get():
                from sklearn.ensemble import RandomForestRegressor
                rfr = RandomForestRegressor(n_estimators=int(self.rfr_n_estimators.get()), criterion=self.rfr_criterion.get(),
                                            max_depth=(int(self.rfr_max_depth.get()) if (self.rfr_max_depth.get() != 'None') else None), 
                                            min_samples_split=(float(self.rfr_min_samples_split.get()) if '.' in self.rfr_min_samples_split.get()
                                                               else int(self.rfr_min_samples_split.get())),
                                            min_samples_leaf=(float(self.rfr_min_samples_leaf.get()) if '.' in self.rfr_min_samples_leaf.get()
                                                               else int(self.rfr_min_samples_leaf.get())),
                                            min_weight_fraction_leaf=float(self.rfr_min_weight_fraction_leaf.get()),
                                            max_features=(float(self.rfr_max_features.get()) if '.' in self.rfr_max_features.get() 
                                                           else int(self.rfr_max_features.get()) if len(self.rfr_max_features.get()) < 4 
                                                           else self.rfr_max_features.get() if (self.rfr_max_features.get() != 'None') 
                                                           else None),
                                            max_leaf_nodes=(int(self.rfr_max_leaf_nodes.get()) if (self.rfr_max_leaf_nodes.get() != 'None') else None),
                                            min_impurity_decrease=float(self.rfr_min_impurity_decrease.get()),
                                            bootstrap=self.rfr_bootstrap.get(), oob_score=self.rfr_oob_score.get(),
                                            n_jobs=(int(self.rfr_n_jobs.get()) if (self.rfr_n_jobs.get() != 'None') else None),
                                            random_state=(int(self.rfr_random_state.get()) if
                                                          (self.rfr_random_state.get() != 'None') else None),
                                            verbose=int(self.rfr_verbose.get()), warm_start=self.rfr_warm_start.get(),
                                            ccp_alpha=float(self.rfr_ccp_alpha.get()),
                                            max_samples=(float(self.rfr_max_samples.get()) if '.' in self.rfr_max_samples.get() 
                                                           else int(self.rfr_max_samples.get()) if (self.rfr_max_samples.get() != 'None') 
                                                           else None))
                if master.x_st_var.get() == 'everywhere':
                    rfr.fit(prev.X_St, prev.y)
                    rfr_scores = cross_val_score(rfr, prev.X_St, prev.y, scoring='r2', cv=folds)
                else:
                    rfr.fit(prev.X, prev.y)
                    rfr_scores = cross_val_score(rfr, prev.X, prev.y, scoring='r2', cv=folds)
                self.scores['rfr'] = rfr_scores.mean()
            # Support vector
            if self.svr_include_comp.get():
                from sklearn.svm import SVR
                svr = SVR(kernel=self.svr_kernel.get(), degree=int(self.svr_degree.get()), 
                          gamma=(float(self.svr_gamma.get()) if '.' in self.svr_gamma.get() else self.svr_gamma.get()),
                          coef0=float(self.svr_coef0.get()), tol=float(self.svr_tol.get()), 
                          C=float(self.svr_C.get()), epsilon=float(self.svr_epsilon.get()), 
                          shrinking=self.svr_shrinking.get(), cache_size=float(self.svr_cache_size.get()), 
                          verbose=self.svr_verbose.get(), max_iter=int(self.svr_max_iter.get()))
                if master.x_st_var.get() == 'None':
                    svr.fit(prev.X, prev.y)
                    svr_scores = cross_val_score(svr, prev.X, prev.y, scoring='r2', cv=folds)
                else:
                    svr.fit(prev.X_St, prev.y)
                    svr_scores = cross_val_score(svr, prev.X_St, prev.y, scoring='r2', cv=folds)
                self.scores['svr'] = svr_scores.mean()
            # Stochastic Gradient Descent
            if self.sgdr_include_comp.get():
                from sklearn.linear_model import SGDRegressor
                sgdr = SGDRegressor(loss=self.sgdr_loss.get(), penalty=self.sgdr_penalty.get(),
                                    alpha=float(self.sgdr_alpha.get()), l1_ratio=float(self.sgdr_l1_ratio.get()),
                                    fit_intercept=self.sgdr_fit_intercept.get(), max_iter=int(self.sgdr_max_iter.get()),
                                    tol=float(self.sgdr_tol.get()), shuffle=self.sgdr_shuffle.get(), 
                                    verbose=int(self.sgdr_verbose.get()), epsilon=float(self.sgdr_epsilon.get()),
                                    random_state=(int(self.sgdr_random_state.get()) if (self.sgdr_random_state.get() 
                                                                                           != 'None') else None),
                                    learning_rate=self.sgdr_learning_rate.get(), eta0=float(self.sgdr_eta0.get()),
                                    power_t=float(self.sgdr_power_t.get()), early_stopping=self.sgdr_early_stopping.get(),
                                    validation_fraction=float(self.sgdr_validation_fraction.get()),
                                    n_iter_no_change=int(self.sgdr_n_iter_no_change.get()), warm_start=self.sgdr_warm_start.get(),
                                    average=(True if self.sgdr_average.get()=='True' else False 
                                             if self.sgdr_average.get()=='False' else int(self.sgdr_average.get())))
                if master.x_st_var.get() == 'None':
                    sgdr.fit(prev.X, prev.y)
                    sgdr_scores = cross_val_score(sgdr, prev.X, prev.y, scoring='r2', cv=folds)
                else:
                    sgdr.fit(prev.X_St, prev.y)
                    sgdr_scores = cross_val_score(sgdr, prev.X_St, prev.y, scoring='r2', cv=folds)
                self.scores['sgdr'] = sgdr_scores.mean()
            # Nearest Neighbor
            if self.knr_include_comp.get():
                from sklearn.neighbors import KNeighborsRegressor
                knr = KNeighborsRegressor(n_neighbors=int(self.knr_n_neighbors.get()), 
                                          weights=self.knr_weights.get(), algorithm=self.knr_algorithm.get(),
                                          leaf_size=int(self.knr_leaf_size.get()), p=int(self.knr_p.get()),
                                          metric=self.knr_metric.get(), 
                                          n_jobs=(int(self.knr_n_jobs.get()) if (self.knr_n_jobs.get() 
                                                                                                != 'None') else None))
                if master.x_st_var.get() == 'None':
                    knr.fit(prev.X, prev.y)
                    knr_scores = cross_val_score(knr, prev.X, prev.y, scoring='r2', cv=folds)
                else:
                    knr.fit(prev.X_St, prev.y)
                    knr_scores = cross_val_score(knr, prev.X_St, prev.y, scoring='r2', cv=folds)
                self.scores['knr'] = knr_scores.mean()
            # Gaussian Process
            if self.gpr_include_comp.get():
                from sklearn.gaussian_process import GaussianProcessRegressor
                gpr = GaussianProcessRegressor(alpha=float(self.gpr_alpha.get()),
                                               n_restarts_optimizer=int(self.gpr_n_restarts_optimizer.get()),
                                               normalize_y=self.gpr_normalize_y.get(), copy_X_train=self.gpr_copy_X_train.get(),
                                               random_state=(int(self.gpr_random_state.get()) if 
                                                             (self.gpr_random_state.get() != 'None') else None))
                if master.x_st_var.get() == 'None':
                    gpr.fit(prev.X, prev.y)
                    gpr_scores = cross_val_score(gpr, prev.X, prev.y, scoring='r2', cv=folds)
                else:
                    gpr.fit(prev.X_St, prev.y)
                    gpr_scores = cross_val_score(gpr, prev.X_St, prev.y, scoring='r2', cv=folds)
                self.scores['gpr'] = gpr_scores.mean()
            # Decision Tree regression
            if self.dtr_include_comp.get():
                from sklearn.tree import DecisionTreeRegressor
                dtr = DecisionTreeRegressor(criterion=self.dtr_criterion.get(), splitter=self.dtr_splitter.get(), 
                                            max_depth=(int(self.dtr_max_depth.get()) if (self.dtr_max_depth.get() != 'None') else None), 
                                            min_samples_split=(float(self.dtr_min_samples_split.get()) if '.' in self.dtr_min_samples_split.get()
                                                              else int(self.dtr_min_samples_split.get())), 
                                            min_samples_leaf=(float(self.dtr_min_samples_leaf.get()) if '.' in self.dtr_min_samples_leaf.get()
                                                             else int(self.dtr_min_samples_leaf.get())),
                                            min_weight_fraction_leaf=float(self.dtr_min_weight_fraction_leaf.get()),
                                            max_features=(float(self.dtr_max_features.get()) if '.' in self.dtr_max_features.get() 
                                                          else int(self.dtr_max_features.get()) if len(self.dtr_max_features.get()) < 4 
                                                          else self.dtr_max_features.get() if (self.dtr_max_features.get() != 'None') 
                                                          else None), 
                                            random_state=(int(self.dtr_random_state.get()) if (self.dtr_random_state.get() != 'None') else None),
                                            max_leaf_nodes=(int(self.dtr_max_leaf_nodes.get()) if (self.dtr_max_leaf_nodes.get() != 'None') else None),
                                            min_impurity_decrease=float(self.dtr_min_impurity_decrease.get()),
                                            ccp_alpha=float(self.dtr_ccp_alpha.get()))
                if master.x_st_var.get() == 'everywhere':
                    dtr.fit(prev.X_St, prev.y)
                    dtr_scores = cross_val_score(dtr, prev.X_St, prev.y, scoring='r2', cv=folds)
                else:
                    dtr.fit(prev.X, prev.y)
                    dtr_scores = cross_val_score(dtr, prev.X, prev.y, scoring='r2', cv=folds)
                self.scores['dtr'] = dtr_scores.mean()
            # MLP
            if self.mlpr_include_comp.get():
                from sklearn.neural_network import MLPRegressor
                mlpr = MLPRegressor(hidden_layer_sizes=tuple([int(self.mlpr_hidden_layer_sizes.get())]),
                                    activation=self.mlpr_activation.get(),solver=self.mlpr_solver.get(),
                                    alpha=float(self.mlpr_alpha.get()), batch_size=(int(self.mlpr_batch_size.get()) if 
                                                                 (self.mlpr_batch_size.get() != 'auto') else 'auto'),
                                    learning_rate=self.mlpr_learning_rate.get(), 
                                    learning_rate_init=float(self.mlpr_learning_rate_init.get()),
                                    power_t=float(self.mlpr_power_t.get()), max_iter=int(self.mlpr_max_iter.get()),
                                    shuffle=self.mlpr_shuffle.get(),
                                    random_state=(int(self.mlpr_random_state.get()) if 
                                                                 (self.mlpr_random_state.get() != 'None') else None),
                                    tol=float(self.mlpr_tol.get()), verbose=self.mlpr_verbose.get(),
                                    warm_start=self.mlpr_warm_start.get(), momentum=float(self.mlpr_momentum.get()),
                                    nesterovs_momentum=self.mlpr_nesterovs_momentum.get(),
                                    early_stopping=self.mlpr_early_stopping.get(), 
                                    validation_fraction=float(self.mlpr_validation_fraction.get()),
                                    beta_1=float(self.mlpr_beta_1.get()), beta_2=float(self.mlpr_beta_2.get()),
                                    epsilon=float(self.mlpr_epsilon.get()), 
                                    n_iter_no_change=int(self.mlpr_n_iter_no_change.get()),
                                    max_fun=int(self.mlpr_max_fun.get()))
                if master.x_st_var.get() == 'everywhere':
                    mlpr.fit(prev.X_St, prev.y)
                    mlpr_scores = cross_val_score(mlpr, prev.X_St, prev.y, scoring='r2', cv=folds)
                else:
                    mlpr.fit(prev.X, prev.y)
                    mlpr_scores = cross_val_score(mlpr, prev.X, prev.y, scoring='r2', cv=folds)
                self.scores['mlpr'] = mlpr_scores.mean()
            self.show_res_button.place(x=400, y=180)

        ttk.Button(self.frame, text="Methods' specifications", 
                  command=lambda: rgr_mtds_specification(self, parent)).place(x=400, y=100)
        ttk.Button(self.frame, text='Perform comparison', command=lambda: compare_methods(self, self.training)).place(x=400, y=140)
        self.show_res_button = ttk.Button(self.frame, text='Show Results', command=lambda: self.Comp_results(self, parent))
        
        ttk.Label(self.frame, text='Choose y', font=myfont1).place(x=30, y=140)
        parent.y_var = tk.StringVar()
        self.combobox1 = ttk.Combobox(self.frame, textvariable=parent.y_var, width=13, values=[])
        self.combobox1.place(x=100,y=140)
        
        ttk.Label(self.frame, text='X from', font=myfont1).place(x=205, y=140)
        self.tr_x_from_combobox = ttk.Combobox(self.frame, textvariable=self.training.x_from_var, width=4, values=[])
        self.tr_x_from_combobox.place(x=255, y=140)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=305, y=140)
        self.tr_x_to_combobox = ttk.Combobox(self.frame, textvariable=self.training.x_to_var, width=4, values=[])
        self.tr_x_to_combobox.place(x=325, y=140)
        self.combobox2 = ttk.Combobox(self.frame, textvariable=parent.x_st_var, width=13,
                                        values=['None', 'Where needed', 'everywhere'])
        self.combobox2.current(1)
        self.combobox2.place(x=150,y=175)
        ttk.Label(self.frame, text='X Standartization', font=myfont1).place(x=30, y=170)
        ttk.Label(self.frame, text='Number of folds:', font=myfont1).place(x=30, y=200)
        cb2.place(x=270, y=175)

        ttk.Label(self.frame, text='Predict Data file:', font=myfont1).place(x=10, y=250)
        pr_data_entry = ttk.Entry(self.frame, font=myfont1, width=38)
        pr_data_entry.place(x=140, y=250)
        # pr_data_entry.insert(0, "C:/Users/csded/Documents/Python/Anaconda/Machine Learning/data/Pokemon.xlsx")
        
        ttk.Button(self.frame, text='Choose file', command=lambda: open_file(self, pr_data_entry)).place(x=490, y=245)
        
        parent.x2_st_var = tk.StringVar()
        
        ttk.Label(self.frame, text='List number', font=myfont1).place(x=120,y=295)
        pr_sheet_entry = ttk.Entry(self.frame, textvariable=self.prediction.sheet, font=myfont1, width=3)
        pr_sheet_entry.place(x=210,y=297)
        
        ttk.Button(self.frame, text='Load data ', command=lambda: load_data(self, self.prediction, pr_data_entry, 'prediction')).place(x=490, y=295)
        
        cb4 = ttk.Checkbutton(self.frame, text="header", variable=self.prediction.header_var, takefocus=False)
        cb4.place(x=10, y=290)
        
        ttk.Label(self.frame, text='Data status:', font=myfont).place(x=10, y=350)
        self.pr_data_status = ttk.Label(self.frame, text='Not Loaded', font=myfont)
        self.pr_data_status.place(x=120, y=350)

        ttk.Button(self.frame, text='View/Change', command=lambda: Data_Preview(self, self.prediction, 'prediction', parent)).place(x=230, y=350)
        
        parent.pr_method = tk.StringVar()
        parent.pr_method.set('Least squares')
        self.combobox9 = ttk.Combobox(self.frame, textvariable=parent.pr_method, values=['Least squares', 'Ridge', 'Lasso', 
                                                                                         'Random Forest', 'Support Vector',
                                                                                         'SGD', 'Nearest Neighbor',
                                                                                         'Gaussian Process', 'Decision Tree',
                                                                                         'Multi-layer Perceptron'
                                                                                        ])
        self.combobox9.place(x=200,y=425)
        ttk.Label(self.frame, text='Choose method', font=myfont1).place(x=30, y=420)
        
        ttk.Label(self.frame, text='X from', font=myfont1).place(x=205, y=390)
        self.pr_x_from_combobox = ttk.Combobox(self.frame, textvariable=self.prediction.x_from_var, width=4, values=[])
        self.pr_x_from_combobox.place(x=255, y=390)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=305, y=390)
        self.pr_x_to_combobox = ttk.Combobox(self.frame, textvariable=self.prediction.x_to_var, width=4, values=[])
        self.pr_x_to_combobox.place(x=325, y=390)
        
        self.combobox10 = ttk.Combobox(self.frame, textvariable=parent.x2_st_var, 
                                        values=['No', 'If needed', 'Yes'])
        self.combobox10.current(1)
        self.combobox10.place(x=200,y=455)
        ttk.Label(self.frame, text='X Standartization', font=myfont1).place(x=30, y=450)

        def make_regression(method):
            if master.dummies_var.get()==0:
                try:
                    training_X = self.training.data.iloc[:,(int(self.training.x_from_var.get())-1): 
                                                         int(self.training.x_to_var.get())]
                    training_y = self.training.data[parent.y_var.get()]
                except:
                    training_X = self.training.data.iloc[:,(int(self.training.x_from_var.get())-1): 
                                                         int(self.training.x_to_var.get())]
                    training_y = self.training.data[int(parent.y_var.get())]
                X = self.prediction.data.iloc[:,(int(self.prediction.x_from_var.get())-1): 
                                              int(self.prediction.x_to_var.get())]
            elif master.dummies_var.get()==1:
                X = pd.get_dummies(self.prediction.data.iloc[:,(int(self.prediction.x_from_var.get())-1): 
                                                             int(self.prediction.x_to_var.get())])
                try:
                    training_X=self.training.data.iloc[:,(int(self.training.x_from_var.get())-1): 
                                                         int(self.training.x_to_var.get())]
                except:
                    training_X=self.training.data.iloc[:,(int(self.training.x_from_var.get())-1): 
                                                         int(self.training.x_to_var.get())]
                try:
                    training_X = pd.get_dummies(training_X)
                    training_y = self.training.data[parent.y_var.get()]
                except:
                    training_X = pd.get_dummies(training_X)
                    training_y = self.training.data[int(parent.y_var.get())]
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            X_St = scaler.fit_transform(X)
            training_X_St = scaler.fit_transform(training_X)
            # Linear regression
            if method == 'Least squares':
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression(fit_intercept=self.lr_fit_intercept.get(), normalize=self.lr_normalize.get(),
                                      copy_X=self.lr_copy_X.get(), 
                                      n_jobs=(int(self.lr_n_jobs.get()) if (self.lr_n_jobs.get() != 'None') else None))
                if master.x2_st_var.get() == 'Yes':
                    lr.fit(training_X_st, training_y)
                    pr_values = lr.predict(X_St)
                else:
                    lr.fit(training_X, training_y)
                    pr_values = lr.predict(X)
            # Ridge regression
            if method == 'Ridge':
                from sklearn.linear_model import Ridge
                rr = Ridge(alpha=float(self.rr_alpha.get()), fit_intercept=self.rr_fit_intercept.get(), normalize=self.rr_normalize.get(),
                           copy_X=self.rr_copy_X.get(), 
                           max_iter=(int(self.rr_max_iter.get()) if (self.rr_max_iter.get() != 'None') else None),
                           tol=float(self.rr_tol.get()), solver=self.rr_solver.get(),
                           random_state=(int(self.rr_random_state.get()) if (self.rr_random_state.get() != 'None') else None))
                if master.x2_st_var.get() == 'Yes':
                    rr.fit(training_X_st, training_y)
                    pr_values = rr.predict(X_St)
                else:
                    rr.fit(training_X, training_y)
                    pr_values = rr.predict(X)
            # Lasso regression
            if method == 'Lasso':
                from sklearn.linear_model import Lasso
                lassor = Lasso(alpha=float(self.lassor_alpha.get()), fit_intercept=self.lassor_fit_intercept.get(),
                               normalize=self.lassor_normalize.get(), precompute=self.lassor_precompute.get(), 
                               copy_X=self.lassor_copy_X.get(), max_iter=(int(self.lassor_max_iter.get()) 
                                                                          if (self.lassor_max_iter.get() != 'None') else None),
                               tol=float(self.lassor_tol.get()), warm_start=self.lassor_warm_start.get(),
                               positive=self.lassor_positive.get(),
                               random_state=(int(self.lassor_random_state.get()) if (self.lassor_random_state.get() != 'None') else None),
                               selection=self.lassor_selection.get())
                if master.x2_st_var.get() == 'No':
                    lassor.fit(training_X, training_y)
                    pr_values = lassor.predict(X)
                else:
                    lassor.fit(training_X_St, training_y)
                    pr_values = lassor.predict(X_St)
            # Random forest
            if method == 'Random Forest':
                from sklearn.ensemble import RandomForestRegressor
                rfr = RandomForestRegressor(n_estimators=int(self.rfr_n_estimators.get()), criterion=self.rfr_criterion.get(),
                                            max_depth=(int(self.rfr_max_depth.get()) if (self.rfr_max_depth.get() != 'None') else None), 
                                            min_samples_split=(float(self.rfr_min_samples_split.get()) if '.' in self.rfr_min_samples_split.get()
                                                               else int(self.rfr_min_samples_split.get())),
                                            min_samples_leaf=(float(self.rfr_min_samples_leaf.get()) if '.' in self.rfr_min_samples_leaf.get()
                                                               else int(self.rfr_min_samples_leaf.get())),
                                            min_weight_fraction_leaf=float(self.rfr_min_weight_fraction_leaf.get()),
                                            max_features=(float(self.rfr_max_features.get()) if '.' in self.rfr_max_features.get() 
                                                           else int(self.rfr_max_features.get()) if len(self.rfr_max_features.get()) < 4 
                                                           else self.rfr_max_features.get() if (self.rfr_max_features.get() != 'None') 
                                                           else None),
                                            max_leaf_nodes=(int(self.rfr_max_leaf_nodes.get()) if (self.rfr_max_leaf_nodes.get() != 'None') else None),
                                            min_impurity_decrease=float(self.rfr_min_impurity_decrease.get()),
                                            bootstrap=self.rfr_bootstrap.get(), oob_score=self.rfr_oob_score.get(),
                                            n_jobs=(int(self.rfr_n_jobs.get()) if (self.rfr_n_jobs.get() != 'None') else None),
                                            random_state=(int(self.rfr_random_state.get()) if
                                                          (self.rfr_random_state.get() != 'None') else None),
                                            verbose=int(self.rfr_verbose.get()), warm_start=self.rfr_warm_start.get(),
                                            ccp_alpha=float(self.rfr_ccp_alpha.get()),
                                            max_samples=(float(self.rfr_max_samples.get()) if '.' in self.rfr_max_samples.get() 
                                                           else int(self.rfr_max_samples.get()) if (self.rfr_max_samples.get() != 'None') 
                                                           else None))
                if master.x2_st_var.get() == 'Yes':
                    rfr.fit(training_X_st, training_y)
                    pr_values = rfr.predict(X_St)
                else:
                    rfr.fit(training_X, training_y)
                    pr_values = rfr.predict(X)
            # Support Vector
            if method == 'Support Vector':
                from sklearn.svm import SVR
                svr = SVR(kernel=self.svr_kernel.get(), degree=int(self.svr_degree.get()), 
                          gamma=(float(self.svr_gamma.get()) if '.' in self.svr_gamma.get() else self.svr_gamma.get()),
                          coef0=float(self.svr_coef0.get()), tol=float(self.svr_tol.get()), 
                          C=float(self.svr_C.get()), epsilon=float(self.svr_epsilon.get()), 
                          shrinking=self.svr_shrinking.get(), cache_size=float(self.svr_cache_size.get()), 
                          verbose=self.svr_verbose.get(), max_iter=int(self.svr_max_iter.get()))
                if master.x2_st_var.get() == 'No':
                    svr.fit(training_X, training_y)
                    pr_values = svr.predict(X)
                else:
                    svr.fit(training_X_St, training_y)
                    pr_values = svr.predict(X_St)
            # Support Vector
            if method == 'SGD':
                from sklearn.linear_model import SGDRegressor
                sgdr = SGDRegressor(loss=self.sgdr_loss.get(), penalty=self.sgdr_penalty.get(),
                                    alpha=float(self.sgdr_alpha.get()), l1_ratio=float(self.sgdr_l1_ratio.get()),
                                    fit_intercept=self.sgdr_fit_intercept.get(), max_iter=int(self.sgdr_max_iter.get()),
                                    tol=float(self.sgdr_tol.get()), shuffle=self.sgdr_shuffle.get(), 
                                    verbose=int(self.sgdr_verbose.get()), epsilon=float(self.sgdr_epsilon.get()),
                                    random_state=(int(self.sgdr_random_state.get()) if (self.sgdr_random_state.get() 
                                                                                           != 'None') else None),
                                    learning_rate=self.sgdr_learning_rate.get(), eta0=float(self.sgdr_eta0.get()),
                                    power_t=float(self.sgdr_power_t.get()), early_stopping=self.sgdr_early_stopping.get(),
                                    validation_fraction=float(self.sgdr_validation_fraction.get()),
                                    n_iter_no_change=int(self.sgdr_n_iter_no_change.get()), warm_start=self.sgdr_warm_start.get(),
                                    average=(True if self.sgdr_average.get()=='True' else False 
                                             if self.sgdr_average.get()=='False' else int(self.sgdr_average.get())))
                if master.x2_st_var.get() == 'No':
                    sgdr.fit(training_X, training_y)
                    pr_values = sgdr.predict(X)
                else:
                    sgdr.fit(training_X_St, training_y)
                    pr_values = sgdr.predict(X_St)
            # Nearest Neighbor
            if method == 'Nearest Neighbor':
                from sklearn.neighbors import KNeighborsRegressor
                knr = KNeighborsRegressor(n_neighbors=int(self.knr_n_neighbors.get()), 
                                          weights=self.knr_weights.get(), algorithm=self.knr_algorithm.get(),
                                          leaf_size=int(self.knr_leaf_size.get()), p=int(self.knr_p.get()),
                                          metric=self.knr_metric.get(), 
                                          n_jobs=(int(self.knr_n_jobs.get()) if (self.knr_n_jobs.get() 
                                                                                                != 'None') else None))
                if master.x2_st_var.get() == 'No':
                    knr.fit(training_X, training_y)
                    pr_values = knr.predict(X)
                else:
                    knr.fit(training_X_St, training_y)
                    pr_values = knr.predict(X_St)
            # Gaussian Process
            if method == 'Gaussian Process':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gpr = GaussianProcessRegressor(alpha=float(self.gpr_alpha.get()),
                                               n_restarts_optimizer=int(self.gpr_n_restarts_optimizer.get()),
                                               normalize_y=self.gpr_normalize_y.get(), copy_X_train=self.gpr_copy_X_train.get(),
                                               random_state=(int(self.gpr_random_state.get()) if 
                                                             (self.gpr_random_state.get() != 'None') else None))
                if master.x2_st_var.get() == 'No':
                    gpr.fit(training_X, training_y)
                    pr_values = gpr.predict(X)
                else:
                    gpr.fit(training_X_St, training_y)
                    pr_values = gpr.predict(X_St)
            # Decision Tree
            if method == 'Decision Tree':
                from sklearn.tree import DecisionTreeRegressor
                dtr = DecisionTreeRegressor(criterion=self.dtr_criterion.get(), splitter=self.dtr_splitter.get(), 
                                            max_depth=(int(self.dtr_max_depth.get()) if (self.dtr_max_depth.get() != 'None') else None), 
                                            min_samples_split=(float(self.dtr_min_samples_split.get()) if '.' in self.dtr_min_samples_split.get()
                                                              else int(self.dtr_min_samples_split.get())), 
                                            min_samples_leaf=(float(self.dtr_min_samples_leaf.get()) if '.' in self.dtr_min_samples_leaf.get()
                                                             else int(self.dtr_min_samples_leaf.get())),
                                            min_weight_fraction_leaf=float(self.dtr_min_weight_fraction_leaf.get()),
                                            max_features=(float(self.dtr_max_features.get()) if '.' in self.dtr_max_features.get() 
                                                          else int(self.dtr_max_features.get()) if len(self.dtr_max_features.get()) < 4 
                                                          else self.dtr_max_features.get() if (self.dtr_max_features.get() != 'None') 
                                                          else None), 
                                            random_state=(int(self.dtr_random_state.get()) if (self.dtr_random_state.get() != 'None') else None),
                                            max_leaf_nodes=(int(self.dtr_max_leaf_nodes.get()) if (self.dtr_max_leaf_nodes.get() != 'None') else None),
                                            min_impurity_decrease=float(self.dtr_min_impurity_decrease.get()),
                                            ccp_alpha=float(self.dtr_ccp_alpha.get()))
                if master.x2_st_var.get() == 'Yes':
                    dtr.fit(training_X_st, training_y)
                    pr_values = dtr.predict(X_St)
                else:
                    dtr.fit(training_X, training_y)
                    pr_values = dtr.predict(X)
            # MLP
            if method == 'Multi-layer Perceptron':
                from sklearn.neural_network import MLPRegressor
                mlpr = MLPRegressor(hidden_layer_sizes=tuple([int(self.mlpr_hidden_layer_sizes.get())]),
                                    activation=self.mlpr_activation.get(),solver=self.mlpr_solver.get(),
                                    alpha=float(self.mlpr_alpha.get()), batch_size=(int(self.mlpr_batch_size.get()) if 
                                                                 (self.mlpr_batch_size.get() != 'auto') else 'auto'),
                                    learning_rate=self.mlpr_learning_rate.get(), 
                                    learning_rate_init=float(self.mlpr_learning_rate_init.get()),
                                    power_t=float(self.mlpr_power_t.get()), max_iter=int(self.mlpr_max_iter.get()),
                                    shuffle=self.mlpr_shuffle.get(),
                                    random_state=(int(self.mlpr_random_state.get()) if 
                                                                 (self.mlpr_random_state.get() != 'None') else None),
                                    tol=float(self.mlpr_tol.get()), verbose=self.mlpr_verbose.get(),
                                    warm_start=self.mlpr_warm_start.get(), momentum=float(self.mlpr_momentum.get()),
                                    nesterovs_momentum=self.mlpr_nesterovs_momentum.get(),
                                    early_stopping=self.mlpr_early_stopping.get(), 
                                    validation_fraction=float(self.mlpr_validation_fraction.get()),
                                    beta_1=float(self.mlpr_beta_1.get()), beta_2=float(self.mlpr_beta_2.get()),
                                    epsilon=float(self.mlpr_epsilon.get()), 
                                    n_iter_no_change=int(self.mlpr_n_iter_no_change.get()),
                                    max_fun=int(self.mlpr_max_fun.get()))
                if master.x2_st_var.get() == 'Yes':
                    mlpr.fit(training_X_st, training_y)
                    pr_values = mlpr.predict(X_St)
                else:
                    mlpr.fit(training_X, training_y)
                    pr_values = mlpr.predict(X)
            self.prediction.data['Y'] = pr_values

        ttk.Button(self.frame, text='Predict values', 
                  command=lambda: make_regression(method=parent.pr_method.get())).place(x=420, y=360)
        
        def save_results():
            files = [('Excel', '*.xlsx'),
                     ('All Files', '*.*'), 
                     ('Text Document', '*.txt')] 
            file = asksaveasfilename(parent=self.frame, filetypes = files, defaultextension = files)
            with pd.ExcelWriter(file, engine="openpyxl", mode='a') as writer: 
                pd.DataFrame(self.prediction.data).to_excel(writer, sheet_name='Result', 
                                                      index=False, header=(False if self.prediction.header_var.get()==0 else True))
        
        ttk.Button(self.frame, text='Save results', 
                  command=lambda: save_results()).place(x=420, y=400)
        ttk.Button(self.frame, text='Quit', 
                  command=lambda: quit_back(self.root, parent)).place(x=420, y=440)
        
class rgr_mtds_specification:
    def __init__(self, prev, parent):
        self.root = tk.Toplevel(parent)

        w = 690
        h = 640
        #setting main window's parameters       
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2) - 30
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.grab_set()
        self.root.focus_force()
        self.root.resizable(False, False)
        self.root.title('Regression methods specification')

        self.canvas = tk.Canvas(self.root)
        self.frame = ttk.Frame(self.canvas, width=1180, height=640)
        self.scrollbar = ttk.Scrollbar(self.canvas, orient="horizontal", command=self.canvas.xview)
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
        
        ttk.Label(self.frame, text='Least Squares', font=myfont_b).place(x=30, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=20, y=40)
        ls_cb1 = ttk.Checkbutton(self.frame, variable=prev.lr_include_comp, takefocus=False)
        ls_cb1.place(x=110, y=50)
        ttk.Label(self.frame, text='Equation type', font=myfont2).place(x=5, y=80)
        ls_combobox1 = ttk.Combobox(self.frame, textvariable=prev.lr_function_type, width=6, values=['Linear', 'Polynomial',
                                                                                                     'Exponential', 'Power',
                                                                                                     'Logarithmic'])
        ls_combobox1.place(x=105,y=80)
        ttk.Label(self.frame, text='Fit intercept', font=myfont2).place(x=5, y=100)
        ls_cb2 = ttk.Checkbutton(self.frame, variable=prev.lr_fit_intercept, takefocus=False)
        ls_cb2.place(x=110, y=100)
        ttk.Label(self.frame, text='Normalize', font=myfont2).place(x=5, y=120)
        ls_cb3 = ttk.Checkbutton(self.frame, variable=prev.lr_normalize, takefocus=False)
        ls_cb3.place(x=110, y=120)
        ttk.Label(self.frame, text='Copy X', font=myfont2).place(x=5, y=140)
        ls_cb4 = ttk.Checkbutton(self.frame, variable=prev.lr_copy_X, takefocus=False)
        ls_cb4.place(x=110, y=140)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=5, y=160)
        ls_e1 = ttk.Entry(self.frame, textvariable=prev.lr_n_jobs, font=myfont2, width=5)
        ls_e1.place(x=110, y=160)
        ttk.Label(self.frame, text='Positive', font=myfont2).place(x=5, y=180)
        ls_cb5 = ttk.Checkbutton(self.frame, variable=prev.lr_positive, takefocus=False)
        ls_cb5.place(x=110, y=180)

        ttk.Label(self.frame, text='Ridge', font=myfont_b).place(x=30, y=200)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=20, y=220)
        rr_cb1 = ttk.Checkbutton(self.frame, variable=prev.rr_include_comp, takefocus=False)
        rr_cb1.place(x=110, y=230)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=5, y=260)
        rr_e1 = ttk.Entry(self.frame, textvariable=prev.rr_alpha, font=myfont2, width=5)
        rr_e1.place(x=110, y=260)
        ttk.Label(self.frame, text='Fit intercept', font=myfont2).place(x=5, y=280)
        rr_cb2 = ttk.Checkbutton(self.frame, variable=prev.rr_fit_intercept, takefocus=False)
        rr_cb2.place(x=110, y=280)
        ttk.Label(self.frame, text='Normalize', font=myfont2).place(x=5, y=300)
        rr_cb3 = ttk.Checkbutton(self.frame, variable=prev.rr_normalize, takefocus=False)
        rr_cb3.place(x=110, y=300)
        ttk.Label(self.frame, text='Copy X', font=myfont2).place(x=5, y=320)
        rr_cb4 = ttk.Checkbutton(self.frame, variable=prev.rr_copy_X, takefocus=False)
        rr_cb4.place(x=110, y=320)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=5, y=340)
        rr_e2 = ttk.Entry(self.frame, textvariable=prev.rr_max_iter, font=myfont2, width=5)
        rr_e2.place(x=110, y=340)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=5, y=360)
        rr_e3 = ttk.Entry(self.frame, textvariable=prev.rr_tol, font=myfont2, width=5)
        rr_e3.place(x=110, y=360)
        ttk.Label(self.frame, text='Solver', font=myfont2).place(x=5, y=380)
        rr_combobox1 = ttk.Combobox(self.frame, textvariable=prev.rr_solver, width=6, values=['auto', 'svd', 'lsqr', 
                                                                                              'sparse_cg', 'sag', 'saga'])
        rr_combobox1.place(x=105,y=380)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=5, y=400)
        rr_e4 = ttk.Entry(self.frame, textvariable=prev.rr_random_state, font=myfont2, width=5)
        rr_e4.place(x=110, y=400)

        ttk.Label(self.frame, text='Gaussian Process', font=myfont_b).place(x=15, y=420)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=15, y=440)
        gp_cb3 = ttk.Checkbutton(self.frame, variable=prev.gpr_include_comp, takefocus=False)
        gp_cb3.place(x=110, y=450)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=5, y=480)
        gp_e2 = ttk.Entry(self.frame, textvariable=prev.gpr_alpha, font=myfont2, width=7)
        gp_e2.place(x=110,y=480)
        ttk.Label(self.frame, text='n restarts\noptimizer', font=myfont2).place(x=5, y=500)
        gp_e1 = ttk.Entry(self.frame, textvariable=prev.gpr_n_restarts_optimizer, font=myfont2, width=7)
        gp_e1.place(x=110,y=510)
        ttk.Label(self.frame, text='Normalize y', font=myfont2).place(x=5, y=540)
        gp_cb1 = ttk.Checkbutton(self.frame, variable=prev.gpr_normalize_y, takefocus=False)
        gp_cb1.place(x=110,y=540)
        ttk.Label(self.frame, text='Copy X train', font=myfont2).place(x=5, y=560)
        gp_cb2 = ttk.Checkbutton(self.frame, variable=prev.gpr_copy_X_train, takefocus=False)
        gp_cb2.place(x=110,y=560)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=5, y=580)
        gp_e3 = ttk.Entry(self.frame, textvariable=prev.gpr_random_state, font=myfont2, width=7)
        gp_e3.place(x=110,y=580)
        
        ttk.Label(self.frame, text='Lasso', font=myfont_b).place(x=200, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=190, y=40)
        lasso_cb1 = ttk.Checkbutton(self.frame, variable=prev.lassor_include_comp, takefocus=False)
        lasso_cb1.place(x=280, y=50)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=175, y=80)
        lasso_e1 = ttk.Entry(self.frame, textvariable=prev.lassor_alpha, font=myfont2, width=5)
        lasso_e1.place(x=280, y=80)
        ttk.Label(self.frame, text='Fit intercept', font=myfont2).place(x=175, y=100)
        lasso_cb1 = ttk.Checkbutton(self.frame, variable=prev.lassor_fit_intercept, takefocus=False)
        lasso_cb1.place(x=280, y=100)
        ttk.Label(self.frame, text='Normalize', font=myfont2).place(x=175, y=120)
        lasso_cb2 = ttk.Checkbutton(self.frame, variable=prev.lassor_normalize, takefocus=False)
        lasso_cb2.place(x=280, y=120)
        ttk.Label(self.frame, text='Precompute', font=myfont2).place(x=175, y=140)
        lasso_cb6 = ttk.Checkbutton(self.frame, variable=prev.lassor_precompute, takefocus=False)
        lasso_cb6.place(x=280, y=140)
        ttk.Label(self.frame, text='Copy X', font=myfont2).place(x=175, y=160)
        lasso_cb3 = ttk.Checkbutton(self.frame, variable=prev.lassor_copy_X, takefocus=False)
        lasso_cb3.place(x=280, y=160)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=175, y=180)
        lasso_e2 = ttk.Entry(self.frame, textvariable=prev.lassor_max_iter, font=myfont2, width=5)
        lasso_e2.place(x=280, y=180)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=175, y=200)
        lasso_e3 = ttk.Entry(self.frame, textvariable=prev.lassor_tol, font=myfont2, width=5)
        lasso_e3.place(x=280, y=200)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=175, y=220)
        lasso_cb4 = ttk.Checkbutton(self.frame, variable=prev.lassor_warm_start, takefocus=False)
        lasso_cb4.place(x=280, y=220)
        ttk.Label(self.frame, text='Positive', font=myfont2).place(x=175, y=240)
        lasso_cb5 = ttk.Checkbutton(self.frame, variable=prev.lassor_positive, takefocus=False)
        lasso_cb5.place(x=280, y=240)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=175, y=260)
        lasso_e4 = ttk.Entry(self.frame, textvariable=prev.lassor_random_state, font=myfont2, width=5)
        lasso_e4.place(x=280, y=260)
        ttk.Label(self.frame, text='Selection', font=myfont2).place(x=175, y=280)
        lasso_combobox2 = ttk.Combobox(self.frame, textvariable=prev.lassor_selection, width=6, values=['cyclic', 'random'])
        lasso_combobox2.place(x=275,y=280)

        ttk.Label(self.frame, text='Support Vector', font=myfont_b).place(x=190, y=310)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=185, y=340)
        sv_cb5 = ttk.Checkbutton(self.frame, variable=prev.svr_include_comp, takefocus=False)
        sv_cb5.place(x=280, y=350)
        ttk.Label(self.frame, text='Kernel', font=myfont2).place(x=175, y=380)
        sv_combobox1 = ttk.Combobox(self.frame, textvariable=prev.svr_kernel, width=6, values=['linear', 'poly', 
                                                                                                 'rbf', 'sigmoid', 'precomputed'])
        sv_combobox1.place(x=270,y=380)
        ttk.Label(self.frame, text='Degree', font=myfont2).place(x=175, y=400)
        sv_e2 = ttk.Entry(self.frame, textvariable=prev.svr_degree, font=myfont2, width=5)
        sv_e2.place(x=280,y=400)
        ttk.Label(self.frame, text='Gamma', font=myfont2).place(x=175, y=420)
        sv_e3 = ttk.Entry(self.frame, textvariable=prev.svr_gamma, font=myfont2, width=5)
        sv_e3.place(x=280,y=420)
        ttk.Label(self.frame, text='coef0', font=myfont2).place(x=175, y=440)
        sv_e4 = ttk.Entry(self.frame, textvariable=prev.svr_coef0, font=myfont2, width=5)
        sv_e4.place(x=280,y=440)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=175, y=460)
        sv_e5 = ttk.Entry(self.frame, textvariable=prev.svr_tol, font=myfont2, width=5)
        sv_e5.place(x=280,y=460)
        ttk.Label(self.frame, text='C', font=myfont2).place(x=175, y=480)
        sv_e1 = ttk.Entry(self.frame, textvariable=prev.svr_C, font=myfont2, width=5)
        sv_e1.place(x=280,y=480)
        ttk.Label(self.frame, text='epsilon', font=myfont2).place(x=175, y=500)
        sv_e8 = ttk.Entry(self.frame, textvariable=prev.svr_epsilon, font=myfont2, width=5)
        sv_e8.place(x=280,y=500)
        ttk.Label(self.frame, text='shrinking', font=myfont2).place(x=175, y=520)
        sv_cb1 = ttk.Checkbutton(self.frame, variable=prev.svr_shrinking, takefocus=False)
        sv_cb1.place(x=280,y=520)
        ttk.Label(self.frame, text='Cache size', font=myfont2).place(x=175, y=540)
        sv_e6 = ttk.Entry(self.frame, textvariable=prev.svr_cache_size, font=myfont2, width=5)
        sv_e6.place(x=280,y=540)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=175, y=560)
        sv_cb3 = ttk.Checkbutton(self.frame, variable=prev.svr_verbose, takefocus=False)
        sv_cb3.place(x=280,y=560)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=175, y=580)
        sv_e7 = ttk.Entry(self.frame, textvariable=prev.svr_max_iter, font=myfont2, width=5)
        sv_e7.place(x=280,y=580)

        ttk.Label(self.frame, text='Random Forest', font=myfont_b).place(x=355, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=350, y=40)
        rf_cb4 = ttk.Checkbutton(self.frame, variable=prev.rfr_include_comp, takefocus=False)
        rf_cb4.place(x=445, y=50)
        ttk.Label(self.frame, text='Trees number', font=myfont2).place(x=340, y=80)
        rf_e1 = ttk.Entry(self.frame, textvariable=prev.rfr_n_estimators, font=myfont2, width=5)
        rf_e1.place(x=445,y=80)
        ttk.Label(self.frame, text='Criterion', font=myfont2).place(x=340, y=100)
        rf_combobox5 = ttk.Combobox(self.frame, textvariable=prev.rfr_criterion, width=5, values=['mse', 'mae'])
        rf_combobox5.place(x=440,y=100)
        ttk.Label(self.frame, text='Max Depth', font=myfont2).place(x=340, y=120)
        rf_e2 = ttk.Entry(self.frame, textvariable=prev.rfr_max_depth, font=myfont2, width=5)
        rf_e2.place(x=445,y=120)
        ttk.Label(self.frame, text='Min samples split', font=myfont2).place(x=340, y=140)
        rf_e3 = ttk.Entry(self.frame, textvariable=prev.rfr_min_samples_split, font=myfont2, width=5)
        rf_e3.place(x=445,y=140)
        ttk.Label(self.frame, text='Min samples leaf', font=myfont2).place(x=340, y=160)
        rf_e4 = ttk.Entry(self.frame, textvariable=prev.rfr_min_samples_leaf, font=myfont2, width=5)
        rf_e4.place(x=445,y=160)
        ttk.Label(self.frame, text='Min weight\nfraction leaf', font=myfont2).place(x=340, y=180)
        rf_e5 = ttk.Entry(self.frame, textvariable=prev.rfr_min_weight_fraction_leaf, font=myfont2, width=5)
        rf_e5.place(x=445,y=190)
        ttk.Label(self.frame, text='Max features', font=myfont2).place(x=340, y=220)
        rf_e6 = ttk.Entry(self.frame, textvariable=prev.rfr_max_features, font=myfont2, width=5)
        rf_e6.place(x=445,y=220)
        ttk.Label(self.frame, text='Max leaf nodes', font=myfont2).place(x=340, y=240)
        rf_e7 = ttk.Entry(self.frame, textvariable=prev.rfr_max_leaf_nodes, font=myfont2, width=5)
        rf_e7.place(x=445,y=240)
        ttk.Label(self.frame, text='Min impurity\ndecrease', font=myfont2).place(x=340, y=260)
        rf_e8 = ttk.Entry(self.frame, textvariable=prev.rfr_min_impurity_decrease, font=myfont2, width=5)
        rf_e8.place(x=445,y=270)
        ttk.Label(self.frame, text='Bootstrap', font=myfont2).place(x=340, y=300)
        rf_cb1 = ttk.Checkbutton(self.frame, variable=prev.rfr_bootstrap, takefocus=False)
        rf_cb1.place(x=445,y=300)
        ttk.Label(self.frame, text='oob score', font=myfont2).place(x=340, y=320)
        rf_cb2 = ttk.Checkbutton(self.frame, variable=prev.rfr_oob_score, takefocus=False)
        rf_cb2.place(x=445,y=320)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=340, y=340)
        rf_e9 = ttk.Entry(self.frame, textvariable=prev.rfr_n_jobs, font=myfont2, width=5)
        rf_e9.place(x=445,y=340)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=340, y=360)
        rf_e10 = ttk.Entry(self.frame, textvariable=prev.rfr_random_state, font=myfont2, width=5)
        rf_e10.place(x=445,y=360)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=340, y=380)
        rf_e11 = ttk.Entry(self.frame, textvariable=prev.rfr_verbose, font=myfont2, width=5)
        rf_e11.place(x=445,y=380)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=340, y=400)
        rf_cb3 = ttk.Checkbutton(self.frame, variable=prev.rfr_warm_start, takefocus=False)
        rf_cb3.place(x=445,y=400)
        ttk.Label(self.frame, text='CCP alpha', font=myfont2).place(x=340, y=420)
        rf_e12 = ttk.Entry(self.frame, textvariable=prev.rfr_ccp_alpha, font=myfont2, width=5)
        rf_e12.place(x=445,y=420)
        ttk.Label(self.frame, text='Max samples', font=myfont2).place(x=340, y=440)
        rf_e13 = ttk.Entry(self.frame, textvariable=prev.rfr_max_samples, font=myfont2, width=5)
        rf_e13.place(x=445,y=440)

        ttk.Label(self.frame, text='SGD', font=myfont_b).place(x=535, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=515, y=40)
        sgd_cb5 = ttk.Checkbutton(self.frame, variable=prev.sgdr_include_comp, takefocus=False)
        sgd_cb5.place(x=610, y=50)
        ttk.Label(self.frame, text='Loss', font=myfont2).place(x=505, y=80)
        sgd_combobox1 = ttk.Combobox(self.frame, textvariable=prev.sgdr_loss, width=11, values=['squared_loss', 'huber', 
                                                                                                'epsilon_insensitive', 
                                                                                                'squared_epsilon_insensitive'])
        sgd_combobox1.place(x=602,y=80)
        ttk.Label(self.frame, text='Penalty', font=myfont2).place(x=505, y=100)
        sgd_combobox2 = ttk.Combobox(self.frame, textvariable=prev.sgdr_penalty, width=7, values=['l2', 'l1', 'elasticnet'])
        sgd_combobox2.place(x=610,y=100)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=505, y=120)
        sgd_e1 = ttk.Entry(self.frame, textvariable=prev.sgdr_alpha, font=myfont2, width=7)
        sgd_e1.place(x=615,y=120)
        ttk.Label(self.frame, text='l1 ratio', font=myfont2).place(x=505, y=140)
        sgd_e2 = ttk.Entry(self.frame, textvariable=prev.sgdr_l1_ratio, font=myfont2, width=7)
        sgd_e2.place(x=615,y=140)
        ttk.Label(self.frame, text='fit intercept', font=myfont2).place(x=505, y=160)
        sgd_cb1 = ttk.Checkbutton(self.frame, variable=prev.sgdr_fit_intercept, takefocus=False)
        sgd_cb1.place(x=615,y=160)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=505, y=180)
        sgd_e3 = ttk.Entry(self.frame, textvariable=prev.sgdr_max_iter, font=myfont2, width=7)
        sgd_e3.place(x=615,y=180)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=505, y=200)
        sgd_e4 = ttk.Entry(self.frame, textvariable=prev.sgdr_tol, font=myfont2, width=7)
        sgd_e4.place(x=615,y=200)
        ttk.Label(self.frame, text='Shuffle', font=myfont2).place(x=505, y=220)
        sgd_cb2 = ttk.Checkbutton(self.frame, variable=prev.sgdr_shuffle, takefocus=False)
        sgd_cb2.place(x=615,y=220)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=505, y=240)
        sgd_e5 = ttk.Entry(self.frame, textvariable=prev.sgdr_verbose, font=myfont2, width=7)
        sgd_e5.place(x=615,y=240)
        ttk.Label(self.frame, text='Epsilon', font=myfont2).place(x=505, y=260)
        sgd_e6 = ttk.Entry(self.frame, textvariable=prev.sgdr_epsilon, font=myfont2, width=7)
        sgd_e6.place(x=615,y=260)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=505, y=280)
        sgd_e8 = ttk.Entry(self.frame, textvariable=prev.sgdr_random_state, font=myfont2, width=7)
        sgd_e8.place(x=615,y=280)
        ttk.Label(self.frame, text='Learning rate', font=myfont2).place(x=505, y=300)
        sgd_combobox3 = ttk.Combobox(self.frame, textvariable=prev.sgdr_learning_rate, width=8, values=['constant', 'optimal',
                                                                                                      'invscaling', 'adaptive'])
        sgd_combobox3.place(x=610,y=300)
        ttk.Label(self.frame, text='eta0', font=myfont2).place(x=505, y=320)
        sgd_e9 = ttk.Entry(self.frame, textvariable=prev.sgdr_eta0, font=myfont2, width=7)
        sgd_e9.place(x=615,y=320)
        ttk.Label(self.frame, text='power t', font=myfont2).place(x=505, y=340)
        sgd_e10 = ttk.Entry(self.frame, textvariable=prev.sgdr_power_t, font=myfont2, width=7)
        sgd_e10.place(x=615,y=340)
        ttk.Label(self.frame, text='Early stopping', font=myfont2).place(x=505, y=360)
        sgd_cb3 = ttk.Checkbutton(self.frame, variable=prev.sgdr_early_stopping, takefocus=False)
        sgd_cb3.place(x=615,y=360)
        ttk.Label(self.frame, text='Validation fraction', font=myfont2).place(x=505, y=380)
        sgd_e11 = ttk.Entry(self.frame, textvariable=prev.sgdr_validation_fraction, font=myfont2, width=7)
        sgd_e11.place(x=615,y=380)
        ttk.Label(self.frame, text='n iter no change', font=myfont2).place(x=505, y=400)
        sgd_e12 = ttk.Entry(self.frame, textvariable=prev.sgdr_n_iter_no_change, font=myfont2, width=7)
        sgd_e12.place(x=615,y=400)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=505, y=420)
        sgd_cb4 = ttk.Checkbutton(self.frame, variable=prev.sgdr_warm_start, takefocus=False)
        sgd_cb4.place(x=615,y=420)
        ttk.Label(self.frame, text='Average', font=myfont2).place(x=505, y=440)
        sgd_e13 = ttk.Entry(self.frame, textvariable=prev.sgdr_average, font=myfont2, width=7)
        sgd_e13.place(x=615,y=440)
        
        ttk.Label(self.frame, text='Nearest Neighbor', font=myfont_b).place(x=705, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=705, y=40)
        kn_cb1 = ttk.Checkbutton(self.frame, variable=prev.knr_include_comp, takefocus=False)
        kn_cb1.place(x=795, y=50)
        ttk.Label(self.frame, text='n neighbors', font=myfont2).place(x=695, y=80)
        kn_e1 = ttk.Entry(self.frame, textvariable=prev.knr_n_neighbors, font=myfont2, width=7)
        kn_e1.place(x=795,y=80)
        ttk.Label(self.frame, text='Weights', font=myfont2).place(x=695, y=100)
        gp_combobox1 = ttk.Combobox(self.frame, textvariable=prev.knr_weights, width=7, values=['uniform', 'distance'])
        gp_combobox1.place(x=795,y=100)
        ttk.Label(self.frame, text='Algorithm', font=myfont2).place(x=695, y=120)
        gp_combobox2 = ttk.Combobox(self.frame, textvariable=prev.knr_algorithm, width=7, values=['auto', 'ball_tree',
                                                                                                'kd_tree', 'brute'])
        gp_combobox2.place(x=795,y=120)
        ttk.Label(self.frame, text='Leaf size', font=myfont2).place(x=695, y=140)
        kn_e2 = ttk.Entry(self.frame, textvariable=prev.knr_leaf_size, font=myfont2, width=7)
        kn_e2.place(x=795,y=140)
        ttk.Label(self.frame, text='p', font=myfont2).place(x=695, y=160)
        kn_e3 = ttk.Entry(self.frame, textvariable=prev.knr_p, font=myfont2, width=7)
        kn_e3.place(x=795,y=160)
        ttk.Label(self.frame, text='Metric', font=myfont2).place(x=695, y=180)
        gp_combobox3 = ttk.Combobox(self.frame, textvariable=prev.knr_metric, width=9, values=['euclidean', 'manhattan', 
                                                                                             'chebyshev', 'minkowski',
                                                                                             'wminkowski', 'seuclidean', 
                                                                                             'mahalanobis'])
        gp_combobox3.place(x=785,y=180)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=695, y=200)
        kn_e4 = ttk.Entry(self.frame, textvariable=prev.knr_n_jobs, font=myfont2, width=7)
        kn_e4.place(x=795,y=200)

        ttk.Label(self.frame, text='Decision Tree', font=myfont_b).place(x=705, y=220)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=705, y=240)
        dt_cb1 = ttk.Checkbutton(self.frame, variable=prev.dtr_include_comp, takefocus=False)
        dt_cb1.place(x=795, y=250)
        ttk.Label(self.frame, text='Criterion', font=myfont2).place(x=695, y=280)
        dt_combobox3 = ttk.Combobox(self.frame, textvariable=prev.dtr_criterion, width=8, values=['mse', 'friedman_mse', 
                                                                                                  'mae', 'poisson'])
        dt_combobox3.place(x=785,y=280)
        ttk.Label(self.frame, text='Splitter', font=myfont2).place(x=695, y=300)
        dt_combobox4 = ttk.Combobox(self.frame, textvariable=prev.dtr_splitter, width=6, values=['best', 'random'])
        dt_combobox4.place(x=790,y=300)
        ttk.Label(self.frame, text='Max Depth', font=myfont2).place(x=695, y=320)
        dt_e1 = ttk.Entry(self.frame, textvariable=prev.dtr_max_depth, font=myfont2, width=5)
        dt_e1.place(x=795,y=320)
        ttk.Label(self.frame, text='Min samples split', font=myfont2).place(x=695, y=340)
        dt_e2 = ttk.Entry(self.frame, textvariable=prev.dtr_min_samples_split, font=myfont2, width=5)
        dt_e2.place(x=795,y=340)
        ttk.Label(self.frame, text='Min samples leaf', font=myfont2).place(x=695, y=360)
        dt_e3 = ttk.Entry(self.frame, textvariable=prev.dtr_min_samples_leaf, font=myfont2, width=5)
        dt_e3.place(x=795,y=360)
        ttk.Label(self.frame, text="Min weight\nfraction leaf", font=myfont2).place(x=695, y=380)
        dt_e4 = ttk.Entry(self.frame, textvariable=prev.dtr_min_weight_fraction_leaf, font=myfont2, width=5)
        dt_e4.place(x=795,y=390)
        ttk.Label(self.frame, text='Max features', font=myfont2).place(x=695, y=420)
        dt_e5 = ttk.Entry(self.frame, textvariable=prev.dtr_max_features, font=myfont2, width=5)
        dt_e5.place(x=795,y=420)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=695, y=440)
        dt_e6 = ttk.Entry(self.frame, textvariable=prev.dtr_random_state, font=myfont2, width=5)
        dt_e6.place(x=795,y=440)
        ttk.Label(self.frame, text='Max leaf nodes', font=myfont2).place(x=695, y=460)
        dt_e7 = ttk.Entry(self.frame, textvariable=prev.dtr_max_leaf_nodes, font=myfont2, width=5)
        dt_e7.place(x=795,y=460)
        ttk.Label(self.frame, text="Min impurity\ndecrease", font=myfont2).place(x=695, y=480)
        dt_e8 = ttk.Entry(self.frame, textvariable=prev.dtr_min_impurity_decrease, font=myfont2, width=5)
        dt_e8.place(x=795,y=490)
        ttk.Label(self.frame, text="CCP alpha", font=myfont2).place(x=695, y=520)
        dt_e9 = ttk.Entry(self.frame, textvariable=prev.dtr_ccp_alpha, font=myfont2, width=5)
        dt_e9.place(x=795,y=520)

        ttk.Label(self.frame, text='Multi-layer Perceptron', font=myfont_b).place(x=875, y=10)
        ttk.Label(self.frame, text=' Include in\ncomparison', font=myfont2).place(x=895, y=40)
        mlp_cb1 = ttk.Checkbutton(self.frame, variable=prev.mlpr_include_comp, takefocus=False)
        mlp_cb1.place(x=985, y=50)
        ttk.Label(self.frame, text='hidden layer\nsizes', font=myfont2).place(x=875, y=80)
        mlp_e1 = ttk.Entry(self.frame, textvariable=prev.mlpr_hidden_layer_sizes, font=myfont2, width=7)
        mlp_e1.place(x=985,y=90)
        ttk.Label(self.frame, text='Activation', font=myfont2).place(x=875, y=120)
        mlp_combobox1 = ttk.Combobox(self.frame, textvariable=prev.mlpr_activation, width=7, values=['identity', 'logistic',
                                                                                                   'tanh', 'relu'])
        mlp_combobox1.place(x=985,y=120)
        ttk.Label(self.frame, text='Solver', font=myfont2).place(x=875, y=140)
        mlp_combobox2 = ttk.Combobox(self.frame, textvariable=prev.mlpr_solver, width=7, values=['lbfgs', 'sgd', 'adam'])
        mlp_combobox2.place(x=985,y=140)
        ttk.Label(self.frame, text='Alpha', font=myfont2).place(x=875, y=160)
        mlp_e2 = ttk.Entry(self.frame, textvariable=prev.mlpr_alpha, font=myfont2, width=7)
        mlp_e2.place(x=985,y=160)
        ttk.Label(self.frame, text='Batch size', font=myfont2).place(x=875, y=180)
        mlp_e3 = ttk.Entry(self.frame, textvariable=prev.mlpr_batch_size, font=myfont2, width=7)
        mlp_e3.place(x=985,y=180)
        ttk.Label(self.frame, text='Learning rate', font=myfont2).place(x=875, y=200)
        mlp_combobox3 = ttk.Combobox(self.frame, textvariable=prev.mlpr_learning_rate, width=7, values=['constant', 'invscaling',
                                                                                                      'adaptive'])
        mlp_combobox3.place(x=985,y=200)
        ttk.Label(self.frame, text='Learning\nrate init', font=myfont2).place(x=875, y=220)
        mlp_e4 = ttk.Entry(self.frame, textvariable=prev.mlpr_learning_rate_init, font=myfont2, width=7)
        mlp_e4.place(x=985,y=230)
        ttk.Label(self.frame, text='Power t', font=myfont2).place(x=875, y=260)
        mlp_e5 = ttk.Entry(self.frame, textvariable=prev.mlpr_power_t, font=myfont2, width=7)
        mlp_e5.place(x=985,y=260)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=875, y=280)
        mlp_e6 = ttk.Entry(self.frame, textvariable=prev.mlpr_max_iter, font=myfont2, width=7)
        mlp_e6.place(x=985,y=280)
        ttk.Label(self.frame, text='Shuffle', font=myfont2).place(x=875, y=300)
        mlp_cb2 = ttk.Checkbutton(self.frame, variable=prev.mlpr_shuffle, takefocus=False)
        mlp_cb2.place(x=985, y=300)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=875, y=320)
        mlp_e7 = ttk.Entry(self.frame, textvariable=prev.mlpr_random_state, font=myfont2, width=7)
        mlp_e7.place(x=985,y=320)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=875, y=340)
        mlp_e8 = ttk.Entry(self.frame, textvariable=prev.mlpr_tol, font=myfont2, width=7)
        mlp_e8.place(x=985,y=340)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=875, y=360)
        mlp_cb3 = ttk.Checkbutton(self.frame, variable=prev.mlpr_verbose, takefocus=False)
        mlp_cb3.place(x=985, y=360)
        ttk.Label(self.frame, text='Warm start', font=myfont2).place(x=875, y=380)
        mlp_cb4 = ttk.Checkbutton(self.frame, variable=prev.mlpr_warm_start, takefocus=False)
        mlp_cb4.place(x=985, y=380)
        ttk.Label(self.frame, text='Momentum', font=myfont2).place(x=875, y=400)
        mlp_e9 = ttk.Entry(self.frame, textvariable=prev.mlpr_momentum, font=myfont2, width=7)
        mlp_e9.place(x=985,y=400)
        ttk.Label(self.frame, text='Nesterovs\nmomentum', font=myfont2).place(x=875, y=420)
        mlp_cb5 = ttk.Checkbutton(self.frame, variable=prev.mlpr_nesterovs_momentum, takefocus=False)
        mlp_cb5.place(x=985, y=430)
        ttk.Label(self.frame, text='Early stopping', font=myfont2).place(x=875, y=460)
        mlp_cb6 = ttk.Checkbutton(self.frame, variable=prev.mlpr_early_stopping, takefocus=False)
        mlp_cb6.place(x=985, y=460)
        ttk.Label(self.frame, text='Validation fraction', font=myfont2).place(x=875, y=480)
        mlp_e10 = ttk.Entry(self.frame, textvariable=prev.mlpr_validation_fraction, font=myfont2, width=7)
        mlp_e10.place(x=985,y=480)
        ttk.Label(self.frame, text='Beta 1', font=myfont2).place(x=875, y=500)
        mlp_e11 = ttk.Entry(self.frame, textvariable=prev.mlpr_beta_1, font=myfont2, width=7)
        mlp_e11.place(x=985,y=500)
        ttk.Label(self.frame, text='Beta 2', font=myfont2).place(x=875, y=520)
        mlp_e12 = ttk.Entry(self.frame, textvariable=prev.mlpr_beta_2, font=myfont2, width=7)
        mlp_e12.place(x=985,y=520)
        ttk.Label(self.frame, text='Epsilon', font=myfont2).place(x=875, y=540)
        mlp_e13 = ttk.Entry(self.frame, textvariable=prev.mlpr_epsilon, font=myfont2, width=7)
        mlp_e13.place(x=985,y=540)
        ttk.Label(self.frame, text='n iter no change', font=myfont2).place(x=875, y=560)
        mlp_e14 = ttk.Entry(self.frame, textvariable=prev.mlpr_n_iter_no_change, font=myfont2, width=7)
        mlp_e14.place(x=985,y=560)
        ttk.Label(self.frame, text='Max fun', font=myfont2).place(x=875, y=580)
        mlp_e15 = ttk.Entry(self.frame, textvariable=prev.mlpr_max_fun, font=myfont2, width=7)
        mlp_e15.place(x=985,y=580)
        
        ttk.Button(self.root, text='OK', command=lambda: quit_back(self.root, prev.root)).place(relx=0.85, rely=0.92)

class Data_Preview:
    def __init__(self, prev, main, full, parent):
        if main.data is not None:
            self.root = tk.Toplevel(parent)
            w = 600
            h = 650
            #setting main window's parameters       
            x = (ws/2) - (w/2)
            y = (hs/2) - (h/2) - 30
            self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
            self.root.lift()
            self.root.resizable(False, False)
            self.root.title('Data Preview')
            self.f = ttk.Frame(self.root, width=w, height=h)
            self.f.place(y=0)
            self.f1 = ttk.Frame(self.root, width=w, height=(h-50))
            if master.current_theme in big_themes:
                self.f1.place(y=0)
            else:
                self.f1.place(y=50)
            self.pt = Table(self.f1, dataframe=main.data, showtoolbar=True, showstatusbar=True)
            self.pt.show()
            ttk.Button(self.f, text='Update Dataframe', command=lambda: 
                       self.update_data(prev, main, self.pt.model.df, full)).place(x=10, y=550)
            ttk.Button(self.f, text='Close', command=lambda: quit_back(self.root, prev.root)).place(x=500, y=610)
            ttk.Button(self.f, text='Drop NA', command=lambda: self.drop_na()).place(x=10, y=580)
            ttk.Button(self.f, text='Fill NA', command=lambda: self.fill_na()).place(x=10, y=610)
    def update_data(self, prev, main, table, full):
        main.data = table
        if full == 'training':
            prev.combobox1.config(values=list(main.data.columns))
            prev.tr_x_from_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            prev.tr_x_from_combobox.set('1')
            prev.tr_x_to_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            prev.tr_x_to_combobox.set(str(main.data.shape[1]-1))
        elif full == 'prediction':
            prev.pr_x_from_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            prev.pr_x_from_combobox.set('1')
            prev.pr_x_to_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            prev.pr_x_to_combobox.set(str(main.data.shape[1]-1))
    def drop_na(self):
        self.df = self.pt.model.df
        self.pt.storeCurrent()
        self.df = self.df.dropna(axis=0)
        self.pt = Table(self.f1, dataframe=self.df, showtoolbar=True, showstatusbar=True)
        self.pt.show()
    def fill_na(self):
        self.fn_win = tk.Toplevel(self.root)
        w=200
        h=150
        #setting main window's parameters       
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2) - 30
        self.fn_win.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.fn_win.lift()
        self.fn_win.resizable(False, False)
        self.fn_win.title('Fill NA')
        self.fill_f = ttk.Frame(self.fn_win, width=w, height=h)
        self.fill_f.place(x=0, y=0)
        ttk.Button(self.fill_f, text='  Mean', command=lambda: fill_w(self, 'mean')).place(x=50, y=10)
        ttk.Button(self.fill_f, text=' Median', command=lambda: fill_w(self, 'median')).place(x=50, y=50)
        ttk.Button(self.fill_f, text='  Mode', command=lambda: fill_w(self, 'mode')).place(x=50, y=90)
        
        def fill_w(self, method):
            self.df = self.pt.model.df
            self.pt.storeCurrent()
            if method == 'mean':
                self.df = self.df.fillna(self.df.mean(axis=0))
            elif method == 'median':
                self.df = self.df.fillna(self.df.median(axis=0))
            elif method == 'mode':
                self.df = self.df.fillna(self.df.mode(axis=0))
            self.pt = Table(self.f1, dataframe=self.df, showtoolbar=True, showstatusbar=True)
            self.pt.show()
            self.fn_win.destroy()
        
def quit_back(current, parent):
    current.destroy()
    parent.lift()

def open_file(app, entry): 
    entry.delete(0, tk.END)
    file1 = askopenfilename(parent=app.root)
    if file1 is not None:
        entry.insert(0, file1)
                
def load_data(app, main, entry, data_type): 
    global data_file
    data_file = entry.get()
    if len(data_file) > 1:
        if main.header_var.get()==1 and (data_file.endswith('.xls') or data_file.endswith('.xlsx')):
            main.data = pd.read_excel(data_file, sheet_name=(int(main.sheet.get())-1))
        elif main.header_var.get()==0 and (data_file.endswith('.xls') or data_file.endswith('.xlsx')):
            main.data = pd.read_excel(data_file, sheet_name=(int(main.sheet.get())-1), header=None)
        elif main.header_var.get()==1 and (data_file.endswith('.csv')):
            main.data = pd.read_csv(data_file)
        elif main.header_var.get()==0 and (data_file.endswith('.csv')):
            main.data = pd.read_csv(data_file, header=None)
        elif main.header_var.get()==1 and (data_file.endswith('.data')):
            main.data = pd.read_table(data_file, sep=',')
        elif main.header_var.get()==0 and (data_file.endswith('.data')):
            main.data = pd.read_table(data_file, sep=',', header=None)
        if data_type=='training':
            app.tr_data_status.config(text='Loaded')
            app.combobox1.config(values=list(main.data.columns))
            app.combobox1.set(main.data.columns[-1])
            app.tr_x_from_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            app.tr_x_from_combobox.set('1')
            app.tr_x_to_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            app.tr_x_to_combobox.set(str(main.data.shape[1]-1))
        elif data_type=='prediction':
            app.pr_data_status.config(text='Loaded')
            app.pr_x_from_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            app.pr_x_from_combobox.set('1')
            app.pr_x_to_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            app.pr_x_to_combobox.set(str(main.data.shape[1]))
start_window();