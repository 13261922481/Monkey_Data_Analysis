# Imports
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
import sys
import numpy as np
import webbrowser
import pandas as pd
import sklearn
from monkey_pt import Table
from utils import *

#fonts
myfont = (None, 13)
myfont_b = (None, 13, 'bold')
myfont1 = (None, 11)
myfont1_b = (None, 11, 'bold')
myfont2 = (None, 10)
myfont2_b = (None, 10, 'bold')

# App to do clustering
class clust_app:
    # sub-class for clustering-related data
    class clust:
        def __init__(self):
            pass
    # elbow method
    class elbow_method:
        def __init__(self, prev, main, parent):
            from sklearn.cluster import KMeans
            from yellowbrick.cluster import KElbowVisualizer
            from sklearn import preprocessing

            try:
                x_from = main.data.columns.get_loc(main.x_from_var.get())
                x_to = main.data.columns.get_loc(main.x_to_var.get()) + 1
            except:
                x_from = int(main.x_from_var.get())
                x_to = int(main.x_to_var.get()) + 1
            
            scaler = preprocessing.StandardScaler()
            X_St = scaler.fit_transform(main.data.iloc[:,x_from : x_to])

            visualizer = KElbowVisualizer(KMeans(), metric=prev.elbow_metric.get(), 
                k=(int(prev.elbow_k_from.get()),int(prev.elbow_k_to.get())))
            visualizer.fit(X_St)
            visualizer.show()
    # dendrogram
    class show_dendrogram:
        def __init__(self, prev, main, parent):
            from scipy.cluster.hierarchy import dendrogram, linkage
            from matplotlib import pyplot as plt
            from sklearn import preprocessing

            try:
                x_from = main.data.columns.get_loc(main.x_from_var.get())
                x_to = main.data.columns.get_loc(main.x_to_var.get()) + 1
            except:
                x_from = int(main.x_from_var.get())
                x_to = int(main.x_to_var.get()) + 1
            
            scaler = preprocessing.StandardScaler()
            X_St = scaler.fit_transform(main.data.iloc[:,x_from : x_to])

            # plt.rcParams["figure.figsize"] = (5,5)

            plt.title('Hierarchical Clustering Dendrogram')
            dendrogram(linkage(X_St, method=prev.dendr_linkage.get()))
            plt.show()
    # Running clustering app itself
    def __init__(self, parent):
        # initialize data
        if not hasattr(self.clust, 'data'):
            self.clust.data = None
            self.clust.sheet = tk.StringVar()
            self.clust.sheet.set('1')
            self.clust.header_var = tk.IntVar()
            self.clust.header_var.set(1)
            self.clust.x_from_var = tk.StringVar()
            self.clust.x_to_var = tk.StringVar()
            self.clust.Viewed = tk.BooleanVar()
            self.clust.view_frame = None
            self.clust.pt = None

        #setting main window's parameters
        w = 620
        h = 350 
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        clust_app.root = tk.Toplevel(parent)
        clust_app.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        clust_app.root.title('Monkey Clustering')
        clust_app.root.lift()
        clust_app.root.tkraise()
        clust_app.root.focus_force()
        clust_app.root.resizable(False, False)
        clust_app.root.protocol("WM_DELETE_WINDOW", lambda: quit_back(clust_app.root, parent))

        parent.iconify()

        # run elbow method
        def try_elbow_method(prev, main, parent):
            try:
                self.elbow_method(prev, main, parent)
            except ValueError as e:
                messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))

        # show dendrogram
        def try_show_dendrogram(prev, main, parent):
            try:
                self.show_dendrogram(prev, main, parent)
            except ValueError as e:
                messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))

        # methods parameters
        self.elbow_metric = tk.StringVar(value='silhouette')
        self.elbow_k_from = tk.StringVar(value='2')
        self.elbow_k_to = tk.StringVar(value='11')
        self.dendr_linkage = tk.StringVar(value='ward')

        self.kmeans_init = tk.StringVar(value='k-means++')
        self.kmeans_n_init = tk.StringVar(value='10')
        self.kmeans_max_iter = tk.StringVar(value='300')
        self.kmeans_tol = tk.StringVar(value='1e-4')
        self.kmeans_verbose = tk.StringVar(value='0')
        self.kmeans_random_state = tk.StringVar(value='None')
        self.kmeans_copy_x = tk.BooleanVar(value=True)
        self.kmeans_algorithm = tk.StringVar(value='auto')

        self.ap_damping = tk.StringVar(value='0.5')
        self.ap_max_iter = tk.StringVar(value='200')
        self.ap_convergence_iter = tk.StringVar(value='15')
        self.ap_copy = tk.BooleanVar(value=True)
        self.ap_verbose = tk.BooleanVar(value=False)
        self.ap_random_state = tk.StringVar(value='0')

        self.ms_bandwidth = tk.StringVar(value='None')
        self.ms_bin_seeding = tk.BooleanVar(value=False)
        self.ms_min_bin_freq = tk.StringVar(value='1')
        self.ms_cluster_all = tk.BooleanVar(value=True)
        self.ms_n_jobs = tk.StringVar(value='None')
        self.ms_max_iter = tk.StringVar(value='300')

        self.sc_eigen_solver = tk.StringVar(value='arpack')
        self.sc_n_components = tk.StringVar(value='None')
        self.sc_random_state = tk.StringVar(value='None')
        self.sc_n_init = tk.StringVar(value='10')
        self.sc_gamma = tk.StringVar(value='1.0')
        self.sc_affinity = tk.StringVar(value='rbf')
        self.sc_n_neighbors = tk.StringVar(value='None')
        self.sc_eigen_tol = tk.StringVar(value='0.0')
        self.sc_assign_labels = tk.StringVar(value='kmeans')
        self.sc_degree = tk.StringVar(value='3')
        self.sc_coef0 = tk.StringVar(value='1')
        self.sc_n_jobs = tk.StringVar(value='None')
        self.sc_verbose = tk.BooleanVar(value=False)

        self.ac_affinity = tk.StringVar(value='euclidean')
        self.ac_compute_full_tree = tk.StringVar(value='auto')
        self.ac_linkage = tk.StringVar(value='ward')
        self.ac_distance_threshold = tk.StringVar(value='None')
        self.ac_compute_distances = tk.BooleanVar(value=False)

        self.dbscan_eps = tk.StringVar(value='0.5')
        self.dbscan_min_samples = tk.StringVar(value='5')
        self.dbscan_metric = tk.StringVar(value='euclidean')
        self.dbscan_algorithm = tk.StringVar(value='auto')
        self.dbscan_leaf_size = tk.StringVar(value='30')
        self.dbscan_p = tk.StringVar(value='None')
        self.dbscan_n_jobs = tk.StringVar(value='None')

        self.opt_min_samples = tk.StringVar(value='5')
        self.opt_max_eps = tk.StringVar(value='np.inf')
        self.opt_metric = tk.StringVar(value='minkowski')
        self.opt_p = tk.StringVar(value='2')
        self.opt_cluster_method = tk.StringVar(value='xi')
        self.opt_eps = tk.StringVar(value='None')
        self.opt_xi = tk.StringVar(value='0.05')
        self.opt_predecessor_correction = tk.BooleanVar(value=True)
        self.opt_min_cluster_size = tk.StringVar(value='None')
        self.opt_algorithm = tk.StringVar(value='auto')
        self.opt_leaf_size = tk.StringVar(value='30')
        self.opt_n_jobs = tk.StringVar(value='None')

        self.bc_threshold = tk.StringVar(value='0.5')
        self.bc_branching_factor = tk.StringVar(value='50')
        self.bc_compute_labels = tk.BooleanVar(value=True)
        self.bc_copy = tk.BooleanVar(value=True)

        # function to do clustering
        def clust_predict_cluster(method):
            try:
                x_from = self.clust.data.columns.get_loc(self.clust.x_from_var.get())
                x_to = self.clust.data.columns.get_loc(self.clust.x_to_var.get()) + 1
            except:
                x_from = int(self.clust.x_from_var.get())
                x_to = int(self.clust.x_to_var.get()) + 1
            if self.dummies_var.get()==0:
                X = self.clust.data.iloc[:,x_from : x_to]
            elif self.dummies_var.get()==1:
                X = pd.get_dummies(self.clust.data.iloc[:,x_from : x_to])
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            X_St = scaler.fit_transform(X)

            if method == 'K-Means':
                from sklearn.cluster import KMeans
                kmeans = KMeans(
                    n_clusters=int(self.n_clusters_entry.get()), init=self.kmeans_init.get(),
                    n_init=int(self.kmeans_n_init.get()), max_iter=int(self.kmeans_max_iter.get()),
                    tol=float(self.kmeans_tol.get()), verbose=int(self.kmeans_verbose.get()),
                    random_state=(int(self.kmeans_random_state.get()) if 
                                 (self.kmeans_random_state.get() != 'None') else None),
                    copy_x=self.kmeans_copy_x.get(), algorithm=self.kmeans_algorithm.get())
                if self.x_st_var.get() == 'No':
                    pr_values = kmeans.fit_predict(X)
                else:
                    pr_values = kmeans.fit_predict(X_St)
            elif method == 'Affinity Propagation':
                from sklearn.cluster import AffinityPropagation
                ap = AffinityPropagation(
                    damping=float(self.ap_damping.get()), max_iter=int(self.ap_max_iter.get()),
                    convergence_iter=int(self.ap_convergence_iter.get()), copy=self.ap_copy.get(),
                    verbose=self.ap_verbose.get(),
                    random_state=(int(self.ap_random_state.get()) if 
                                  (self.ap_random_state.get() != 'None') else None))
                if self.x_st_var.get() == 'No':
                    pr_values = ap.fit_predict(X)
                else:
                    pr_values = ap.fit_predict(X_St)
            elif method == 'Mean Shift':
                from sklearn.cluster import MeanShift
                ms = MeanShift(
                    bandwidth=(float(self.ms_bandwidth.get()) 
                        if (self.ms_bandwidth.get() != 'None') else None), 
                    bin_seeding=self.ms_bin_seeding.get(),
                    min_bin_freq=int(self.ms_min_bin_freq.get()), 
                    cluster_all=self.ms_cluster_all.get(),
                    n_jobs=(int(self.ms_n_jobs.get()) 
                        if (self.ms_n_jobs.get() != 'None') else None),
                    max_iter=int(self.ms_max_iter.get()))
                if self.x_st_var.get() == 'No':
                    pr_values = ms.fit_predict(X)
                else:
                    pr_values = ms.fit_predict(X_St)
            elif method == 'Spectral clustering':
                from sklearn.cluster import SpectralClustering
                sc = SpectralClustering(
                    n_clusters=int(self.n_clusters_entry.get()), 
                    eigen_solver=self.sc_eigen_solver.get(),
                    n_components=(int(self.sc_n_components.get()) 
                        if (self.sc_n_components.get() != 'None') else None),
                    random_state=(int(self.sc_random_state.get()) if 
                                  (self.sc_random_state.get() != 'None') else None),
                    n_init=int(self.sc_n_init.get()), gamma=float(self.sc_gamma.get()),
                    affinity=self.sc_affinity.get(), 
                    n_neighbors=(int(self.sc_n_neighbors.get()) 
                        if (self.sc_n_neighbors.get() != 'None') else None),
                    eigen_tol=float(self.sc_eigen_tol.get()), 
                    assign_labels=self.sc_assign_labels.get(),
                    degree=float(self.sc_degree.get()), coef0=float(self.sc_coef0.get()),
                    n_jobs=(int(self.sc_n_jobs.get()) 
                        if (self.sc_n_jobs.get() != 'None') else None),
                    verbose=self.sc_verbose.get())
                if self.x_st_var.get() == 'No':
                    pr_values = sc.fit_predict(X)
                else:
                    pr_values = sc.fit_predict(X_St)
            elif method == 'Hierarchical clustering':
                from sklearn.cluster import AgglomerativeClustering
                ac = AgglomerativeClustering(
                    n_clusters=int(self.n_clusters_entry.get()), affinity=self.ac_affinity.get(),
                    compute_full_tree=self.ac_compute_full_tree.get(), 
                    linkage=self.ac_linkage.get(),
                    distance_threshold=(float(self.ac_distance_threshold.get()) 
                        if (self.ac_distance_threshold.get() != 'None') else None),
                    compute_distances=self.ac_compute_distances.get())
                if self.x_st_var.get() == 'No':
                    pr_values = ac.fit_predict(X)
                else:
                    pr_values = ac.fit_predict(X_St)
            elif method == 'DBSCAN':
                from sklearn.cluster import DBSCAN
                dbscan = DBSCAN(
                    eps=float(self.dbscan_eps.get()), 
                    min_samples=int(self.dbscan_min_samples.get()),
                    metric=self.dbscan_metric.get(), algorithm=self.dbscan_algorithm.get(),
                    leaf_size=int(self.dbscan_leaf_size.get()),
                    p=(float(self.dbscan_p.get()) 
                        if (self.dbscan_p.get() != 'None') else None),
                    n_jobs=(int(self.dbscan_n_jobs.get()) 
                        if (self.dbscan_n_jobs.get() != 'None') else None))
                if self.x_st_var.get() == 'No':
                    pr_values = dbscan.fit_predict(X)
                else:
                    pr_values = dbscan.fit_predict(X_St)
            elif method == 'OPTICS':
                from sklearn.cluster import OPTICS
                optics = OPTICS(
                    min_samples=(float(self.opt_min_samples.get()) 
                        if '.' in self.opt_min_samples.get()
                        else int(self.opt_min_samples.get())),
                    max_eps=(float(self.opt_max_eps.get()) 
                        if (self.opt_max_eps.get() != 'np.inf') else np.inf),
                    metric=self.opt_metric.get(), p=int(self.opt_p.get()),
                    cluster_method=self.opt_cluster_method.get(), 
                    eps=(float(self.opt_eps.get()) 
                        if (self.opt_eps.get() != 'None') else None),
                    xi=float(self.opt_xi.get()), 
                    predecessor_correction=self.opt_predecessor_correction.get(),
                    min_cluster_size=(float(self.opt_min_cluster_size.get()) 
                        if '.' in self.opt_min_cluster_size.get() 
                        else int(self.opt_min_cluster_size.get()) 
                        if (self.opt_min_cluster_size.get() != 'None') else None),
                    algorithm=self.opt_algorithm.get(), 
                    leaf_size=int(self.opt_leaf_size.get()),
                    n_jobs=(int(self.opt_n_jobs.get()) if 
                            (self.opt_n_jobs.get() != 'None') else None))
                if self.x_st_var.get() == 'No':
                    pr_values = optics.fit_predict(X)
                else:
                    pr_values = optics.fit_predict(X_St)
            elif method == 'Birch':
                from sklearn.cluster import Birch
                bc = Birch(
                    threshold=float(self.bc_threshold.get()), 
                    branching_factor=int(self.bc_branching_factor.get()),
                    n_clusters=int(self.n_clusters_entry.get()),
                    compute_labels=self.bc_compute_labels.get(), copy=self.bc_copy.get())
                if self.x_st_var.get() == 'No':
                    pr_values = bc.fit_predict(X)
                else:
                    pr_values = bc.fit_predict(X_St)
            
            if self.place_result_var.get() == 'Start':
                self.clust.data.insert(0, 'Cluster', pr_values)
            elif self.place_result_var.get() == 'End':
                self.clust.data['Cluster'] = pr_values

            if self.clust.Viewed.get() == True:
                self.clust.pt = Table(self.clust.view_frame, dataframe=self.clust.data, 
                    showtoolbar=True, showstatusbar=True, height=350, 
                    notebook=Data_Preview.notebook.nb, dp_main=self.clust)
                self.clust.pt.show()
                self.clust.pt.redraw()
                Data_Preview.notebook.nb.select(self.clust.view_frame)
                Data_Preview.root.lift()
            
        # run clustering
        def try_clust_predict_cluster(method):
            try:
                clust_predict_cluster(method)
            except ValueError as e:
                messagebox.showerror(parent=self.root, message='Error: "{}"'.format(e))

        # Application interface

        self.bg_frame = ttk.Frame(clust_app.root, width=10, height=h)
        self.bg_frame.place(x=0, y=0)

        self.frame = ttk.Frame(clust_app.root, width=w, height=h)
        self.frame.place(x=10, y=0)

        ttk.Label(self.frame, text='Data file:').place(x=30, y=10)
        e1 = ttk.Entry(self.frame, font=myfont1, width=40)
        e1.place(x=120, y=10)

        ttk.Button(self.frame, text='Choose file', 
            command=lambda: open_file(self, e1)).place(x=490, y=10)

        ttk.Label(self.frame, text='List number:').place(x=120,y=50)
        training_sheet_entry = ttk.Entry(self.frame, 
            textvariable=self.clust.sheet, font=myfont1, width=3)
        training_sheet_entry.place(x=215,y=52)

        ttk.Button(self.frame, text='Load data ', 
            command=lambda: load_data(self, self.clust, e1, 'clust')).place(x=490, y=50)
        
        cb1 = ttk.Checkbutton(self.frame, text="header", 
            variable=self.clust.header_var, takefocus=False)
        cb1.place(x=10, y=50)

        ttk.Label(self.frame, text='Data status:').place(x=10, y=95)
        self.clust.data_status = ttk.Label(self.frame, text='Not Loaded')
        self.clust.data_status.place(x=120, y=95)

        ttk.Button(self.frame, text='View/Change', 
            command=lambda: Data_Preview(self, self.clust, 'clust', parent)).place(x=230, y=95)

        ttk.Label(self.frame, text='Number of clusters:', font=myfont1).place(x=30, y=140)
        self.n_clusters_var = tk.StringVar(value='2')
        self.n_clusters_entry = ttk.Entry(self.frame, 
            textvariable=self.n_clusters_var, font=myfont1, width=3)
        self.n_clusters_entry.place(x=170, y=142)

        ttk.Button(self.frame, text="Methods' specifications", 
                  command=lambda: clust_mtds_specification(self, parent)).place(x=400, y=100)

        ttk.Button(self.frame, text='Elbow method', width=12, 
            command=lambda: try_elbow_method(self, self.clust, parent)).place(x=400, y=140)

        ttk.Button(self.frame, text='Dendrogram', width=12, 
            command=lambda: try_show_dendrogram(self, self.clust, parent)).place(x=400, y=180)

        ttk.Label(self.frame, text='X from', font=myfont1).place(x=225, y=130)
        self.clust_x_from_combobox = ttk.Combobox(self.frame, 
            textvariable=self.clust.x_from_var, width=14, values=[])
        self.clust_x_from_combobox.place(x=275, y=132)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=225, y=155)
        self.clust_x_to_combobox = ttk.Combobox(self.frame, 
            textvariable=self.clust.x_to_var, width=14, values=[])
        self.clust_x_to_combobox.place(x=275, y=157)

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
        self.clust_method = tk.StringVar(value='K-Means')
        self.combobox9 = ttk.Combobox(self.frame, textvariable=self.clust_method, width=15, 
            values=['K-Means', 'Affinity Propagation', 'Mean Shift', 'Spectral clustering',
                'Hierarchical clustering', 'DBSCAN', 'OPTICS', 'Birch'])
        self.combobox9.place(x=150,y=232)

        ttk.Label(self.frame, text='Place result', font=myfont1).place(x=30, y=260)
        self.place_result_var = tk.StringVar(value='End')
        self.combobox9 = ttk.Combobox(self.frame, textvariable=self.place_result_var, width=10, 
            values=['Start', 'End'])
        self.combobox9.place(x=150,y=262)

        ttk.Button(self.frame, text='Predict', width=7,
            command=lambda: 
                try_clust_predict_cluster(method=self.clust_method.get())).place(x=400, y=240)

        ttk.Button(self.frame, text='Save to file', width=9,
            command=lambda: save_results(self, self.clust, 'clust result')).place(x=500, y=240)

        ttk.Button(self.frame, text='Save to sql', width=9,
            command=lambda: save_to_sql(self, self.clust, 'clust_result')).place(x=500, y=275)

        ttk.Button(self.frame, text='Quit', width=7,
                  command=lambda: quit_back(clust_app.root, parent)).place(x=400, y=275)

# sub-app for methods' specifications
class clust_mtds_specification:
    def __init__(self, prev, parent):
        self.root = tk.Toplevel(parent)

        #setting main window's parameters      
        w = 840
        h = 630 
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.focus_force()
        self.root.resizable(False, False)
        self.root.title('Clustering methods specification')

        self.canvas = tk.Canvas(self.root)
        self.frame = ttk.Frame(self.canvas, width=w, height=h)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
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

        ttk.Label(self.frame, text='Elbow method', font=myfont_b).place(x=40, y=10)
        ttk.Label(self.frame, text='Metric', font=myfont2).place(x=20, y=40)
        elbow_combobox1 = ttk.Combobox(self.frame, textvariable=prev.elbow_metric, width=11, 
            values=['distortion', 'silhouette', 'calinski_harabasz'])
        elbow_combobox1.place(x=125,y=40)
        ttk.Label(self.frame, text='K from', font=myfont2).place(x=20, y=65)
        elbow_e1 = ttk.Entry(self.frame, textvariable=prev.elbow_k_from, font=myfont2, width=7)
        elbow_e1.place(x=140, y=65)
        ttk.Label(self.frame, text='K to', font=myfont2).place(x=20, y=90)
        elbow_e2 = ttk.Entry(self.frame, textvariable=prev.elbow_k_to, font=myfont2, width=7)
        elbow_e2.place(x=140, y=90)

        ttk.Label(self.frame, text='Dendrogram', font=myfont_b).place(x=40, y=120)
        ttk.Label(self.frame, text='Linkage', font=myfont2).place(x=20, y=150)
        dendr_combobox1 = ttk.Combobox(self.frame, textvariable=prev.dendr_linkage, width=7, 
            values=['ward', 'complete', 'average', 'single'])
        dendr_combobox1.place(x=140,y=150)

        ttk.Label(self.frame, text='K-Means', font=myfont_b).place(x=30, y=180)
        ttk.Label(self.frame, text='Init', font=myfont2).place(x=20, y=210)
        kmeans_combobox1 = ttk.Combobox(self.frame, textvariable=prev.kmeans_init, width=10, 
            values=['k-means++', 'random'])
        kmeans_combobox1.place(x=130,y=210)
        ttk.Label(self.frame, text='N init', font=myfont2).place(x=20, y=235)
        kmeans_e1 = ttk.Entry(self.frame, textvariable=prev.kmeans_n_init, font=myfont2, width=7)
        kmeans_e1.place(x=140, y=235)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=20, y=260)
        kmeans_e2 = ttk.Entry(self.frame, textvariable=prev.kmeans_max_iter, font=myfont2, width=7)
        kmeans_e2.place(x=140, y=260)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=20, y=285)
        kmeans_e3 = ttk.Entry(self.frame, textvariable=prev.kmeans_tol, font=myfont2, width=7)
        kmeans_e3.place(x=140, y=285)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=20, y=310)
        kmeans_e4 = ttk.Entry(self.frame, textvariable=prev.kmeans_verbose, font=myfont2, width=7)
        kmeans_e4.place(x=140, y=310)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=20, y=335)
        kmeans_e5 = ttk.Entry(self.frame, 
            textvariable=prev.kmeans_random_state, font=myfont2, width=7)
        kmeans_e5.place(x=140, y=335)
        ttk.Label(self.frame, text='Copy X', font=myfont2).place(x=20, y=360)
        kmeans_cb1 = ttk.Checkbutton(self.frame, variable=prev.kmeans_copy_x, takefocus=False)
        kmeans_cb1.place(x=140, y=360)
        ttk.Label(self.frame, text='Algorithm', font=myfont2).place(x=20, y=385)
        kmeans_combobox2 = ttk.Combobox(self.frame, textvariable=prev.kmeans_algorithm, width=7, 
            values=['auto', 'full', 'elkan'])
        kmeans_combobox2.place(x=135,y=385)

        ttk.Label(self.frame, text='Mean Shift', font=myfont_b).place(x=240, y=10)
        ttk.Label(self.frame, text='Bandwidth', font=myfont2).place(x=220, y=40)
        ms_e1 = ttk.Entry(self.frame, textvariable=prev.ms_bandwidth, font=myfont2, width=7)
        ms_e1.place(x=340, y=40)
        ttk.Label(self.frame, text='Bin seeding', font=myfont2).place(x=220, y=65)
        ms_cb1 = ttk.Checkbutton(self.frame, variable=prev.ms_bin_seeding, takefocus=False)
        ms_cb1.place(x=340, y=65)
        ttk.Label(self.frame, text='Min bin freq', font=myfont2).place(x=220, y=90)
        ms_e2 = ttk.Entry(self.frame, textvariable=prev.ms_min_bin_freq, font=myfont2, width=7)
        ms_e2.place(x=340, y=90)
        ttk.Label(self.frame, text='Cluster all', font=myfont2).place(x=220, y=115)
        ms_cb2 = ttk.Checkbutton(self.frame, variable=prev.ms_cluster_all, takefocus=False)
        ms_cb2.place(x=340, y=115)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=220, y=140)
        ms_e3 = ttk.Entry(self.frame, textvariable=prev.ms_n_jobs, font=myfont2, width=7)
        ms_e3.place(x=340, y=140)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=220, y=165)
        ms_e4 = ttk.Entry(self.frame, textvariable=prev.ms_max_iter, font=myfont2, width=7)
        ms_e4.place(x=340, y=165)

        ttk.Label(self.frame, text='Spectral Clustering', font=myfont_b).place(x=230, y=195)
        ttk.Label(self.frame, text='Eigen Solver', font=myfont2).place(x=220, y=225)
        sc_combobox1 = ttk.Combobox(self.frame, textvariable=prev.sc_eigen_solver, width=7, 
            values=['arpack', 'lobpcg', 'amg'])
        sc_combobox1.place(x=340,y=225)
        ttk.Label(self.frame, text='N components', font=myfont2).place(x=220, y=250)
        sc_e1 = ttk.Entry(self.frame, textvariable=prev.sc_n_components, font=myfont2, width=7)
        sc_e1.place(x=340, y=250)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=220, y=275)
        sc_e2 = ttk.Entry(self.frame, textvariable=prev.sc_random_state, font=myfont2, width=7)
        sc_e2.place(x=340, y=275)
        ttk.Label(self.frame, text='n init', font=myfont2).place(x=220, y=300)
        sc_e3 = ttk.Entry(self.frame, textvariable=prev.sc_n_init, font=myfont2, width=7)
        sc_e3.place(x=340, y=300)
        ttk.Label(self.frame, text='Gamma', font=myfont2).place(x=220, y=325)
        sc_e4 = ttk.Entry(self.frame, textvariable=prev.sc_gamma, font=myfont2, width=7)
        sc_e4.place(x=340, y=325)
        ttk.Label(self.frame, text='Affinity', font=myfont2).place(x=220, y=350)
        sc_combobox2 = ttk.Combobox(self.frame, textvariable=prev.sc_affinity, width=13, 
            values=['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'])
        sc_combobox2.place(x=315,y=350)
        ttk.Label(self.frame, text='N neighbors', font=myfont2).place(x=220, y=375)
        sc_e5 = ttk.Entry(self.frame, textvariable=prev.sc_n_neighbors, font=myfont2, width=7)
        sc_e5.place(x=340, y=375)
        ttk.Label(self.frame, text='Eigen tol', font=myfont2).place(x=220, y=400)
        sc_e6 = ttk.Entry(self.frame, textvariable=prev.sc_eigen_tol, font=myfont2, width=7)
        sc_e6.place(x=340, y=400)
        ttk.Label(self.frame, text='Assign labels', font=myfont2).place(x=220, y=425)
        sc_combobox3 = ttk.Combobox(self.frame, textvariable=prev.sc_assign_labels, width=10, 
            values=['kmeans', 'kmeans'])
        sc_combobox3.place(x=340,y=425)
        ttk.Label(self.frame, text='Degree', font=myfont2).place(x=220, y=450)
        sc_e7 = ttk.Entry(self.frame, textvariable=prev.sc_degree, font=myfont2, width=7)
        sc_e7.place(x=340, y=450)
        ttk.Label(self.frame, text='coef0', font=myfont2).place(x=220, y=475)
        sc_e8 = ttk.Entry(self.frame, textvariable=prev.sc_coef0, font=myfont2, width=7)
        sc_e8.place(x=340, y=475)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=220, y=500)
        sc_e9 = ttk.Entry(self.frame, textvariable=prev.sc_n_jobs, font=myfont2, width=7)
        sc_e9.place(x=340, y=500)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=220, y=525)
        sc_cb1 = ttk.Checkbutton(self.frame, variable=prev.sc_verbose, takefocus=False)
        sc_cb1.place(x=340, y=525)

        ttk.Label(self.frame, text='DBSCAN', font=myfont_b).place(x=470, y=10)
        ttk.Label(self.frame, text='eps', font=myfont2).place(x=420, y=40)
        dbscan_e1 = ttk.Entry(self.frame, textvariable=prev.dbscan_eps, font=myfont2, width=7)
        dbscan_e1.place(x=540,y=40)
        ttk.Label(self.frame, text='Min samples', font=myfont2).place(x=420, y=65)
        dbscan_e2 = ttk.Entry(self.frame, 
            textvariable=prev.dbscan_min_samples, font=myfont2, width=7)
        dbscan_e2.place(x=540,y=65)
        ttk.Label(self.frame, text='Metric', font=myfont2).place(x=420, y=90)
        dbscan_combobox1 = ttk.Combobox(self.frame, textvariable=prev.dbscan_metric, width=8, 
            values=['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'cityblock'])
        dbscan_combobox1.place(x=535,y=90)
        ttk.Label(self.frame, text='Algorithm', font=myfont2).place(x=420, y=115)
        dbscan_combobox2 = ttk.Combobox(self.frame, textvariable=prev.dbscan_algorithm, width=7, 
            values=['auto', 'ball_tree', 'kd_tree', 'brute'])
        dbscan_combobox2.place(x=535,y=115)
        ttk.Label(self.frame, text='Leaf size', font=myfont2).place(x=420, y=140)
        dbscan_e3 = ttk.Entry(self.frame, textvariable=prev.dbscan_leaf_size, font=myfont2, width=7)
        dbscan_e3.place(x=540,y=140)
        ttk.Label(self.frame, text='P', font=myfont2).place(x=420, y=165)
        dbscan_e4 = ttk.Entry(self.frame, textvariable=prev.dbscan_p, font=myfont2, width=7)
        dbscan_e4.place(x=540,y=165)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=420, y=190)
        dbscan_e5 = ttk.Entry(self.frame, textvariable=prev.dbscan_n_jobs, font=myfont2, width=7)
        dbscan_e5.place(x=540,y=190)

        ttk.Label(self.frame, text='OPTICS', font=myfont_b).place(x=470, y=220)
        ttk.Label(self.frame, text='Min samples', font=myfont2).place(x=420, y=250)
        opt_e1 = ttk.Entry(self.frame, textvariable=prev.opt_min_samples, font=myfont2, width=7)
        opt_e1.place(x=540,y=250)
        ttk.Label(self.frame, text='Max eps', font=myfont2).place(x=420, y=275)
        opt_e2 = ttk.Entry(self.frame, textvariable=prev.opt_max_eps, font=myfont2, width=7)
        opt_e2.place(x=540,y=275)
        ttk.Label(self.frame, text='Metric', font=myfont2).place(x=420, y=300)
        opt_combobox1 = ttk.Combobox(self.frame, textvariable=prev.opt_metric, width=8, 
            values=['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'cityblock'])
        opt_combobox1.place(x=535,y=300)
        ttk.Label(self.frame, text='P', font=myfont2).place(x=420, y=325)
        opt_e3 = ttk.Entry(self.frame, textvariable=prev.opt_p, font=myfont2, width=7)
        opt_e3.place(x=540,y=325)
        ttk.Label(self.frame, text='Cluster method', font=myfont2).place(x=420, y=350)
        opt_combobox2 = ttk.Combobox(self.frame, textvariable=prev.opt_cluster_method, width=6, 
            values=['xi', 'dbscan'])
        opt_combobox2.place(x=540,y=350)
        ttk.Label(self.frame, text='Eps', font=myfont2).place(x=420, y=375)
        opt_e4 = ttk.Entry(self.frame, textvariable=prev.opt_eps, font=myfont2, width=7)
        opt_e4.place(x=540,y=375)
        ttk.Label(self.frame, text='Xi', font=myfont2).place(x=420, y=400)
        opt_e5 = ttk.Entry(self.frame, textvariable=prev.opt_xi, font=myfont2, width=7)
        opt_e5.place(x=540,y=400)
        ttk.Label(self.frame, text='Predecessor\ncorrection', font=myfont2).place(x=420, y=425)
        opt_cb1 = ttk.Checkbutton(self.frame, 
            variable=prev.opt_predecessor_correction, takefocus=False)
        opt_cb1.place(x=540, y=435)
        ttk.Label(self.frame, text='Min cluster size', font=myfont2).place(x=420, y=470)
        opt_e6 = ttk.Entry(self.frame, textvariable=prev.opt_min_cluster_size, font=myfont2, width=7)
        opt_e6.place(x=540,y=470)
        ttk.Label(self.frame, text='Algorithm', font=myfont2).place(x=420, y=495)
        opt_combobox3 = ttk.Combobox(self.frame, textvariable=prev.opt_algorithm, width=7, 
            values=['auto', 'ball_tree', 'kd_tree', 'brute'])
        opt_combobox3.place(x=535,y=495)
        ttk.Label(self.frame, text='Leaf size', font=myfont2).place(x=420, y=520)
        opt_e7 = ttk.Entry(self.frame, textvariable=prev.opt_leaf_size, font=myfont2, width=7)
        opt_e7.place(x=540,y=520)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=420, y=545)
        opt_e8 = ttk.Entry(self.frame, textvariable=prev.opt_n_jobs, font=myfont2, width=7)
        opt_e8.place(x=540,y=545)

        ttk.Label(self.frame, text='Affinity Propagation', font=myfont_b).place(x=620, y=10)
        ttk.Label(self.frame, text='Damping', font=myfont2).place(x=620, y=40)
        ap_e1 = ttk.Entry(self.frame, textvariable=prev.ap_damping, font=myfont2, width=7)
        ap_e1.place(x=740, y=40)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=620, y=65)
        ap_e2 = ttk.Entry(self.frame, textvariable=prev.ap_max_iter, font=myfont2, width=7)
        ap_e2.place(x=740, y=65)
        ttk.Label(self.frame, text='Convergence iter', font=myfont2).place(x=620, y=90)
        ap_e3 = ttk.Entry(self.frame, textvariable=prev.ap_convergence_iter, font=myfont2, width=7)
        ap_e3.place(x=740, y=90)
        ttk.Label(self.frame, text='Copy', font=myfont2).place(x=620, y=115)
        ap_cb1 = ttk.Checkbutton(self.frame, variable=prev.ap_copy, takefocus=False)
        ap_cb1.place(x=740, y=115)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=620, y=140)
        ap_cb2 = ttk.Checkbutton(self.frame, variable=prev.ap_verbose, takefocus=False)
        ap_cb2.place(x=740, y=140)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=620, y=165)
        ap_e4 = ttk.Entry(self.frame, textvariable=prev.ap_random_state, font=myfont2, width=7)
        ap_e4.place(x=740, y=165)

        ttk.Label(self.frame, text='Agglomerative Clustering', font=myfont_b).place(x=620, y=195)
        ttk.Label(self.frame, text='Affinity', font=myfont2).place(x=620, y=225)
        ac_combobox1 = ttk.Combobox(self.frame, textvariable=prev.ac_affinity, width=8, 
            values=['euclidean', 'l1', 'l2', 'manhattan', 'cosine'])
        ac_combobox1.place(x=740,y=225)
        ttk.Label(self.frame, text='Compute full tree', font=myfont2).place(x=620, y=250)
        ac_combobox2 = ttk.Combobox(self.frame, textvariable=prev.ac_compute_full_tree, width=6, 
            values=['auto', 'True', 'False'])
        ac_combobox2.place(x=740,y=250)
        ttk.Label(self.frame, text='Linkage', font=myfont2).place(x=620, y=275)
        ac_combobox3 = ttk.Combobox(self.frame, textvariable=prev.ac_linkage, width=7, 
            values=['ward', 'complete', 'average', 'single'])
        ac_combobox3.place(x=740,y=275)
        ttk.Label(self.frame, text='Distance threshold', font=myfont2).place(x=620, y=300)
        ac_e1 = ttk.Entry(self.frame, textvariable=prev.ac_distance_threshold, font=myfont2, width=7)
        ac_e1.place(x=740, y=300)
        ttk.Label(self.frame, text='Compute distances', font=myfont2).place(x=620, y=325)
        sc_cb1 = ttk.Checkbutton(self.frame, variable=prev.ac_compute_distances, takefocus=False)
        sc_cb1.place(x=740, y=325)

        ttk.Label(self.frame, text='Birch', font=myfont_b).place(x=650, y=355)
        ttk.Label(self.frame, text='Threshold', font=myfont2).place(x=620, y=385)
        bc_e1 = ttk.Entry(self.frame, textvariable=prev.bc_threshold, font=myfont2, width=7)
        bc_e1.place(x=740,y=385)
        ttk.Label(self.frame, text='Branching factor', font=myfont2).place(x=620, y=410)
        bc_e2 = ttk.Entry(self.frame, textvariable=prev.bc_branching_factor, font=myfont2, width=7)
        bc_e2.place(x=740,y=410)
        ttk.Label(self.frame, text='Compute labels', font=myfont2).place(x=620, y=435)
        bc_cb1 = ttk.Checkbutton(self.frame, variable=prev.bc_compute_labels, takefocus=False)
        bc_cb1.place(x=740, y=435)
        ttk.Label(self.frame, text='Copy', font=myfont2).place(x=620, y=460)
        bc_cb2 = ttk.Checkbutton(self.frame, variable=prev.bc_copy, takefocus=False)
        bc_cb2.place(x=740, y=460)

        ttk.Button(self.root, text='OK', width=5,
            command=lambda: quit_back(self.root, clust_app.root)).place(relx=0.85, rely=0.92)

    # function to restore default parameters
    def restore_defaults(self, prev, parent):
        if tk.messagebox.askyesno("Restore", "Restore default settings?"):
            prev.kmeans_init = tk.StringVar(value='k-means++')
            prev.kmeans_n_init = tk.StringVar(value='10')
            prev.kmeans_max_iter = tk.StringVar(value='300')
            prev.kmeans_tol = tk.StringVar(value='1e-4')
            prev.kmeans_verbose = tk.StringVar(value='0')
            prev.kmeans_random_state = tk.StringVar(value='None')
            prev.kmeans_copy_x = tk.BooleanVar(value=True)
            prev.kmeans_algorithm = tk.StringVar(value='auto')

            prev.ap_damping = tk.StringVar(value='0.5')
            prev.ap_max_iter = tk.StringVar(value='200')
            prev.ap_convergence_iter = tk.StringVar(value='15')
            prev.ap_copy = tk.BooleanVar(value=True)
            prev.ap_verbose = tk.BooleanVar(value=False)
            prev.ap_random_state = tk.StringVar(value='0')

            prev.ms_bandwidth = tk.StringVar(value='None')
            prev.ms_bin_seeding = tk.BooleanVar(value=False)
            prev.ms_min_bin_freq = tk.StringVar(value='1')
            prev.ms_cluster_all = tk.BooleanVar(value=True)
            prev.ms_n_jobs = tk.StringVar(value='None')
            prev.ms_max_iter = tk.StringVar(value='300')

            prev.sc_eigen_solver = tk.StringVar(value='arpack')
            prev.sc_n_components = tk.StringVar(value='None')
            prev.sc_random_state = tk.StringVar(value='None')
            prev.sc_n_init = tk.StringVar(value='10')
            prev.sc_gamma = tk.StringVar(value='1.0')
            prev.sc_affinity = tk.StringVar(value='rbf')
            prev.sc_n_neighbors = tk.StringVar(value='None')
            prev.sc_eigen_tol = tk.StringVar(value='0.0')
            prev.sc_assign_labels = tk.StringVar(value='kmeans')
            prev.sc_degree = tk.StringVar(value='3')
            prev.sc_coef0 = tk.StringVar(value='1')
            prev.sc_n_jobs = tk.StringVar(value='None')
            prev.sc_verbose = tk.BooleanVar(value=False)

            prev.ac_affinity = tk.StringVar(value='euclidean')
            prev.ac_compute_full_tree = tk.StringVar(value='auto')
            prev.ac_linkage = tk.StringVar(value='ward')
            prev.ac_distance_threshold = tk.StringVar(value='None')
            prev.ac_compute_distances = tk.BooleanVar(value=False)

            prev.dbscan_eps = tk.StringVar(value='0.5')
            prev.dbscan_min_samples = tk.StringVar(value='5')
            prev.dbscan_metric = tk.StringVar(value='euclidean')
            prev.dbscan_algorithm = tk.StringVar(value='auto')
            prev.dbscan_leaf_size = tk.StringVar(value='30')
            prev.dbscan_p = tk.StringVar(value='None')
            prev.dbscan_n_jobs = tk.StringVar(value='None')

            prev.opt_min_samples = tk.StringVar(value='5')
            prev.opt_max_eps = tk.StringVar(value='np.inf')
            prev.opt_metric = tk.StringVar(value='minkowski')
            prev.opt_p = tk.StringVar(value='2')
            prev.opt_cluster_method = tk.StringVar(value='xi')
            prev.opt_eps = tk.StringVar(value='None')
            prev.opt_xi = tk.StringVar(value='0.05')
            prev.opt_predecessor_correction = tk.BooleanVar(value=True)
            prev.opt_min_cluster_size = tk.StringVar(value='None')
            prev.opt_algorithm = tk.StringVar(value='auto')
            prev.opt_leaf_size = tk.StringVar(value='30')
            prev.opt_n_jobs = tk.StringVar(value='None')

            prev.bc_threshold = tk.StringVar(value='0.5')
            prev.bc_branching_factor = tk.StringVar(value='50')
            prev.bc_compute_labels = tk.BooleanVar(value=True)
            prev.bc_copy = tk.BooleanVar(value=True)

            quit_back(self.root, prev.root)
            clust_mtds_specification(prev, parent)

    def save_settings(self, prev, parent):
        save_file = open(asksaveasfilename(parent=self.root, defaultextension=".txt",
            filetypes = (("Text files","*.txt"),)
            ), 'w')
        save_file.write(
            str({
                'kmeans_init' : prev.kmeans_init.get(),
                'kmeans_n_init' : prev.kmeans_n_init.get(),
                'kmeans_max_iter' : prev.kmeans_max_iter.get(),
                'kmeans_tol' : prev.kmeans_tol.get(),
                'kmeans_verbose' : prev.kmeans_verbose.get(),
                'kmeans_random_state' : prev.kmeans_random_state.get(),
                'kmeans_copy_x' : prev.kmeans_copy_x.get(),
                'kmeans_algorithm' : prev.kmeans_algorithm.get(),

                'ap_damping' : prev.ap_damping.get(),
                'ap_max_iter' : prev.ap_max_iter.get(),
                'ap_convergence_iter' : prev.ap_convergence_iter.get(),
                'ap_copy' : prev.ap_copy.get(),
                'ap_verbose' : prev.ap_verbose.get(),
                'ap_random_state' : prev.ap_random_state.get(),

                'ms_bandwidth' : prev.ms_bandwidth.get(),
                'ms_bin_seeding' : prev.ms_bin_seeding.get(),
                'ms_min_bin_freq' : prev.ms_min_bin_freq.get(),
                'ms_cluster_all' : prev.ms_cluster_all.get(),
                'ms_n_jobs' : prev.ms_n_jobs.get(),
                'ms_max_iter' : prev.ms_max_iter.get(),

                'sc_eigen_solver' : prev.sc_eigen_solver.get(),
                'sc_n_components' : prev.sc_n_components.get(),
                'sc_random_state' : prev.sc_random_state.get(),
                'sc_n_init' : prev.sc_n_init.get(),
                'sc_gamma' : prev.sc_gamma.get(),
                'sc_affinity' : prev.sc_affinity.get(),
                'sc_n_neighbors' : prev.sc_n_neighbors.get(),
                'sc_eigen_tol' : prev.sc_eigen_tol.get(),
                'sc_assign_labels' : prev.sc_assign_labels.get(),
                'sc_degree' : prev.sc_degree.get(),
                'sc_coef0' : prev.sc_coef0.get(),
                'sc_n_jobs' : prev.sc_n_jobs.get(),
                'sc_verbose' : prev.sc_verbose.get(),

                'ac_affinity' : prev.ac_affinity.get(),
                'ac_compute_full_tree' : prev.ac_compute_full_tree.get(),
                'ac_linkage' : prev.ac_linkage.get(),
                'ac_distance_threshold' : prev.ac_distance_threshold.get(),
                'ac_compute_distances' : prev.ac_compute_distances.get(),

                'dbscan_eps' : prev.dbscan_eps.get(),
                'dbscan_min_samples' : prev.dbscan_min_samples.get(),
                'dbscan_metric' : prev.dbscan_metric.get(),
                'dbscan_algorithm' : prev.dbscan_algorithm.get(),
                'dbscan_leaf_size' : prev.dbscan_leaf_size.get(),
                'dbscan_p' : prev.dbscan_p.get(),
                'dbscan_n_jobs' : prev.dbscan_n_jobs.get(),

                'opt_min_samples' : prev.opt_min_samples.get(),
                'opt_max_eps' : prev.opt_max_eps.get(),
                'opt_metric' : prev.opt_metric.get(),
                'opt_p' : prev.opt_p.get(),
                'opt_cluster_method' : prev.opt_cluster_method.get(),
                'opt_eps' : prev.opt_eps.get(),
                'opt_xi' : prev.opt_xi.get(),
                'opt_predecessor_correction' : prev.opt_predecessor_correction.get(),
                'opt_min_cluster_size' : prev.opt_min_cluster_size.get(),
                'opt_algorithm' : prev.opt_algorithm.get(),
                'opt_leaf_size' : prev.opt_leaf_size.get(),
                'opt_n_jobs' :prev.opt_n_jobs.get(),

                'bc_threshold' : prev.bc_threshold.get(),
                'bc_branching_factor' : prev.bc_branching_factor.get(),
                'bc_compute_labels' : prev.bc_compute_labels.get(),
                'bc_copy' : prev.bc_copy.get(),
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
            # setattr(prev, key, tk.StringVar(value=settings_dict[key]))

        quit_back(self.root, prev.root)
        clust_mtds_specification(prev, parent)