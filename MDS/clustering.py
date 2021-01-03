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
from .utils import Data_Preview, quit_back, open_file, load_data, save_results

myfont = (None, 13)
myfont_b = (None, 13, 'bold')
myfont1 = (None, 11)
myfont1_b = (None, 11, 'bold')
myfont2 = (None, 10)
myfont2_b = (None, 10, 'bold')

class cls_app:
    class clust:
        def __init__(self):
            pass
    class prediction:
        def __init__(self):
            pass
    class elbow_method:
        def __init__(self, prev, main, parent):
            from sklearn.cluster import KMeans
            from yellowbrick.cluster import KElbowVisualizer
            from sklearn import preprocessing
            
            scaler = preprocessing.StandardScaler()
            X_St = scaler.fit_transform(main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())])

            visualizer = KElbowVisualizer(KMeans(), metric=prev.elbow_metric.get(), 
                                          k=(int(prev.elbow_k_from.get()),int(prev.elbow_k_to.get())))
            visualizer.fit(X_St)
            visualizer.show()
    class show_dendrogram:
        def __init__(self, prev, main, parent):
            from scipy.cluster.hierarchy import dendrogram, linkage
            from matplotlib import pyplot as plt
            from sklearn import preprocessing
            
            scaler = preprocessing.StandardScaler()
            X_St = scaler.fit_transform(main.data.iloc[:,(int(main.x_from_var.get())-1): int(main.x_to_var.get())])

            # plt.rcParams["figure.figsize"] = (5,5)

            plt.title('Hierarchical Clustering Dendrogram')
            dendrogram(linkage(X_St, method=prev.dendr_linkage.get()))
            plt.show()
    def __init__(self, parent):
        self.clust.data = None
        self.clust.sheet = tk.StringVar()
        self.clust.sheet.set('1')
        self.clust.header_var = tk.IntVar()
        self.clust.header_var.set(1)
        self.clust.x_from_var = tk.StringVar()
        self.clust.x_to_var = tk.StringVar()
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
        self.root.title('Monkey Clustering')
        self.root.lift()
        self.root.tkraise()
        self.root.focus_force()
        self.root.resizable(False, False)
        
        self.frame = ttk.Frame(self.root, width=w, height=h)
        self.frame.place(x=0, y=0)

        ttk.Label(self.frame, text='Data file:').place(x=30, y=10)
        e1 = ttk.Entry(self.frame, font=myfont1, width=40)
        e1.place(x=120, y=10)

        ttk.Button(self.frame, text='Choose file', command=lambda: open_file(self, e1)).place(x=490, y=10)
        
        self.dummies_var = tk.IntVar(value=0)

        ttk.Label(self.frame, text='List number:').place(x=120,y=50)
        training_sheet_entry = ttk.Entry(self.frame, textvariable=self.clust.sheet, font=myfont1, width=3)
        training_sheet_entry.place(x=215,y=52)

        ttk.Button(self.frame, text='Load data ', command=lambda: load_data(self, self.clust, e1, 'clust')).place(x=490, y=50)
        
        cb2 = ttk.Checkbutton(self.frame, text="Dummies", variable=self.dummies_var, takefocus=False)
        cb2.place(x=270, y=175)
        cb1 = ttk.Checkbutton(self.frame, text="header", variable=self.clust.header_var, takefocus=False)
        cb1.place(x=10, y=50)

        ttk.Label(self.frame, text='Data status:').place(x=10, y=100)
        self.cls_data_status = ttk.Label(self.frame, text='Not Loaded')
        self.cls_data_status.place(x=120, y=100)

        ttk.Button(self.frame, text='View/Change', 
                   command=lambda: Data_Preview(self, self.clust, 'clust', parent)).place(x=230, y=100)

        ttk.Label(self.frame, text='Number of clusters:', font=myfont1).place(x=30, y=140)
        self.n_clusters_var = tk.StringVar(value='2')
        self.n_clusters_entry = ttk.Entry(self.frame, textvariable=self.n_clusters_var, font=myfont1, width=3)
        self.n_clusters_entry.place(x=170, y=142)

        ttk.Button(self.frame, text="Methods' specifications", 
                  command=lambda: cls_mtds_specification(self, parent)).place(x=400, y=100)

        self.elbow_metric = tk.StringVar(value='silhouette')
        self.elbow_k_from = tk.StringVar(value='2')
        self.elbow_k_to = tk.StringVar(value='11')
        self.dendr_linkage = tk.StringVar(value='ward')

        ttk.Button(self.frame, text='Elbow method', command=lambda: self.elbow_method(self, self.clust, parent)).place(x=400, y=140)

        ttk.Button(self.frame, text='Dendrogram', command=lambda: self.show_dendrogram(self, self.clust, parent)).place(x=400, y=180)

        ttk.Label(self.frame, text='X from', font=myfont1).place(x=205, y=140)
        self.cls_x_from_combobox = ttk.Combobox(self.frame, textvariable=self.clust.x_from_var, width=4, values=[])
        self.cls_x_from_combobox.place(x=255, y=142)
        ttk.Label(self.frame, text='to', font=myfont1).place(x=305, y=140)
        self.cls_x_to_combobox = ttk.Combobox(self.frame, textvariable=self.clust.x_to_var, width=4, values=[])
        self.cls_x_to_combobox.place(x=325, y=142)

        self.x_st_var = tk.StringVar(value='If needed')
        ttk.Label(self.frame, text='X Standartization', font=myfont1).place(x=30, y=170)
        self.combobox2 = ttk.Combobox(self.frame, textvariable=self.x_st_var, width=13,
                                        values=['No', 'If needed', 'Yes'])
        self.combobox2.place(x=150,y=175)

        ttk.Label(self.frame, text='Choose method', font=myfont1).place(x=30, y=230)
        self.cls_method = tk.StringVar(value='K-Means')
        self.combobox9 = ttk.Combobox(self.frame, textvariable=self.cls_method, values=['K-Means', 'Affinity Propagation', 
                                                                                        'Mean Shift', 'Spectral clustering',
                                                                                        'Hierarchical clustering', 'DBSCAN',
                                                                                        'OPTICS', 'Birch'
                                                                                        ])
        self.combobox9.place(x=200,y=232)

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

        def cls_predict_cluster(method):
            if self.dummies_var.get()==0:
                X = self.clust.data.iloc[:,(int(self.clust.x_from_var.get())-1): int(self.clust.x_to_var.get())]
            elif self.dummies_var.get()==1:
                X = pd.get_dummies(self.clust.data.iloc[:,(int(self.clust.x_from_var.get())-1): int(self.clust.x_to_var.get())])
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            X_St = scaler.fit_transform(X)

            if method == 'K-Means':
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=int(self.n_clusters_entry.get()), init=self.kmeans_init.get(),
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
                ap = AffinityPropagation(damping=float(self.ap_damping.get()), max_iter=int(self.ap_max_iter.get()),
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
                ms = MeanShift(bandwidth=(float(self.ms_bandwidth.get()) if (self.ms_bandwidth.get() != 'None') else None), 
                               bin_seeding=self.ms_bin_seeding.get(),
                               min_bin_freq=int(self.ms_min_bin_freq.get()), cluster_all=self.ms_cluster_all.get(),
                               n_jobs=(int(self.ms_n_jobs.get()) if 
                                       (self.ms_n_jobs.get() != 'None') else None),
                               max_iter=int(self.ms_max_iter.get()))
                if self.x_st_var.get() == 'No':
                    pr_values = ms.fit_predict(X)
                else:
                    pr_values = ms.fit_predict(X_St)
            elif method == 'Spectral clustering':
                from sklearn.cluster import SpectralClustering
                sc = SpectralClustering(n_clusters=int(self.n_clusters_entry.get()), eigen_solver=self.sc_eigen_solver.get(),
                                        n_components=(int(self.sc_n_components.get()) if 
                                                      (self.sc_n_components.get() != 'None') else None),
                                        random_state=(int(self.sc_random_state.get()) if 
                                                      (self.sc_random_state.get() != 'None') else None),
                                        n_init=int(self.sc_n_init.get()), gamma=float(self.sc_gamma.get()),
                                        affinity=self.sc_affinity.get(), 
                                        n_neighbors=(int(self.sc_n_neighbors.get()) if 
                                                     (self.sc_n_neighbors.get() != 'None') else None),
                                        eigen_tol=float(self.sc_eigen_tol.get()), assign_labels=self.sc_assign_labels.get(),
                                        degree=float(self.sc_degree.get()), coef0=float(self.sc_coef0.get()),
                                        n_jobs=(int(self.sc_n_jobs.get()) if 
                                                (self.sc_n_jobs.get() != 'None') else None),
                                        verbose=self.sc_verbose.get())
                if self.x_st_var.get() == 'No':
                    pr_values = sc.fit_predict(X)
                else:
                    pr_values = sc.fit_predict(X_St)
            elif method == 'Hierarchical clustering':
                from sklearn.cluster import AgglomerativeClustering
                ac = AgglomerativeClustering(n_clusters=int(self.n_clusters_entry.get()), affinity=self.ac_affinity.get(),
                                             compute_full_tree=self.ac_compute_full_tree.get(), linkage=self.ac_linkage.get(),
                                             distance_threshold=(float(self.ac_distance_threshold.get()) if 
                                                                 (self.ac_distance_threshold.get() != 'None') else None),
                                             compute_distances=self.ac_compute_distances.get())
                if self.x_st_var.get() == 'No':
                    pr_values = ac.fit_predict(X)
                else:
                    pr_values = ac.fit_predict(X_St)
            elif method == 'DBSCAN':
                from sklearn.cluster import DBSCAN
                dbscan = DBSCAN(eps=float(self.dbscan_eps.get()), min_samples=int(self.dbscan_min_samples.get()),
                                metric=self.dbscan_metric.get(), algorithm=self.dbscan_algorithm.get(),
                                leaf_size=int(self.dbscan_leaf_size.get()),
                                p=(float(self.dbscan_p.get()) if 
                                   (self.dbscan_p.get() != 'None') else None),
                                n_jobs=(int(self.dbscan_n_jobs.get()) if 
                                        (self.dbscan_n_jobs.get() != 'None') else None))
                if self.x_st_var.get() == 'No':
                    pr_values = dbscan.fit_predict(X)
                else:
                    pr_values = dbscan.fit_predict(X_St)
            elif method == 'OPTICS':
                from sklearn.cluster import OPTICS
                optics = OPTICS()
                if self.x_st_var.get() == 'No':
                    pr_values = optics.fit_predict(X)
                else:
                    pr_values = optics.fit_predict(X_St)
            elif method == 'Birch':
                from sklearn.cluster import Birch
                bc = Birch(n_clusters=int(self.n_clusters_entry.get()))
                if self.x_st_var.get() == 'No':
                    pr_values = bc.fit_predict(X)
                else:
                    pr_values = bc.fit_predict(X_St)

            self.clust.data['Cluster'] = pr_values
            

        ttk.Button(self.frame, text='Predict clusters', 
                  command=lambda: cls_predict_cluster(method=self.cls_method.get())).place(x=400, y=230)

        ttk.Button(self.frame, text='Quit', 
                  command=lambda: quit_back(self.root, parent)).place(x=420, y=290)

class cls_mtds_specification:
    def __init__(self, prev, parent):
        self.root = tk.Toplevel(parent)

        w = 690
        h = 640
        #setting main window's parameters       
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.grab_set()
        self.root.focus_force()
        self.root.resizable(False, False)
        self.root.title('Clustering methods specification')

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

        ttk.Label(self.frame, text='Elbow method', font=myfont_b).place(x=30, y=10)
        ttk.Label(self.frame, text='Metric', font=myfont2).place(x=5, y=40)
        elbow_combobox1 = ttk.Combobox(self.frame, textvariable=prev.elbow_metric, width=11, values=['distortion', 'silhouette',
                                                                                                     'calinski_harabasz'])
        elbow_combobox1.place(x=95,y=40)
        ttk.Label(self.frame, text='K from', font=myfont2).place(x=5, y=60)
        elbow_e1 = ttk.Entry(self.frame, textvariable=prev.elbow_k_from, font=myfont2, width=5)
        elbow_e1.place(x=110, y=60)
        ttk.Label(self.frame, text='K to', font=myfont2).place(x=5, y=80)
        elbow_e2 = ttk.Entry(self.frame, textvariable=prev.elbow_k_to, font=myfont2, width=5)
        elbow_e2.place(x=110, y=80)

        ttk.Label(self.frame, text='Dendrogram', font=myfont_b).place(x=30, y=110)
        ttk.Label(self.frame, text='Linkage', font=myfont2).place(x=5, y=140)
        dendr_combobox1 = ttk.Combobox(self.frame, textvariable=prev.dendr_linkage, width=7, values=['ward', 'complete', 'average',
                                                                                                      'single'])
        dendr_combobox1.place(x=105,y=140)

        ttk.Label(self.frame, text='K-Means', font=myfont_b).place(x=30, y=170)
        ttk.Label(self.frame, text='Init', font=myfont2).place(x=5, y=200)
        kmeans_combobox1 = ttk.Combobox(self.frame, textvariable=prev.kmeans_init, width=10, values=['k-means++', 'random'])
        kmeans_combobox1.place(x=100,y=200)
        ttk.Label(self.frame, text='N init', font=myfont2).place(x=5, y=220)
        kmeans_e1 = ttk.Entry(self.frame, textvariable=prev.kmeans_n_init, font=myfont2, width=5)
        kmeans_e1.place(x=110, y=220)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=5, y=240)
        kmeans_e2 = ttk.Entry(self.frame, textvariable=prev.kmeans_max_iter, font=myfont2, width=5)
        kmeans_e2.place(x=110, y=240)
        ttk.Label(self.frame, text='tol', font=myfont2).place(x=5, y=260)
        kmeans_e3 = ttk.Entry(self.frame, textvariable=prev.kmeans_tol, font=myfont2, width=5)
        kmeans_e3.place(x=110, y=260)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=5, y=280)
        kmeans_e4 = ttk.Entry(self.frame, textvariable=prev.kmeans_verbose, font=myfont2, width=5)
        kmeans_e4.place(x=110, y=280)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=5, y=300)
        kmeans_e5 = ttk.Entry(self.frame, textvariable=prev.kmeans_random_state, font=myfont2, width=5)
        kmeans_e5.place(x=110, y=300)
        ttk.Label(self.frame, text='Copy X', font=myfont2).place(x=5, y=320)
        kmeans_cb1 = ttk.Checkbutton(self.frame, variable=prev.kmeans_copy_x, takefocus=False)
        kmeans_cb1.place(x=110, y=320)
        ttk.Label(self.frame, text='Algorithm', font=myfont2).place(x=5, y=340)
        kmeans_combobox2 = ttk.Combobox(self.frame, textvariable=prev.kmeans_algorithm, width=7, values=['auto', 'full', 'elkan'])
        kmeans_combobox2.place(x=105,y=340)

        ttk.Label(self.frame, text='Affinity Propagation', font=myfont_b).place(x=5, y=370)
        ttk.Label(self.frame, text='Damping', font=myfont2).place(x=5, y=400)
        ap_e1 = ttk.Entry(self.frame, textvariable=prev.ap_damping, font=myfont2, width=5)
        ap_e1.place(x=110, y=400)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=5, y=420)
        ap_e2 = ttk.Entry(self.frame, textvariable=prev.ap_max_iter, font=myfont2, width=5)
        ap_e2.place(x=110, y=420)
        ttk.Label(self.frame, text='Convergence iter', font=myfont2).place(x=5, y=440)
        ap_e3 = ttk.Entry(self.frame, textvariable=prev.ap_convergence_iter, font=myfont2, width=5)
        ap_e3.place(x=110, y=440)
        ttk.Label(self.frame, text='Copy', font=myfont2).place(x=5, y=460)
        ap_cb1 = ttk.Checkbutton(self.frame, variable=prev.ap_copy, takefocus=False)
        ap_cb1.place(x=110, y=460)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=5, y=480)
        ap_cb2 = ttk.Checkbutton(self.frame, variable=prev.ap_verbose, takefocus=False)
        ap_cb2.place(x=110, y=480)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=5, y=500)
        ap_e4 = ttk.Entry(self.frame, textvariable=prev.ap_random_state, font=myfont2, width=5)
        ap_e4.place(x=110, y=500)

        ttk.Label(self.frame, text='Mean Shift', font=myfont_b).place(x=205, y=10)
        ttk.Label(self.frame, text='Bandwidth', font=myfont2).place(x=185, y=40)
        ms_e1 = ttk.Entry(self.frame, textvariable=prev.ms_bandwidth, font=myfont2, width=5)
        ms_e1.place(x=290, y=40)
        ttk.Label(self.frame, text='Bin seeding', font=myfont2).place(x=185, y=60)
        ms_cb1 = ttk.Checkbutton(self.frame, variable=prev.ms_bin_seeding, takefocus=False)
        ms_cb1.place(x=290, y=60)
        ttk.Label(self.frame, text='Min bin freq', font=myfont2).place(x=185, y=80)
        ms_e2 = ttk.Entry(self.frame, textvariable=prev.ms_min_bin_freq, font=myfont2, width=5)
        ms_e2.place(x=290, y=80)
        ttk.Label(self.frame, text='Cluster all', font=myfont2).place(x=185, y=100)
        ms_cb2 = ttk.Checkbutton(self.frame, variable=prev.ms_cluster_all, takefocus=False)
        ms_cb2.place(x=290, y=100)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=185, y=120)
        ms_e3 = ttk.Entry(self.frame, textvariable=prev.ms_n_jobs, font=myfont2, width=5)
        ms_e3.place(x=290, y=120)
        ttk.Label(self.frame, text='Max iter', font=myfont2).place(x=185, y=140)
        ms_e4 = ttk.Entry(self.frame, textvariable=prev.ms_max_iter, font=myfont2, width=5)
        ms_e4.place(x=290, y=140)

        ttk.Label(self.frame, text='Spectral Clustering', font=myfont_b).place(x=195, y=170)
        ttk.Label(self.frame, text='Eigen Solver', font=myfont2).place(x=185, y=200)
        sc_combobox1 = ttk.Combobox(self.frame, textvariable=prev.sc_eigen_solver, width=7, values=['arpack', 'lobpcg', 'amg'])
        sc_combobox1.place(x=285,y=200)
        ttk.Label(self.frame, text='N components', font=myfont2).place(x=185, y=220)
        sc_e1 = ttk.Entry(self.frame, textvariable=prev.sc_n_components, font=myfont2, width=5)
        sc_e1.place(x=290, y=220)
        ttk.Label(self.frame, text='Random state', font=myfont2).place(x=185, y=240)
        sc_e2 = ttk.Entry(self.frame, textvariable=prev.sc_random_state, font=myfont2, width=5)
        sc_e2.place(x=290, y=240)
        ttk.Label(self.frame, text='n init', font=myfont2).place(x=185, y=260)
        sc_e3 = ttk.Entry(self.frame, textvariable=prev.sc_n_init, font=myfont2, width=5)
        sc_e3.place(x=290, y=260)
        ttk.Label(self.frame, text='Gamma', font=myfont2).place(x=185, y=280)
        sc_e4 = ttk.Entry(self.frame, textvariable=prev.sc_gamma, font=myfont2, width=5)
        sc_e4.place(x=290, y=280)
        ttk.Label(self.frame, text='Affinity', font=myfont2).place(x=185, y=300)
        sc_combobox2 = ttk.Combobox(self.frame, textvariable=prev.sc_affinity, width=13, values=['nearest_neighbors', 'rbf',
                                                                                                 'precomputed',
                                                                                                 'precomputed_nearest_neighbors'])
        sc_combobox2.place(x=265,y=300)
        ttk.Label(self.frame, text='N neighbors', font=myfont2).place(x=185, y=320)
        sc_e5 = ttk.Entry(self.frame, textvariable=prev.sc_n_neighbors, font=myfont2, width=5)
        sc_e5.place(x=290, y=320)
        ttk.Label(self.frame, text='Eigen tol', font=myfont2).place(x=185, y=340)
        sc_e6 = ttk.Entry(self.frame, textvariable=prev.sc_eigen_tol, font=myfont2, width=5)
        sc_e6.place(x=290, y=340)
        ttk.Label(self.frame, text='Assign labels', font=myfont2).place(x=185, y=360)
        sc_combobox3 = ttk.Combobox(self.frame, textvariable=prev.sc_assign_labels, width=10, values=['kmeans', 'kmeans'])
        sc_combobox3.place(x=285,y=360)
        ttk.Label(self.frame, text='Degree', font=myfont2).place(x=185, y=380)
        sc_e7 = ttk.Entry(self.frame, textvariable=prev.sc_degree, font=myfont2, width=5)
        sc_e7.place(x=290, y=380)
        ttk.Label(self.frame, text='coef0', font=myfont2).place(x=185, y=400)
        sc_e8 = ttk.Entry(self.frame, textvariable=prev.sc_coef0, font=myfont2, width=5)
        sc_e8.place(x=290, y=400)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=185, y=420)
        sc_e9 = ttk.Entry(self.frame, textvariable=prev.sc_n_jobs, font=myfont2, width=5)
        sc_e9.place(x=290, y=420)
        ttk.Label(self.frame, text='Verbose', font=myfont2).place(x=185, y=440)
        sc_cb1 = ttk.Checkbutton(self.frame, variable=prev.sc_verbose, takefocus=False)
        sc_cb1.place(x=290, y=440)

        ttk.Label(self.frame, text='Agglomerative Clustering', font=myfont_b).place(x=175, y=470)
        ttk.Label(self.frame, text='Affinity', font=myfont2).place(x=185, y=500)
        ac_combobox1 = ttk.Combobox(self.frame, textvariable=prev.ac_affinity, width=8, values=['euclidean', 'l1', 'l2',
                                                                                                    'manhattan', 'cosine'])
        ac_combobox1.place(x=295,y=500)
        ttk.Label(self.frame, text='Compute full tree', font=myfont2).place(x=185, y=520)
        ac_combobox2 = ttk.Combobox(self.frame, textvariable=prev.ac_compute_full_tree, width=6, values=['auto', 'True', 'False'])
        ac_combobox2.place(x=300,y=520)
        ttk.Label(self.frame, text='Linkage', font=myfont2).place(x=185, y=540)
        ac_combobox3 = ttk.Combobox(self.frame, textvariable=prev.ac_linkage, width=7, values=['ward', 'complete', 'average',
                                                                                               'single'])
        ac_combobox3.place(x=300,y=540)
        ttk.Label(self.frame, text='Distance threshold', font=myfont2).place(x=185, y=560)
        ac_e1 = ttk.Entry(self.frame, textvariable=prev.ac_distance_threshold, font=myfont2, width=5)
        ac_e1.place(x=300, y=560)
        ttk.Label(self.frame, text='Compute distances', font=myfont2).place(x=185, y=580)
        sc_cb1 = ttk.Checkbutton(self.frame, variable=prev.ac_compute_distances, takefocus=False)
        sc_cb1.place(x=300, y=580)

        ttk.Label(self.frame, text='DBSCAN', font=myfont_b).place(x=420, y=10)
        ttk.Label(self.frame, text='eps', font=myfont2).place(x=390, y=40)
        dbscan_e1 = ttk.Entry(self.frame, textvariable=prev.dbscan_eps, font=myfont2, width=5)
        dbscan_e1.place(x=495,y=40)
        ttk.Label(self.frame, text='Min samples', font=myfont2).place(x=390, y=60)
        dbscan_e2 = ttk.Entry(self.frame, textvariable=prev.dbscan_min_samples, font=myfont2, width=5)
        dbscan_e2.place(x=495,y=60)
        ttk.Label(self.frame, text='Metric', font=myfont2).place(x=390, y=80)
        dbscan_combobox1 = ttk.Combobox(self.frame, textvariable=prev.dbscan_metric, width=8, values=['euclidean', 'l1', 'l2',
                                                                                                    'manhattan', 'cosine', 'cityblock'])
        dbscan_combobox1.place(x=490,y=80)
        ttk.Label(self.frame, text='Algorithm', font=myfont2).place(x=390, y=100)
        dbscan_combobox2 = ttk.Combobox(self.frame, textvariable=prev.dbscan_algorithm, width=7, values=['auto', 'ball_tree', 'kd_tree',
                                                                                                         'brute'])
        dbscan_combobox2.place(x=490,y=100)
        ttk.Label(self.frame, text='Leaf size', font=myfont2).place(x=390, y=120)
        dbscan_e3 = ttk.Entry(self.frame, textvariable=prev.dbscan_leaf_size, font=myfont2, width=5)
        dbscan_e3.place(x=495,y=120)
        ttk.Label(self.frame, text='P', font=myfont2).place(x=390, y=140)
        dbscan_e4 = ttk.Entry(self.frame, textvariable=prev.dbscan_p, font=myfont2, width=5)
        dbscan_e4.place(x=495,y=140)
        ttk.Label(self.frame, text='n jobs', font=myfont2).place(x=390, y=160)
        dbscan_e5 = ttk.Entry(self.frame, textvariable=prev.dbscan_n_jobs, font=myfont2, width=5)
        dbscan_e5.place(x=495,y=160)

        ttk.Button(self.root, text='OK', command=lambda: quit_back(self.root, prev.root)).place(relx=0.85, rely=0.92)