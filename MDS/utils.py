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

# master = ThemedTk(theme=open('settings\style.txt', 'r').read(), fonts=True)

myfont = (None, 13)
myfont_b = (None, 13, 'bold')
myfont1 = (None, 11)
myfont1_b = (None, 11, 'bold')
myfont2 = (None, 10)
myfont2_b = (None, 10, 'bold')

# style = ttk.Style()
#style.theme_use('xpnative')
# style.configure('.', font=myfont)

big_themes = ['blue', 'clam', 'kroc', 'radiance', 'smog', 'ubuntu']

class Data_Preview:
    def __init__(self, prev, main, full, parent):
        if main.data is not None:
            self.root = tk.Toplevel(parent)
            w = 600
            h = 650
            #setting main window's parameters       
            x = (parent.ws/2) - (w/2)
            y = (parent.hs/2) - (h/2) - 30
            self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
            self.root.lift()
            self.root.resizable(False, False)
            self.root.title('Data Preview')
            self.f = ttk.Frame(self.root, width=w, height=h)
            self.f.place(y=0)
            self.f1 = ttk.Frame(self.root, width=w, height=(h-50))
            if parent.current_theme in big_themes:
                self.f1.place(y=0)
            else:
                self.f1.place(y=50)
            self.pt = Table(self.f1, dataframe=main.data, showtoolbar=True, showstatusbar=True)
            self.pt.show()
            ttk.Button(self.f, text='Update Data', command=lambda: 
                       self.update_data(prev, main, self.pt.model.df, full)).place(x=10, y=580)
            ttk.Button(self.f, text='Close', command=lambda: quit_back(self.root, prev.root)).place(x=500, y=610)
            ttk.Button(self.f, text='Drop NA', command=lambda: self.drop_na()).place(x=130, y=580)
            ttk.Button(self.f, text='Fill NA', command=lambda: self.fill_na(parent)).place(x=130, y=610)
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
    def fill_na(self, parent):
        self.fn_win = tk.Toplevel(self.root)
        w=200
        h=150
        #setting main window's parameters       
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
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
        elif data_type=='clust':
            app.cls_data_status.config(text='Loaded')
            app.cls_x_from_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            app.cls_x_from_combobox.set('1')
            app.cls_x_to_combobox.config(values=list(range(1, main.data.shape[1]+1)))
            app.cls_x_to_combobox.set(str(main.data.shape[1]))

def save_results(prev, main):
    files = [('Excel', '*.xlsx'),
             ('All Files', '*.*'), 
             ('Text Document', '*.txt')] 
    file = asksaveasfilename(parent=prev.frame, filetypes = files, defaultextension = files)
    if os.path.isfile(file):
        with pd.ExcelWriter(file, engine="openpyxl", mode='a') as writer: 
            pd.DataFrame(main.data).to_excel(writer, sheet_name='Result', 
                                                  index=False, header=(False if main.header_var.get()==0 else True))
    else:
        pd.DataFrame(main.data).to_excel(file, sheet_name='Result', 
                                                  index=False, header=(False if main.header_var.get()==0 else True))