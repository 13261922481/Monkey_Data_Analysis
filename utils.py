# Imports
import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
import sys
import numpy as np
import webbrowser
import pandas as pd
import sklearn
from monkey_pt import Table

#fonts to be used
myfont = (None, 13)
myfont_b = (None, 13, 'bold')
myfont1 = (None, 11)
myfont1_b = (None, 11, 'bold')
myfont2 = (None, 10)
myfont2_b = (None, 10, 'bold')

# App to view/change data. Based on dmnfarrell's pandastable
class DataPreview:
    class notebook:
        def __init__(self):
            pass
    def __init__(self, prev, main, full, parent):
        global data_view_frame, w, h
        if main.data is not None and main.Viewed.get()==False:
            if not hasattr(DataPreview, 'root'):
                DataPreview.root = tk.Toplevel(parent)
                w = 650
                h = 650
                data_view_frame = ttk.Frame(DataPreview.root, width=w, height=h)
                data_view_frame.place(y=0)
                #setting main window's parameters       
                x = (parent.ws/2) - (w/2)
                y = (parent.hs/2) - (h/2) - 30
                DataPreview.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
                DataPreview.root.resizable(False, False)
                DataPreview.root.title('Data Viewer')

            if not hasattr(DataPreview.notebook, 'nb'):
                DataPreview.notebook.nb = ttk.Notebook(data_view_frame)
                DataPreview.notebook.nb.place(y=0)
            
            main.view_frame = ttk.Frame(DataPreview.root, width=w, height=(h-50))
            DataPreview.notebook.nb.add(main.view_frame, text=full)
            DataPreview.notebook.nb.select(main.view_frame)
            
            main.pt = Table(main.view_frame, dataframe=main.data, showtoolbar=True, 
                showstatusbar=True, height=450, notebook=DataPreview.notebook.nb, dp_main=main)
            main.pt.show()
            ttk.Button(main.view_frame, text='Update Data', command=lambda: 
                       self.update_data(prev, main, main.pt.model.df, full)).place(x=10, y=530)
            ttk.Button(main.view_frame, text='Update and Close', 
                command=lambda: self.upd_quit(main.view_frame, prev, main, 
                main.pt.model.df, full)).place(x=450, y=530)
            
            ttk.Button(main.view_frame, text='Drop NA', width=7,
                command=lambda: self.drop_na(main)).place(x=140, y=530)
            ttk.Button(main.view_frame, text='Fill NA', width=7,
                command=lambda: self.fill_na(parent, main)).place(x=235, y=530)

            ttk.Button(data_view_frame, text='Create Table', 
                       command=lambda: self.create_new_table(DataPreview.notebook.nb)).place(x=10, y=610)
            ttk.Button(data_view_frame, text='Back', 
                command=lambda: DataPreview.root.withdraw()).place(x=480, y=610)

            DataPreview.root.deiconify()
            DataPreview.root.lift()

            def on_closing():
                DataPreview.root.withdraw()

            DataPreview.root.protocol("WM_DELETE_WINDOW", on_closing)

            main.Viewed.set(True)

        elif main.data is not None and main.Viewed.get()==True:
            DataPreview.root.deiconify()
            DataPreview.root.lift()

        elif main.data is None:
            if not hasattr(DataPreview, 'root'):
                DataPreview.root = tk.Toplevel(parent)
                w = 650
                h = 650
                data_view_frame = ttk.Frame(DataPreview.root, width=w, height=h)
                data_view_frame.place(y=0)
                #setting main window's parameters       
                x = (parent.ws/2) - (w/2)
                y = (parent.hs/2) - (h/2) - 30
                DataPreview.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
                DataPreview.root.resizable(False, False)
                DataPreview.root.title('Data Viewer')

            if not hasattr(DataPreview.notebook, 'nb'):
                DataPreview.notebook.nb = ttk.Notebook(data_view_frame)
                DataPreview.notebook.nb.place(y=0)

            ttk.Button(data_view_frame, text='Create Table', 
                       command=lambda: self.create_new_table(DataPreview.notebook.nb)).place(x=10, y=610)
            ttk.Button(data_view_frame, text='Back', 
                command=lambda: DataPreview.root.withdraw()).place(x=480, y=610)

            DataPreview.root.deiconify()
            DataPreview.root.lift()

            def on_closing():
                DataPreview.root.withdraw()

            DataPreview.root.protocol("WM_DELETE_WINDOW", on_closing)

        if not hasattr(self, 'tables'):
            self.tables = 1

    # function to create new clear table
    def create_new_table(self, notebook):
        nt_frame = ttk.Frame(DataPreview.root, width=w, height=(h-50))
        DataPreview.notebook.nb.add(nt_frame, text='New table{}'.format(self.tables))
        self.tables += 1
        new_pt = Table(nt_frame, dataframe=None, showtoolbar=True, showstatusbar=True, 
            height=450, notebook=DataPreview.notebook.nb)
        new_pt.show()
        DataPreview.notebook.nb.select(nt_frame)

    # function to update viewed data
    def update_data(self, prev, main, table, full):
        main.data = table
        if full=='clf training' or full=='rgr training':
            prev.combobox1.config(values=list(main.data.columns))
            prev.combobox1.set(main.data.columns[-1])
            prev.tr_x_from_combobox.config(values=list(main.data.columns))
            prev.tr_x_from_combobox.set(main.data.columns[0])
            prev.tr_x_to_combobox.config(values=list(main.data.columns))
            prev.tr_x_to_combobox.set(main.data.columns[-2])
        elif full=='clf prediction' or full=='rgr prediction':
            prev.pr_x_from_combobox.config(values=list(main.data.columns))
            prev.pr_x_from_combobox.set(main.data.columns[0])
            prev.pr_x_to_combobox.config(values=list(main.data.columns))
            prev.pr_x_to_combobox.set(main.data.columns[-1])
        elif full=='clust':
            prev.clust_x_from_combobox.config(values=list(main.data.columns))
            prev.clust_x_from_combobox.set(main.data.columns[0])
            prev.clust_x_to_combobox.config(values=list(main.data.columns))
            prev.clust_x_to_combobox.set(main.data.columns[-1])
        elif full=='dcmp':
            prev.dcmp_x_from_combobox.config(values=list(main.data.columns))
            prev.dcmp_x_from_combobox.set(main.data.columns[0])
            prev.dcmp_x_to_combobox.config(values=list(main.data.columns))
            prev.dcmp_x_to_combobox.set(main.data.columns[-1])
    # function to close currently viewed table and update it
    def upd_quit(self, frame, prev, main, table, full):
        self.update_data(prev, main, table, full)
        frame.destroy()
        main.Viewed.set(False)
    # function to close currently viewed table without updating it
    def quit(self, frame, prev, main, table, full):
        # self.update_data(prev, main, table, full)
        frame.destroy()
        main.Viewed.set(False)
    # function to drop all nan's from table
    def drop_na(self, main):
        self.df = main.pt.model.df
        main.pt.storeCurrent()
        self.df = self.df.dropna(axis=0)
        main.pt = Table(main.view_frame, dataframe=self.df, showtoolbar=True, showstatusbar=True, 
            height=450, notebook=DataPreview.notebook.nb, dp_main=main)
        main.pt.show()
    # function to fill nan's in different ways
    def fill_na(self, parent, main):
        self.fn_win = tk.Toplevel(DataPreview.root)
        w=200
        h=250
        #setting main window's parameters       
        x = (parent.ws/2) - (w/2)
        y = (parent.hs/2) - (h/2) - 30
        self.fn_win.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.fn_win.lift()
        self.fn_win.resizable(False, False)
        self.fn_win.title('Fill NA')
        self.fill_f = ttk.Frame(self.fn_win, width=w, height=h)
        self.fill_f.place(x=0, y=0)
        ttk.Button(self.fill_f, text='Mean', width=10, 
            command=lambda: fill_w(self, 'mean')).place(x=50, y=10)
        ttk.Button(self.fill_f, text=' Median', width=10, 
            command=lambda: fill_w(self, 'median')).place(x=50, y=40)
        ttk.Button(self.fill_f, text='Mode', width=10, 
            command=lambda: fill_w(self, 'mode')).place(x=50, y=70)
        ttk.Button(self.fill_f, text='Min', width=10, 
            command=lambda: fill_w(self, 'min')).place(x=50, y=100)
        ttk.Button(self.fill_f, text='Max', width=10, 
            command=lambda: fill_w(self, 'max')).place(x=50, y=130)
        ttk.Button(self.fill_f, text='     None\n(object type\n     only)', width=10, 
            command=lambda: fill_w(self, 'none')).place(x=50, y=160)
        
        def fill_w(self, method):
            self.df = main.pt.model.df
            main.pt.storeCurrent()
            if method == 'mean':
                self.df = self.df.fillna(self.df.mean(axis=0))
            elif method == 'median':
                self.df = self.df.fillna(self.df.median(axis=0))
            elif method == 'mode':
                self.df = self.df.fillna(self.df.mode(axis=0))
            elif method == 'min':
                self.df = self.df.fillna(self.df.min(axis=0))
            elif method == 'max':
                self.df = self.df.fillna(self.df.max(axis=0))
            elif method == 'none':
                str_cols = self.df.select_dtypes(include=['object']).columns
                self.df.loc[:, str_cols] = self.df.loc[:, str_cols].fillna('None')
                # self.df = self.df.astype(object).replace(np.nan, 'None')
            main.pt = Table(main.view_frame, dataframe=self.df, showtoolbar=True, showstatusbar=True)
            main.pt.show()
            self.fn_win.destroy()
        
#function to quit back without destroying root
def quit_back(current, parent):
    current.grab_release()
    current.withdraw()
    parent.deiconify()
    parent.lift()
    parent.focus_force()

#function to open files and insert path to entry
def open_file(app, entry): 
    entry.delete(0, tk.END)
    file1 = askopenfilename(parent=app.root, filetypes = (("Supported","*.csv *.xlsx *.xls *.data *.sql"), ("all files","*.*")))
    if file1 is not None:
        entry.insert(0, file1)

#function to load data from files             
def load_data(app, main, entry, data_type): 
    global data_file
    data_file = entry.get()
    if len(data_file) > 1 and not data_file.endswith('.sql'):
        if (main.header_var.get()==1 and (data_file.endswith('.xls') or data_file.endswith('.xlsx'))):
            main.data = pd.read_excel(data_file, sheet_name=(int(main.sheet.get())-1))
        elif (main.header_var.get()==0 and (data_file.endswith('.xls') or data_file.endswith('.xlsx'))):
            main.data = pd.read_excel(data_file, sheet_name=(int(main.sheet.get())-1), header=None)
        elif main.header_var.get()==1 and (data_file.endswith('.csv')):
            main.data = pd.read_csv(data_file)
        elif main.header_var.get()==0 and (data_file.endswith('.csv')):
            main.data = pd.read_csv(data_file, header=None)
        elif main.header_var.get()==1 and (data_file.endswith('.data')):
            main.data = pd.read_table(data_file, sep=',')
        elif main.header_var.get()==0 and (data_file.endswith('.data')):
            main.data = pd.read_table(data_file, sep=',', header=None)
        if data_type=='clf training' or data_type=='rgr training':
            app.training.data_status.config(text='Loaded')
            app.combobox1.config(values=list(main.data.columns))
            app.combobox1.set(main.data.columns[-1])
            app.tr_x_from_combobox.config(values=list(main.data.columns))
            app.tr_x_from_combobox.set(main.data.columns[0])
            app.tr_x_to_combobox.config(values=list(main.data.columns))
            app.tr_x_to_combobox.set(main.data.columns[-2])
        elif data_type=='clf prediction' or data_type=='rgr prediction':
            app.prediction.data_status.config(text='Loaded')
            app.pr_x_from_combobox.config(values=list(main.data.columns))
            app.pr_x_from_combobox.set(main.data.columns[0])
            app.pr_x_to_combobox.config(values=list(main.data.columns))
            app.pr_x_to_combobox.set(main.data.columns[-1])
        elif data_type=='clust':
            app.clust.data_status.config(text='Loaded')
            app.clust_x_from_combobox.config(values=list(main.data.columns))
            app.clust_x_from_combobox.set(main.data.columns[0])
            app.clust_x_to_combobox.config(values=list(main.data.columns))
            app.clust_x_to_combobox.set(main.data.columns[-1])
        elif data_type=='dcmp':
            app.decomposition.data_status.config(text='Loaded')
            app.dcmp_x_from_combobox.config(values=list(main.data.columns))
            app.dcmp_x_from_combobox.set(main.data.columns[0])
            app.dcmp_x_to_combobox.config(values=list(main.data.columns))
            app.dcmp_x_to_combobox.set(main.data.columns[-1])
        if main.Viewed.get() == True:
            main.pt = Table(main.view_frame, dataframe=main.data, showtoolbar=True, 
                showstatusbar=True, height=450, notebook=DataPreview.notebook.nb, dp_main=main)
            main.pt.show()
            DataPreview.root.lift()
    if len(data_file) > 1 and data_file.endswith('.sql'):
        app.opensql = OpenSQL(app, main, entry, data_type)
        app.opensql
        # OpenSQL(app, main, entry, data_type)

class OpenSQL:
    def __init__(self, app, main, entry, data_type):
        from sqlalchemy import create_engine
        import pymysql

        if not hasattr(self, 'root'):

            self.root = tk.Toplevel(app.root)
            w=200
            h=250
            ws = self.root.winfo_screenwidth() # width of the screen
            hs = self.root.winfo_screenheight() # height of the screen
            #setting main window's parameters       
            x = (ws/2) - (w/2)
            y = (hs/2) - (h/2) - 30
            self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
            self.root.lift()
            self.root.focus_force()
            self.root.resizable(False, False)
            self.root.title('Connect to SQL Server')
            self.frame = ttk.Frame(self.root, width=w, height=h)
            self.frame.place(x=0, y=0)

            ttk.Label(self.frame, text='Connect to SQL Server', font=myfont).place(x=10, y=10)

            ttk.Label(self.frame, text='SQL Server', font=myfont1).place(x=10, y=50)
            OpenSQL.sql_server_var = tk.StringVar(value='MySQL')
            self.combobox1 = ttk.Combobox(self.frame, textvariable=OpenSQL.sql_server_var, 
                width=10, values=['MySQL'])
            self.combobox1.place(x=100,y=53)

            ttk.Label(self.frame, text='User', font=myfont1).place(x=10, y=80)
            OpenSQL.sql_user_var = tk.StringVar()
            self.user_entry = ttk.Entry(self.frame, textvariable=OpenSQL.sql_user_var, 
                width=10, font=myfont2)
            self.user_entry.place(x=100,y=83)

            ttk.Label(self.frame, text='Password', font=myfont1).place(x=10, y=110)
            OpenSQL.sql_pw_var = tk.StringVar()
            self.pw_entry = ttk.Entry(self.frame, textvariable=OpenSQL.sql_pw_var, 
                width=10, font=myfont2)
            self.pw_entry.place(x=100,y=113)

            ttk.Label(self.frame, text='Host', font=myfont1).place(x=10, y=140)
            OpenSQL.sql_host_var = tk.StringVar()
            self.host_entry = ttk.Entry(self.frame, textvariable=OpenSQL.sql_host_var, 
                width=10, font=myfont2)
            self.host_entry.place(x=100,y=143)

            ttk.Label(self.frame, text='Database', font=myfont1).place(x=10, y=170)
            OpenSQL.sql_db_var = tk.StringVar()
            self.db_entry = ttk.Entry(self.frame, textvariable=OpenSQL.sql_db_var, 
                width=10, font=myfont2)
            self.db_entry.place(x=100,y=173)

            def load_sql():
                if OpenSQL.sql_server_var.get()=='MySQL':
                    db_connection_str = ('mysql+pymysql://' + OpenSQL.sql_user_var.get() + 
                        ':' + OpenSQL.sql_pw_var.get() + '@' + OpenSQL.sql_host_var.get() +
                        '/' + OpenSQL.sql_db_var.get())
                    db_connection = create_engine(db_connection_str)

                    query = open(data_file, 'r').read()
                    main.data = pd.read_sql(query, db_connection)

                if data_type=='clf training' or data_type=='rgr training':
                    app.training.data_status.config(text='Loaded')
                    app.combobox1.config(values=list(main.data.columns))
                    app.combobox1.set(main.data.columns[-1])
                    app.tr_x_from_combobox.config(values=list(main.data.columns))
                    app.tr_x_from_combobox.set(main.data.columns[0])
                    app.tr_x_to_combobox.config(values=list(main.data.columns))
                    app.tr_x_to_combobox.set(main.data.columns[-2])
                elif data_type=='clf prediction' or data_type=='rgr prediction':
                    app.prediction.data_status.config(text='Loaded')
                    app.pr_x_from_combobox.config(values=list(main.data.columns))
                    app.pr_x_from_combobox.set(main.data.columns[0])
                    app.pr_x_to_combobox.config(values=list(main.data.columns))
                    app.pr_x_to_combobox.set(main.data.columns[-1])
                elif data_type=='clust':
                    app.clust.data_status.config(text='Loaded')
                    app.clust_x_from_combobox.config(values=list(main.data.columns))
                    app.clust_x_from_combobox.set(main.data.columns[0])
                    app.clust_x_to_combobox.config(values=list(main.data.columns))
                    app.clust_x_to_combobox.set(main.data.columns[-1])
                elif data_type=='dcmp':
                    app.decomposition.data_status.config(text='Loaded')
                    app.dcmp_x_from_combobox.config(values=list(main.data.columns))
                    app.dcmp_x_from_combobox.set(main.data.columns[0])
                    app.dcmp_x_to_combobox.config(values=list(main.data.columns))
                    app.dcmp_x_to_combobox.set(main.data.columns[-1])
                if main.Viewed.get() == True:
                    main.pt = Table(main.view_frame, dataframe=main.data, showtoolbar=True, 
                        showstatusbar=True, height=450, notebook=DataPreview.notebook.nb, dp_main=main)
                    main.pt.show()
                    DataPreview.root.lift()
                self.root.withdraw()

            ttk.Button(self.frame, text='Load SQL', width=10,
                command=lambda: load_sql()).place(x=50, y=210)

            self.root.protocol("WM_DELETE_WINDOW", lambda: self.root.withdraw())
        else:
            self.root.deiconify()
            self.root.lift()

#function to save data to files
def save_results(prev, main, name='Result'):
    files = [('csv files', '*.csv'),
             ('Excel', '*.xlsx'),
             ('All Files', '*.*')] 
    file = asksaveasfilename(parent=prev.frame, filetypes = files, defaultextension = files)
    if file.endswith('.csv'):
        pd.DataFrame(main.data).to_csv(file, 
            index=False, header=(False if main.header_var.get()==0 else True))
    elif os.path.isfile(file):
        with pd.ExcelWriter(file, engine="openpyxl", mode='a') as writer: 
            pd.DataFrame(main.data).to_excel(writer, sheet_name=name, 
                index=False, header=(False if main.header_var.get()==0 else True))
    else:
        pd.DataFrame(main.data).to_excel(file, sheet_name=name, 
            index=False, header=(False if main.header_var.get()==0 else True))

class SaveToSQL:
    def __init__(self, prev, main, name='Result'):
        from sqlalchemy import create_engine
        import pymysql

        SaveToSQL.root = tk.Toplevel(prev.root)
        w=200
        h=280
        ws = self.root.winfo_screenwidth() # width of the screen
        hs = self.root.winfo_screenheight() # height of the screen
        #setting main window's parameters       
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2) - 30
        SaveToSQL.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        SaveToSQL.root.lift()
        SaveToSQL.root.focus_force()
        SaveToSQL.root.resizable(False, False)
        SaveToSQL.root.title('Connect to SQL Server')
        self.frame = ttk.Frame(SaveToSQL.root, width=w, height=h)
        self.frame.place(x=0, y=0)

        ttk.Label(self.frame, text='Connect to SQL Server', font=myfont).place(x=10, y=10)

        ttk.Label(self.frame, text='SQL Server', font=myfont1).place(x=10, y=50)
        SaveToSQL.sql_server_var = tk.StringVar(value='MySQL')
        self.combobox1 = ttk.Combobox(self.frame, textvariable=SaveToSQL.sql_server_var, 
            width=10, values=['MySQL'])
        self.combobox1.place(x=100,y=53)

        ttk.Label(self.frame, text='User', font=myfont1).place(x=10, y=80)
        SaveToSQL.sql_user_var = tk.StringVar()
        self.user_entry = ttk.Entry(self.frame, textvariable=SaveToSQL.sql_user_var, 
            width=10, font=myfont2)
        self.user_entry.place(x=100,y=83)

        ttk.Label(self.frame, text='Password', font=myfont1).place(x=10, y=110)
        SaveToSQL.sql_pw_var = tk.StringVar()
        self.pw_entry = ttk.Entry(self.frame, textvariable=SaveToSQL.sql_pw_var, 
            width=10, font=myfont2)
        self.pw_entry.place(x=100,y=113)

        ttk.Label(self.frame, text='Host', font=myfont1).place(x=10, y=140)
        SaveToSQL.sql_host_var = tk.StringVar()
        self.host_entry = ttk.Entry(self.frame, textvariable=SaveToSQL.sql_host_var, 
            width=10, font=myfont2)
        self.host_entry.place(x=100,y=143)

        ttk.Label(self.frame, text='Database', font=myfont1).place(x=10, y=170)
        SaveToSQL.sql_db_var = tk.StringVar()
        self.db_entry = ttk.Entry(self.frame, textvariable=SaveToSQL.sql_db_var, 
            width=10, font=myfont2)
        self.db_entry.place(x=100,y=173)

        ttk.Label(self.frame, text='Table name', font=myfont1).place(x=10, y=200)
        SaveToSQL.sql_table_var = tk.StringVar()
        self.db_entry = ttk.Entry(self.frame, textvariable=SaveToSQL.sql_table_var, 
            width=10, font=myfont2)
        self.db_entry.place(x=100,y=203)
        self.db_entry.insert(0, name)

        def _save_to_sql():
            if SaveToSQL.sql_server_var.get()=='MySQL':
                db_connection_str = ('mysql+pymysql://' + SaveToSQL.sql_user_var.get() + 
                    ':' + SaveToSQL.sql_pw_var.get() + '@' + SaveToSQL.sql_host_var.get() +
                    '/' + SaveToSQL.sql_db_var.get())
                db_connection = create_engine(db_connection_str)

                from sqlalchemy import event
                @event.listens_for(db_connection, "before_cursor_execute")
                def receive_before_cursor_execute(
                   conn, cursor, statement, params, context, executemany
                    ):
                        if executemany:
                            cursor.fast_executemany = True

                # def set_d_type_dict(df):
                #     type_dict = {}
                #     for i, j in zip(df.columns, df.dtypes):
                #         if "object" in str(j):
                #             type_dict.update({i: VARCHAR(512)})
                #         if "float" in str(j):
                #             type_dict.update({i: DECIMAL(19, 2)})
                #         if "int" in str(j):
                #             type_dict.update({i: DECIMAL(19)})
                #     return type_dict

                # d_type = set_d_type_dict(pd.DataFrame(main.data))

                pd.DataFrame(main.data).to_sql(SaveToSQL.sql_table_var.get(), con=db_connection,
                    method='multi', chunksize=2000, index=False
                )

            SaveToSQL.root.withdraw()

        ttk.Button(self.frame, text='Save to SQL', width=10,
            command=lambda: _save_to_sql()).place(x=50, y=240)

# function to restart the program from inside
def restart_app():
    if messagebox.askyesno("Restart", "Restart Application?"):
        os.execl(sys.executable, sys.executable, *sys.argv)