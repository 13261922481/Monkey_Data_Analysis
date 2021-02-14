# Imports
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import multiprocessing
import os
import sys
import numpy as np
import webbrowser
import pandas as pd
import sklearn
from classification import *
from regression import *
from clustering import *
from decomposition import *
from utils import restart_app

# getting the right path to images for frozen version
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.abspath('')

#themed tkinter master
master = ThemedTk(theme=open('settings\style.txt', 'r').read(), fonts=True)

#fonts to be used
myfont = (None, 12)
myfont_b = (None, 12, 'bold')
myfont1 = (None, 11)
myfont1_b = (None, 11, 'bold')
myfont2 = (None, 10)
myfont2_b = (None, 10, 'bold')

#configuring main font to be used
style = ttk.Style()
style.configure('.', font=myfont)
style.configure('TButton', padding=(5,1), relief='raised')

#start window class
class start_window:
    def __init__(self):
        # Setting geometrical things
        w = 500
        h = 400

        master.ws = master.winfo_screenwidth() # width of the screen
        master.hs = master.winfo_screenheight() # height of the screen
        
        start_window.frame =ttk.Frame(master, width=w, height=h)
        start_window.frame.place(x=0, y=0)
        
        # placing funny monkey
        img_name1 = 'imgs/ML/monkey1.png'
        img_path1 = os.path.join(application_path, img_name1)
        img1 = tk.PhotoImage(file=img_path1)
        panel = tk.Label(start_window.frame, image = img1)
        panel.place(relx=0.04, rely=0.15)

        # instead of closing apps, which leads to losing data, we just iconify/deiconify it
        def run_app(app):
            if not hasattr(app, 'root'):
                app(master)
            else:
                app.root.deiconify()
        
        ttk.Button(start_window.frame, text='Classification', width=12, 
            command=lambda: run_app(clf_app)).place(x=300, y=80)
        ttk.Button(start_window.frame, text='Regression', width=12, 
            command=lambda: run_app(rgr_app)).place(x=300, y=130)
        ttk.Button(start_window.frame, text='Clustering', width=12, 
            command=lambda: run_app(clust_app)).place(x=300, y=180)
        ttk.Button(start_window.frame, text='Decomposition', width=12, 
            command=lambda: run_app(dcmp_app)).place(x=300, y=230)

        #setting main window's parameters       
        x = (master.ws/2) - (w/2)
        y = (master.hs/2) - (h/2)  - 30
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
        main_menu.add_command(label='Restart', command=lambda: restart_app())
        settings_menu = tk.Menu(main_menu, tearoff=False)
        def set_theme(theme):
            master.set_theme(theme)
            style.configure('.', font=myfont)
            style.configure('TButton', padding=(5,1), relief='raised')
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
        # style_menu.add_command(label='Ubuntu', command=lambda: set_theme('ubuntu')) #big
        style_menu.add_command(label='Vista', command=lambda: set_theme('vista'))
        style_menu.add_command(label='winnative', command=lambda: set_theme('winnative'))
        style_menu.add_command(label='winxpblue', command=lambda: set_theme('winxpblue'))
        style_menu.add_command(label='xpnative', command=lambda: set_theme('xpnative'))
        settings_menu.add_cascade(label="Style", menu=style_menu)
        help_menu = tk.Menu(main_menu, tearoff=False)
        def open_site(site):
            webbrowser.open(site)
        def open_about_window():
            about_message = 'Machine Learning tool 1.2'
            about_detail = "by Monkey22\na.k.a Osipov A."
            tk.messagebox.showinfo(title='About', message=about_message, detail=about_detail)
        docs_menu = tk.Menu(help_menu, tearoff=False)
        help_menu.add_cascade(label="Docs", menu=docs_menu)
        docs_menu.add_command(label='Pandas', 
            command=lambda: open_site('https://pandas.pydata.org'))
        docs_menu.add_command(label='Numpy', 
            command=lambda: open_site('https://numpy.org'))
        docs_menu.add_command(label='Scikit-learn', 
            command=lambda: open_site('https://scikit-learn.org'))
        docs_menu.add_command(label='XGBoost', 
            command=lambda: open_site('https://xgboost.readthedocs.io/en/latest/'))
        docs_menu.add_command(label='CatBoost', 
            command=lambda: open_site('https://catboost.ai/docs/concepts/about.html'))
        docs_menu.add_command(label='LightGBM', 
            command=lambda: open_site('https://lightgbm.readthedocs.io/en/latest/'))
        help_menu.add_command(label='About', command=open_about_window)
        main_menu.add_cascade(label="Settings", menu=settings_menu)
        main_menu.add_cascade(label="Help", menu=help_menu)
        
        master.mainloop()

#run the program with multiprocessing support
if __name__ == '__main__':
    multiprocessing.freeze_support()
    start_window()