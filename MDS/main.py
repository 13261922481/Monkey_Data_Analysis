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
from .classification import *
from .regression import *
from .clustering import *

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

        master.ws = master.winfo_screenwidth() # width of the screen
        master.hs = master.winfo_screenheight() # height of the screen
        
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
        ttk.Button(sw_frame, text='   Clustering  ', command=lambda: cls_app(master)).place(relx=0.6, rely=0.55)
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