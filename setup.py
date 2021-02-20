from cx_Freeze import setup, Executable
import sys


#main
exe = Executable(script="main.py", targetName="ML.exe", base="Win32GUI", icon="icon1.ico")
exe_con = Executable(script="main.py", targetName="ML_consoled.exe", icon="icon1.ico")
buildOptions = dict(includes =["monkey_pt.util", "monkey_pt.images", "monkey_pt.config", 
    'scipy.spatial.transform._rotation_groups', "seaborn.cm", 'sqlalchemy.dialects.mysql'], optimize=1)
setup(name = "ML",version = "0.7", description = "", executables = [exe, exe_con], options =dict(build_exe = buildOptions))

# "monkey_pt.util", "monkey_pt.images", "monkey_pt.config"