import os
import tkinter as tk
import tkinter.ttk as ttk
import pygubu

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_UI = os.path.join(PROJECT_PATH, "newproject")


class NewprojectApp:
    def __init__(self, master=None):
        # build ui
        self.frame1 = ttk.Frame(master)
        self.label1 = ttk.Label(self.frame1)
        self.label1.configure(text='image to go here')
        self.label1.pack(side='top')
        self.frame2 = ttk.Frame(self.frame1)
        self.frame3 = ttk.Frame(self.frame2)
        self.label2 = ttk.Label(self.frame3)
        self.label2.configure(text='feature number')
        self.label2.grid(column='0', row='0')
        self.scale2 = ttk.Scale(self.frame3)
        self.scale2.configure(from_='0', orient='horizontal', to='64')
        self.scale2.grid(column='1', row='0')
        self.scale2.configure(command=self.feature_changed)
        self.frame3.configure(height='200', width='200')
        self.frame3.grid(column='0', row='0')
        self.frame4 = ttk.Frame(self.frame2)
        self.label3 = ttk.Label(self.frame4)
        self.label3.configure(text='threshold')
        self.label3.grid(column='0', row='0')
        self.scale1 = ttk.Scale(self.frame4)
        self.scale1.configure(from_='0', orient='horizontal', to='255')
        self.scale1.grid(column='1', row='0', sticky='e')
        self.scale1.configure(command=self.scale_changed)
        self.frame4.configure(height='200', width='200')
        self.frame4.grid(column='0', row='1')
        self.frame2.configure(height='200', width='200')
        self.frame2.pack(side='top')
        self.frame1.configure(height='200', width='200')
        self.frame1.pack(side='top')

        # Main widget
        self.mainwindow = self.frame1

    def feature_changed(self, scale_value):
        print(round(float(scale_value)))

    def scale_changed(self, scale_value):
        print(round(float(scale_value)))

    def run(self):
        self.mainwindow.mainloop()


if __name__ == '__main__':
    root = tk.Tk()
    app = NewprojectApp(root)
    app.run()

