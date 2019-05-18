from kivy.app import App

from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from datepicker import DatePicker
from kivy.uix.textinput import TextInput
from kivy.properties import NumericProperty 
import concurrent.futures

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.garden.mapview import MapView, MapMarker

import train
from kivy.clock import Clock

from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_name = "preprocessed_471_2017.csv"


class MapApp(App):
    
    def build(self):
        self.title = "Trafik Akis Hizi Tahmini"
        sm = ScreenManager()

        hyper = HyperScreen(name='tfs')
        train = TrainScreen(name="train")
        
        sm.add_widget(hyper)
        sm.add_widget(train)

        return sm
        

class HyperScreen(Screen):

    def __init__(self, *args, **kwargs):
        super(HyperScreen, self).__init__(*args, **kwargs)
        self.popup = SettingsPopUp()

    def settings_button_click(self):
        self.popup.open()

    def start_train_button(self):
        self.manager.current = self.manager.screens[1].name
        self.weekday = True
        if self.popup.ids.day.text == 'Hayir':
            self.weekday = False
        
        self.daypart = True
        if self.popup.ids.daypart.text == "Hayir":
            self.daypart = False

        t = Thread(target=self.fit, args=(50,))
        t.setDaemon(True)
        t.start()

        #t.join()
    def fit(self, epochs):  
        train_loss_history = np.empty(shape=(epochs))
        test_loss_history = np.empty(shape=(epochs))
        epoch_history = {"train": train_loss_history, "test":test_loss_history}

        self.train_object = train.prepare_model(
        "./speed_data/FSM/preprocessed_471_2017.csv",
        self.weekday, 
        self.ids.train_start.text,
        self.ids.train_end.text,
        self.ids.test_start.text,
        self.ids.test_end.text,
        int(self.popup.ids.time_step.text),
        0,
        self.daypart
        )       
        for i in range(epochs):
            history = self.train_object.regressor.fit(
                self.train_object.x_train,
                self.train_object.y_train,
                epochs=1,
                batch_size=int(self.manager.screens[0].popup.ids.batch.text),
                validation_data=(self.train_object.x_test, self.train_object.y_test)
            )
            epoch_history["train"][i] = history.history["loss"][0]
            epoch_history["test"][i] = history.history["val_loss"][0]
            self.manager.screens[1].update_results(epoch_history, i+1, epochs)

    


class SettingsPopUp(Popup):
    pass

class TrainScreen(Screen):

    def update_results(self, epoch_history, current_epoch, max_epoch):
        self.ids.header.text = "EPOCH: " + str(current_epoch) + "/" + str(max_epoch)
        self.ids.figure.figure.clf()
        indexes = [i for i in range(1, 1 + len(epoch_history["train"]))]
        plt.plot(indexes, epoch_history["train"], label="Eğitim MAPE")
        plt.plot(indexes, epoch_history["test"], label="Test MAPE")
        plt.show()
        self.ids.figure.figure.canvas.draw_idle()
        epoch_info_text = self.ids.header.text + "\n"
        epoch_info_text += "Eğitim MAPE: " + str(epoch_history["train"][current_epoch-1]) + "\n"
        epoch_info_text += "Test MAPE: " + str(epoch_history["test"][current_epoch-1])
        self.ids.result.text = epoch_info_text


class MyFigure(FigureCanvasKivyAgg):

    def __init__(self,**kwargs):
        super(MyFigure, self).__init__(plt.gcf(),**kwargs)


MapApp().run()