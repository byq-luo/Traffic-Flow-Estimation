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
from kivy.properties import NumericProperty, ObjectProperty, ListProperty, BooleanProperty
import concurrent.futures

from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recyclegridlayout import RecycleGridLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.garden.mapview import MapView, MapMarker

from train import Train
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_name = "preprocessed_471_2017.csv"

file_name="results.csv"

class MapApp(App):
    
    def build(self):
        self.title = "Trafik Akis Hizi Tahmini"
        sm = ScreenManager()

        hyper = HyperScreen(name='tfs')
        train = TrainScreen(name="train")
        results = ResultsScreen(name='res')

        sm.add_widget(hyper)
        sm.add_widget(train)
        sm.add_widget(results)
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

        t = Thread(target=self.fit, args=(int(self.popup.ids.epoch.text),))
        t.setDaemon(True)
        t.start()

        #t.join()
    def fit(self, epochs):  
        train_loss_history = np.empty(shape=(epochs))
        test_loss_history = np.empty(shape=(epochs))
        epoch_history = {"train": train_loss_history, "test":test_loss_history}

        self.train_object = Train(
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
            epoch_history["train"][i], epoch_history["test"][i] = self.train_object.fit(
                int(self.popup.ids.batch.text)
            )
            self.manager.screens[1].update_results(epoch_history, i+1, epochs)
        self.train_object.save_estimations(file_name)
        self.manager.screens[2].get_dataframe()



class SettingsPopUp(Popup):
    pass

class TrainScreen(Screen):

    def update_results(self, epoch_history, current_epoch, max_epoch):
        self.ids.figure.figure.clf()
        self.ids.header.text = "EPOCH: " + str(current_epoch) + "/" + str(max_epoch)
        indexes = [i for i in range(1, 1 + len(epoch_history["train"]))]
        plt.plot(indexes, epoch_history["train"], label="Eğitim MAPE")
        plt.plot(indexes, epoch_history["test"], label="Test MAPE")
        #self.ids.figure.figure.canvas.draw_idle()
        epoch_info_text = self.ids.header.text + "\n"
        epoch_info_text += "Eğitim MAPE: " + str(epoch_history["train"][current_epoch-1]) + "\n"
        epoch_info_text += "Test MAPE: " + str(epoch_history["test"][current_epoch-1])
        self.ids.result.text = epoch_info_text

    def see_all_results_button_click(self):
        self.manager.current = self.manager.screens[2].name



class ResultsScreen(Screen):
    frame_list = ObjectProperty()
    column_headings =ObjectProperty()
    rv_data = ListProperty([])
    
    def __init__(self, **kwargs):
        super(ResultsScreen, self).__init__(**kwargs)

    def get_dataframe(self):
        df = pd.read_csv(file_name)

        for heading in df.columns:
            self.column_headings.add_widget(Label(text=heading))

        data = []
        for row in df.itertuples():
            for i in range(1, len(row)):
                data.append([row[i], row[0]])
        self.rv_data = [{'text': str(x[0]), 'Index': str(x[1]), 'selectable': True} for x in data]

class MyFigure(FigureCanvasKivyAgg):

    def __init__(self,**kwargs):
        super(MyFigure, self).__init__(plt.gcf(),**kwargs)
        self.figure = plt.figure()


MapApp().run()