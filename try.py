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
from kivy.uix.image import Image
from kivy.core.image import Image as CoreI


from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recyclegridlayout import RecycleGridLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.garden.mapview import MapView, MapMarker

from kivy.clock import Clock

from train import Train
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kivy.clock import mainthread
file_name = "preprocessed_471_2017.csv"

file_name="results.csv"

class MapApp(App):
    
    def build(self):
        self.title = "Trafik Akis Hizi Tahmini"
        sm = ScreenManager()

        home = HomeScreen(name="home")
        hyper = HyperScreen(name='tfs')
        train = TrainScreen(name="train")
        results = ResultsScreen(name='res')


        sm.add_widget(hyper)
        sm.add_widget(train)
        sm.add_widget(results)
        sm.add_widget(home)

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

        self.t = Thread(target=self.fit, args=(int(self.popup.ids.epoch.text),))
        self.t.setDaemon(True)
        self.t.start()

        #t.join()
    def fit(self, epochs):  
        train_loss_history = []
        test_loss_history = []
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
            temp_train, temp_test = self.train_object.fit(
                int(self.popup.ids.batch.text)
            )
            epoch_history["train"].append(temp_train)
            epoch_history["test"].append(temp_test)
            self.manager.screens[1].update_results(epoch_history, i+1, epochs)
        self.train_object.save_estimations(file_name)
        self.manager.screens[2].get_dataframe()
        self.manager.screens[1].ids.results.disabled = False




class SettingsPopUp(Popup):
    pass

class TrainScreen(Screen):

    def update_results(self, epoch_history, current_epoch, max_epoch):
        self.ids.header.text = "EPOCH: " + str(current_epoch) + "/" + str(max_epoch)
        indexes = [i for i in range(1, 1 + len(epoch_history["train"]))]
        fig = plt.figure()
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        ax.set_xlim([1,max_epoch])
        ax.set_ylim([0,100])
        ax.set_xticks(indexes)
        ax.plot(indexes, epoch_history["train"], label="Eğitim MAPE")
        ax.plot(indexes, epoch_history["test"], label="Test MAPE")
        self.fig_path = "epoch_his/fig_" + str(current_epoch) + ".png" 
        fig.savefig(self.fig_path)    
        ax.cla()
        self.update_image()
        epoch_info_text = self.ids.header.text + "\n"
        epoch_info_text += "Eğitim MAPE: " + str(epoch_history["train"][current_epoch-1]) + "\n"
        epoch_info_text += "Test MAPE: " + str(epoch_history["test"][current_epoch-1])
        self.ids.result.text = epoch_info_text

    def see_all_results_button_click(self):
        self.manager.current = self.manager.screens[2].name

    @mainthread
    def update_image(self):
        self.ids.graph.source = self.fig_path

class ResultsScreen(Screen):
    frame_list = ObjectProperty()
    column_headings =ObjectProperty()
    rv_data = ListProperty([])
    
    def __init__(self, **kwargs):
        super(ResultsScreen, self).__init__(**kwargs)
        self.column_headings.add_widget(Label(text="Tarih"))
        self.column_headings.add_widget(Label(text="Gercek Deger"))
        self.column_headings.add_widget(Label(text="Tahmin"))

    def get_dataframe(self):
        df = pd.read_csv(file_name)

      

        data = []
        for row in df.itertuples():
            for i in range(1, len(row)):
                data.append([row[i], row[0]])
        self.rv_data = [{'text': str(x[0]), 'Index': str(x[1]), 'selectable': True} for x in data]


class HomeScreen(Screen):
    pass

MapApp().run()