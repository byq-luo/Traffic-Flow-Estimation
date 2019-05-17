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

from kivy.garden.mapview import MapView, MapMarker
import train
from kivy.clock import Clock
import threading

from threading import Thread



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
            self.manager.screens[1].ids.result.text = str(history.history["val_loss"])

    


class SettingsPopUp(Popup):
    pass

class TrainScreen(Screen):

    def __init__(self, *args, **kwargs):
        super(TrainScreen, self).__init__(*args, **kwargs)
        self.train_object = None

    def start_train(self):
        pass


MapApp().run()