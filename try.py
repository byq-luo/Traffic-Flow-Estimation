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
from kivy.properties import NumericProperty, ObjectProperty, StringProperty, ListProperty, BooleanProperty
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar      
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recyclegridlayout import RecycleGridLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.garden.mapview import MapView, MapMarker
from kivy.clock import Clock
from train import Train, RegionSelector
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kivy.clock import mainthread
from tkinter.messagebox import showerror
import tkinter as tk
from datetime import datetime


tk.Tk().withdraw()
file_name="results.csv"
datetime_format = "%d-%m-%Y %H:%M"

class MapApp(App):
    
    def build(self):
        self.default_title = "Trafik Akis Hizi Tahmini" 
        self.title = self.default_title
        sm = ScreenManager()

        home = HomeScreen(name="home")
        hyper = HyperScreen(self, name='hyper')
        train = TrainScreen(name="train")
        results = ResultsScreen(name='res')

 

        sm.add_widget(hyper)
        sm.add_widget(train)
        sm.add_widget(results)
        sm.add_widget(home)
        
        sm.current = "home"
        
        return sm
        

class HyperScreen(Screen):

    def __init__(self, app,*args, **kwargs):
        super(HyperScreen, self).__init__(*args, **kwargs)
        self.app = app
        self.regions = RegionSelector()
        self.settings_popup = SettingsPopUp()
        self.preprocess_popup = PreprocessPopup()
        self.markers = {}
        self.current_marker_id = -1
        self.ids.region.values = self.regions.get_provinces()
    
    def settings_button_click(self):
        self.settings_popup.open()

    def check_parameters_for_train(self):
        if self.current_marker_id == -1:
            showerror("HATA", "Lutfen bolge seciniz")
            return False
        elif self.markers[self.current_marker_id].default_source == "./marker_ims/red.png":
            showerror("HATA", "Sectiginiz bolgede yeterli veri yok. Kirmizi veri noktalari secilememektedir.")
            return False

        train_start_date = datetime.strptime(self.ids.train_start.text + " 00:00", datetime_format)
        train_end_date = datetime.strptime(self.ids.train_end.text + " 00:00", datetime_format)
        test_start_date = datetime.strptime(self.ids.test_start.text + " 00:00", datetime_format)
        test_end_date = datetime.strptime(self.ids.test_end.text + " 00:00", datetime_format)
        
        if train_end_date < train_start_date or test_end_date < test_start_date:
            showerror("HATA", "Baslangic tarihleri bitis tarihlerinden once olmalidir")
            return False
        
        else:
            return True


    def start_train_button(self):
        if self.check_parameters_for_train() == False:
            return
        
        self.manager.screens[1].ids.progress.max = int(self.settings_popup.ids.epoch.text)
        self.manager.screens[1].ids.progress.value = 0
        self.manager.current = self.manager.screens[1].name
        self.app.title = self.build_result_title()
        self.weekday = True
        if self.settings_popup.ids.day.text == 'Hayir':
            self.weekday = False
        
        self.daypart = True
        if self.settings_popup.ids.daypart.text == "Hayir":
            self.daypart = False
        
        self.prev_weeks = 0
        if self.settings_popup.ids.prev_weeks.text == "1 ve 2 hafta":
            self.prev_weeks = 1
        elif self.settings_popup.ids.prev_weeks.text == "1, 2 ve 3 hafta":
            self.prev_weeks = 2
        self.train_thread = Thread(target=self.fit, args=(int(self.settings_popup.ids.epoch.text),))
        self.train_thread.setDaemon(True)
        self.train_thread.start()

    def fit(self, epochs):  
        train_loss_history = []
        test_loss_history = []
        epoch_history = {"train": train_loss_history, "test":test_loss_history}
        id = self.regions.find_id_from_address(self.ids.sensor.text)
        source_name = RegionSelector.build_file_name(self.ids.region.text, id)
        
        if RegionSelector.check_data_exist(source_name) == False:
            self.preprocess_popup.open()
            RegionSelector.pull_data_from_database(id, self.ids.region.text)
            RegionSelector.preprocess(id, self.ids.region.text, self.preprocess_popup)
            self.preprocess_popup.dismiss()

        self.train_object = Train(
        source_name,
        self.weekday, 
        self.ids.train_start.text,
        self.ids.train_end.text,
        self.ids.test_start.text,
        self.ids.test_end.text,
        int(self.settings_popup.ids.time_step.text),
        self.prev_weeks,
        self.daypart
        )       
        for i in range(epochs):
            temp_train, temp_test = self.train_object.fit(
                int(self.settings_popup.ids.batch.text)
            )
            epoch_history["train"].append(temp_train)
            epoch_history["test"].append(temp_test)
            self.manager.screens[1].update_results(epoch_history, i+1, epochs)
            self.manager.screens[1].ids.progress.value = i+1
        
        self.train_object.save_estimations(file_name)
        self.manager.screens[2].get_dataframe()
        self.manager.screens[1].ids.results.disabled = False
        self.manager.screens[1].ids.home.disabled = False
        self.build_result_title()
        self.unpin_map()
        self.reset_spinners()
        self.reset_map_zoom()
        self.current_marker_id = -1

    def pin_map(self):
        self.ids.sensor.text = "Sensor Seciniz"
        self.unpin_map()
        self.ids.sensor.values = []
        sensors = self.regions.get_sensors(self.ids.region.text)
        for index, row in sensors.iterrows():
            file = RegionSelector.build_file_name(self.ids.region.text, row.ID)
            exists = RegionSelector.check_data_exist(file)
            marker = MyMarker(self, row.percentage, exists, row.address, lon=row.long, lat=row.lat)
            self.markers[row.ID] = marker
            self.ids.map.add_marker(marker)
            self.ids.sensor.values.append(row.address)
        self.reset_map_zoom()
        avg_lat, avg_lon = self.get_center_of_markers()
        self.ids.map.lat = avg_lat
        self.ids.map.lon = avg_lon
        self.ids.map.zoom = 12

    def focus_marker(self):
        id = self.regions.find_id_from_address(self.ids.sensor.text)
        self.markers[id].source = "./marker_ims/blue.png"
        self.current_marker_id = id

    def release_marker_focus(self):
        self.markers[self.current_marker_id].source = self.markers[self.current_marker_id].default_source
        self.current_marker_id = -1

    def get_center_of_markers(self):
        if len(self.markers) == 0:
            return 41.091602, 29.066435
        avg_lat = 0
        avg_lon = 0
        for marker in self.markers.values():
            avg_lat += marker.lat
            avg_lon += marker.lon
        avg_lat /= len(self.markers)
        avg_lon /= len(self.markers)
        return avg_lat, avg_lon

    def reset_map_zoom(self):
        self.ids.map.lat = 41.091602
        self.ids.map.lon = 29.066435
        self.ids.map.zoom = 10

    def build_result_title(self):
        return self.ids.sensor.text + "/" + self.ids.region.text
    
    def unpin_map(self):
        if len(self.markers) > 0:
            for marker in self.markers.values():
                self.ids.map.remove_marker(marker)
            self.markers = {}

    def reset_spinners(self):
        self.ids.region.text = "Bolge Seciniz"
        self.ids.sensor.text = "Sensor Seciniz"
        self.ids.sensor.values = []

    def on_sensor_spinner_text_change(self):
        if self.manager.current == "hyper":
            if self.current_marker_id != -1:
                self.release_marker_focus()
            
            if self.ids.sensor.text != "Sensor Seciniz":
                self.focus_marker()
   
class SettingsPopUp(Popup):
    pass


class TrainScreen(Screen):

    def update_results(self, epoch_history, current_epoch, max_epoch):     
        self.ids.header.text = "EPOK: " + str(current_epoch) + "/" + str(max_epoch)
        self.save_epoch_history_figure(epoch_history, current_epoch, max_epoch)
        self.update_image()
        epoch_info_text = "Eğitim MAPE: " + str(round(epoch_history["train"][current_epoch-1],3)) + "\n"
        epoch_info_text += "Test    MAPE: " + str(round(epoch_history["test"][current_epoch-1],3))
        self.ids.result.text = epoch_info_text

    def see_all_results_button_click(self):
        self.manager.current = self.manager.screens[2].name

    @mainthread
    def update_image(self):
        self.ids.graph.source = self.fig_path

    def go_back_button(self):
        self.reset_screen()
        self.manager.screens[0].app.title = self.manager.screens[0].app.default_title
        self.manager.screens[1].ids.results.disabled = True
        self.manager.screens[1].ids.home.disabled = True
        self.manager.current = "hyper"
        
    def save_epoch_history_figure(self, epoch_history, current_epoch, max_epoch):
        indexes = [i for i in range(1, 1 + len(epoch_history["train"]))]
        fig = plt.figure()
        plt.style.use('dark_background')
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        ax.set_xlim([1,max_epoch])
        ax.set_ylim([0,100])
        plt.title(label="Epok Gecmisi")
        ax.plot(indexes, epoch_history["train"], label="Eğitim MAPE")
        ax.plot(indexes, epoch_history["test"], label="Test MAPE")
        ax.legend()
        self.fig_path = "epoch_his/fig_" + str(current_epoch) + ".png" 
        fig.savefig(self.fig_path)    
        ax.cla()

    @mainthread
    def reset_screen(self):
        self.fig_path = "epoch_his/default.png"
        self.update_image()
        self.ids.header.text = "Ilk epok sonuclari bekleniyor"
        self.ids.result.text = ""


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

    def go_back_button(self):
        self.manager.current = "train"

    def go_home_button(self):
        self.manager.screens[0].app.title = self.manager.screens[0].app.default_title
        self.manager.screens[1].ids.results.disabled = True
        self.manager.screens[1].ids.home.disabled = True
        self.manager.screens[1].reset_screen()
        self.manager.current = "home"


class HomeScreen(Screen):
    
    def go_to_hyper_screen(self):
        self.manager.current = self.manager.screens[0].name


class MyMarker(MapMarker):

    def __init__(self, screen, percentage,has_data, address,*args, **kwargs):
        super(MyMarker, self).__init__(*args, **kwargs)
        self.screen = screen
        if percentage < 80:
            self.source = "./marker_ims/red.png"
        elif has_data:
            self.source = "./marker_ims/green.png"
        else:
            self.source = "./marker_ims/yellow.png"
        self.default_source = self.source
        self.address = address
        

    def update_sensor_spinner_text(self):
        self.screen.ids.sensor.text = self.address


class PreprocessPopup(Popup):
    pass


MapApp().run()

