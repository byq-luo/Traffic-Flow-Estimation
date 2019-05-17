from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.base import runTouchApp
from kivy.graphics import Rectangle, Color
from datepicker import DatePicker
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.dropdown import DropDown
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.garden.mapview import MapView, MapMarker, MapSource


dct = {'dct1' : [29.1,40.0],'dct2' : [29.1,40.0],'dct3' : [29.1,40.0],'dct4' : [29.1,40.0]}
dct1 = {'Besiktas' : [29.1,40.0],'Besiktas1' : [29.1,40.0],'Besiktas2' : [29.1,40.0]}
dct2 = {'Besiktasa' : [29.1,40.0],'Besiktas1' : [29.1,40.0],'Besiktas2' : [29.1,40.0]}
current_marker = []


#-----------------------CLASS TO CREATE GUI-------------------------------------------#
class tfe(App):
    def build(Self):

#-------MAIN LAYOUT WHICH INCLUDES ALL----------#
        superbox = BoxLayout(orientation = 'horizontal', padding = (1,1))
        with superbox.canvas:
        	Color(1, 1, 1)

#-------WIDGET LAYOUT AREA ----------------#
        widget_box = BoxLayout(orientation = 'vertical', size_hint_x = 3/10, padding = (2,2))
        gridlayout_one 	 = GridLayout(cols = 1,size_hint_y = 7 /10, padding = (2,2))
        gridlayout_two 	 = GridLayout(cols = 2,size_hint_y = 50/100,padding = (2,2))
        gridlayout_three = GridLayout(cols = 1,size_hint_y = 30/100, padding = (2,2))


#-------GRID LAYOUT ONE AREA-----------------#

		#----Spinner defition for Regioon and related tasks area----#
        def get_region_choice(spinner,text):
        	# text = user region choice
        	sensor_spinner.values = text
        region_spinner = Spinner(text = 'Select Region', values = dct.keys(), background_color = (1,1,1,1), color = (1,1,1,1),sync_height = False, padding = (5,5))
        region_spinner.bind(text = get_region_choice)

        #----Spinner defition for Sensor Choice and related tasks area----#
        def get_sensor_choice(spinner,text):
        	# Kullanıcı sensor secımı text ıcınde, buradan koordınat alman lazım harıtada ısaretleme ıcın
        	print('Kullanıcı sensor Seçimi',text)
        	map.remove_marker(current_marker[0])
        	del current_marker[0]
        	s = MapMarker(lon=29.096435, lat= 41.091602)  # Sydney
        	map.add_marker(s)
        	current_marker.append(s)
        	#map.center_on(41.091602,28.066435)

        sensor_spinner = Spinner(text = 'Select Sensor', values = dct.keys(),background_color = (1,1,1,1), color = (1,1,1,1),sync_height = False, padding = (5,5))
        sensor_spinner.bind(text = get_sensor_choice)


        #---Add Spinners to grid layout----#
        gridlayout_one.add_widget(region_spinner)
        gridlayout_one.add_widget(sensor_spinner)

#-------GRID LAYOUT TWO AREA-----------------#
        
        #-------Functions on User Date Choice------#
        def on__train_start_text(instance,value):
        	print(value)

        def on__train_end_text(instance,value):
        	print(value)

        def on__test_start_text(instance,value):
        	print(value)

        def on__test_end_text(instance,value):
        	print(value)

       	#-----Labels for Training-----#
       	train_start_label = Label(text = 'Train Start',color = (1,1,1,1))
       	gridlayout_two.add_widget(train_start_label)
       	
       	train_end_label = Label(text = 'Train End')
       	gridlayout_two.add_widget(train_end_label)

        #------DatePickers to Get User Date Choices----#
        train_start = DatePicker()
        train_start.bind(text=on__train_start_text)
        gridlayout_two.add_widget(train_start)
        
        train_end = DatePicker()
        train_end.bind(text=on__train_end_text)
        gridlayout_two.add_widget(train_end)


       	test_start_label = Label(text = 'Test Start')
       	gridlayout_two.add_widget(test_start_label)
       	test_end_label = Label(text = 'Test End')
       	gridlayout_two.add_widget(test_end_label)

        test_start = DatePicker()
        test_start.bind(text = on__test_start_text)
        gridlayout_two.add_widget(test_start)
        

        test_end = DatePicker()
        test_end.bind(text = on__test_end_text)
        gridlayout_two.add_widget(test_end)


#-------GRID LAYOUT THREE AREA-----------------#

        hyper_layout = GridLayout(cols = 2, padding = (10,10))

        top_label_one = Label()
        top_label_two = Label()
        hyper_layout.add_widget(top_label_one)
        hyper_layout.add_widget(top_label_two)
        
        def get_epoch(spinner,text):
        	print('Kullanıcı Girisli Epoch Sayısı', text)
        epoch_label = Label(text = 'Epok Sayısı Seçiniz',font_size='15sp', bold = True, halign = 'justify')
        epoch_spinner = Spinner(text = 'Epoch Sayısı', values=('Home', 'Work', 'Other', 'Custom'))
        epoch_spinner.bind(text =get_epoch)
        hyper_layout.add_widget(epoch_label)
        hyper_layout.add_widget(epoch_spinner)


        def get_time_stamp(spinner,text):
        	print('Kullanıcı Zaman Aralığı Seçimi',text)
        zaman_araligi_label = Label(text = 'Zaman Aralığı Seçiniz',font_size='15sp', bold = True,halign = 'justify')
        zaman_araligi = Spinner(text = 'Zaman Aralığı', values=('Home', 'Work', 'Other', 'Custom'))
        zaman_araligi.bind(text = get_time_stamp)
        hyper_layout.add_widget(zaman_araligi_label)
        hyper_layout.add_widget(zaman_araligi)


        def get_batch_size(spinner,text):
        	print('Kullanıcı batch size ', text)
        batch_label = Label(text = 'Batch Boyutu Seçiniz',font_size='15sp', bold = True,halign = 'justify')
        batch = Spinner(text = 'Batch Boyutu', values=('Home', 'Work'))
        batch.bind(text = get_batch_size)
        hyper_layout.add_widget(batch_label)
        hyper_layout.add_widget(batch)


        def get_hafta_vector(spinner,text):
        	print('Kullanıcı hafta vektor Seçimi', text)
        hafta_vektor_label = Label(text = 'Hafta Vektörü Eklensin Mi ?',font_size='15sp', bold = True,halign = 'justify')
        hafta_vektor = Spinner(text = 'Hafta Vektörü', values=('Evet', 'Hayır'))        
        hafta_vektor.bind(text = get_hafta_vector)
        hyper_layout.add_widget(hafta_vektor_label)
        hyper_layout.add_widget(hafta_vektor)


        def get_sekans(spinner,text):
        	print('Kullanıcı hafta sekans secimi',text)
        hafta_sekans_label = Label(text = 'Kaç Hafta Eklensin ?',font_size='15sp', bold = True,halign = 'justify')
        hafta_sekans = Spinner(text = 'Hafta Sayısı', values = ('1 Hafta', '1+2 Hafta', '1+2+3 Hafta'))
        hafta_sekans.bind(text = get_sekans)
        hyper_layout.add_widget(hafta_sekans_label)
        hyper_layout.add_widget(hafta_sekans)


        bottom_label_one = Label()
        bottom_label_two = Label()
        hyper_layout.add_widget(bottom_label_one)
        hyper_layout.add_widget(bottom_label_two)


        def open_hyper_popup(instance):
        	popup.open()

        popup = Popup(title = 'Set Hyper Parameters', content = hyper_layout,size_hint=(None, None), size=(425, 425))
        hyper_button = Button(text = 'Set Hyper Parameters')
        hyper_button.bind(on_release = open_hyper_popup)

        start_button = Button(text = 'Start Training')
        gridlayout_three.add_widget(hyper_button)
        gridlayout_three.add_widget(start_button)


#-------MAP LAYOUT AREA -----------------------#
        
        map_box = BoxLayout(orientation='horizontal',padding = (1,1))
        map = MapView(zoom=10, lon=28.979530, lat=41.015137,double_tap_zoom = True)
        m = MapMarker(lon=29.066435, lat= 41.091602)  # Sydney
        mapsource = MapSource()
        map.add_marker(m)
        current_marker.append(m)
        map_box.add_widget(map)


#-------CREATE LAYOUT TREE ----------------#
        widget_box.add_widget(gridlayout_one)
        widget_box.add_widget(gridlayout_two)
        widget_box.add_widget(gridlayout_three)
        superbox.add_widget(widget_box)
        superbox.add_widget(map_box)

        return superbox

if __name__ == '__main__':
    tfe().run()