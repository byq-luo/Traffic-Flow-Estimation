import pandas as pd
"""
temp = pd.read_csv(".\\hava_durumu\\sicaklik.csv", encoding = "latin")
prep = pd.read_csv(".\\hava_durumu\\toplam_yagis.csv", encoding = "latin")
wind = pd.read_csv(".\\hava_durumu\\ruzgar_yonu_ve_hizi.csv", encoding = "latin")

def create_date(year, month, day, hour):
    date = str(year) + "/" + str(month) + "/" + str(day)
    date +=  " " + str(hour) + ":00"
    return date

temp['DATE'] = temp.apply(lambda x: create_date(
        x['YEAR'], x['MONTH'], x['DAY'], x['HOUR'],          
    ), axis = 1)

temp.drop(['NAME', 'YEAR', 'MONTH', 'DAY', 'HOUR'], axis = 'columns', inplace = True)

prep['DATE'] = prep.apply(lambda x: create_date(
        x['YEAR'], x['MONTH'], x['DAY'], x['HOUR'],          
    ), axis = 1)

prep.drop(['NAME', 'YEAR', 'MONTH', 'DAY', 'HOUR'], axis = 'columns', inplace = True)

wind['DATE'] = wind.apply(lambda x: create_date(
        x['YEAR'], x['MONTH'], x['DAY'], x['HOUR'],          
    ), axis = 1)

wind.drop(['NAME', 'YEAR', 'MONTH', 'DAY', 'HOUR'], axis = 'columns', inplace = True)

def divide_direction_and_speed(str):
    
    i = 0
    while str[i].isdigit() == False:
        i += 1
    speed = float(str[i:])
    if str[i - 1:i] == " ":
        direction = str[:i - 1]
    else:
        direction = str[:i]
    return speed

wind['SPEED'] = wind['WIND'].apply(divide_direction_and_speed)
wind.drop(['WIND'], axis = 1, inplace = True)



merged = pd.merge(temp, prep, how = 'outer' ,on = ['ID', 'DATE'])
merged = pd.merge(merged, wind,how = 'outer' ,on = ['ID', 'DATE'])

merged.to_csv("merged_weather.csv")

"""
def build_insert(table_name, col_names, col_vals, col_of_datetime):
    query = "INSERT INTO "
    query += table_name
    query += " ("
    for name in col_names:
        query += name
        query += ","
    query = query[:len(query) - 1] + ")"
    query += " VALUES ("
    for i in range(len(col_vals)):
        if i == col_of_datetime:
            query += "convert(smalldatetime, '" + col_vals[i] + "'),"
        else:
            query += col_vals[i] + ","
    query = query[:len(query) - 1] + ");"
    return query


merged = pd.read_csv("merged_weather.csv")
all_cols = ['ID', 'Date', 'temperature', 'precipitation', 'windSpeed']
queries = ""
for index, row in merged.iterrows():
    not_null_col_names = []
    not_null_col_vals = []
    for col in all_cols:
        if row.isnull()[col] == False:
            not_null_col_names.append(col)
            not_null_col_vals.append(str(row[col]))
    queries += build_insert("WeatherInfo",
                            not_null_col_names,
                            not_null_col_vals,
                            1
        )
    queries += "\n"

with open("weather_insert_query.txt", "w") as f:
    f.write(queries)