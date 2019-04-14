import pandas as pd
"""
daily_sd = pd.read_csv("daily_sd.csv")
print("File was read..")

def merge_date(year, month, day):
    return str(int(year)) + "/" + str(int(month)) + "/" + str(int(day))

daily_sd['Date'] = daily_sd.apply(lambda x: merge_date(
        x['Year'], x['Month'], x['Day']
    ), axis = 'columns')
print("Column date was created")

daily_sd.drop(['Year', 'Month', 'Day'], axis = 'columns', inplace = True)
print("Useless columns were dropped")

daily_sd.to_csv("daily_sd_v2.csv")

"""

daily_sd = pd.read_csv("daily_sd_v2.csv")
table_name = "DailyStDOfSpeedData"
col_names = ['ID', 'Direction','Date', 'Count', 'StD']

def build_insert(table_name, col_names, col_vals, col_of_datetime):
    query = "INSERT INTO "
    query += table_name
    query += " ("
    for name in col_names:
        query += name
        query += ","
    query = query[:len(query) - 1] + ")"
    query += " VALUES ("
    for i in range(len(col_names)):
        if i == col_of_datetime:
            query += "convert(date, '" + str(col_vals[col_names[i]]) + "'),"
        else:
            query += str(col_vals[col_names[i]]) + ","
    query = query[:len(query) - 1] + ");"
    return query

queries = ""

def round_2(n):
    return round(n,2)

daily_sd['StD'] = daily_sd['StD'].apply(round_2)

for index, row in daily_sd.iterrows():
    queries += build_insert(table_name, col_names, row, 2)
    print(index)
    queries += "\n"
    

print("queries were buildt")
with open("daily_sd.txt", "w") as f:
    f.write(queries)
    
