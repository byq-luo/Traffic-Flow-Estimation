from datetime import datetime
import sql_server_processing as ssp
import file_operations as fo
import data_preprocessing as dp
from numpy import hstack


id = 442
direction = 0
starting_date = '2017-04-03 00:00:00'
ending_date =  '2017-04-03 23:59:00'
time_intervals = [10, 20, 30]
sigmas = [2, 3, 4]
'''
cnxn = ssp.connect_to_server(ssp.produce_connection_str(ssp.server_name, ssp.database_name))
cursor = cnxn.cursor()

query = (
        "select fusedDate, fusedSpeed from fusedData2017 "
        r"where vSegID = " + str(id) + " and vSegDir = 0 and "
        r"fusedDate between '" + starting_date + "' and '" + ending_date + "'"
        )

cursor = ssp.run_sql_query(cursor, query)
fo.save_rows_to_csv(cursor, 2, "outlier_test_file_csv")
'''
data_str = fo.read_from_csv("outlier_test_file.csv")
number_of_elements = len(data_str)
data_converted = dp.str_to_datetime(data_str, (0,), dp.datetime_format)
data_converted = dp.str_to_int(data_converted, (1,))
outlier_indexes = []
for sigma in sigmas:
    for time_interval in time_intervals:
        outlier_indexes.append(dp.find_outliers_with_convolution(
            data_converted, time_interval, sigma
        ))

for i in range(len(outlier_indexes)):
    print(len(outlier_indexes[i]))


parameter_combinations = len(sigmas) * len(time_intervals)
outlier_vectors = []

for current_combination in range(parameter_combinations):
    outlier_vectors.append([])
    i = 0
    j = 0
    while(j < len(outlier_indexes[current_combination])):
        if outlier_indexes[current_combination][j] == i:
            outlier_vectors[current_combination].append(1)
            j += 1
        else:
            outlier_vectors[current_combination].append(0)
        i += 1
    while i < number_of_elements:
        outlier_vectors[current_combination].append(0)
        i += 1

values = hstack((data_str, dp.transpose_the_list(outlier_vectors)))
j = 0
for i in range(number_of_elements):
    if values[i][3] == 1 and values[i][4] == 1:
        j += 1

print(j)


#fo.save_list_to_csv(valuesq, "outliers.csv", 2 + parameter_combinations) 
