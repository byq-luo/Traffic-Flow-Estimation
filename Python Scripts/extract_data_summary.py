from datetime import datetime
import sql_server_processing as ssp
import file_operations as fo
import data_preprocessing as dp



'''
segIDs = [760,442,2918,2919,521,608,443,574,
          444,472,471,1773,248,249,192,1762,1738,1739,75,789,790,
          2,450,106,1765,497,433,302,303,304,553,64]

'''
segIDs = dp.transpose_the_list(dp.str_to_int(fo.read_from_csv("good_data_points.csv"), (0,)))[0]


starting_dates = ['2017-02-20 00:00:00', '2017-04-03 00:00:00', 
                  '2017-07-03 00:00:00', '2017-10-02 00:00:00']

ending_dates =   ['2017-02-25 00:00:00', '2017-04-08 00:00:00', 
                  '2017-07-08 00:00:00', '2017-10-07 00:00:00']
summary_of_data_points = []
outlier_time_interval = 20
outlier_sigma = 2
cnxn = ssp.connect_to_server(ssp.produce_connection_str(ssp.server_name, ssp.database_name))
cursor = cnxn.cursor()
print('connected')
for id in segIDs:
    for i in range(len(starting_dates)):
        
        query = (
            "select fusedDate, fusedSpeed from fusedData2017 "
            r"where vSegID = " + str(id) + " and vSegDir = 0 and "
            r"fusedDate between '" + starting_dates[i] + "' and '" + ending_dates[i] + "'"
        )
        file_name = fo.construct_file_name(
            id, 0, starting_dates[i][5:10], ending_dates[i][5:10], False)
        print('saving data to ' + file_name)
        cursor = ssp.run_sql_query(cursor, query)
        fo.save_rows_to_csv(cursor, 2, file_name)
        
        print('pulling data from ' + file_name)
        
        #pulling data from csv 
        data_str = fo.read_from_csv(file_name)
        elements_in_data = len(data_str)
        if len(data_str) == 0:
            print('----------------')
            print('THERE IS NO DATA')
            print('----------------')
            summary_of_data_points.append(
                [id, 0, starting_dates[i], ending_dates[i], -1, -1, elements_in_data]
            )
            
        else:
            print('converting')
            #converting data types to correct ones
            data_converted = dp.str_to_datetime(data_str, (0,), dp.datetime_format)
            data_converted = dp.str_to_int(data_converted, (1,))
        
            print('removing outliers')
            #detecting and removing outliers
            outlier_indexes = dp.find_outliers_with_convolution(data_converted, outlier_time_interval, outlier_sigma)
            data_without_outliers = dp.clean_outliers(data_converted, outlier_indexes)
        
            print('subsampling data')
            #subsampling the data
            subsampled_data = dp.subsampling_the_data(
                data_without_outliers,
                datetime.strptime(starting_dates[i], dp.datetime_format), 5)
        
            print('calculating summary')
            #calculating and saving the summary of the data
            mean, sd = dp.mean_and_std(dp.transpose_the_list(subsampled_data)[1])
            summary_of_data_points.append(
                [id, 0, starting_dates[i], ending_dates[i], mean, sd, elements_in_data]
            )
        
            print('saving prerocessed data to files')
            #saving the data to file
            file_name = fo.construct_file_name(
                id, 0, starting_dates[i][5:10], ending_dates[i][5:10], True
            )
            fo.save_list_to_csv(subsampled_data, file_name, 2)

ssp.close_server_connection(cnxn)
fo.save_list_to_csv(summary_of_data_points, "summary.csv", 7)

