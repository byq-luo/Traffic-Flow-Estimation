import csv
from datetime import datetime


def save_rows_to_csv(cursor, number_of_columns, file_name):
    """Saves the results of an sql insert query to a CSV file.

    :param cursor: Results of the query
    :param number_of_columns: Number of columns in the result
    :param file_name: Name of the CSV file.
    """
    with open(file_name, 'w') as f:
        row = cursor.fetchone()
        while row:
            line = construct_line(row, number_of_columns)            
            row = cursor.fetchone()
            if(row is not None):
                line += "\n"
            f.write(line)


def construct_line(row, number_of_columns):
    """Constructs a CSV line from a row of cursor.

    :param row: Row of sql cursor
    :param number_of_columns: Number of columns in the row
    :return: String of csv line
    """
    line = ''
    for i in range(number_of_columns):
        line += str(row[i])
        if i + 1 != number_of_columns:
            line += ","
    return line


def read_from_csv(file_name):
    """Reads CSV file into a matrix.
    """
    return list(csv.reader(open(file_name, "r"), delimiter=","))


def construct_file_name(segID, segDir, starting_date, ending_date, is_preprocessed):
    """Constructs a file name string for consistency for the
    data read from SQL.
    """
    file_name = ".\\Data\\" + str(segID) + "_" + str(segDir) + "_"
    file_name += starting_date + "_" + ending_date + "_" + str(is_preprocessed) + ".csv"
    return file_name
    

