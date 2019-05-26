import pyodbc
import file_operations as fo


class DB_Definitions:
    server_name = "localhost\MSSQLSERVER02"
    database_name = "FusedData-2016-2017-2018"
    number_of_data_points = 7252




def build_connection_str(server, database):
    """ builds the string that
    requiread for connecting the ms sql server.

    :param server: Name of the server
    :param database: Name of the database
    :return: Connection string
    """
    conn_str =  (
    r'Driver={SQL Server};'
    r'Server=' + server + ';'
    r'Database=' + database + ';'
    r'Trusted_Connection=yes;'
    )    
    return conn_str


def connect_to_server(connection_string):
    """Connects the MS SQL Server.

    :param connection_string: Required sring to connect the server.
    :return: a DSN connection
    """
    return pyodbc.connect(connection_string)


def run_sql_query(cursor, query):
    """Runs the given sql query and returns the cursor back with the results.
    """
    cursor.execute(query)
    return cursor


def close_server_connection(cnxn):
    """Closes the given SQL connection."""
    cnxn.close()


def build_insert_query(table_name, col_names, col_vals, should_it_be_quoted):
    """Builds an SQL insert query.

    :param table_name: Name of the table.
    :param col_names: Names of the columns in a list
    :param col_vals: Corresponding values for the columns in a list
    :param should_it_be_quoted: A list that holds if the corresponding
    column should be quoted in the query or not. 1 if it should, 0 otherwise.
    :return: SQL insert query string.
    """
    query = "INSERT INTO "
    query += table_name
    query += " ("
    for name in col_names:
        query += name
        query += ","
    query = query[:len(query) - 1] + ")"
    query += " VALUES ("
    for i in range(len(col_vals)):
        if(should_it_be_quoted[i] == 1):
            query += "'"
        query += str(col_vals[i])
        if(should_it_be_quoted[i] == 1):
            query += "'"
        query += ","
    query = query[:len(query) - 1] + ");"
    return query
