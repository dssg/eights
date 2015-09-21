import numpy as np class ArrayEmitter(object):
    """
    Array emitter is a tool that accepts tables from either SQL or CSVs in the 
    RG format, then generates Numpy structured arrays in the M format based on 
    selection criteria on those tables.
    
    **RG Tables**

    Tables can be specified from either a CSV file (using the 
    get_rg_from_csv method) or from a SQL query (using the
    get_rg_from_SQL method). Imported tables must adhere to the *RG* format:

    *Table 1--an example RG-format table*

    +------------+------------+----------+-------------+-------+
    | student_id | start_year | end_year |     feature | value |
    +============+============+==========+=============+=======+
    |          0 |       2005 |     2006 |    math_gpa |   2.3 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2005 |     2006 | english_gpa |   4.0 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2005 |     2006 |    absences |     7 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2006 |     2007 |    math_gpa |   2.1 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2006 |     2007 | english_gpa |   3.9 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2006 |     2007 |    absences |     8 |
    +------------+------------+----------+-------------+-------+
    |          1 |       2005 |     2006 |    math_gpa |   3.4 |
    +------------+------------+----------+-------------+-------+
    |          1 |       2005 |     2006 |    absenses |     0 |
    +------------+------------+----------+-------------+-------+
    |          1 |       2006 |     2007 |    math_gpa |   3.5 |
    +------------+------------+----------+-------------+-------+
    |          1 |       2007 |     2008 | english_gpa |   2.4 |
    +------------+------------+----------+-------------+-------+
    |          2 |       2004 |     2005 |    math_gpa |   2.4 |
    +------------+------------+----------+-------------+-------+
    |          2 |       2005 |     2006 |    math_gpa |   3.4 |
    +------------+------------+----------+-------------+-------+
    |          2 |       2005 |     2006 |    absenses |    14 |
    +------------+------------+----------+-------------+-------+
    |          2 |       2006 |     2007 |    absenses |    96 |
    +------------+------------+----------+-------------+-------+

    In an RG-formatted table, there are five columns:
    
    1. The unique identifier of a unit. By "unit," we mean unit in a
       statistical sense, where a population consists of a number of units.
       In Table 1, a unit is a student, and each student is uniquely 
       identified by a value that appears in the student_id column.
       Table 1 defines data for students 0, 1, and 2.
    2. The time at which a certain record begins to be applicable. In Table 1,
       start_year is this start time.
    3. The time at which a certain record ceases to be applicable. In Table 1,
       end_year is this stop time.
    4. The name of a feature applicable to that unit at that time. In Table 1,
       this is "feature" 
    5. The value of the feature for that unit at that time. In Table 1, this is
       Value

    The values in the first column uniquely identify each unit, but there
    can be more than one row in the table per unit. These tables give us
    information in the form of: "For unit u, from time t1 to time t2, feature f 
    had value x"

    In Table 1, the values of the student_id column each correspond to
    one student. Each student may have multiple rows on this table 
    corresponding to multiple features at multiple times. For example, during
    2005-2006, student 0 had a math_gpa of 2.3 and an english_gpa of 4.0.
    During 2006-2007, student 0's math_gpa dropped to 2.1, while his or her
    english_gpa dropped to 3.9.

    **M Tables**

    ArrayEmitter generates M formatted tables based on RG formatted tables. 
    For example, the RG-formatted table Table 1 might result in the following 
    M-formatted table:

    *Table 2*

    +------------+----------+-------------+----------+
    | student_id | math_gpa | english_gpa | absences |
    +============+==========+=============+==========+
    |          0 |      2.2 |        3.95 |        8 |
    +------------+----------+-------------+----------+
    |          1 |     3.45 |         nan |        0 |
    +------------+----------+-------------+----------+
    |          2 |      3.4 |         nan |       94 |
    +------------+----------+-------------+----------+

    In an M-formatted table, each unit has a single row, and each feature has
    its own column. Notice that the student_ids in Table 2 correspond to the
    student_ids in Table 1, and the names of the columns in Table 2 correspond
    to the entries in the "feature" column of Table 1. The process used to 
    determine the values in these columns is elucidated below.

    **Converting an RG-formatted table to an M-formatted table.**

    In order to decide what values appear in our M-formatted table, we:

    1. Optionally select a aggregation methods with set_aggregation and 
       set_default_aggregation
    2. Select a timeframe with emit_M

    When creating the M table, we first take only entries in the RG table
    table that fall within the timeframe specified in emit_M, then we aggregate 
    those entries using the user_specified aggretation method. If an aggreagation 
    method is not specified, ArrayGenerator will take the mean. For example, if
    we have Table 1 stored in table1.csv, and run the following:

    >>> ag = ArrayGenerator()
    >>> ag.get_rg_from_csv('table1.csv')
    >>> ag.set_aggregation('math_gpa', 'mean')
    >>> ag.set_aggregation('absences', 'max')
    >>> table2 = ag.emit_M(2005, 2006)

    we end up with Table 2

    Notice that math_gpa and english_gpa are the average for 2005 and 2006
    per student, while absenses is the max over 2005 and 2006. Also notice
    that english_gpa for student 1 is nan, since the only english_gpa for
    student 1 is from 2007, which is outside of our range. For student 2,
    english_gpa is nan because student 2 has no entries in the table for
    english_gpa.

    **Taking subsets of units**

    In addition to taking subsets of items in RG tables, we might also 
    want to take subsets of units (i.e. rows in M-format tables) according
    to some perameter. For example, we might want to consider only 
    students with a math_gpa at or below 3.4. In order to subset units, we use 
    the select_rows_in_M function. For example:

    >>> ag = ArrayGenerator()
    >>> ag.get_rg_from_csv('table1.csv')
    >>> ag.set_aggregation('math_gpa', 'mean')
    >>> ag.set_aggregation('absences', 'max')
    >>> ag = ag.select_rows_in_M('math_gpa <= 3.4')
    >>> table3 = ag.to_sa(2005, 2006)

    Gives us 
    
    *Table 3:*

    +------------+----------+-------------+----------+
    | student_id | math_gpa | english_gpa | absences |
    +============+==========+=============+==========+
    |          0 |      2.2 |        3.95 |        8 |
    +------------+----------+-------------+----------+
    |          2 |      3.4 |         nan |       94 |
    +------------+----------+-------------+----------+

    Notice that Table 3 is identical to Table 2, except student 1 has been
    omitted because his/her GPA is higher than 3.4.
   
    Parameters
    ----------
    conn_string : str or None
        SQLAlchemy connection string used to connect.
        If None, conn_string must be specified for each SQL query

    """

    def __init__(self, conn_string=None):
        pass

    def get_rg_from_SQL(self, query, unit_id_col=None, start_time_col=None,
                        stop_time_col=None, feature_col=None, val_col=None):
                        conn_string=None):
        """ Gets an RG-formatted matrix from a CSV file
           
        Parameters
        ----------
        query : str
            An SQL query that returns the RG-formatted table.

        unit_id_col : str or None
            The name of the column containing unique unit IDs. For example,
            in Table 1, this is 'student_id'. If None, ArrayEmitter will
            pick the first column

        start_time_col : str or None
            The name of the column containing start time. In Table 1,
            this is 'start_year'. If None, ArrayEmitter will pick the second
            column.

        end_time_col : str or None
            The name of the column containing the stop time. In Table 1,
            this is 'end_year'. If None, ArrayEmitter will pick the third
            column.

        feature_col : str or None
            The name of the column containing the feature name. In Table 1,
            this is 'feature'. If None, ArrayEmitter will pick the fourth
            column.

        val_col : str or None
            The name of the column containing the value for the given
            feature for the given user at the given time. In Table 1,
            this is 'value'. If None, ArrayEmitter will pick the fifth
            column.

        conn_str : str or None
            SQLAlchemy connection string to connect to the database and run
            the query. If None, the conn_str used to initialize the ArrayGenerator
            will be used
            
        Examples
        --------
        >>> conn_str = ...
        >>> ag = ArrayGenerator()
        >>> ag.get_rg_from_SQL('SELECT * FROM table_1', 'student_id', 
        ...                    conn_str=conn_str)

        """
        pass

    def get_rg_from_csv(self, csv_file_path, unit_id_col=None, 
                        start_time_col=None, stop_time_col=None, 
                        feature_col=None, val_col=None):
        """ Get an RG-formatted table from a CSV file.
       
        Parameters
        ----------
        csv_file_path : str
            Path of the csv file to import table from

        unit_id_col : str or None
            The name of the column containing unique unit IDs. For example,
            in Table 1, this is 'student_id'. If None, ArrayEmitter will
            pick the first column

        start_time_col : str or None
            The name of the column containing start time. In Table 1,
            this is 'start_year'. If None, ArrayEmitter will pick the second
            column.

        end_time_col : str or None
            The name of the column containing the stop time. In Table 1,
            this is 'end_year'. If None, ArrayEmitter will pick the third
            column.

        feature_col : str or None
            The name of the column containing the feature name. In Table 1,
            this is 'feature'. If None, ArrayEmitter will pick the fourth
            column.

        val_col : str or None
            The name of the column containing the value for the given
            feature for the given user at the given time. In Table 1,
            this is 'value'. If None, ArrayEmitter will pick the fifth
            column.


        Examples
        --------
            
        >>> ag = ArrayGenerator()
        >>> ag.get_rg_from_csv('table_1.csv')             
        """
        pass

    def set_aggregation(self, feature_name, method):
        """Sets the method used to aggregate across dates in a RG table.

        If set_aggregation is not called for a given feature, the method will
        default to 'mean'
        
        Parameters
        ----------
        feature_name : str
            Name of feature for which we are aggregating
        method : str or list -> float
            Method used to aggregate the feature across year. If a str, must
            be one of:

                * 'mean'
                * 'median'
                * 'mode'
                * 'min'
                * 'max'
                * 'latest'
                * 'earliest'

            If a function, must take a list and return a float

        Examples
        --------
        >>> ag = ArrayGenerator(...)
        >>> ... # Populate ag with Table 1 and Table 2
        >>> ag.set_aggregation('math_gpa', 'mean')
        >>> ag.set_aggregation('absences', 'max')
        >>> sa = ag.to_sa(2005, 2006)
        """
        pass

    def set_default_aggregation(self, method):
        pass
    
    def select_rows_in_M(self, where):
        """
        
        Specifies a subset of the units to be returned in the M-table 
        according to some constraint.

        Parameters
        ----------
        where : str
            A statement required to be true about the returned table using
            at least one column name, constant values, parentheses and the 
            operators: ==, !=, <, >, <=, >=, and, or, not.

        Returns
        -------
        ArrayGenerator
            A copy of the current ArrayGenerator with the additional where 
            condition added

        Examples
        --------
        >>> ag = ArrayGenerator(...)
        >>> ... # Populate ag with Table 1 and Table 2
        >>> ag.set_aggregation('math_gpa', 'mean')
        >>> ag.set_aggregation('absences', 'max')
        >>> ag = ag.set_rows_in_M('grad_year == 2007')
        >>> sa = ag.to_sa(2005, 2006)
        """
        # Note that this copies the original rather than mutating it, so
        # taking a subset does not permanently lose data.

        # We can recycle the mini-language from UPSG Query
        # https://github.com/dssg/UPSG/blob/master/upsg/transform/split.py#L210
        return self

    def select_cols_in_M(self, where):
        pass
        
    def emit_M(self, start_time, stop_time):
        """Creates a structured array in M-format

        Start times and stop times are inclusive

        Parameters
        ----------
        start_time : number or datetime.datetime
            Start time of log tables to include in this sa
        stop_time : number or datetime.datetime
            Stop time of log tables to include in this sa
        Returns
        -------
        np.ndarray
            Numpy structured array constructed using the specified queries and
            subsets
        """
        return np.array([0], dtype=[('f0', int)])

    def subset_over(self, directive):
        """
        Generates ArrayGenerators according to some subsetting directive.

        Parameters
        ----------
        directive : ?
        
        Returns
        -------
        ?
        """
        pass
