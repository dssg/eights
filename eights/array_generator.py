import numpy as np

class ArrayGenerator(object):
    """
    Array generator is a tool to confederate tables from either SQL or CSV,
    then generate Numpy structured arrays based on subsets of that 
    confederation of tables.

    **Tables**

    Tables can be specified from either a CSV file (using the 
    import_from_csv method) or from a SQL query (using the
    import_from_SQL method). Imported tables may adhere to one of two formats:

    1. The Static Format. 

        In The static format, each row represents one unit of the population.
        The table contains one column that uniquely identifies each unit. For
        example, 

        *Table 1:*

        +------------+-----------+----------------+
        | student_id | grad_year |        address |
        +============+===========+================+
        |          0 |      2007 | 1234 N Halsted |
        +------------+-----------+----------------+
        |          1 |      2008 | 5555 W Addison |
        +------------+-----------+----------------+
        |          2 |      2007 |   2726 N Clark |
        +------------+-----------+----------------+

        In Table 1, each unit is a student. Each student has his or her own 
        row, which is uniquely identified by its entry in the student_id 
        column.

        It is expected that all tables provided to the ArrayGenerator
        will have the same concept of a unit and the same unique numbering
        scheme. For example, if the ArrayGenerator is provided with Table 1,
        then every further table provided will also have a column for 
        student_id, where student X in Table 1 corresponds to student X in
        the new table.

    2. The Log Format

        In a log format table, there are four columns:
        
        1. The unique identifier of a unit. Unit here means the same thing
           as it did static format tables.
        2. A time frame for which the column is applicable.
        3. The name of a feature applicable to that unit at that time.
        4. The value of the feature for that unit at that time. 

        The values in column one still uniquely identify each unit, but there
        can be more than one row in the table per unit. These tables give us
        information in the form of: "For unit X, during time y, feature z had
        value w"

        For example,

        *Table 2:*

        +------------+------+-------------+-------+
        | student_id | year |     feature | value |
        +============+======+=============+=======+
        |          0 | 2005 |    math_gpa |   2.3 |
        +------------+------+-------------+-------+
        |          0 | 2005 | english_gpa |   4.0 |
        +------------+------+-------------+-------+
        |          0 | 2005 |    absences |     7 |
        +------------+------+-------------+-------+
        |          0 | 2006 |    math_gpa |   2.1 |
        +------------+------+-------------+-------+
        |          0 | 2006 | english_gpa |   3.9 |
        +------------+------+-------------+-------+
        |          0 | 2006 |    absences |     8 |
        +------------+------+-------------+-------+
        |          1 | 2005 |    math_gpa |   3.4 |
        +------------+------+-------------+-------+
        |          1 | 2005 |    absenses |     0 |
        +------------+------+-------------+-------+
        |          1 | 2006 |    math_gpa |   3.5 |
        +------------+------+-------------+-------+
        |          1 | 2007 | english_gpa |   2.4 |
        +------------+------+-------------+-------+
        |          2 | 2004 |    math_gpa |   2.4 |
        +------------+------+-------------+-------+
        |          2 | 2005 |    math_gpa |   3.4 |
        +------------+------+-------------+-------+
        |          2 | 2005 |    absenses |    14 |
        +------------+------+-------------+-------+
        |          2 | 2006 |    absenses |    96 |
        +------------+------+-------------+-------+

        In Table 2, the values of the student_id column each correspond to
        one student, and, consequently, one row in feature 1. However, each
        student may have multiple rows on this table corresponding to 
        multiple features at multiple times. For example, during
        2005, student 0 had a math_gpa of 2.3 and an english_gpa of 4.0.
        During 2006, student 0's math_gpa dropped to 2.1, while his or her
        english_gpa dropped to 3.9.

    **Feature Generation And Subsetting features:**

        In the structured array that ultimately results, for each unit, 
        there will be at most one column corresponding to each unique entry 
        of the 'feature' column in each log table. For example, if we 
        confederate Table 1 and Table 2, we will have columns for:

        * student_id
        * grad_year
        * address
        * absences
        * math_pga
        * english_gpa

        Each row in the resulting structured array will represent one student 
        and will have a unique student_id.
        
        In order to decide what value appears in the columns originating from 
        the log-format table, we:

        1. Optionally select an aggregation method with set_aggregation
        2. Select a timeframe with to_sa

        When creating the structured array, we first take only entries of log
        tables that fall within the timeframe, then we aggregate those entries
        using the user_specified aggretation method. If an aggreagation method
        is not specified, ArrayGenerator will take the mean. For example:

        >>> ag = ArrayGenerator(...)
        >>> ... # Populate ag with Table 1 and Table 2
        >>> ag.set_aggregation('math_gpa', 'mean')
        >>> ag.set_aggregation('absences', 'max')
        >>> sa = ag.to_sa(2005, 2006)

        Gives us 
        
        *Table 3:*

        +------------+-----------+----------------+----------+-------------+----------+
        | student_id | grad_year |        address | math_gpa | english_gpa | absences |
        +============+===========+================+==========+=============+==========+
        |          0 |      2007 | 1234 N Halsted |      2.2 |        3.95 |        8 |
        +------------+-----------+----------------+----------+-------------+----------+
        |          1 |      2008 | 5555 W Addison |     3.45 |         nan |        0 |
        +------------+-----------+----------------+----------+-------------+----------+
        |          2 |      2007 |   2726 N Clark |      3.4 |         nan |       94 |
        +------------+-----------+----------------+----------+-------------+----------+

        Notice that math_gpa and english_gpa are the average for 2005 and 2006
        per student, while absenses is the max over 2005 and 2006. Also notice
        that english_gpa for student 1 is nan, since the only english_gpa for
        student 1 is from 2007, which is outside of our range. For student 2,
        english_gpa is nan because student 2 has no entries in the table for
        english_gpa.

    **Taking subsets of units**

        In addition to taking subsets of items in log tables, we might also 
        want to take subsets of units (i.e. rows in static tables) according
        to some perameter. For example, we might want to consider only 
        students with grad_year of 2007. In order to subset units, we use the
        add_constraint function. For example:

        >>> ag = ArrayGenerator(...)
        >>> ... # Populate ag with Table 1 and Table 2
        >>> ag.set_aggregation('math_gpa', 'mean')
        >>> ag.set_aggregation('absences', 'max')
        >>> ag = ag.set_constraint('grad_year == 2007')
        >>> sa = ag.to_sa(2005, 2006)

        Gives us 
        
        *Table 4:*

        +------------+-----------+----------------+----------+-------------+----------+
        | student_id | grad_year |        address | math_gpa | english_gpa | absences |
        +============+===========+================+==========+=============+==========+
        |          0 |      2007 | 1234 N Halsted |      2.2 |        3.95 |        8 |
        +------------+-----------+----------------+----------+-------------+----------+
        |          2 |      2007 |   2726 N Clark |      3.4 |         nan |       94 |
        +------------+-----------+----------------+----------+-------------+----------+

        
    Parameters
    ----------
    conn_string : str or None
        SQLAlchemy connection string used to connect.
        If None, conn_string must be specified for each SQL query

    """"

    def __init__(self, conn_string=None):
        pass

    def import_from_SQL(self, query, unit_id_col, table_type='static', 
                        conn_string=None):
       """
       Add tables that will be confederated into structured arrays from
       SQL.
       
        Parameters
        ----------
        query : str
            An SQL query that returns at least two columns. One column must
            have the name passed to unit_id_col

        unit_id_col : str
            The name of the column containing unique unit IDs. For example,
            in Table 1:

            *Table 1:*

            +------------+-----------+----------------+
            | student_id | grad_year |        address |
            +============+===========+================+
            |          0 |      2007 | 1234 N Halsted |
            +------------+-----------+----------------+
            |          1 |      2008 | 5555 W Addison |
            +------------+-----------+----------------+
            |          2 |      2007 |   2726 N Clark |
            +------------+-----------+----------------+

            unit_id_col should be 'student_id'

        table_type : 'static' or 'log'
            Specifies whether the table is static or log format. See 
            the doc string for ArrayGenerator for an explanation

        conn_str : str or None
            SQLAlchemy connection string to connect to the database and run
            the query. If None, the conn_str used to initialize the ArrayGenerator
            will be used
            
        Examples
        --------
        >>> conn_str = ...
        >>> ag = ArrayGenerator()
        >>> ag.import_from_SQL('SELECT * FROM table_1', 'student_id', 
        ...                    table_type='static', conn_str=conn_str)
        >>> ag.import_from_SQL('SELECT * FROM table_2', 'student_id',
                               table_type='log', conn_str=conn_str)

        """
        pass

    def import_from_csv(self, csv_file_path, unit_id_col, table_type='static')
       """
       Add tables that will be confederated into structured arrays from
       a csv file.
       
        Parameters
        ----------
        csv_file_path : str
            Path of the csv file to import table from

        unit_id_col : str
            The name of the column containing unique unit IDs. For example,
            in Table 1:

            *Table 1:*

            +------------+-----------+----------------+
            | student_id | grad_year |        address |
            +============+===========+================+
            |          0 |      2007 | 1234 N Halsted |
            +------------+-----------+----------------+
            |          1 |      2008 | 5555 W Addison |
            +------------+-----------+----------------+
            |          2 |      2007 |   2726 N Clark |
            +------------+-----------+----------------+

            unit_id_col should be 'student_id'

        col_type : 'static' or 'log'
            Specifies whether the table is static or log format. See 
            the doc string for ArrayGenerator for an explanation

        Examples
        --------
            
        >>> ag = ArrayGenerator()
        >>> ag.import_from_csv('table_1.csv', 'student_id', 
        ...                    table_type='static')
        >>> ag.import_from_csv('table_2.csv', 'student_id',
                               table_type='log')
            
        """
        pass
    
    def take_subset(self, where):
        """Specifies a subset of the query to return when calling to_sa

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
        >>> tb = ArrayGenerator(...)
        ...
        >>> tbsub1 = tb.take_subset('Graduated == 1 and GPA_10 > 2.3')
        >>> tbsub2 = tb.take_subset('Graduated == 0 and not GPA_9 < 2.0')
        """
        # Note that this copies the original rather than mutating it, so
        # taking a subset does not permanently lose data.

        # We can recycle the mini-language from UPSG Query
        # https://github.com/dssg/UPSG/blob/master/upsg/transform/split.py#L210
        return self
        
    def to_sa(self):
        """
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
