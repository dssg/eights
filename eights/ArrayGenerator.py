import numpy as np

class ArrayGenerator(object):
    """
    Parameters
    ----------
    uid_col : str
        The name of the column that is the unique ID across queries.
        All queries must contain a column with the provided for 
        uid_col.
    conn_string : str or None
        SQLAlchemy connection string used to connect.
        If None, conn_string must be specified for each query

    """"

    def __init__(self, uid_col, conn_string=None):
        pass

    def add_query(self, query, transpose_on=None, conn_string=None):
       """
       Add column(s) to the arrays that will be constructed.
       
        Parameters
        ----------
        query : str
            An SQL query that returns at least two columns. One must have the
            name provided as uid_col on initialization
        transpose_on : str or list of str or None
            For queries in "log format," this is the column that will be used
            to create multiple columns per row.

            By log format, we mean that the same value may appear multiple 
            times in the column named uid_col, but, if there is a value that
            appears multiple times in the uid_col, there will be a value in
            a secondary column which disambiguates the entries. For example,
            consider the following tables:

            Table 1:            
            
            +------+-----------------+-------------+
            | ID   | Date of Birth   | Graduated   |
            +======+=================+=============+
            | 1    | 1988-09-22      | 1           |
            +------+-----------------+-------------+
            | 2    | 1989-08-08      | 0           |
            +------+-----------------+-------------+

            Table 2: 

            +------+---------+-------+
            | ID   | Grade   | GPA   |
            +======+=========+=======+
            | 1    | 9       | 3.2   |
            +------+---------+-------+
            | 1    | 10      | 3.4   |
            +------+---------+-------+
            | 1    | 11      | 4.0   |
            +------+---------+-------+
            | 2    | 9       | 2.1   |
            +------+---------+-------+
            | 2    | 10      | 2.0   |
            +------+---------+-------+
            | 2    | 11      | 2.3   |
            +------+---------+-------+
            | 2    | 12      | 2.5   |
            +------+---------+-------+

            Table 2 is in log format while table 1 is not necessarily in log
            format. In table 2, Values in The ID column are non-unique.
            (Our value for uid_col would be "ID"). Instead, there is one
            entry per Grade. When we call to_sa, we would want one row per ID
            and one column per grade. Something like:

            Table 3:

            +------+-----------------+-------------+----------+-----------+-----------+-----------+
            | ID   | Date of Birth   | Graduated   | GPA\_9   | GPA\_10   | GPA\_11   | GPA\_12   |
            +======+=================+=============+==========+===========+===========+===========+
            | 1    | 1988-09-22      | 1           | 3.2      | 3.4       | 4.0       |           |
            +------+-----------------+-------------+----------+-----------+-----------+-----------+
            | 2    | 1989-08-08      | 0           | 2.1      | 2.0       | 2.3       | 2.5       |
            +------+-----------------+-------------+----------+-----------+-----------+-----------+

            To get this result, we set transpose_on='Grade' for the query
            that returns Table 2. For the query that returns Table 1, 
            we set transpose_on=None, because Table 1 has a single row per
            ID, so this transpose is unnecessary.

            We also support multiple transpose_on columns, for example if 
            table 2 also had a column for "subject" like:

            Table 4:

            +------+---------+-----------+-------+
            | ID   | Grade   | Subject   | GPA   |
            +======+=========+===========+=======+
            | 1    | 9       | Math      | 3.2   |
            +------+---------+-----------+-------+
            | 1    | 10      | Math      | 3.4   |
            +------+---------+-----------+-------+
            | 2    | 9       | Math      | 2.1   |
            +------+---------+-----------+-------+
            | 2    | 9       | History   | 2.0   |
            +------+---------+-----------+-------+
            | 2    | 10      | Math      | 2.3   |
            +------+---------+-----------+-------+
            | 2    | 10      | History   | 2.5   |
            +------+---------+-----------+-------+

            then we could set transpose_on=['Grade', 'Subject'] to generate
            columns like GPA_9_Math, GPA_9_Histor, GPA_10_Math etc.

        conn_str : str or None
            SQLAlchemy connection string to connect to the database and run
            the query. If None, the conn_str used to initialize the ArrayGenerator
            will be used
            
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
