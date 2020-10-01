from multiprocessing import Pool, cpu_count
from tqdm.notebook import tqdm as tqdm_notebook
from sqlite3 import Connection
from IPython import get_ipython
from warnings import warn
from tqdm import tqdm
import pandas as pd
import numpy as np


def which_environment() -> str:
    """
    Test if module is being executed in the Jupyter environment.
    Returns
    -------
    str
        'jupyter', 'ipython' or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


def progress_bar(x: iter,
                 verbose: bool = True,
                 **kwargs) -> callable:
    """
    Generate a progress bar using the tqdm library. If execution environment is Jupyter, return tqdm_notebook
    otherwise used tqdm.
    Parameters
    -----------
    x: iterable
        some iterable to pass to tqdm function
    verbose: bool, (default=True)
        Provide feedback (if False, no progress bar produced)
    kwargs:
        additional keyword arguments for tqdm
    :return: tqdm or tqdm_notebook, depending on environment
    """
    if not verbose:
        return x
    if which_environment() == 'jupyter':
        return tqdm_notebook(x, **kwargs)
    return tqdm(x, **kwargs)


def format_datetime(datetime: str):
    """
    Given a timestamp string from CHAD database, remove timezone and return a Pandas Timestamp
    Parameters
    ----------
    datetime: str

    Returns
    -------
    pd.Timestamp
    """
    if not datetime:
        warn(f"Invalid datetime {datetime}, returning Null")
        return None
    try:
        return pd.to_datetime(datetime, format="%Y-%m-%dT%H:%M:%SZ").tz_localize(None)
    except ValueError:
        warn(f"Invalid datetime {datetime}, returning Null")
        return None


def load_events(con: Connection,
                start_date: str,
                patient_ids: list or None = None):
    """
    Load the events table as a Pandas DataFrame with events occurring after the given event date
    Parameters
    ----------
    con: Connection
        SQLite3 connection object
    start_date: str
        Date string of the format "%d/%m/%Y"
    patient_ids: list or None (optional)
        If provided, search is limited to given patient IDs
    Returns
    -------
    pd.DataFrame
    """
    if patient_ids is None:
        q = """SELECT patient_id, component, event_type, event_datetime, death, critical_care, source FROM Events;"""
        events = pd.read_sql(q, con=con)
    else:
        q = "SELECT patient_id, component, event_type, event_datetime, death, critical_care " \
            "FROM Events WHERE patient_id IN (" + ",".join("?" * len(patient_ids)) + ");"
        events = pd.read_sql(q, con=con, params=patient_ids)
    with Pool(cpu_count()) as pool:
        events["event_datetime"] = list(pool.map(format_datetime, events.event_datetime.values))
    try:
        start_date = pd.to_datetime(start_date, format="%d/%m/%Y")
    except ValueError:
        raise ValueError(f"{start_date} is an invalid datetime, must be of format '%d/%m/%Y'")
    return events[events.event_datetime >= start_date]


def load_covid_results(con: Connection):
    """
    Load the COVID-19 PCR results table as a Pandas DataFrame. Where collection datetime is missing,
    values will be filled by the test datetime

    Parameters
    ----------
    con: Connection
        SQLite3 Connection object

    Returns
    -------
    pd.DataFrame
    """
    q = "SELECT patient_id, test_datetime, collection_datetime, test_result " \
        "FROM Microbiology WHERE test_name='Covid19-PCR' AND test_result='Positive';"
    covid_results = pd.read_sql(q, con=con)
    covid_results["collection_datetime"].fillna(covid_results.test_datetime, inplace=True)
    with Pool(cpu_count()) as pool:
        covid_results["test_datetime"] = list(pool.map(format_datetime, covid_results.test_datetime.values))
        covid_results["collection_datetime"] = list(pool.map(format_datetime, covid_results.collection_datetime.values))
    return covid_results


def convert_to_float(value: str or int or float, errors: str = "raise"):
    """
    Given a value from a Pathology test result, convert to a float, handling > or < symbols by
    removing and returning the raw value. Null value or empty strings return None. Erroneous values
    in the database hold a value of 'Issue with result', these values will also return None.

    Parameters
    ----------
    value: str or int or float
        Value to convert
    errors: str (default="raise")
        How to handle errors. If "ignore" then errors are ignored and the original value is returned.
    Returns
    -------
    float or None
    """
    if isinstance(value, float) or isinstance(value, int):
        return float(value)
    if value is None or value == "Issue with result":
        return None
    if len(value.strip()) == 0:
        return None
    try:
        return float(value.replace(">", "").replace("<", ""))
    except ValueError:
        if errors == "ignore":
            return value
        raise ValueError(f"Invalid value {value}, could not convert to float")


def load_pathology(con: Connection, patient_ids: list):
    """
    Given a list of patient identifiers, load the pathology table and return a dataframe containing
    results for only the given patients.

    Parameters
    ----------
    con: Connection
        SQLite3 Connection object
    patient_ids: list
        List of patient IDs (as strings)

    Returns
    -------
    pd.DataFrame
    """
    q = "SELECT * FROM Pathology WHERE patient_id IN (" + ",".join("?" * len(patient_ids)) + ");"
    pathology = pd.read_sql(q, con=con, params=patient_ids)
    pathology["collection_datetime"].fillna(pathology.test_datetime, inplace=True)
    with Pool(cpu_count()) as pool:
        pathology["test_datetime"] = list(pool.map(format_datetime, pathology.test_datetime.values))
        pathology["collection_datetime"] = list(pool.map(format_datetime, pathology.collection_datetime.values))
    pathology = pathology[pathology.valid == 1].drop(["valid", "request_location"], axis=1)
    pathology["test_result"] = pathology["test_result"].apply(lambda x: convert_to_float(x, errors="ignore"))
    return pathology


def filter_events(patient_id: str or None,
                  events: pd.DataFrame,
                  datetime_filter: tuple or None = None,
                  filters: dict or None = None):
    """
    Filter the events dataframe for event type and location (e.g. EU or IP) for a single patient or all
    patients if patient_id is None

    Parameters
    ----------
    patient_id: str pr None
    events: pd.DataFrame
    filters: dict (optional)
    datetime_filter
    Returns
    -------
    pd.DataFrame
    """
    events = events.copy()
    if patient_id is not None:
        events = events[events.patient_id == patient_id]
    if filters is not None:
        for k, v in filters.items():
            events = events[events[k] == v]
    if datetime_filter is not None:
        if datetime_filter[0] is not None:
            events = events[events.event_datetime >= datetime_filter[0]]
        if datetime_filter[1] is not None:
            events = events[events.event_datetime <= datetime_filter[1]]
    return events


def eu_admissions(pt_events: pd.DataFrame,
                  attendance_only: bool = False,
                  first_record_only: bool = True,
                  admission_window: int = 48,
                  covid_positivity_window: tuple or None = None,
                  covid_results: pd.DataFrame or None = None):
    """
    Given the events dataframe of a single patient, return attendance events with a
    subsequent admission within the admission_window (default=48 hours)

    Parameters
    ----------
    pt_events: pd.DataFrame
    attendance_only: bool (default=False)
    admission_window: int (default=48)
        Timeframe (in hours) after attendance event to search for subsequent admission
    covid_positivity_window: tuple or None (optional)
    covid_results
    first_record_only
    Returns
    -------
    pd.DataFrame
    """
    assert len(set(pt_events.patient_id.values)) == 1, "pt_events should contain records for only a single patient"
    pt_events = pt_events.sort_values("event_datetime")
    attendance = pt_events[(pt_events.event_type == "ATTENDANCE") &
                           (pt_events.component == "EU")]
    if attendance_only:
        return attendance
    admissions = pt_events[(pt_events.event_type == "ADMISSION") &
                           (pt_events.component == "IP")]
    if covid_positivity_window is not None:
        assert covid_results is not None, "Must provide positive COVID-19 pcr results (covid_results)"
        covid_results = covid_results[covid_results.patient_id == pt_events.patient_id.values[0]]
        if covid_results.shape[0] == 0:
            return pd.DataFrame(columns=pt_events.columns)
    idx = list()
    for i, dt in enumerate(attendance.event_datetime.values):
        limit = dt + np.timedelta64(admission_window, "h")
        if any([dt <= x <= limit for x in admissions.event_datetime.values]):
            if covid_results is not None:
                earliest_covid = dt - np.timedelta64(covid_positivity_window[0], "D")
                latest_covid = dt + np.timedelta64(covid_positivity_window[1], "D")
                if any([earliest_covid <= x <= latest_covid for x in covid_results.collection_datetime.values]):
                    if first_record_only:
                        return attendance.iloc[i]
                    else:
                        idx.append(i)
                continue
            elif first_record_only:
                return attendance.iloc[i]
            idx.append(i)
    return attendance.iloc[idx]


def label_endpoint(pt_event: pd.Series,
                   events: pd.DataFrame,
                   timelimit: int = 28,
                   endpoint: dict or None = None):
    endpoint = endpoint or dict(death=1)
    all_pt_events = events[events.patient_id == pt_event.patient_id]
    event_dt = pt_event.event_datetime
    limit = event_dt + np.timedelta64(timelimit, "D")
    in_window_events = all_pt_events[(all_pt_events.event_datetime >= event_dt) &
                                     (all_pt_events.event_datetime <= limit)]
    endpoint_ = {}
    for k, v in endpoint.items():
        endpoint_[k] = in_window_events[in_window_events[k] == v].shape[0] > 0
    return endpoint_


def append_pathology(pt_event: pd.Series,
                     pathology: pd.DataFrame,
                     time_window: tuple = (48, 72)):
    """
    Given event for a single patient, search the given dataframe of pathology results and append
    results to pt_events in the given time window.

    Parameters
    ----------
    pt_event
    pathology
    time_window

    Returns
    -------

    """
    patient_path = pathology[pathology.patient_id == pt_event.patient_id]
    earliest = pt_event.event_datetime - np.timedelta64(time_window[0], "h")
    latest = pt_event.event_datetime + np.timedelta64(time_window[1], "h")
    in_window_pathology = patient_path[(patient_path.collection_datetime >= earliest) &
                                       (patient_path.collection_datetime <= latest)]
    return pd.DataFrame(pt_event).T.merge(in_window_pathology, on="patient_id")


def load_demographics(con: Connection,
                      patient_ids: list or None = None):
    if patient_ids is None:
        q = "SELECT patient_id, age, gender, wimd FROM Patients;"
        return pd.read_sql(q, con=con)
    sql = "SELECT patient_id, age, gender, wimd FROM Patients WHERE patient_id IN (" + ",".join("?" * len(patient_ids)) + ")"
    return pd.read_sql(sql, con=con, params=patient_ids)
