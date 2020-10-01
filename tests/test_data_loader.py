from ..data_loader import format_datetime, load_events, load_covid_results, convert_to_float, load_pathology, \
    filter_events, eu_admissions, label_endpoint
import pandas as pd
import pytest


@pytest.mark.parametrize("datetime", ["2020-01-05T10:20:00Z",
                                      "2020-05-17T22:55:00Z",
                                      "2020-05-05T07:14:00Z"])
def test_format_datetime_poscase(datetime):
    formatted_datetime = format_datetime(datetime)
    assert isinstance(formatted_datetime, pd.Timestamp)
    assert formatted_datetime.tz is None


@pytest.mark.parametrize("datetime", [None, "", 42124, "not a date", "13:45"])
def test_format_datetime_negcase(datetime):
    with pytest.warns(UserWarning) as warn:
        formatted_datetime = format_datetime(datetime)
    assert formatted_datetime is None
    assert warn[0].message.args[0] == f"Invalid datetime {datetime}, returning Null"


@pytest.mark.parametrize("start_date,test_dates", [("17/04/2020", ("25/04/2020", "01/01/2020")),
                                                   ("04/04/2020", ("18/05/2020", "02/02/2020")),
                                                   ("9/6/2020", ("10/6/2020", "21/04/2020"))])
def test_load_events_poscase(con, start_date: str, test_dates: tuple):
    events = load_events(con, start_date)
    assert isinstance(events, pd.DataFrame)
    columns = ["patient_id", "component", "event_type", "event_datetime", "death", "critical_care", "source"]
    assert all([x in events.columns for x in columns])
    test_dates = [pd.to_datetime(x, format="%d/%m/%Y") for x in test_dates]
    assert events.event_datetime.min() < test_dates[0]
    assert events.event_datetime.min() > test_dates[1]


@pytest.mark.parametrize("start_date", ["16/03/2020 15:44", 48723742938, "5/5/20", "15:43", "48/2/2013", "2020/03/1"])
def test_load_events_negcase(con, start_date: str):
    with pytest.raises(ValueError) as exp:
        load_events(con, start_date)
    assert str(exp.value) == f"{start_date} is an invalid datetime, must be of format '%d/%m/%Y'"


def test_load_covid_results(con):
    covid_results = load_covid_results(con)
    assert isinstance(covid_results, pd.DataFrame)
    columns = ["patient_id", "test_datetime", "collection_datetime", "test_result"]
    assert all(x in covid_results.columns for x in columns)
    assert sum(covid_results.collection_datetime.isnull()) == 0


@pytest.mark.parametrize("value", ["74382.3242", "3234", 43.232, 341312, ">42.43", "<300"])
def test_convert_to_float_poscase(value):
    assert isinstance(convert_to_float(value), float)


@pytest.mark.parametrize("value", ["words", "some@email.co.uk", "(433)", "382-34239-1232", "23,232"])
def test_convert_to_float_negcase(value):
    with pytest.raises(ValueError) as exp:
        convert_to_float(value, errors="raise")
    assert str(exp.value) == f"Invalid value {value}, could not convert to float"
    x = convert_to_float(value, errors="ignore")
    assert x == value


@pytest.mark.parametrize("value", [None, "", "   ", "Issue with result"])
def test_convert_to_float_null(value):
    assert convert_to_float(value) is None


def test_load_pathology(con):
    patient_ids = ["10111077", "10165698", "10108462", "10147498"]
    pathology = load_pathology(con, patient_ids)
    assert isinstance(pathology, pd.DataFrame)
    x = pathology[(pathology.patient_id == "10111077") &
                  (pathology.test_datetime == format_datetime("2020-02-21T02:49:00Z")) &
                  (pathology.test_name == "Bilirubin")].test_result.values[0]
    assert x == 4.0
    x = pathology[(pathology.patient_id == "10165698") &
                  (pathology.test_datetime == format_datetime("2020-03-14T19:36:00Z")) &
                  (pathology.test_name == "Temperature [POCT]")].test_result.values[0]
    assert x == 37.5
    x = pathology[(pathology.patient_id == "10108462") &
                  (pathology.test_datetime == format_datetime("2020-05-08T19:02:00Z")) &
                  (pathology.test_name == "APTT")].test_result.values[0]
    assert x == 34.4
    x = pathology[(pathology.patient_id == "10147498") &
                  (pathology.test_datetime == format_datetime("2020-02-17T13:12:00Z")) &
                  (pathology.test_name == "Haemoglobin (Hb)")].test_result.values[0]
    assert x == 122.0


@pytest.mark.parametrize("patient_id,expected_admissions,expected_attendance",
                         [("10145015", 0, 1),
                          ("10090226", 0, 3),
                          ("10194072", 1, 1),
                          ("10130106", 3, 3)])
def test_filter_events(con, patient_id, expected_admissions, expected_attendance):
    events = load_events(con=con, start_date="01/01/2020")
    admissions = filter_events(events=events, patient_id=patient_id, filters=dict(event_type="ADMISSION", component="IP"))
    assert admissions.shape[0] == expected_admissions
    attendance = filter_events(events=events, patient_id=patient_id, filters=dict(event_type="ATTENDANCE", component="EU"))
    assert attendance.shape[0] == expected_attendance


@pytest.mark.parametrize("patient_id, expected_n, expected_covid_n, expected_attend", [("10090465", 0, 0, 0),
                                                                                       ("10090361", 0, 0, 0),
                                                                                       ("10092553", 1, 0, 0)])
def test_eu_admissions(con, patient_id, expected_n, expected_covid_n, expected_attend):
    events = load_events(con=con, start_date="01/01/2020")
    events = events[events.patient_id == patient_id]
    covid_results = load_covid_results(con=con)
    df = eu_admissions(pt_events=events, attendance_only=False, admission_window=48, covid_positivity_window=None)
    assert df.shape[0] == expected_n
    df = eu_admissions(pt_events=events, attendance_only=False, admission_window=48, covid_positivity_window=(30, 30),
                       covid_results=covid_results)
    assert df.shape[0] == expected_n
    df = eu_admissions(pt_events=events, attendance_only=True, admission_window=48, covid_positivity_window=None)
    assert df.shape[0] == expected_n

