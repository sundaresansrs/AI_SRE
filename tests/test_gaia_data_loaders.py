from pathlib import Path

import pandas as pd

from src.data_loaders.business_loader import load_business
from src.data_loaders.metric_loader import load_metric
from src.data_loaders.trace_loader import load_trace
from src.data_loaders import run_truth

TARGET_COLUMNS = ["timestamp", "service_name", "metric_name", "value", "is_anomaly", "fault_type"]


def assert_standard_schema(df: pd.DataFrame) -> None:
    assert list(df.columns) == TARGET_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].notna().all()
    assert pd.api.types.is_float_dtype(df["value"])
    assert pd.api.types.is_bool_dtype(df["is_anomaly"])
    assert df["service_name"].dtype == object
    assert df["metric_name"].dtype == object
    assert df["fault_type"].dtype == object


def test_business_loader(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample_business.csv"
    csv_path.write_text(
        "datetime,service,message\n"
        "2021-07-01 12:44:56,dbservice2,2021-07-01 12:44:56 | INFO | 0.0.0.2 | dbservice2 | permission succeeded\n"
        "2021-07-01 12:45:05,dbservice2,2021-07-01 12:45:05 | ERROR | 0.0.0.2 | dbservice2 | retry failed\n",
        encoding="utf-8",
    )

    df = load_business(csv_path)
    assert_standard_schema(df)
    assert set(df["service_name"]) == {"dbservice2"}
    assert df["metric_name"].tolist() == ["business_event", "business_event"]
    assert df["is_anomaly"].tolist() == [True, True]
    assert df["fault_type"].tolist() == ["memory_anomalies", "memory_anomalies"]


def test_business_loader_handles_quote_only_lines(tmp_path: Path) -> None:
    csv_path = tmp_path / "malformed_business.csv"
    csv_path.write_text(
        "datetime,service,message\n"
        "2021-07-01 12:44:56,dbservice2,\"2021-07-01 12:44:56 | INFO | 0.0.0.2 | dbservice2 | permission succeeded\n"
        "\"\n"
        "2021-07-01 12:45:05,dbservice2,\"2021-07-01 12:45:05 | ERROR | 0.0.0.2 | dbservice2 | retry failed\n"
        "\"\n",
        encoding="utf-8",
    )

    df = load_business(csv_path)
    assert_standard_schema(df)
    assert len(df) == 2
    assert df["is_anomaly"].tolist() == [True, True]
    assert df["fault_type"].tolist() == ["memory_anomalies", "memory_anomalies"]


def test_business_loader_uses_run_truth_for_known_service_outside_fault_window(tmp_path: Path) -> None:
    csv_path = tmp_path / "outside_window_business.csv"
    csv_path.write_text(
        "datetime,service,message\n"
        "2021-07-01 00:00:00,dbservice2,2021-07-01 00:00:00 | ERROR | 0.0.0.2 | dbservice2 | retry failed\n",
        encoding="utf-8",
    )

    df = load_business(csv_path)
    assert_standard_schema(df)
    assert df["is_anomaly"].tolist() == [False]
    assert df["fault_type"].tolist() == ["normal"]


def test_business_loader_uses_embedded_timestamp_for_real_run_truth_match(tmp_path: Path) -> None:
    csv_path = tmp_path / "real_window_business.csv"
    csv_path.write_text(
        "datetime,service,message\n"
        "2021-07-02,webservice1,\"2021-07-02 10:24:40,123 | INFO | 0.0.0.1 | 172.17.0.3 | webservice1 | 5869227cf332787e | request http://0.0.0.2:9387/set_key_value_into_redis\"\n"
        "2021-07-02,webservice1,\"2021-07-02 10:24:41,456 | INFO | 0.0.0.1 | 172.17.0.3 | webservice1 | 5869227cf332787e | uuid: 8ff943fc-dadc-11eb-92c0-0242ac110003 write redis successfully\"\n",
        encoding="utf-8",
    )

    df = load_business(csv_path)
    assert_standard_schema(df)
    assert df["is_anomaly"].tolist() == [True, True]
    assert df["fault_type"].tolist() == ["memory_anomalies", "memory_anomalies"]


def test_overlapping_fault_windows_choose_latest_start(monkeypatch) -> None:
    truth = pd.DataFrame(
        {
            "service_name": ["testservice", "testservice"],
            "fault_type": ["long_fault", "nested_fault"],
            "start": [pd.Timestamp("2021-01-01 00:00:00"), pd.Timestamp("2021-01-01 00:05:00")],
            "end": [pd.Timestamp("2021-01-01 01:00:00"), pd.Timestamp("2021-01-01 00:15:00")],
        }
    )
    monkeypatch.setattr(run_truth, "load_run_truth", lambda: truth)

    timestamp = pd.Timestamp("2021-01-01 00:10:00")
    assert run_truth.is_anomaly_for("testservice", timestamp) == (True, "nested_fault")
    batch = run_truth.is_anomaly_for_batch(
        pd.Series(["testservice"]),
        pd.Series([timestamp]),
    )
    assert batch.iloc[0].to_dict() == {"is_anomaly": True, "fault_type": "nested_fault"}


def test_metric_loader(tmp_path: Path) -> None:
    csv_path = tmp_path / "dbservice1_0.0.0.4_docker_cpu_core_0_norm_pct_2021-07-01_2021-07-15.csv"
    csv_path.write_text(
        "timestamp,value\n"
        "1625133601000,0.0\n"
        "1625133631000,0.5\n",
        encoding="utf-8",
    )

    df = load_metric(csv_path)
    assert_standard_schema(df)
    assert set(df["service_name"]) == {"dbservice1"}
    assert df["metric_name"].tolist() == ["docker_cpu_core_0_norm_pct", "docker_cpu_core_0_norm_pct"]
    assert df["is_anomaly"].tolist() == [False, False]
    assert df["fault_type"].tolist() == ["normal", "normal"]


def test_metric_loader_uses_run_truth_for_known_service_in_window(tmp_path: Path) -> None:
    csv_path = tmp_path / "webservice1_0.0.0.1_docker_cpu_core_8_norm_pct_2021-07-01_2021-07-15.csv"
    csv_path.write_text(
        "timestamp,value\n"
        f"{int(pd.Timestamp('2021-07-02 10:24:35').value / 1_000_000)},0.91\n"
        f"{int(pd.Timestamp('2021-07-02 10:24:45').value / 1_000_000)},0.88\n",
        encoding="utf-8",
    )

    df = load_metric(csv_path)
    assert_standard_schema(df)
    assert df["is_anomaly"].tolist() == [True, True]
    assert df["fault_type"].tolist() == ["memory_anomalies", "memory_anomalies"]


def test_metric_loader_uses_run_truth_for_known_service_outside_fault_window(tmp_path: Path) -> None:
    csv_path = tmp_path / "webservice1_0.0.0.1_docker_cpu_core_8_norm_pct_2021-07-01_2021-07-15.csv"
    csv_path.write_text(
        "timestamp,value\n"
        f"{int(pd.Timestamp('2021-07-01 00:00:00').value / 1_000_000)},0.1\n",
        encoding="utf-8",
    )

    df = load_metric(csv_path)
    assert_standard_schema(df)
    assert df["is_anomaly"].tolist() == [False]
    assert df["fault_type"].tolist() == ["normal"]


def test_trace_loader(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace_table_dbservice1_2021-07.csv"
    csv_path.write_text(
        "timestamp,host_ip,service_name,trace_id,span_id,parent_id,start_time,end_time,url,status_code,message\n"
        "2021-07-01 11:44:27,0.0.0.4,dbservice1,c124e30fb40651dc,58ac80ceea500f66,8b3e4a4003c5119c,2021-07-01 11:44:26.632751,2021-07-01 11:44:27.151922,http://0.0.0.4:9388/db_login_methods,200,request call function 1 dbservice1.db_login_methods\n"
        "2021-07-01 14:58:24,0.0.0.4,dbservice1,fc80cb49734064c,fa6ac0cfb325b70,bf8f0b5ac0800698,2021-07-01 14:58:03.840086,2021-07-01 14:58:24.147245,http://0.0.0.4:9388/db_login_methods,500,request call function 1 dbservice1.db_login_methods\n",
        encoding="utf-8",
    )

    df = load_trace(csv_path)
    assert_standard_schema(df)
    assert set(df["service_name"]) == {"dbservice1"}
    assert df["metric_name"].tolist() == ["request_status_code", "request_status_code"]
    assert df["value"].tolist() == [200.0, 500.0]
    assert df["is_anomaly"].tolist() == [True, False]
    assert df["fault_type"].tolist() == ["memory_anomalies", "normal"]


def test_trace_loader_uses_run_truth_for_known_service_outside_fault_window(tmp_path: Path) -> None:
    csv_path = tmp_path / "trace_table_dbservice1_outside_fault.csv"
    csv_path.write_text(
        "timestamp,host_ip,service_name,trace_id,span_id,parent_id,start_time,end_time,url,status_code,message\n"
        "2021-07-01 14:58:24,0.0.0.4,dbservice1,fc80cb49734064c,fa6ac0cfb325b70,bf8f0b5ac0800698,2021-07-01 14:58:03.840086,2021-07-01 14:58:24.147245,http://0.0.0.4:9388/db_login_methods,500,request call function 1 dbservice1.db_login_methods\n",
        encoding="utf-8",
    )

    df = load_trace(csv_path)
    assert_standard_schema(df)
    assert df["is_anomaly"].tolist() == [False]
    assert df["fault_type"].tolist() == ["normal"]
