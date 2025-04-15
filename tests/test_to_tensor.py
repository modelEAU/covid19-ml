import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append("src")
sys.path.append("tests")

from fixtures import (  # noqa: E402 F401
    parquet_path_short,
    sample_tensor_config_regression,
)

from covid19_ml.to_tensor import (  # noqa: E402
    compute_centreau_health,  # noqa: E402
    compute_curvature,  # noqa: E402
    compute_slope,  # noqa: E402
    compute_smooth,  # noqa: E402
    compute_stpaul_health,  # noqa: E402
    compute_trend_curvature,  # noqa: E402
    compute_trend_slope,  # noqa: E402
    compute_ts_types,  # noqa: E402
    create_load_variables,  # noqa: E402
    extract_poly_population,  # noqa: E402
    find_key,  # noqa: E402
    get_variable_rank,  # noqa: E402
    interpolate_values,  # noqa: E402
    shift_back,  # noqa: E402
    shift_forward,  # noqa: E402
    to_tensor,  # noqa: E402
)
from covid19_ml.types_ml import City, TimeSeriesType, Variable  # noqa: E402


def test_find_key():
    assert find_key({"a": 1}, 1) == "a"
    assert find_key({"a": 1}, 3) is None
    assert find_key({"a": 1, "b": 1}, 1) == "a"


def test_shift_back():
    input_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = 2
    expected = np.array([[np.nan, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]]).T

    np.testing.assert_allclose(shift_back(input_vec, n), expected, equal_nan=True)


def test_shift_forward():
    input_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = 2
    expected = np.array(
        [[2.0, 3.0, 4.0, 5.0, np.nan], [3.0, 4.0, 5.0, np.nan, np.nan]]
    ).T
    print(shift_forward(input_vec, n))
    print(expected)
    np.testing.assert_allclose(shift_forward(input_vec, n), expected, equal_nan=True)


def test_extract_poly_population():
    # Create a test DataFrame
    df = pd.DataFrame(
        {
            "OverlappingPoly-1-Polygon_pop": ["100", "100", "100", "", "100"],
            "OverlappingPoly-2-Polygon_pop": ["200", "200", "300", "400", "200"],
            "OverlappingPoly-3-Polygon_pop": ["", "", "", "", ""],
            "SewershedPoly-Polygon_pop": ["300", "300", "300", "300", "300"],
        }
    )

    # Test extracting population of a non-sewershed polygon
    assert extract_poly_population(df, "1") == 100.0

    # Test extracting population of a sewershed polygon
    assert extract_poly_population(df, "SewershedPoly", sewershed=True) == 300.0

    # Test raising an error when population is not defined for the polygon
    with pytest.raises(ValueError, match="Population is not defined for the polygon"):
        extract_poly_population(df, "3")

    # Test raising an error when population is not constant for the whole time range
    with pytest.raises(
        ValueError, match="Population is not constant for the whole time range"
    ):
        extract_poly_population(df, "2")


def test_compute_smooth():
    # Create a test DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [5, 4, 3, 2, 1]})

    # Test smoothing with an even window size
    with pytest.raises(
        ValueError,
        match="The window size for the Savitzky-Golay filter must be an odd integer greater than or equal to 3",
    ):
        compute_smooth(df, 4)

    # Test smoothing with an odd window size
    smoothed_df = compute_smooth(df, 5)
    assert isinstance(smoothed_df, pd.DataFrame)
    assert smoothed_df.shape == (5, 2)

    expected_1 = np.array([1.171428571, 1.914285714, 3.0, 4.085714286, 4.828571429])
    expected_2 = np.array([1.171428571, 1.914285714, 3.0, 4.085714286, 4.828571429])[
        ::-1
    ]
    assert np.allclose(smoothed_df["col1_SMOOTH"].to_numpy(), expected_1)
    assert np.allclose(smoothed_df["col2_SMOOTH"].to_numpy(), expected_2)


def test_compute_slope():
    # Create a test DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [5, 4, 3, 2, 1]})

    # Test computing slope
    df_with_slope = compute_slope(df, smooth_window=5, model_window=5)
    expected_df_with_slopes = pd.DataFrame(
        {
            "col1_SLOPE": [0.910367, 0.910367, 0.910367, 0.910367, 0.910367],
            "col2_SLOPE": [-0.910367, -0.910367, -0.910367, -0.910367, -0.910367],
        }
    )

    assert isinstance(df_with_slope, pd.DataFrame)
    assert df_with_slope.shape == (5, 2)
    np.testing.assert_allclose(
        df_with_slope["col1_SLOPE"].to_numpy(),
        expected_df_with_slopes["col1_SLOPE"],
        equal_nan=True,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        df_with_slope["col2_SLOPE"].to_numpy(),
        expected_df_with_slopes["col2_SLOPE"],
        equal_nan=True,
        rtol=1e-5,
    )


def test_compute_curvature():
    # Create a test DataFrame
    df = pd.DataFrame({"col1": [1, 8, 27, 64, 125], "col2": [-125, -64, -27, -8, -1]})

    # Test computing slope
    df_with_curvature = compute_curvature(
        df, smooth_window=5, model_window=5, alpha=0.5
    )
    expected_df_with_curvatures = pd.DataFrame(
        {
            "col1_CURVATURE": [8.1043032, 8.1043032, 8.1043032, 8.1043032, 8.1043032],
            "col2_CURVATURE": [
                -8.1043032,
                -8.1043032,
                -8.1043032,
                -8.1043032,
                -8.1043032,
            ],
        }
    )

    assert isinstance(df_with_curvature, pd.DataFrame)
    assert df_with_curvature.shape == (5, 2)
    np.testing.assert_allclose(
        df_with_curvature["col1_CURVATURE"].to_numpy(),
        expected_df_with_curvatures["col1_CURVATURE"],
        equal_nan=True,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        df_with_curvature["col2_CURVATURE"].to_numpy(),
        expected_df_with_curvatures["col2_CURVATURE"],
        equal_nan=True,
        rtol=1e-5,
    )


def test_compute_trend_slope():
    # Create a test DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [5, 4, 3, 2, 1]})

    # Test computing slope
    df_with_trend_slope = compute_trend_slope(df, smooth_window=5, model_window=5)
    expected_df_with_trend_slopes = pd.DataFrame(
        {
            "col1_TREND_SLOPE": [1, 1, 2, 1, 1],
            "col2_TREND_SLOPE": [1, 1, 0, 1, 1],
        }
    )

    assert isinstance(df_with_trend_slope, pd.DataFrame)
    assert df_with_trend_slope.shape == (5, 2)
    np.testing.assert_allclose(
        df_with_trend_slope["col1_TREND_SLOPE"].to_numpy(),
        expected_df_with_trend_slopes["col1_TREND_SLOPE"],
        equal_nan=True,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        df_with_trend_slope["col2_TREND_SLOPE"].to_numpy(),
        expected_df_with_trend_slopes["col2_TREND_SLOPE"],
        equal_nan=True,
        rtol=1e-5,
    )


def test_compute_trend_curvature():
    # Create a test DataFrame
    df = pd.DataFrame({"col1": [1, 8, 27, 64, 125], "col2": [-125, -64, -27, -8, -1]})

    # Test computing slope
    df_with_trend_curvature = compute_trend_curvature(
        df, smooth_window=5, model_window=5, alpha=0.5
    )
    expected_df_with_trend_curvatures = pd.DataFrame(
        {
            "col1_TREND_CURVATURE": [1, 1, 2, 1, 1],
            "col2_TREND_CURVATURE": [1, 1, 0, 1, 1],
        }
    )

    assert isinstance(df_with_trend_curvature, pd.DataFrame)
    assert df_with_trend_curvature.shape == (5, 2)
    np.testing.assert_allclose(
        df_with_trend_curvature["col1_TREND_CURVATURE"].to_numpy(),
        expected_df_with_trend_curvatures["col1_TREND_CURVATURE"],
        equal_nan=True,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        df_with_trend_curvature["col2_TREND_CURVATURE"].to_numpy(),
        expected_df_with_trend_curvatures["col2_TREND_CURVATURE"],
        equal_nan=True,
        rtol=1e-5,
    )


def test_compute_ts_types():
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [1, 8, 27, 64, 125]})

    smooth_window = 5
    model_window = 5
    pd.testing.assert_frame_equal(
        df,
        compute_ts_types(df, None, smooth_window, model_window, period=0),  # type: ignore
    )
    pd.testing.assert_frame_equal(
        df, compute_ts_types(df, [], smooth_window, model_window, period=0)
    )
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [1, 8, 27, 64, 125]})
    expected_raw = pd.DataFrame(
        {
            "col1_RAW": [1, 2, 3, 4, 5],
            "col2_RAW": [1, 8, 27, 64, 125],
        }
    )
    pd.testing.assert_frame_equal(
        expected_raw,
        compute_ts_types(
            df, [TimeSeriesType.RAW], smooth_window, model_window, period=0
        ),
    )
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [1, 8, 27, 64, 125]})

    expected_smooth = pd.DataFrame(
        {
            "col1_SMOOTH": [1.171428571, 1.914285714, 3.0, 4.085714286, 4.828571429],
            "col2_SMOOTH": [1.171428571, 7.914285714, 27.0, 71.8, 112.4857143],
        }
    )
    pd.testing.assert_frame_equal(
        expected_smooth,
        compute_ts_types(
            df, [TimeSeriesType.SMOOTH], smooth_window, model_window, period=0
        ),
    )
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [1, 8, 27, 64, 125]})
    expected_slope = pd.DataFrame(
        {
            "col1_SLOPE": [0.910367, 0.910367, 0.910367, 0.910367, 0.910367],
            "col2_SLOPE": [27.431836, 27.431836, 27.431836, 27.431836, 27.431836],
        }
    )
    pd.testing.assert_frame_equal(
        expected_slope,
        compute_ts_types(
            df, [TimeSeriesType.SLOPE], smooth_window, model_window, period=0
        ),
    )
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [1, 8, 27, 64, 125]})
    expected_curvature = pd.DataFrame(
        {
            "col1_CURVATURE": [0.0, 0.0, 0.0, 0.0, 0.0],
            "col2_CURVATURE": [8.1043032, 8.1043032, 8.1043032, 8.1043032, 8.1043032],
        }
    )
    pd.testing.assert_frame_equal(
        expected_curvature,
        compute_ts_types(
            df,
            [TimeSeriesType.CURVATURE],
            smooth_window,
            model_window,
            alpha=0.5,
            period=0,
        ),
    )
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [-1, -8, -27, -64, -125]})
    expected_trend_slope = pd.DataFrame(
        {
            "col1_TREND_SLOPE": [1, 1, 2, 1, 1],
            "col2_TREND_SLOPE": [1, 1, 0, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(
        expected_trend_slope,
        compute_ts_types(
            df, [TimeSeriesType.TREND_SLOPE], smooth_window, model_window, period=0
        ),
    )
    df = pd.DataFrame({"col1": [1, 8, 27, 64, 125], "col2": [-1, -8, -27, -64, -125]})

    expected_trend_curvature = pd.DataFrame(
        {
            "col1_TREND_CURVATURE": [1, 1, 2, 1, 1],
            "col2_TREND_CURVATURE": [1, 1, 0, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(
        expected_trend_curvature,
        compute_ts_types(
            df,
            [TimeSeriesType.TREND_CURVATURE],
            smooth_window,
            model_window,
            alpha=0.5,
            period=0,
        ),
    )


def test_create_load_variables():
    data = {
        "Calculated_timestamp": pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03"]
        ),
        "CPHD-conf_report_value": [1, 2, 3],
        "CPHD-death_report_value": [4, 5, 6],
        "CPHD-hospcen_report_value": [7, 8, 9],
        "CPHD-pposrt_report_value": [10, 11, 12],
        "CPHD-pctvaccinedose1_report_value": [13, 14, 15],
        "CPHD-pctvaccinedose2_report_value": [16, 17, 18],
        "SiteMeasure_wwbod5c_mgl_single-to-mean_value": [19, 20, 21],
        "SiteMeasure_wwcod_mgl_single-to-mean_value": [22, 23, 24],
        "SiteMeasure_wwtss_mgl_single-to-mean_value": [25, 26, 27],
        "SiteMeasure_wwnh4n_mgl_single-to-mean_value": [28, 29, 30],
        "SiteMeasure_wwtemp_degc_single-to-mean_value": [31, 32, 33],
        "SiteMeasure_wwflow_m3d_single-to-mean_value": [1, 2, 3],
        "WWMeasure_covn1_gcml_single-to-mean_value": [37, 38, 39],
    }
    df = pd.DataFrame(data)
    result = create_load_variables(df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_series_equal(
        result[Variable.N1_FLOW.value], pd.Series([37, 76, 117], name="N1_FLOW")
    )

    pd.testing.assert_series_equal(
        result[Variable.NH4_FLOW.value], pd.Series([28, 58, 90], name="NH4_FLOW")
    )
    pd.testing.assert_series_equal(
        result[Variable.TSS_FLOW.value], pd.Series([25, 52, 81], name="TSS_FLOW")
    )


def test_interpolate_values():
    # Create a test dataset with missing values
    data = {
        "date": pd.date_range(start="2021-01-01", end="2021-01-10"),
        "value1": [1, 8, 27, None, None, 216, 343, None, 729, 1000],
        "value2": [None, None, None, None, 5, None, None, None, None, 10],
        "value3": [1, 2, 3, 4, 5, 6, 7, 8, None, None],
    }
    df = pd.DataFrame(data)
    df = df.set_index("date")

    expected_data = {
        "date": pd.date_range(start="2021-01-01", end="2021-01-10"),
        "value1": [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000],
        "value2": [5, 5, 5, 5, 5, 5, 5, 10, 10, 10],
        "value3": [1, 2, 3, 4, 5, 6, 7, 8, 8, 8],
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df = expected_df.set_index("date")
    expected_df = expected_df.astype(float)

    # Test interpolate_values with a small dataset
    interpolated_df = interpolate_values(df)
    assert isinstance(interpolated_df, pd.DataFrame)
    assert not interpolated_df.isna().any().any()
    pd.testing.assert_frame_equal(interpolated_df, expected_df)


def test_extract_poly_population2():
    # Using mtl_01
    cols = {
        "OverlappingPoly-prov_qc_hlthreg_montreal-Polygon_link": [0, "", 8],
        "OverlappingPoly-prov_qc_hlthreg_montreal-Polygon_name": [0, "", 8],
        "OverlappingPoly-prov_qc_hlthreg_montreal-Polygon_notes": [0, "", 8],
        "OverlappingPoly-prov_qc_hlthreg_montreal-Polygon_polygonID": [0, "", 8],
        "OverlappingPoly-prov_qc_hlthreg_montreal-Polygon_pop": [14, "", 8],
        "OverlappingPoly-prov_qc_hlthreg_montreal-Polygon_type": [0, "", 8],
        "OverlappingPoly-prov_qc_hlthreg_montreal-Polygon_wkt": [0, "", 8],
        "SewershedPoly-Polygon_link": [0, "", 8],
        "SewershedPoly-Polygon_name": [0, "", 8],
        "SewershedPoly-Polygon_notes": [0, "", 8],
        "SewershedPoly-Polygon_polygonID": [0, "", 8],
        "SewershedPoly-Polygon_pop": [17, "", 8],
        "SewershedPoly-Polygon_type": [0, "", 8],
        "SewershedPoly-Polygon_wkt": [0, "", 8],
    }
    df = pd.DataFrame(cols)

    poly_id = "mtl_01"
    is_sewershed = True
    # should raise Value error
    with pytest.raises(
        ValueError, match="Population is not constant for the whole time range"
    ):
        extract_poly_population(df, poly_id, is_sewershed)
    df = df.iloc[:-1]
    expected = 17.0
    assert extract_poly_population(df, poly_id, is_sewershed) == expected
    df = df.iloc[1:]
    # should raise Value error
    with pytest.raises(ValueError, match="Population is not defined for the polygon"):
        extract_poly_population(df, poly_id, is_sewershed)
    df = df.iloc[:-1]
    with pytest.raises(ValueError, match="Population is not defined for the polygon"):
        extract_poly_population(df, poly_id, is_sewershed)

    is_sewershed = False
    with pytest.raises(
        KeyError, match="Could not locate column OverlappingPoly-mtl_01-Polygon_pop"
    ):
        extract_poly_population(df, poly_id, is_sewershed)


def test_compute_centreau_health():
    data = {
        "OverlappingPoly-prov_qc_hlthreg_capitale_nationale-Polygon_pop": [100],
        "CPHD-prov_qc_hlthreg_capitale_nationale_conf_report_value": [1],
        "CPHD-prov_qc_hlthreg_capitale_nationale_death_report_value": [2],
        "CPHD-prov_qc_hlthreg_capitale_nationale_hospcen_report_value": [3],
        "CPHD-prov_qc_hlthreg_capitale_nationale_pposrt_report_value": [4],
        "CPHD-prov_qc_hlthreg_capitale_nationale_postest_report_value": [5],
        "CPHD-prov_qc_hlthreg_capitale_nationale_test_report_value": [6],
        "CPHD-prov_qc_hlthreg_capitale_nationale_pctvaccinedose1_report_value": [7],
        "CPHD-prov_qc_hlthreg_capitale_nationale_pctvaccinedose2_report_value": [8],
    }
    df = pd.DataFrame(data)

    expected = pd.DataFrame.from_dict(
        {
            "OverlappingPoly-prov_qc_hlthreg_capitale_nationale-Polygon_pop": [100],
            "CPHD-prov_qc_hlthreg_capitale_nationale_conf_report_value": [1],
            "CPHD-prov_qc_hlthreg_capitale_nationale_death_report_value": [2],
            "CPHD-prov_qc_hlthreg_capitale_nationale_hospcen_report_value": [3],
            "CPHD-prov_qc_hlthreg_capitale_nationale_pposrt_report_value": [4],
            "CPHD-prov_qc_hlthreg_capitale_nationale_postest_report_value": [5],
            "CPHD-prov_qc_hlthreg_capitale_nationale_test_report_value": [6],
            "CPHD-prov_qc_hlthreg_capitale_nationale_pctvaccinedose1_report_value": [7],
            "CPHD-prov_qc_hlthreg_capitale_nationale_pctvaccinedose2_report_value": [8],
            "CASES": [1000.0],
            "DEATHS": [2000.0],
            "HOSPITALIZED": [3000.0],
            "POS_TESTS_RATE": [4.0],
            "POS_TESTS": [5000.0],
            "TESTS": [6.0],
            "PCT_VAX1": [7.0],
            "PCT_VAX2": [8.0],
        },
        orient="columns",
    )

    df = compute_centreau_health(df, City.QUEBEC_EAST)
    pd.testing.assert_frame_equal(df, expected)


def test_compute_stpaul_health():
    data = {
        "SewershedPoly-Polygon_pop": [100],
        "OverlappingPoly-hennepin_county_hlthreg-Polygon_pop": [100],
        "CPHD-hennepin_county_hlthreg_conf_report_value": [1],
        "CPHD-hennepin_county_hlthreg_death_report_value": [2],
        "CPHD-hennepin_county_hlthreg_hospcen_report_value": [3],
        "CPHD-hennepin_county_hlthreg_pposrt_report_value": [4],
        "CPHD-hennepin_county_hlthreg_postest_report_value": [5],
        "CPHD-hennepin_county_hlthreg_test_report_value": [6],
        "CPHD-hennepin_county_hlthreg_pctvaccinedose1_report_value": [7],
        "CPHD-hennepin_county_hlthreg_pctvaccinedose2_report_value": [8],
        "OverlappingPoly-anoka_county_hlthreg-Polygon_pop": [100],
        "CPHD-anoka_county_hlthreg_conf_report_value": [1],
        "CPHD-anoka_county_hlthreg_death_report_value": [2],
        "CPHD-anoka_county_hlthreg_hospcen_report_value": [3],
        "CPHD-anoka_county_hlthreg_pposrt_report_value": [4],
        "CPHD-anoka_county_hlthreg_postest_report_value": [5],
        "CPHD-anoka_county_hlthreg_test_report_value": [6],
        "CPHD-anoka_county_hlthreg_pctvaccinedose1_report_value": [7],
        "CPHD-anoka_county_hlthreg_pctvaccinedose2_report_value": [8],
        "OverlappingPoly-dakota_county_hlthreg-Polygon_pop": [100],
        "CPHD-dakota_county_hlthreg_conf_report_value": [1],
        "CPHD-dakota_county_hlthreg_death_report_value": [2],
        "CPHD-dakota_county_hlthreg_hospcen_report_value": [3],
        "CPHD-dakota_county_hlthreg_pposrt_report_value": [4],
        "CPHD-dakota_county_hlthreg_postest_report_value": [5],
        "CPHD-dakota_county_hlthreg_test_report_value": [6],
        "CPHD-dakota_county_hlthreg_pctvaccinedose1_report_value": [7],
        "CPHD-dakota_county_hlthreg_pctvaccinedose2_report_value": [8],
        "OverlappingPoly-ramsey_county_hlthreg-Polygon_pop": [100],
        "CPHD-ramsey_county_hlthreg_conf_report_value": [1],
        "CPHD-ramsey_county_hlthreg_death_report_value": [2],
        "CPHD-ramsey_county_hlthreg_hospcen_report_value": [3],
        "CPHD-ramsey_county_hlthreg_pposrt_report_value": [4],
        "CPHD-ramsey_county_hlthreg_postest_report_value": [5],
        "CPHD-ramsey_county_hlthreg_test_report_value": [6],
        "CPHD-ramsey_county_hlthreg_pctvaccinedose1_report_value": [7],
        "CPHD-ramsey_county_hlthreg_pctvaccinedose2_report_value": [8],
        "OverlappingPoly-washington_county_hlthreg-Polygon_pop": [100],
        "CPHD-washington_county_hlthreg_conf_report_value": [1],
        "CPHD-washington_county_hlthreg_death_report_value": [2],
        "CPHD-washington_county_hlthreg_hospcen_report_value": [3],
        "CPHD-washington_county_hlthreg_pposrt_report_value": [4],
        "CPHD-washington_county_hlthreg_postest_report_value": [5],
        "CPHD-washington_county_hlthreg_test_report_value": [6],
        "CPHD-washington_county_hlthreg_pctvaccinedose1_report_value": [7],
        "CPHD-washington_county_hlthreg_pctvaccinedose2_report_value": [8],
        "CPHD-metro_plant_sw_conf_report_value": [1],
        "CPHD-metro_plant_sw_death_report_value": [2],
        "CPHD-metro_plant_sw_hospcen_report_value": [3],
        "CPHD-metro_plant_sw_pposrt_report_value": [4],
        "CPHD-metro_plant_sw_postest_report_value": [5],
        "CPHD-metro_plant_sw_test_report_value": [6],
        "CPHD-metro_plant_sw_pctvaccinedose1_report_value": [7],
        "CPHD-metro_plant_sw_pctvaccinedose2_report_value": [8],
    }

    df = pd.DataFrame(data)

    expected = pd.DataFrame.from_dict(
        data={
            "OverlappingPoly-hennepin_county_hlthreg-Polygon_pop": [100],
            "CPHD-hennepin_county_hlthreg_conf_report_value": [1000.0],
            "CPHD-hennepin_county_hlthreg_death_report_value": [2000.0],
            "CPHD-hennepin_county_hlthreg_hospcen_report_value": [3000.0],
            "CPHD-hennepin_county_hlthreg_pposrt_report_value": [4.0],
            "CPHD-hennepin_county_hlthreg_postest_report_value": [5000.0],
            "CPHD-hennepin_county_hlthreg_test_report_value": [6.0],
            "CPHD-hennepin_county_hlthreg_pctvaccinedose1_report_value": [7.0],
            "CPHD-hennepin_county_hlthreg_pctvaccinedose2_report_value": [8.0],
            "OverlappingPoly-anoka_county_hlthreg-Polygon_pop": [100],
            "CPHD-anoka_county_hlthreg_conf_report_value": [1000.0],
            "CPHD-anoka_county_hlthreg_death_report_value": [2000.0],
            "CPHD-anoka_county_hlthreg_hospcen_report_value": [3000.0],
            "CPHD-anoka_county_hlthreg_pposrt_report_value": [4.0],
            "CPHD-anoka_county_hlthreg_postest_report_value": [5000.0],
            "CPHD-anoka_county_hlthreg_test_report_value": [6.0],
            "CPHD-anoka_county_hlthreg_pctvaccinedose1_report_value": [7.0],
            "CPHD-anoka_county_hlthreg_pctvaccinedose2_report_value": [8.0],
            "OverlappingPoly-dakota_county_hlthreg-Polygon_pop": [100],
            "CPHD-dakota_county_hlthreg_conf_report_value": [1000.0],
            "CPHD-dakota_county_hlthreg_death_report_value": [2000.0],
            "CPHD-dakota_county_hlthreg_hospcen_report_value": [3000.0],
            "CPHD-dakota_county_hlthreg_pposrt_report_value": [4.0],
            "CPHD-dakota_county_hlthreg_postest_report_value": [5000.0],
            "CPHD-dakota_county_hlthreg_test_report_value": [6.0],
            "CPHD-dakota_county_hlthreg_pctvaccinedose1_report_value": [7.0],
            "CPHD-dakota_county_hlthreg_pctvaccinedose2_report_value": [8.0],
            "OverlappingPoly-ramsey_county_hlthreg-Polygon_pop": [100],
            "CPHD-ramsey_county_hlthreg_conf_report_value": [1000.0],
            "CPHD-ramsey_county_hlthreg_death_report_value": [2000.0],
            "CPHD-ramsey_county_hlthreg_hospcen_report_value": [3000.0],
            "CPHD-ramsey_county_hlthreg_pposrt_report_value": [4.0],
            "CPHD-ramsey_county_hlthreg_postest_report_value": [5000.0],
            "CPHD-ramsey_county_hlthreg_test_report_value": [6.0],
            "CPHD-ramsey_county_hlthreg_pctvaccinedose1_report_value": [7.0],
            "CPHD-ramsey_county_hlthreg_pctvaccinedose2_report_value": [8.0],
            "OverlappingPoly-washington_county_hlthreg-Polygon_pop": [100],
            "CPHD-washington_county_hlthreg_conf_report_value": [1000.0],
            "CPHD-washington_county_hlthreg_death_report_value": [2000.0],
            "CPHD-washington_county_hlthreg_hospcen_report_value": [3000.0],
            "CPHD-washington_county_hlthreg_pposrt_report_value": [4.0],
            "CPHD-washington_county_hlthreg_postest_report_value": [5000.0],
            "CPHD-washington_county_hlthreg_test_report_value": [6.0],
            "CPHD-washington_county_hlthreg_pctvaccinedose1_report_value": [7.0],
            "CPHD-washington_county_hlthreg_pctvaccinedose2_report_value": [8.0],
            "SewershedPoly-Polygon_pop": [100],
            "CPHD-metro_plant_sw_conf_report_value": [1000.0],
            "CPHD-metro_plant_sw_death_report_value": [2000.0],
            "CPHD-metro_plant_sw_hospcen_report_value": [3000.0],
            "CPHD-metro_plant_sw_pposrt_report_value": [4.0],
            "CPHD-metro_plant_sw_postest_report_value": [5000.0],
            "CPHD-metro_plant_sw_test_report_value": [6.0],
            "CPHD-metro_plant_sw_pctvaccinedose1_report_value": [7.0],
            "CPHD-metro_plant_sw_pctvaccinedose2_report_value": [8.0],
            "CASES": [1000.0],
            "DEATHS": [2000.0],
            "HOSPITALIZED": [3000.0],
            "POS_TESTS_RATE": [4.0],
            "POS_TESTS": [5000.0],
            "TESTS": [6.0],
            "PCT_VAX1": [7.0],
            "PCT_VAX2": [8.0],
        }
    )
    expected = expected.sort_index(axis=1)
    df = compute_stpaul_health(df)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 62)

    pd.testing.assert_frame_equal(df, expected)


def test_get_variable_rank():
    # Test get_variable_rank with a small dataset
    variables = [Variable("HOSPITALIZED"), Variable("CASES")]
    ts_types = [TimeSeriesType("RAW"), TimeSeriesType("SMOOTH")]
    result = get_variable_rank(variables, ts_types)
    assert isinstance(result, dict)
    assert len(result) == 4
    assert result[(Variable("CASES"), TimeSeriesType("RAW"))] == 0
    assert result[(Variable("CASES"), TimeSeriesType("SMOOTH"))] == 1
    assert result[(Variable("HOSPITALIZED"), TimeSeriesType("RAW"))] == 2
    assert result[(Variable("HOSPITALIZED"), TimeSeriesType("SMOOTH"))] == 3

    # Test get_variable_rank with a larger dataset
    variables = [Variable("CASES"), Variable("HOSPITALIZED"), Variable("COD")]
    ts_types = [
        TimeSeriesType("SMOOTH"),
        TimeSeriesType("RAW"),
        TimeSeriesType("TREND_SLOPE"),
    ]
    result = get_variable_rank(variables, ts_types)
    assert isinstance(result, dict)
    assert len(result) == 9
    assert result[(Variable("CASES"), TimeSeriesType("RAW"))] == 0
    assert result[(Variable("CASES"), TimeSeriesType("SMOOTH"))] == 1
    assert result[(Variable("CASES"), TimeSeriesType("TREND_SLOPE"))] == 2
    assert result[(Variable("COD"), TimeSeriesType("RAW"))] == 3
    assert result[(Variable("COD"), TimeSeriesType("SMOOTH"))] == 4
    assert result[(Variable("COD"), TimeSeriesType("TREND_SLOPE"))] == 5
    assert result[(Variable("HOSPITALIZED"), TimeSeriesType("RAW"))] == 6
    assert result[(Variable("HOSPITALIZED"), TimeSeriesType("SMOOTH"))] == 7
    assert result[(Variable("HOSPITALIZED"), TimeSeriesType("TREND_SLOPE"))] == 8


def test_to_tensor(parquet_path_short, sample_tensor_config_regression):  # noqa: C901
    # Test to_tensor with default configuration
    X, y, info = to_tensor(
        parquet_path_short,
        City.EASYVILLE_1,
        "2020-01-05",
        "2020-01-08",
        sample_tensor_config_regression,
    )
    assert X.shape == (4, 4, 5)
    assert y.shape == (4, 4, 4)
    assert len(info) == 4
    assert info[0]["first_input_date"] == "2020-01-01"
    assert info[0]["last_input_date"] == "2020-01-05"
    assert info[0]["first_target_date"] == "2020-01-06"
    assert info[0]["last_target_date"] == "2020-01-09"
    assert info[0]["city"] == "EASYVILLE_1"
    assert info[0]["augmented"] is False

    assert info[0]["x_column_lookup"] == {
        "CASES_RAW": 0,
        "CASES_SMOOTH": 1,
        "HOSPITALIZED_RAW": 2,
        "HOSPITALIZED_SMOOTH": 3,
    }
    assert info[0]["y_column_lookup"] == {
        "BOD_TREND_CURVATURE": 0,
        "BOD_TREND_SLOPE": 1,
        "COD_TREND_CURVATURE": 2,
        "COD_TREND_SLOPE": 3,
    }

    assert info[-1]["first_input_date"] == "2020-01-04"
    assert info[-1]["last_input_date"] == "2020-01-08"
    assert info[-1]["first_target_date"] == "2020-01-09"
    assert info[-1]["last_target_date"] == "2020-01-12"
    assert info[-1]["city"] == "EASYVILLE_1"
    assert info[-1]["augmented"] is False

    assert info[0]["x_column_lookup"] == {
        "CASES_RAW": 0,
        "CASES_SMOOTH": 1,
        "HOSPITALIZED_RAW": 2,
        "HOSPITALIZED_SMOOTH": 3,
    }
    assert info[0]["y_column_lookup"] == {
        "BOD_TREND_CURVATURE": 0,
        "BOD_TREND_SLOPE": 1,
        "COD_TREND_CURVATURE": 2,
        "COD_TREND_SLOPE": 3,
    }
    # assert that tensors are equal

    assert torch.allclose(
        X[0, info[0]["x_column_lookup"]["CASES_RAW"], :],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
    ), "Tensors are not close"

    assert torch.allclose(
        X[1, info[0]["x_column_lookup"]["CASES_RAW"], :],
        torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]),
    ), "Tensors are not close"

    assert torch.allclose(
        y[0, info[0]["y_column_lookup"]["BOD_TREND_CURVATURE"], :],
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
    ), "Tensors are not close"
    assert torch.allclose(
        y[0, info[0]["y_column_lookup"]["BOD_TREND_SLOPE"], :],
        torch.tensor([2.0, 2.0, 2.0, 2.0]),
    ), "Tensors are not close"
