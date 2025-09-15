from __future__ import annotations

import argparse
import datetime
import heapq
import os
import time
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd
from traffic.algorithms.prediction.flightplan import FlightPlanPredict
from traffic.core import Flight, FlightPlan, Traffic
from traffic.core.mixins import DataFrameMixin

extent = "LFBBBDX"
prefix_sector = "LFBB"


def basename(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0]


def default(defaultfname: str, fname: Optional[str]) -> str:
    return defaultfname if fname is None else fname


class Metadata(DataFrameMixin):
    def __getitem__(self, key: str) -> None | FlightPlan:
        df = self.data.query(f'flight_id == "{key}"')
        if df.shape[0] == 0:
            return None

        fps = df.iloc[0]['route']
        fpg = fps.split()
        return FlightPlan(" ".join([fpg[0], *fpg[2:]]) if fpg[1] == "DCT" else fps)



metadata = pd.read_parquet("/home/kim/kimthesis/scripts/download_data/A22_metadata")

metadata_simple = Metadata(
    metadata.groupby("flight_id", as_index=False)
    .last()
    .eval("icao24 = icao24.str.lower()")
)


def dist_lat_min(f1: Flight, f2: Flight) -> Any:
    try:
        if f1 & f2 is None:  # no overlap
            return None
        return cast(pd.DataFrame, f1.distance(f2))["lateral"].min()
    except TypeError:
        print(
            f"exception in dist_lat_min for flights {f1.flight_id} and {f2.flight_id}"
        )
        return None


def extract_flight_deviations(
    flight: Flight,
    flightplan: FlightPlan,
    context_traffic: Traffic,
    margin_fl: int = 50,
    margin_neighbours: int = 100,
    angle_precision: int = 2,
    min_distance: int = 200,
    forward_time: int = 20,
) -> None | pd.DataFrame:
    """
    Examines all deviations in flight and returns selected ones in a dataframe.

    :param flight: Flight of interest
    :param flightplan: Flight plan of flight
    :param context_traffic: Surrounding flights
    :param margin_fl: Margin in ft to check altitude stability
    :param angle_precision: Desired precision in alignment computation
    :param min_distance: Distance from which we consider a navpoint for alignment
    :param forward_time: Duration of trajectory prediction

    :return: None or DataFrame containing selected deviations
    """
    cast(str, flight.flight_id)
    list_dicts = []

    for hole in flight - flight.aligned_on_navpoint(
        flightplan,
        angle_precision=angle_precision,
        min_distance=min_distance,
    ):
        if (
            hole is not None
            and hole.duration > pd.Timedelta("30s")
            and hole.altitude_max - hole.altitude_min < margin_fl
            and hole.start > flight.start
            and hole.stop < flight.stop
        ):

            flight = flight.resample("1s")
            hole = hole.resample("1s")
            flmin = hole.altitude_min - margin_fl
            flmax = hole.altitude_max + margin_fl
            altmin_neighbours = hole.altitude_min - margin_neighbours
            altmax_neighbours = hole.altitude_max + margin_neighbours

            stop_neighbours = min(
                hole.start + pd.Timedelta(minutes=forward_time),
                flight.stop,
            )
            flight_interest = flight.between(hole.start, stop_neighbours)
            assert flight_interest is not None

            offlimits = flight_interest.query(
                f"altitude>{flmax} or altitude<{flmin}"
            )
            # if there is at least one off-limits portion, we cut
            if offlimits is not None:
                istop = offlimits.data.index[0]
                flight_interest.data = flight_interest.data.loc[:istop]
                stop_neighbours = flight_interest.stop

            # we select neighbours which altitude intersects flight
            # find intersecting portions
            neighbours = (
                cast(Traffic, context_traffic - flight)
                .between(
                    start=hole.start,
                    stop=stop_neighbours,
                    strict=False,
                )
                .iterate_lazy()
                .query(
                    f"{altmin_neighbours} <= altitude <= {altmax_neighbours}"
                )
                .feature_gt("duration", datetime.timedelta(seconds=2))
                .eval()
            )

            pred_possible = flight.before(hole.start) is not None

            if neighbours is None and not pred_possible:
                continue

            if pred_possible:
                # compute prediction
                predictor = FlightPlanPredict(fp=flightplan, 
                                              start=hole.start, 
                                              horizon_minutes=20)
                pred_fp = predictor.predict(flight).convert_dtypes(dtype_backend="pyarrow")


            if neighbours is not None:
                # distance to closest neighbor + flight_id + timestamp
                if pred_possible:
                    # find the ten closest neighbours in the margins limit
                    neighbour_ids = []
                    neighbour_dist = []
                    neighbour_time = []
                    for n in neighbours:
                        distance_df = pred_fp.distance(n)
                        min_index = distance_df["lateral"].idxmin()

                        neighbour_ids.append(n.flight_id)
                        neighbour_dist.append(
                            distance_df.loc[min_index, "lateral"]
                        )
                        neighbour_time.append(
                            distance_df.loc[min_index, "timestamp"]
                        )

                    distance_id_pairs = list(
                        zip(neighbour_dist, neighbour_ids, neighbour_time)
                    )
                    smallest_10 = heapq.nsmallest(
                        10, distance_id_pairs, key=lambda x: x[0]
                    )

                    # iterate on 10 closest neighbours
                    for n in smallest_10:
                        temp_dict = hole.summary(
                            ["flight_id", "start", "stop", "duration"]
                        )
                        temp_dict = {
                            **temp_dict,
                            **dict(
                                min_f_dist=None,
                                min_fp_dist=None,
                                min_fp_time=None,
                                min_f_time=None,
                                neighbour_id=None,
                                difference=None,
                            ),
                        }
                        temp_dict["min_fp_dist"] = n[0]
                        temp_dict["neighbour_id"] = n[1]
                        temp_dict["min_fp_time"] = n[2]
                        closest = neighbours[n[1]]
                        temp_dict["alt_pred_cpa"] = pred_fp.at(n[2]).altitude
                        temp_dict["alt_neighbour_cpa"] = closest.at(
                            n[2]
                        ).altitude

                        # calculation for real trajectory and neighbour
                        distance_real = flight_interest.distance(closest)
                        min_index = distance_real["lateral"].idxmin()

                        temp_dict["min_f_time"] = distance_real.loc[
                            min_index, "timestamp"
                        ]
                        temp_dict["min_f_dist"] = distance_real.loc[
                            min_index, "lateral"
                        ]

                        list_dicts.append(temp_dict)
    if len(list_dicts) == 0:
        return None
    deviations = pd.DataFrame(list_dicts)
    # we compute the difference between actual and predicted separation
    deviations["difference"] = (
        deviations["min_f_dist"] - deviations["min_fp_dist"]
    )
    # we clear the cases for which trajectories exist more than once
    # deviations = deviations[deviations.min_f_dist != 0.0]
    return deviations


def extract_traffic_deviations(
    flights: Traffic,
    metadata_file: Path,
    context_traffic: Traffic,
    margin_fl: int = 50,
    angle_precision: int = 2,
    min_distance: int = 200,
    forward_time: int = 20,
) -> pd.DataFrame:
    cumul_deviations = []
    metadata = pd.read_parquet(metadata_file)

    metadata_simple = Metadata(
        metadata.groupby("flight_id", as_index=False)
        .last()
        .eval("icao24 = icao24.str.lower()")
    )
    # metadata_simple = metadata_simple[metadata_simple["state"] != "CANCELLED"]

    for flight in flights:
        try:
            df = extract_flight_deviations(
                flight,
                cast(FlightPlan, metadata_simple[cast(str, flight.flight_id)]),
                context_traffic,
            )
            if df is not None:
                cumul_deviations.append(df)
        except AssertionError:
            print(f"AssertionError in main for flight{flight.flight_id}")
        except TypeError as e:
            print(f"TypeError in main for flight {flight.flight_id} : {e}")
        except AttributeError as e:
            print(f"AttributeError in main for flight {flight.flight_id} : {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
    if cumul_deviations:
        all_deviations = pd.concat(cumul_deviations, ignore_index=True)
    else:
        all_deviations = pd.DataFrame()
    return all_deviations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract deviations from dataframe",
    )
    parser.add_argument("-trajs")
    parser.add_argument("-out", default=None)
    args = parser.parse_args()
    trajs = Traffic.from_file(args.trajs)
    trajs.data = trajs.data.dropna()

    time.time()
    deviations = extract_traffic_deviations(
        cast(Traffic, trajs),
        Path("../download_data/A22_metadata"),
        cast(Traffic, trajs),
    )

    out = args.out
    deviations.to_parquet(out)


if __name__ == "__main__":
    main()
