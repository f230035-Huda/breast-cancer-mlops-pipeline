import argparse
import json
import urllib.request

import pandas as pd


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode())


def post_prediction(url, features):
    data = json.dumps(features).encode()

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
    )
    parser.add_argument("--requests", type=int, default=3)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    df = pd.read_csv("data/raw/breast_cancer.csv")
    features = df.drop(columns=["target"]).iloc[0].to_dict()

    before = get_json(f"{base_url}/stats")
    print("Initial stats:", before)

    for number in range(1, args.requests + 1):
        result = post_prediction(
            f"{base_url}/predict",
            features,
        )
        print(f"Prediction {number}:", result)

    after = get_json(f"{base_url}/stats")
    print("Final stats:", after)

    expected = before["prediction_count"] + args.requests

    if after["prediction_count"] != expected:
        raise AssertionError(
            f"Expected count {expected}, got {after['prediction_count']}"
        )

    print("\nPASS: All predictions succeeded and counter increased correctly.")


if __name__ == "__main__":
    main()
