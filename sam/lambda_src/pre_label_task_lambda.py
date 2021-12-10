# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import json

import boto3

s3 = boto3.resource("s3")

def uri_to_s3_obj(s3_uri):
    if not s3_uri.startswith("s3://"):
        # This is a local path, indicate using None
        return None
    bucket, key = s3_uri.split("s3://")[1].split("/", 1)
    return s3.Object(bucket, key)


def fetch_s3(s3_uri):
    print(f"FETCH {s3_uri}")
    obj = uri_to_s3_obj(s3_uri)
    body = obj.get()["Body"]
    return body.read()


def lambda_handler(event, context):
    print("Received event: ", event)
    source = json.loads(fetch_s3(event["dataObject"]["source"]))
    return {
        "taskInput": {
            "source": source,
        }
    }
