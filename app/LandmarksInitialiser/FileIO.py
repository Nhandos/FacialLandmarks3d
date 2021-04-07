import json


def ImportMetadata(metadatafile):

    if metadatafile.endswith(".json"):
        with open(metadatafile, "r") as fp:
            data = json.loads(fp)
    else:
        

    return data




