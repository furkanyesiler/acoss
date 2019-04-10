import json
import argparse
from SHS_scraper.musicbrainz_querying import load_json


def get_stats(input_filename):
    data = load_json(input_filename)

    fields = ['']

    work_counter = 0
    perf_counter = 0
    have_length = 0
    have_mbid = 0
    have_tags = 0

    for work_id, work in data.items():
        work_counter += 1
        for perf_id, performance in work.items():
            perf_counter += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gives you the statistics of presence of MusicBrainz data")
    parser.add_argument("-i", "--input", action="store", default="metadata/whatisacover_subset.json",
                        help="JSON file with metadata")

    args = parser.parse_args()

    get_stats(args.input)

