import json
import argparse


def load_json(filename):
    with open(filename) as fp:
        data = json.load(fp)
    return data


def get_stats(input_filename):
    data = load_json(input_filename)

    fields = ['']

    work_counter = 0
    perf_counter = 0
    have_length = 0
    have_mbids = 0
    have_many_mbids = 0
    have_tags = 0

    for work_id, work in data.items():
        work_counter += 1
        for perf_id, performance in work.items():
            perf_counter += 1

            if 'mb_performances' in performance:
                have_mbids += 1

                if len(performance['mb_performances']) > 1:
                    have_many_mbids += 1

                has_length = False
                has_tags = False
                for mbid, mb_performance in performance['mb_performances'].items():
                    has_length |= 'length' in mb_performance
                    has_tags |= 'tags' in mb_performance

                have_length += int(has_length)
                have_tags += int(has_tags)

    print('Total works: {}'.format(work_counter))
    print('Total performances: {}'.format(perf_counter))
    print('- have mbids: {}'.format(have_mbids))
    print('- have more than one mbid: {}'.format(have_many_mbids))
    print('- have length: {}'.format(have_length))
    print('- have tags: {}'.format(have_tags))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gives you the statistics of presence of MusicBrainz data")
    parser.add_argument("-i", "--input", action="store", default="metadata/whatisacover_subset.json",
                        help="JSON file with metadata")

    args = parser.parse_args()

    get_stats(args.input)

