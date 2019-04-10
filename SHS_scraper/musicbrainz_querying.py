import json
import argparse
import musicbrainzngs as mb


# TODO make helpers project-wide
def load_json(filename):
    with open(filename) as fp:
        data = json.load(fp)
    return data


def write_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp, indent='\t')


def process_results(results, threshold):
    good_results = []
    for result in results:
        if int(result['ext:score']) >= threshold:
            good_results.append(result)
        else:
            break
    return good_results


def get_artists(artist_query, threshold):
    results = mb.search_artists(artist_query)
    return process_results(results['artist-list'], threshold)


def get_recordings(recording_query, artist_query, threshold):
    results = mb.search_recordings(recording_query, artist=artist_query)
    return process_results(results['recording-list'], threshold)


def init_mb_client(api_rate):
    # TODO create global config with app name and version
    mb.set_useragent('CoverSongDataset', 'v0.1')
    mb.set_rate_limit(api_rate)


def add_ids(performance, threshold):
    # check artist individually
    # TODO: handle multiple artists (with & or - in name)
    artists = get_artists(performance['perf_artist'], threshold)
    artist_only_set = set()

    for artist in artists:
        # print('artist_id: {}, artist_name: {}, score: {}'.format(artist['id'], artist['name'],
        #                                                          artist['ext:score']))
        artist_only_set.add(artist['id'])

    # check recording with artist together
    recordings = get_recordings(performance['perf_title'], performance['perf_artist'], threshold)
    artist_recording_dict = {}  # holds mappings from artist_id to recording_id

    for recording in recordings:
        # print('recording_id: {}'.format(recording['id']))
        for credited_artist in recording['artist-credit']:
            try:
                artist = credited_artist['artist']
                # print('recording artist_id: {}, artist_name: {}'
                #       .format(artist['id'], artist['name']))

                if artist['id'] not in artist_recording_dict:
                    artist_recording_dict[artist['id']] = {}
                artist_recording_dict[artist['id']][recording['id']] = {}

                if 'length' in recording:
                    artist_recording_dict[artist['id']][recording['id']]['length'] = recording['length']
            except TypeError:
                print('Weird credited artist: {}'.format(credited_artist))

    # TODO: add possible work_mbid

    artists = artist_only_set & artist_recording_dict.keys()

    if len(artists) == 0:
        print('Didn\'t find match')
        return False, 'no_match'

    if len(artists) > 1:
        # TODO: properly handle multiple artists
        print('Artist has multiple mbids')
        return False, 'multiple_artists'

    (artist_id,) = artists
    performance['perf_artist_mbid'] = artist_id

    performances = artist_recording_dict[artist_id]
    performance['mb_performances'] = performances

    print('artist_mbid={} perf_mbids={}'.format(artist_id, performances.keys()))

    return True, 'single_recording' if len(performances) == 1 else 'multiple_recordings'


def add_tags(performance):
    tags_all = {}
    for recording_id, recording in performance['mb_performances'].items():
        result = mb.get_recording_by_id(recording_id, includes=['tags'])
        if 'tag-list' in result['recording']:
            print(result['recording'])
            tags = result['recording']['tag-list']
            recording['tags'] = {}
            for tag in tags:
                if tag['name'] in tags_all:
                    tags_all[tag['name']] += int(tag['count'])
                else:
                    tags_all[tag['name']] = int(tag['count'])
                recording['tags'][tag['name']] = int(tag['count'])

    if len(tags_all) > 0:
        performance['perf_mb_tags'] = tags_all
        return True, tags_all
    return False


def augment_metadata(input_filename, output_filename, threshold, api_rate):
    data = load_json(input_filename)

    total_works = len(data)
    work_counter = 0
    perf_counter = 0
    stats = {
        'no_match': 0,
        'single_recording': 0,
        'multiple_recordings': 0,
        'multiple_artists': 0,
        'has_tags': 0,
        'total': 0
    }

    init_mb_client(api_rate)

    for work_id, work in data.items():
        work_counter += 1
        for perf_id, performance in work.items():
            perf_counter += 1
            print('- {}: {} ({}/{})'.format(performance['perf_artist'], performance['perf_title'], work_counter,
                                            total_works))

            success, status = add_ids(performance, threshold)
            stats[status] += 1
            if success:
                success = add_tags(performance)
                if success:
                    stats['has_tags'] += 1

            stats['total'] += 1

    # TODO: add safeguard against overwriting a file
    write_json(data, output_filename)
    print(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Queries musicbrainz to enhance the metadata of cover song list")
    parser.add_argument("-i", "--input", action="store", default="metadata/whatisacover_subset.json",
                        help="Path to the input file with metadata. If subset file is not specified, whole input file "
                             "will be processed.")
    parser.add_argument("-o", "--output", action="store", default="metadata/metadata_augmented.json",
                        help="Path to the output file")
    parser.add_argument("-t", "--threshold", action="store", default=90,
                        help="Score threshold to filter the results from MusicBrainz API (0~100)")
    parser.add_argument("-r", "--rate", action="store", default=0.1,
                        help="Rate limit for MusicBrainz API (one response per specified period in seconds)")
    # TODO: add skip options

    args = parser.parse_args()

    augment_metadata(args.input, args.output, args.threshold, args.rate)
