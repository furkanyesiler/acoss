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


def augment_metadata(input_filename, subset_filename, output_filename, threshold, api_rate):
    data = load_json(input_filename)
    subset = load_json(subset_filename) if subset_filename else None

    if subset:
        # TODO: iterate through subset instead of data
        pass

    total_works = len(data)
    work_counter = 0
    stats = {
        'no_match': 0,
        'single_recording': 0,
        'multiple_recordings': 0,
        'multiple_artists': 0
    }

    init_mb_client(api_rate)

    for work_id, work in data.items():
        work_counter += 1
        for perf_id, performance in work.items():
            print('- {}: {} ({}/{})'.format(performance['perf_artist'], performance['perf_title'], work_counter,
                                            total_works))

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
                        if artist['id'] in artist_recording_dict:
                            artist_recording_dict[artist['id']].add(recording['id'])
                        else:
                            artist_recording_dict[artist['id']] = {recording['id']}
                    except TypeError:
                        print('Weird credited artist: {}'.format(credited_artist))

            # TODO: add possible work_mbid

            artists = artist_only_set & artist_recording_dict.keys()

            if len(artists) == 0:
                print('Didn\'t find match')
                stats['no_match'] += 1
            elif len(artists) > 1:
                print('Artist has multiple mbids')
                stats['multiple_artists'] += 1
                # TODO: properly handle multiple artists
            else:
                (artist_id,) = artists
                work[perf_id]['perf_artist_mbid'] = artist_id

                performance_ids = list(artist_recording_dict[artist_id])
                work[perf_id]['perf_mbids'] = list(artist_recording_dict[artist_id])
                if len(performance_ids) > 1:
                    stats['single_recording'] += 1
                else:
                    stats['multiple_recordings'] += 1

                print('artist_mbid={} perf_mbids={}'.format(artist_id, performance_ids))

        print('-'*60)

    write_json(data, output_filename)
    print(stats)


def check_tags(input_filename, output_filename, api_rate):
    data = load_json(input_filename)

    init_mb_client(api_rate)

    perf_counter = 0
    perf_with_tags = 0

    for work_id, work in data.items():
        for perf_id, performance in work.items():
            perf_counter += 1
            if 'perf_mbids' in performance:
                tags_all = {}
                for recording_id in performance['perf_mbids']:
                    result = mb.get_recording_by_id(recording_id, includes=['tags'])
                    if 'tag-list' in result['recording']:
                        print(result['recording'])
                        tags = result['recording']['tag-list']
                        for tag in tags:
                            if tag['name'] in tags_all:
                                tags_all[tag['name']] += int(tag['count'])
                            else:
                                tags_all[tag['name']] = int(tag['count'])
                if len(tags_all) > 0:
                    performance['perf_mb_tags'] = tags_all
                    print(perf_counter, tags_all)
                    perf_with_tags += 1
        if perf_counter >= 10:
            break

    write_json(data, output_filename)
    print('Found tags for {} performances'.format(perf_with_tags))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Queries musicbrainz to enhance the metadata of cover song list")
    parser.add_argument("-i", "--input", action="store", default="metadata/whatisacover_subset.json",
                        help="Path to the input file with metadata. If subset file is not specified, whole input file "
                             "will be processed.")
    parser.add_argument("-o", "--output", action="store", default="metadata/metadata_augmented.json",
                        help="Path to the output file")
    parser.add_argument("-s", "--subset", action="store", default=None,
                        help="Path to json file with subset list")
    parser.add_argument("-t", "--threshold", action="store", default=90,
                        help="Score threshold to filter the results from MusicBrainz API (0~100)")
    parser.add_argument("-r", "--rate", action="store", default=0.1,
                        help="Rate limit for MusicBrainz API (one response per specified period in seconds)")

    args = parser.parse_args()

    # augment_metadata(args.input, args.subset, args.output, args.threshold, args.rate)
    check_tags(args.output, "metadata/metadata_augmented_with_tags.json", args.rate)
