from bs4 import BeautifulSoup
import urllib
import youtube_dl
import json
import os
import argparse


def download_track(yt_link, work_title, perf_artist):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'songs/' + work_title + ' - ' + perf_artist + '.%(ext)s',
        'quiet': True
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_link])


def work_page_parser(url, all_songs):
    work_page = urllib.request.urlopen('https://secondhandsongs.com' + url + '/versions')

    work_soup = BeautifulSoup(work_page, 'html.parser')

    header = work_soup.find_all('header', {'class': 'row'})[0]
    header_strong = header.find_all('strong', {'itemprop': 'name'})
    work_title = header_strong[0].get_text()

    div = work_soup.find_all('a', {'class': 'link-artist'})[0]
    div_span = div.find_all('span', {'itemprop': 'name'})
    work_artist = div_span[0].get_text()

    url_id = url.split('/')[2]
    work_id = 'W_' + url_id
    print(work_title)

    all_songs[work_id] = dict()

    versions_table = work_soup.find_all('table', {'id': 'versions' + url_id})

    instrumentals_table = work_soup.find_all('table', {'id': 'instrumentals' + url_id})

    return versions_table, instrumentals_table, all_songs, work_title, work_artist, work_id


def perf_page_parser(table, all_songs, work_title, work_artist, work_id, inst_flag, download_songs):
    track_list = table.find_all('tr', {'itemprop': 'recordedAs'})

    for row_item in track_list:
        field_icon = row_item.find('td', {'class': 'field-icon'})
        if field_icon.find('a') is not None:
            release_year = row_item.find('td', {'class': 'field-date'}).get_text().split(' ')[-1]
            perf_link = field_icon.find('a')['href']
            perf_page = urllib.request.urlopen('https://secondhandsongs.com' + perf_link)

            perf_soup = BeautifulSoup(perf_page, 'html.parser')

            perf_id = 'P_' + perf_link.split('/')[-1]

            header = perf_soup.find_all('header', {'class': 'row'})[0]
            span_tags = header.find_all('span', {'itemprop': 'name'})
            try:
                perf_title = span_tags[0].get_text()
            except:
                perf_title = 'None'
            try:
                perf_artist = span_tags[1].get_text()
            except:
                perf_artist = 'None'

            yt_link_shs = perf_soup.find_all("iframe")[0]['src']
            if yt_link_shs.split('/')[2] == 'www.youtube.com':

                yt_link = 'http:' + yt_link_shs.split('?', 1)[0]

                all_songs[work_id][perf_id] = {}
                all_songs[work_id][perf_id]['work_title'] = work_title
                all_songs[work_id][perf_id]['work_artist'] = work_artist
                all_songs[work_id][perf_id]['perf_title'] = perf_title
                all_songs[work_id][perf_id]['perf_artist'] = perf_artist
                all_songs[work_id][perf_id]['release_year'] = release_year
                all_songs[work_id][perf_id]['work_id'] = work_id
                all_songs[work_id][perf_id]['perf_id'] = perf_id
                all_songs[work_id][perf_id]['yt_link'] = yt_link
                all_songs[work_id][perf_id]['instrumental'] = inst_flag

                if download_songs == 1:
                    try:
                        download_track(yt_link, work_title, perf_artist)

                        all_songs[work_id][perf_id]['filename'] = work_title + ' - ' + perf_artist + '.mp3'
                        all_songs[work_id][perf_id]['downloaded'] = 'Yes'

                    except:
                        all_songs[work_id][perf_id]['filename'] = 'None'
                        all_songs[work_id][perf_id]['downloaded'] = 'No'

    return all_songs


def main(start_idx, end_idx, download_songs):
    if not os.path.exists('songs/'):
        os.mkdir('songs/')
    if not os.path.exists('metadata/'):
        os.mkdir('metadata/')

    all_songs = dict()

    for page_idx in range(start_idx, end_idx):
        database_page = urllib.request.urlopen(
            'https://secondhandsongs.com/statistics?adaptationExcluded=1&christmasExcluded=1&page=' + str(page_idx)
            + '&sort=covers&list=stats_work_covered')
        database_soup = BeautifulSoup(database_page, 'html.parser')

        print('PAGE ---------------------' + str(page_idx))
        works_in_page = database_soup.find_all('a', {'class': 'link-work'})

        for item in works_in_page:
            work_url_short = item.get('href')
            versions_table, instrumentals_table, all_songs, work_title, work_artist, work_id = work_page_parser(
                work_url_short, all_songs)

            if len(versions_table) != 0:
                all_songs = perf_page_parser(versions_table[0], all_songs,
                                             work_title, work_artist, work_id, 'No', download_songs)
            if len(instrumentals_table) != 0:
                all_songs = perf_page_parser(instrumentals_table[0], all_songs,
                                             work_title, work_artist, work_id, 'Yes', download_songs)

    with open('metadata/metadata_shs_'+str(start_idx)+'-'+str(end_idx)+'.json', 'w') as fp:
        json.dump(all_songs, fp, indent='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script scrapes Secondhandsongs.com website to obtain metadata '
                                                 'and download available songs from youtube')
    parser.add_argument('-s',
                        '--start_idx',
                        type=int,
                        default='0',
                        help='Index of the first page for scraping. Default is 0.')
    parser.add_argument('-e',
                        '--end_idx',
                        type=int,
                        default='750',
                        help='Index of the last page for scraping. Default is 750.')

    parser.add_argument('-d',
                        '--download_songs',
                        type=int,
                        default=0,
                        help='Whether to download songs or not. 1 for yes, 0 for no.')

    args = parser.parse_args()
    main(start_idx=args.start_idx,
         end_idx=args.end_idx,
         download_songs=args.download_songs)
