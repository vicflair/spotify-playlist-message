import spotipy
import sys
import time
from collections import namedtuple
from difflib import SequenceMatcher
from functools import partial, wraps
from multiprocessing.dummy import Pool

sp = spotipy.Spotify()


# A NamedTuple object to represent songs/tracks in Spotify. 'title' is the
# normalized form of the 'Title'.
Track = namedtuple('Track', ['Title', 'Artists', 'Album', 'Popularity',
                             'title'])


def timeit(func):
    """ Decorator function used to time execution. For testing purposes only.
    """

    @wraps(func)
    def timed_function(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print '%s execution time: %f secs' % (func.__name__, end - start)
        return output

    return timed_function


def memorize(func):
    """ Decorator function for caching outputs of a function to speed up
     retrieval time. If an argument has been previously used, the corresponding
     output will be looked up instead of calculated.
    """
    cache = {}

    @wraps(func)
    def cached_function(*args, **kwargs):
        if args not in cache:
            cache[args] = func(*args, **kwargs)
        return cache[args]

    return cached_function


def normalize(text):
    """ Normalize text (e.g. song titles or input message) so we can
    compare apples to apples.
    """
    # Remove endings of song titles with extra info, e.g., (radio edit)
    text, foo, bar = text.partition(' - ')
    text, foo, bar = text.partition(' (')
    # Set to lower case and strip extra spaces on the right
    norm = text.lower().rstrip().decode('utf-8')
    # Optionally stem using NTLK
    return norm


@memorize
def search_tracks(phrase):
    """ Submit a query to Spotify for a given <phrase>. NOTE: the maximum
    number of results is apparently limited to 50.
    """
    # Make query with Spotify metadata API
    results = sp.search(q=phrase, limit=50)
    # Extract track information: title, artist, album
    tracks = [t for t in results['tracks']['items']]
    songs = [get_song(track) for track in tracks]
    return songs


def get_song(track):
    """ Extract song information from a JSON track/song object from the
    Spotify metadata API and put it into a Track namedTuple for convenience.
    """
    # Extract some identifying track information
    Title = track['name'].encode('utf-8')
    title = normalize(Title)
    Artist = [a['name'].encode('utf-8') for a in track['artists']]
    Album = track['name'].encode('utf-8')
    Popularity = track['popularity']
    # Put information into a namedTuple for convenience
    song = Track(Title, Artist, Album, Popularity, title)
    return song


#@timeit # Querying takes the most time
def ngram_search(words, n=2):
    """ Create a set of potential songs to form the playlist by querying every
    n-gram, i.e. every consecutive group of n words, in the message. Thus,
    there are N-n+1 queries to Spotify, where N is the # of words in the
    message. This function serves several purposes:
    1) It limits the number of queries made since they take much more time than
    forming the optimal playlist.
    2) It allows the querying to occur up-front rather than on-demand (perhaps
    unpredictably) in the DP parsing function, which in turns allows querying
    to be performed in parallel, if desired.
    3) N-grams where n > 1 but sufficiently allow many songs to be queried,
    but using word context to increase the likelihood of finding good matches
    for the playlist. n = 2 or 3 seem to work well.
    """
    N = len(words)
    # Queries are the first 1 to N words
    queries = [' '.join(words[i:i + n]) for i in range(N - n + 1)]
    # Create multiple threads to parallelize querying
    p = Pool(4)
    results = p.map(search_tracks, queries)
    p.close()
    p.join()
    # Merge results and sort in popularity order (descending) to ensure that
    # more popular songs are selected for play lists
    merged = reduce(list.__add__, results)
    songs = sorted(merged, key=lambda x: x.Popularity, reverse=True)
    return songs


def score_match(phrase, song):
    """ Score matches between a phrase and a song title based purely on the
    difflib ratio score. Simple and fast. Could also create a score based on
    phrase words missing in the song  title, extra words in the song title,
    perfect word matches, etc.
    """
    return SequenceMatcher(None, phrase, song.title).ratio()
    ## Examples of other score metrics and modifiers:
    ## Penalize based on difference in phrase length (word count)
    #  return -abs(len(song.split()) - len(phrase.split()))
    ## Penalize based on missing words
    #  return -len([w for w in phrase.split() if w not in song.split()])


@memorize
def greedy_match(phrase, songs=None):
    words = phrase.split()
    # Greedy search
    best_score = 0
    num_words = 1
    for i in range(len(words)):
        phrase = ' '.join(words[0:i + 1])
        scores = [score_match(phrase, song) for song in songs]
        if max(scores) >= best_score:
            best_score = max(scores)
            num_words = i + 1

    # Best matching songs and phrase
    best_phrase = ' '.join(words[0:num_words])
    scores = [score_match(best_phrase, song) for song in songs]
    best_song = songs[scores.index(best_score)]

    return best_score, best_song, num_words


@timeit
def create_greedy_playlist(msg):
    print ''
    print '------'
    print '***Greedy method***'
    print 'Original message: ', msg
    words = normalize(msg).split()
    songs = ngram_search(words)
    playlist = []
    i = 0
    while i < len(words):
        _, best_song, num_words = greedy_match(' '.join(words[i:]),
                                               songs=songs)
        i += num_words
        playlist.append(best_song)
    print 'Playlist: '
    print '# | SONG TITLE | ARTIST | ALBUM'
    for i, p in enumerate(playlist):
        song_info = '{0} | {1} | {2}'.format(p.Title, ', '.join(p.Artists),
                                             p.Album)
        print '{0}. | '.format(i + 1) + song_info


@memorize
def dp_match(phrase, songs=None):
    """ For a given <phrase>, a string, find the song in the list <songs>
    which has the best match score, according to the score_match() function.
    Return both the best matching song and its score.
    """
    scores = [score_match(phrase, song) for song in songs]
    i = scores.index(max(scores))
    return scores[i], [phrase], [songs[i]]


@memorize
def dp_parse(phrase, songs=None):
    """ Optimally parse a sentence into phrases and matching songs (titles)
    using a dynamic programming approach. Test each possible parse by
    recursively checking the best match for a phrase of length i and the best
    parse for remaining N-i words. The optimal solution is acquired by giving
    'local' scores to phrase-song pairs, via dp_match(), and a 'global' score
    for the entire song parse for the whole message via a function of these
    local scores.
    """
    # If phrase is None, there are no parses: return empty set
    if not phrase:
        return 0, [], []
    # Tokenize into words based on spaces
    words = phrase.split()
    # If only one word, no song parsing needed: get best song match
    if len(words) == 1:
        return dp_match(phrase, songs=songs)
    # If multiple words, recursively test possible song parses
    else:
        # Initialize candidate paths, i.e. parses
        candidates = []
        N = len(words)
        for i in range(N, 0, -1):
            # Partition the phrase in two segments at position i
            words_a = ' '.join(words[0:i])
            words_b = ' '.join(words[i:])
            # Match the first part with a song, and get the results of parsing
            # the second part
            part_a = dp_match(words_a, songs=songs)
            part_b = dp_parse(words_b, songs=songs)
            # Compute the resulting total score, phrases parsings, and
            # matching songs for each phrase. Do this for each possible
            # partition.
            score_path = part_a[0] * i + part_b[0]
            word_path = part_a[1] + part_b[1]
            song_path = part_a[2] + part_b[2]
            path = (score_path, word_path, song_path)
            candidates.append(path)
        # Sort in order of highest scores, return the best path
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        return candidates[0]


@timeit  # Playlist creation from a message given a set of songs is quick
def create_dp_playlist(msg):
    """ Query some songs, then create a playlist using a dynamic programming
    approach. Print out the results.
    """
    print ''
    print '------'
    print '***Dynamic Programming method***'
    print 'Original message: ', msg
    # Normalize and tokenize message and use it to query songs
    words = normalize(msg).split(' ')
    songs = ngram_search(words)
    # Form playlist and print
    playlist = dp_parse(normalize(msg), songs=songs)
    print 'Playlist: '
    print '# | SONG TITLE | ARTIST | ALBUM'
    for i, p in enumerate(playlist[2]):
        song_info = '{0} | {1} | {2}'.format(p.Title, ', '.join(p.Artists),
                                             p.Album)
        print '{0}. | '.format(i + 1) + song_info


def test_msgs():
    """ A set of messages used to test this script.
    """
    msgs = [
        "Hi, how are you doing today? Do you want to get a beer? See you soon!",
        "If I can't let it go out of my mind?",
        "I would lie to you but never for you.",
        "Man I'm so bored. Let's do something fun. Call me.",
        "Hey babe, somebody told me you're not coming home tonight.",
        "What Katie did sometimes makes me wonder.",
        "All the time I lie awake questioning my decision.",
        "I am what I am. Love me.",
        "I love New York! Let's go there! ",
        "So are you coming now? Let's party until midnight!",
        "Happy Valentine's Day!",
        "One good thing about music. When it hits you, you feel no pain.",
        "Where words fail, music speaks.",
        "Life without you is a mistake. Please take me back.",
        "I'm glad I have a friend like you."
    ]
    return msgs


def main():
    """ Test the playlist creator for a set of pre-determined messages.
    """
    msgs = test_msgs()
    for msg in msgs:
        # Greedy method has about the same results, but usually much faster,
        # Compared the dynamic programming method.
        create_greedy_playlist(msg)
        #create_dp_playlist(msg)

if __name__ == '__main__':
    main()
