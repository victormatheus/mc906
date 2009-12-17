#!/usr/bin/env python

import os
import gc
import numpy
from util import *
from re import split
from sys import stdout
from parse import Parser
from operator import itemgetter
from kmeans import Kmeans, choose_initial, choose_initial_pp
from guppy import hpy
import util

# Path to the directory containing the messages
path = './cluster-txt/messages/'
fname_stemmed = 'stemparsed.txt'
fname = 'parsed.txt'
fstopname = 'english'

def print_result(freqs, errors):
    for i, (f, e) in enumerate(zip(freqs, errors)):
        print 'Cluster #%d' % i
        print ', '.join(f.keys())
        print 'Error: %d\n' % e

def get_docs_frequencies(clusters):
    docs = []
    for c in clusters:
        freq = {}
        for doc in c:
            name = os.path.basename(split(r'-[0-9]+.txt$', doc.filename)[0])
            freq[name] = freq.get(name, 0) + 1
        docs.append(freq)
        gc.collect()
    return docs

def calc_error(freqs):
    errors = []
    for freq in freqs:
        f, name = max((f, name) for name, f in freq.iteritems())
        err = sum(freq[f] for f in filter(lambda n: n != name, freq.keys()))
        errors.append(err)
    return errors

def slice_sorted_words(dictio, delpercent):
    """
    Function to return a sorted version of a dictionary sorted
    by values and sliced in the beginning and end by delpercent
    """

    n = int((len(dictio) * (delpercent / 100.0)) / 2)
    return dict(sorted(dictio.iteritems(),
                key=itemgetter(1))[n:len(dictio) - n])


if __name__ == "__main__":
    filelist = [(path + f) for f in os.listdir(path)]

    parser = Parser(fstopname)
    for stem in True, False:
        for idf in True, False:
            print 'Parsing files...',
            stdout.flush()
            parser.parse(filelist, stem)
            # Ignore the 10% least and most frequent words
            parser.words = slice_sorted_words(parser.words, 10)
            print 'done'

            print 'Normalizing frequencies...',
            stdout.flush()
            # Don't modify the original set
            for i, doc in enumerate(parser.docset):
                parser.docset[i].char_vector = normalize(doc, parser.words, idf)
            gc.collect()
            print 'done'

            h=hpy()
            print h.heap()
            
            print 'Dimension: ', len(parser.docset[0].char_vector)
            print 'Vector*1000: ', parser.docset[0].char_vector*1000
            print 'Norm: ', util.norm(parser.docset[0].char_vector)
            
            for chooser in choose_initial, choose_initial_pp:
                for k in [2,]:
                    errors = []
                    print '\nStemming words: %s' % stem
                    print 'Using IDF: %s' % idf
                    print 'Running with %d centroids' % k
                    if chooser is choose_initial:
                        print 'Chooser: normal'
                    else:
                        print 'Chooser: plusplus'
                    stdout.flush()
                    for _ in xrange(2):
                        kmeans = Kmeans(parser.docset, k, distance,
                                        calc_centroid, chooser)
                        clusters = get_clusters(kmeans.result(), parser.docset)
                        freqs = get_docs_frequencies(clusters)
                        errors.append(sum(calc_error(freqs)))
                    print 'Error mean: %d and median: %d' % \
                        (numpy.mean(errors), numpy.median(errors))
            gc.collect()
