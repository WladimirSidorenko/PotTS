#!/usr/bin/env python2.7
# -*- coding: utf-8-unix; mode: python; -*-

##################################################################
# Imports
from __future__ import print_function, unicode_literals

from merge_conll_mmax import WSPAN_PREFIX_RE
from measure_corpus_agreement import BINARY_OVERLAP, PROPORTIONAL_OVERLAP, \
    BASEDATA_SFX, MRKBL_NAME_RE, MARK_SFX_RE, _markables2tuples, \
    _compute_kappa

from collections import defaultdict, Counter
# from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import measure_corpus_agreement
import argparse
import codecs
import glob
import numpy as np
import os
import re
import sys
import xml.etree.ElementTree as ET

##################################################################
# Variables and Constants
# topic matching
OVERLAP_MTOK_IDX = 0
TOTAL_MTOK_IDX = 1
TOTAL_MARK_IDX = 2

USCORE_RE = re.compile('_+')
TOP2TOP = {"addition": "general",
           "politics_addition": "politics",
           "federal_election_addition": "federal_election",
           "pope_election_addition": "pope_election"}
REL_MARKABLES = ("sentiment", "emo-expression")
TOP2COL = {"federal_election": 0,
           "pope_election": 1,
           "politics": 2,
           "general": 3}
XTICK_MARKS = np.arange(len(TOP2COL))
XLABELS = [USCORE_RE.sub(' ', k) for k, v in
           sorted(TOP2COL.iteritems(), key=lambda el: el[1])]
CAT2ROW = {"emotional words": 0,
           "emoticons": 1,
           "random": 2}
YTICK_MARKS = np.arange(len(CAT2ROW))
YLABELS = [k for k, v in sorted(CAT2ROW.iteritems(), key=lambda el: el[1])]
MTX_DIM = (len(CAT2ROW), len(TOP2COL))
POLARITY = "polarity"
POS = "positive"
INTENSITY = "intensity"
INT2INT = {"weak": 0, "medium": 1, "strong": 2}

ENCODING = "utf-8"
# set of possible categories
CATEGORIES = set()
# mapping from tweet id to corpus category
TWEETID2CAT = {}
# mapping from token id to tweet id
TOKID2TWEETID = {}
# mapping from token id to corpus category
TOKID2CAT = {}
# suffixes of file names
BASEDATA_SFX_RE = re.compile(re.escape(BASEDATA_SFX))
SRC_SFX_RE = ".xml"
EOL = "EOL"
TOK = "tokens"
TOTAL = "total"
ANNOTATOR = "annotators"
STAT_MTX_SFX = "_stat.png"
AGR_MTX_SFX = "_agreement.png"
CMAP = plt.cm.PuBu

# Prepare containers for storing information about matching and
# mismatching annotations ([0, 0, 0]):
# the 0-th element in the list is the number of matching annotations,
# the 1-st element is the total number of tokens annotated with that
# markable,
# the 2-nd element is the total number of markables
STATISTICS = defaultdict(lambda: {TOK: 0, ANNOTATOR:
                                  [defaultdict(lambda: [0, 0, 0]),
                                   defaultdict(lambda: [0, 0, 0])]})

# TOTAL: number of matching sentiments/emotional expressions
# ANNOTATOR: [number of matching annotations,
# total number of sentiments annotated with the positive class]
POL_STAT = defaultdict(lambda: {TOTAL: 0, ANNOTATOR: ([0, 0], [0, 0])})
# statistics on intensities
INT_STAT = defaultdict(lambda: {i: {j: 0 for j in INT2INT.itervalues()}
                                for i in INT2INT.itervalues()})
INT_MIN = min(INT2INT.itervalues())
INT_MAX = max(INT2INT.itervalues())
INT_LST = sorted(INT2INT.itervalues())


##################################################################
# Methods
def _compute_alpha(a_istat):
    """Compute Krippendorff's alpha.

    Args:
    a_istat (n x n dict): coincidence matrix

    Returns:
    (float):
    Krippendorff's alpha

    """
    marginals = {i: float(sum(a_istat[i].itervalues())) for i in
                 a_istat}
    # compute delta for the ordinal scale
    imax = INT_MAX + 1
    avg_ij = m_i = m_j = m_k = 0.
    delta = defaultdict(lambda: defaultdict(float))
    for i in xrange(INT_MIN, imax):
        m_i = marginals[i]
        for j in xrange(i + 1, imax):
            m_j = marginals[j]
            avg_ij = (m_i + m_j) / 2.
            delta[i][j] = delta[j][i] = \
                sum(marginals[k] - avg_ij for k in xrange(i, j + 1))**2
    num = float(sum(a_istat[i][j] * delta[i][j] for i in a_istat
                for j in a_istat[i]))
    if not num:
        return 0.
    denom = float(sum(marginals[i] * marginals[j] *
                      delta[i][j] for i in marginals for j in marginals))
    _denom = sum(marginals.itervalues()) - 1.
    if not _denom:
        return 0.
    denom /= _denom
    alpha = 1. - num / denom
    return alpha


def plot_mtx(a_mtx, a_fname):
    """Plot matrix and store it in PNG format.

    Args:
    a_mtx (np.array): array with source data
    a_fname (str): path to the file for storing the model

    Returns:
    (void):

    """
    plt.imshow(a_mtx, interpolation='nearest', cmap=CMAP)
    plt.colorbar()
    plt.xlabel("Topics")
    plt.xticks(XTICK_MARKS, XLABELS, fontsize="small")
    plt.ylabel("Formal categories")
    plt.yticks(YTICK_MARKS, YLABELS, rotation=90, fontsize="small")
    plt.savefig(a_fname)
    plt.cla()
    plt.clf()


def read_src_corpus(a_src_corpus):
    """Read raw corpus, generating mappings from token ids to cat.

    Args:
    a_src_corpus (str): path to source corpus

    Returns:
    (void):
    updates global variable `TWEETID2CAT`

    """
    global CATEGORIES, TWEETID2CAT
    with codecs.open(a_src_corpus, 'r', ENCODING) as ifile:
        itopic = itype = cat = None
        corpus = ET.parse(ifile).getroot()
        for isubcorpus in corpus.iterfind("./subcorpus"):
            itopic = isubcorpus.get("name")
            itopic = TOP2TOP.get(itopic, itopic)
            for isubsubcorpus in isubcorpus.iterfind("./subsubcorpus"):
                itype = isubsubcorpus.get("type")
                cat = (itopic, itype)
                CATEGORIES.add(cat)
                for itweet in isubsubcorpus.iterfind("./tweet"):
                    TWEETID2CAT[itweet.get("id")] = cat


def _get_tweet_id(a_tweet_it):
    """Obtain id of a tweet from an iterator.

    Args:
    a_tweet_it (iterator): iterator over XML elements

    Returns:
    (str):
    id of the next tweet returned by iterator

    Raises:
    StopIteration

    """
    return a_tweet_it.next().get("id")


def read_basedata(a_basedata_dir, a_src_dir, a_ptrn="*.xml"):
    """Read files from source directory.

    Args:
    a_basedata_dir (str): path to directory with base data
    a_src_dir (str): path to directory with source files
    a_ptrn (str): iglobbing pattern for iteratin over files

    Returns:
    (void):
    updates global variable `TWEETID2CAT`

    """
    global STATISTICS, TOKID2CAT

    last_tweet_seen = False
    cat = tweet_it = None
    srctree = basetree = ""
    srcfname = basefname = tweet_id = tok_id = tok_txt = ""
    dir_it = glob.iglob(os.path.join(a_basedata_dir, a_ptrn))
    for fname in dir_it:
        basetree = ET.parse(fname).getroot()
        basefname = BASEDATA_SFX_RE.sub("", os.path.basename(fname))
        srcfname = os.path.join(a_src_dir, basefname + SRC_SFX_RE)
        srctree = ET.parse(srcfname).getroot()
        tweet_it = srctree.iterfind("./tweet")
        tweet_id = _get_tweet_id(tweet_it)
        cat = TWEETID2CAT[tweet_id]
        last_tweet_seen = False
        for itok in basetree.iterfind("./word"):
            tok_id = WSPAN_PREFIX_RE.sub("", itok.get("id"))
            tok_txt = itok.text
            STATISTICS[cat][TOK] += 1
            if tok_txt == EOL:
                TOKID2CAT[(basefname, tok_id)] = cat
                try:
                    tweet_id = _get_tweet_id(tweet_it)
                except StopIteration:
                    last_tweet_seen = True
                cat = TWEETID2CAT[tweet_id]
            else:
                if last_tweet_seen:
                    raise RuntimeError(
                        "Unequal number of tweets for file: {:s}".format(
                            fname))
                # print("basefname =", repr(basefname), file=sys.stderr)
                # print("tok_id =", repr(tok_id), file=sys.stderr)
                TOKID2CAT[(basefname, tok_id)] = cat


def _update_polarity_intensity(a_mtuple, a_mname, a_anno_id,
                               a_tokid2markid2, a_mtuples2,
                               a_update_cnt=False):
    """Update statistics for one annotator.

    Args:
    a_mtuple (tuple): tuples of the markable
    a_mname (str): name of the markable level
    a_anno_id (int): id of the annotator whose statistics should be updated
    a_tokid2markdid2 (dict): mapping from token ids to competing markable
                             indices
    a_mtuples2 (list(tuple)): competing markables
    a_update_cnt (bool): update counter of intersecting markables

    Returns:
    (void): updates global variable in place

    """
    global INT_STAT, POL_STAT

    # print("a_anno_id =", repr(a_anno_id), file=sys.stderr)
    # print("a_mtuple =", repr(a_mtuple), file=sys.stderr)
    candidates = Counter()
    for t_id in a_mtuple[0]:
        if t_id in a_tokid2markid2:
            for m_id in a_tokid2markid2[t_id]:
                candidates[m_id] += 1
    # print("candidates =", repr(candidates), file=sys.stderr)
    if not candidates:
        return
    # choose the best competing candidate
    best_ratio = ratio = -1.
    best_cnt = best_m_id = -1
    for m_id, t_cnt in candidates.iteritems():
        # print("candidate =", repr(a_mtuples2[m_id]), file=sys.stderr)
        if t_cnt > best_cnt:
            best_cnt = t_cnt
            best_m_id = m_id
            best_ratio = float(t_cnt)/float(len(a_mtuples2[m_id]))
        elif t_cnt == best_cnt:
            ratio = float(t_cnt)/float(len(a_mtuples2[m_id]))
            if ratio > best_ratio:
                best_cnt = t_cnt
                best_m_id = m_id
                best_ratio = ratio
    # get best candidate
    mtuple2 = a_mtuples2[best_m_id]
    # print("mtuple2 =", repr(mtuple2), file=sys.stderr)
    # update polarity
    if a_mtuple[-1][POLARITY] == POS:
        POL_STAT[a_mname][ANNOTATOR][a_anno_id][TOTAL_MTOK_IDX] += 1
        POL_STAT[a_mname][ANNOTATOR][a_anno_id][OVERLAP_MTOK_IDX] += \
            mtuple2[-1][POLARITY] == POS
    # update intensity only once
    if a_update_cnt:
        POL_STAT[a_mname][TOTAL] += 1
        int1 = INT2INT[a_mtuple[-1][INTENSITY]]
        int2 = INT2INT[mtuple2[-1][INTENSITY]]
        INT_STAT[a_mname][int1][int2] += 1
        INT_STAT[a_mname][int2][int1] += 1
    # print("POL_STAT[TOTAL] =", repr(POL_STAT[a_mname][TOTAL]))
    # print("POL_STAT[ANNOTATOR] =", repr(POL_STAT[a_mname][ANNOTATOR]))
    # print("INT_STAT =", repr(INT_STAT))


def _update_annotator_stat(a_mtuples, a_word_ids2, a_basefname, a_anno_id,
                           a_mname, a_cmp, a_compute_agr=True,
                           a_tokid2markdid2=None, a_mtuples2=None,
                           a_update_cnt=False):
    """Update statistics for one annotator.

    Args:
    a_mtuples (list(tuple)): tuples of markables
    a_word_ids2 (set): set of word ids from alternative annotation
    a_basefname (str): base file name (needed to construct the token id)
    a_anno_id (int): id of the annotator whose statistics should be updated
    a_mname (str): name of the markable level
    a_cmp (int): comparison scheme
    a_compute_agr (bool): boolean flag indicating whether to compute agreement
    a_tokid2markdid2 (dict): mapping from token ids to competing markable
                             indices
    a_mtuples2 (list(tuple)): competing markables
    a_update_cnt (bool): update counter of intersecting markables

    Returns:
    (void): updates global variable in place

    """
    global STATISTICS

    valid_toks = set()
    added_toks = set()
    active_cats = set()
    unseen_toks = set()
    tok_id = cat_stat = None
    for imtuple in a_mtuples:
        # check category of the tweet, to which the given markable pertains
        for itok_id in imtuple[0]:
            tok_id = (a_basefname, unicode(itok_id))
            if tok_id in TOKID2CAT:
                active_cats.add(TOKID2CAT[tok_id])
                valid_toks.add(itok_id)
        assert len(active_cats) == 1, \
            "One tweet is pertaining to more than one category" \
            " {:s}".format(repr(tok_id))
        imtuple[0] = [w for w in imtuple[0] if w in valid_toks]
        cat_stat = \
            STATISTICS[list(active_cats)[0]][ANNOTATOR][a_anno_id][a_mname]
        cat_stat[TOTAL_MARK_IDX] += 1
        if a_compute_agr:
            if a_cmp & PROPORTIONAL_OVERLAP:
                unseen_toks = set([w for w in imtuple[0]
                                   if w not in added_toks])
                cat_stat[OVERLAP_MTOK_IDX] += len(unseen_toks & a_word_ids2)
                added_toks.update(unseen_toks)
            else:
                # we do not consider exact match here
                unseen_toks = set(imtuple[0])
                if unseen_toks & a_word_ids2:
                    cat_stat[OVERLAP_MTOK_IDX] += len(unseen_toks)
            cat_stat[TOTAL_MTOK_IDX] += len(unseen_toks)
            if a_mname in REL_MARKABLES:
                _update_polarity_intensity(imtuple, a_mname, a_anno_id,
                                           a_tokid2markdid2, a_mtuples2,
                                           a_update_cnt)
        active_cats.clear()
        valid_toks.clear()


def _generate_tok2mark(a_tuples):
    """Generate mappings from tone ids to markable indices.

    Args:
    a_tuples (list(tuple)): markable tuples

    Returns:
    (dict):
    mappings from token ids to markable indices

    """
    tokid2markid = defaultdict(list)
    for i, tpl in enumerate(a_tuples):
        for tok_id in tpl[0]:
            tokid2markid[tok_id].append(i)
    return tokid2markid


def _update_stat(a_t1, a_t2, a_basefname, a_mname, a_cmp=BINARY_OVERLAP):
    """Compare annotations present in two XML trees.

    Args:
    a_t1 (xml.ElementTree): first XML tree to compare
    a_t2 (xml.ElementTree):  second XML tree to compare
    a_basefname (str): base file name
    a_mname (str): name of the markable level
    a_cmp (int): mode for comparing two spans

    Returns:
    (void):
    updates global statistics in place

    """
    global STATISTICS
    # convert markables in files to lists of tuples
    m_tuples1 = _markables2tuples(a_t1)
    m_tuples2 = _markables2tuples(a_t2)
    # generate lists of all indices in markables
    m1_set = set([w for mt in m_tuples1 for w in mt[0]])
    m2_set = set([w for mt in m_tuples2 for w in mt[0]])
    # generate mappings from token ids to the markbale indices
    if a_mname in REL_MARKABLES:
        tokid2markdid1 = _generate_tok2mark(m_tuples1)
        tokid2markdid2 = _generate_tok2mark(m_tuples2)
    else:
        tokid2markdid1 = tokid2markdid2 = None
    # update statistics for each single annotator
    # print("*** filename =", repr(a_basefname), file=sys.stderr)
    _update_annotator_stat(m_tuples1, m2_set, a_basefname, 0,
                           a_mname, a_cmp, a_t2 is not None,
                           tokid2markdid2, m_tuples2, a_update_cnt=True)
    _update_annotator_stat(m_tuples2, m1_set, a_basefname, 1,
                           a_mname, a_cmp, a_t1 is not None,
                           tokid2markdid1, m_tuples1)


def compute_stat(a_basedata_dir, a_dir1, a_dir2, a_ptrn="",
                 a_cmp=BINARY_OVERLAP):
    """Compare markables in two annotation directories.

    Args:
    a_basedata_dir (str): directory containing basedata for MMAX project
    a_dir1 (str): directory containing markables for the first annotator
    a_dir2 (str): directory containing markables for the second annotator
    a_ptrn (str): shell pattern for markable files
    a_cmp (int): mode for comparing two annotation spans

    Returns:
    (void)

    """
    global STATISTICS
    # find annotation files from first directory
    if a_ptrn:
        dir1_iterator = glob.iglob(os.path.join(a_dir1, a_ptrn))
    else:
        dir1_iterator = os.listdir(a_dir1)
    # iterate over files from the first directory
    f1 = f2 = ""
    basename1 = ""
    basedata_fname = base_key = ""
    fd1 = fd2 = basedata_fd = None
    t1 = t2 = None
    annotations = None
    n = 0                       # total number of words in a file

    for f1 in dir1_iterator:
        # get name of second file
        basename1 = os.path.basename(f1)
        # print("Processing file '{:s}'".format(f1), file=sys.stderr)
        f2 = os.path.join(a_dir2, basename1)
        # open both files for reading
        fd1 = open(os.path.join(a_dir1, basename1), 'r')
        try:
            t1 = ET.parse(fd1)
        except (IOError, ET.ParseError):
            t1 = None
        finally:
            fd1.close()
        # read XML information from second file ignoring non-existent, empty,
        # and wrong formatted files
        try:
            fd2 = open(f2, 'r')
            try:
                t2 = ET.parse(fd2)
            finally:
                fd2.close()
        except (IOError, ET.ParseError):
            t2 = None

        if t1 is None and t2 is None:
            continue
        # determine the name of the markable for which we should calculate
        # annotations
        mname = MRKBL_NAME_RE.match(basename1).group(1).lower()
        base_key = MARK_SFX_RE.sub("", basename1)
        # compare two XML trees
        _update_stat(t1, t2, base_key, mname, a_cmp)


def main():
    """Main method for measuring agreement and marking differences in corpus.

    Args:
    (void)

    """
    argparser = argparse.ArgumentParser(description=
                                        "Script for plotting corpus statistics"
                                        " and agreement.")
    argparser.add_argument("src_corpus",
                           help="XML corpus of source files")
    argparser.add_argument("src_dir",
                           help="directory containing XML corpus files")
    argparser.add_argument("basedata_dir",
                           help="directory containing basedata (tokens) "
                           "for MMAX project")
    argparser.add_argument("directory1",
                           help="directory containing markables"
                           " from the first annotator")
    argparser.add_argument("directory2",
                           help="directory containing markables"
                           "from the second annotator")
    # agreement schemes for spans
    argparser.add_argument("-b", "--binary-overlap",
                           help="consider two spans to agree on all"
                           " of tokens of their respective spans if"
                           " they overlap by at least one token"
                           " (default comparison mode)",
                           action="store_const", const=BINARY_OVERLAP,
                           default=0)
    argparser.add_argument("-p", "--proportional-overlap",
                           help="count as agreement only tokens that actually"
                           " ovelap in two spans", action="store_const",
                           const=PROPORTIONAL_OVERLAP, default=0)
    argparser.add_argument("--pattern",
                           help="shell pattern for files with markables",
                           type=str, default="*.xml")
    args = argparser.parse_args()
    # process raw corpus
    read_src_corpus(args.src_corpus)
    # process basedata
    read_basedata(args.basedata_dir, args.src_dir, args.pattern)
    # check if comparison scheme was specified
    cmp_scheme = args.binary_overlap | args.proportional_overlap
    if cmp_scheme == 0:
        cmp_scheme = BINARY_OVERLAP
    # check existence and readability of directory
    dir1 = args.directory1
    dir2 = args.directory2

    assert os.path.isdir(dir1) and os.access(dir1, os.X_OK), \
        "Directory '{:s}' does not exist or cannot be accessed.".format(dir1)
    assert os.path.isdir(dir2) and os.access(dir2, os.X_OK), \
        "Directory '{:s}' does not exist or cannot be accessed.".format(dir2)
    compute_stat(args.basedata_dir, dir1, dir2, a_ptrn=args.pattern,
                 a_cmp=cmp_scheme)
    sorted_cats = STATISTICS.keys()
    sorted_cats.sort()

    n_toks = 0
    ikappa = 0.
    agr_mtx = None
    stat_mtx = None
    pp = cat = stat1 = stat2 = None
    # plot statistics

    # small hard-code of topics and categories, but that's what I really need
    # to plot
    for mname in REL_MARKABLES:
        stat_mtx = np.zeros(MTX_DIM)
        agr_mtx = np.zeros(MTX_DIM)
        for icat, irow in CAT2ROW.iteritems():
            for itopic, icol in TOP2COL.iteritems():
                cat = (itopic, icat)
                n_toks = STATISTICS[cat][TOK]
                stat1 = STATISTICS[cat][ANNOTATOR][0][mname]
                stat2 = STATISTICS[cat][ANNOTATOR][-1][mname]
                total1, total2 = stat1[TOTAL_MTOK_IDX], \
                    stat2[TOTAL_MTOK_IDX]
                overlap1, overlap2 = stat1[OVERLAP_MTOK_IDX], \
                    stat2[OVERLAP_MTOK_IDX]
                ikappa = _compute_kappa(overlap1, total1, overlap2,
                                        total2, n_toks, cmp_scheme)
                agr_mtx[irow, icol] = ikappa
                stat_mtx[irow, icol] = stat2[TOTAL_MARK_IDX]
        print("mname = {:s}".format(mname), file=sys.stderr)
        print(repr(stat_mtx), file=sys.stderr)
        plot_mtx(stat_mtx, mname + STAT_MTX_SFX)
        plot_mtx(agr_mtx, mname + AGR_MTX_SFX)

    # print("STATISTICS =", repr(STATISTICS))
    # print("POL_STAT =", repr(POL_STAT))
    # print("INT_STAT =", repr(INT_STAT))
    n = 0
    ialpha = 0.
    ipol_stat = iint_stat = None
    for mname in REL_MARKABLES:
        ipol_stat, iint_stat = POL_STAT[mname], INT_STAT[mname]
        n_toks = ipol_stat[TOTAL]
        stat1, stat2 = ipol_stat[ANNOTATOR]
        total1, total2 = stat1[TOTAL_MTOK_IDX], stat2[TOTAL_MTOK_IDX]
        overlap1, overlap2 = stat1[OVERLAP_MTOK_IDX], stat2[OVERLAP_MTOK_IDX]
        ikappa = _compute_kappa(overlap1, total1, overlap2,
                                total2, n_toks, BINARY_OVERLAP)
        ialpha = _compute_alpha(INT_STAT[mname])
        print("Kappa ({:s}): {:f}".format(mname, ikappa), file=sys.stderr)
        print("Alpha ({:s}): {:f}".format(mname, ialpha), file=sys.stderr)

##################################################################
# Main
if __name__ == "__main__":
    main()
