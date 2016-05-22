#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from measure_corpus_agreement import BINARY_OVERLAP, PROPORTIONAL_OVERLAP, \
    MRKBL_NAME_RE, MARK_SFX_RE, _markables2tuples, \
    _compute_kappa
from plot_stat import CATEGORIES, TWEETID2CAT, TOKID2TWEETID, \
    read_src_corpus, read_basedata, _generate_tok2mark

from collections import defaultdict, Counter
from math import sqrt
from xml.etree import ElementTree as ET
import argparse
import glob
import numpy as np
import os
import sys

##################################################################
# Variables and Constants

# counter of tweet tokens
TWEETTOK_CNT = None

# statistics on frequency and agreement of markables divided by tweet category
CAT_STAT = defaultdict()

# statistics on the number of elements
TWEETID2CNT_KAPPA = defaultdict(lambda:
                                defaultdict(lambda: None))

# statistics on frequencies and agreement of markables for single tweets
TWEETID2MSTAT = defaultdict(lambda:
                            defaultdict(lambda:
                                        [0, 0, 0, 0, set()]))
# counter of double annotations
DBL_ANNO = defaultdict(lambda:
                       defaultdict(lambda: [0, 0]))

A1_OFFS = 0
A2_OFFS = 2

A_IDX = 0
M_IDX = 1
A1_IDX = 0
M1_IDX = 1
A2_IDX = 2
M2_IDX = 3
MRKBLS_IDX = -1

# relevant markables
REL_MARKABLES = set(["emo-expression", "sentiment"])


##################################################################
# Methods
def compute_agr_stat(a_basedata_dir, a_dir1, a_dir2, a_ptrn="",
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
    # find annotation files from first directory
    if a_ptrn:
        dir1_iterator = glob.iglob(os.path.join(a_dir1, a_ptrn))
    else:
        dir1_iterator = os.listdir(a_dir1)
    # iterate over files from the first directory
    f1 = f2 = ""
    t1 = t2 = None
    fd1 = fd2 = basedata_fd = None
    basename1 = mname = base_key = ""

    for f1 in dir1_iterator:
        # get name of second file
        basename1 = os.path.basename(f1)
        mname = MRKBL_NAME_RE.match(basename1).group(1).lower()
        if mname is None or mname not in REL_MARKABLES:
            continue
        print("Processing file '{:s}'".format(basename1), file=sys.stderr)
        f2 = os.path.join(a_dir2, basename1)
        if not os.path.exists(f2) or not os.access(f2, os.R_OK):
            continue
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
        fd2 = open(f2, 'r')
        try:
            t2 = ET.parse(fd2)
        finally:
            fd2.close()

        if t1 is None or t2 is None:
            continue
        # determine the name of the markable for which we should calculate
        # annotations
        base_key = MARK_SFX_RE.sub("", basename1)
        # compare two XML trees
        _update_stat(t1, t2, base_key, mname, a_cmp)


def _update_stat(a_t1, a_t2, a_basefname, a_mname, a_cmp=BINARY_OVERLAP):
    """Compare annotations present in two XML trees.

    Args:
    a_t1 (xml.ElementTree):
      first XML tree to compare
    a_t2 (xml.ElementTree):
      second XML tree to compare
    a_basefname (str):
      base file name
    a_mname (str):
      name of the markable level
    a_cmp (int):
      mode for comparing two spans

    Returns:
    (void):
      updates global statistics in place

    """
    # convert markables in files to lists of tuples
    m_tuples1 = _markables2tuples(a_t1)
    m_tuples2 = _markables2tuples(a_t2)
    # generate lists of all indices in markables
    m1_set = set([w for mt in m_tuples1 for w in mt[0]])
    m2_set = set([w for mt in m_tuples2 for w in mt[0]])
    # generate mappings from token ids to the markbale indices
    tokid2markdid1 = _generate_tok2mark(m_tuples1)
    tokid2markdid2 = _generate_tok2mark(m_tuples2)
    # update statistics for each single annotator
    _update_annotator_stat(m_tuples1, m2_set, a_basefname, A1_OFFS,
                           a_mname, a_cmp, a_t2 is not None,
                           tokid2markdid2, m_tuples2)
    _update_annotator_stat(m_tuples2, m1_set, a_basefname, A2_OFFS,
                           a_mname, a_cmp, a_t1 is not None,
                           tokid2markdid1, m_tuples1, a_update_cnt=True)


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
    a_update_cnt (bool): update counter of tweet markables

    Returns:
    (void): updates global variable in place

    """
    global TWEETID2MSTAT

    added_toks = set()
    unseen_toks = None
    tok_id = cat_stat = None
    # update counters of markables
    tweet_id = ""
    tweet_ids = set()
    tweet_stat = None
    a_idx = a_anno_id + A_IDX
    m_idx = a_anno_id + M_IDX
    # update counters for computing agreement
    for imtuple in a_mtuples:
        # check category of the tweet which the given markable pertains to
        for itok_id in imtuple[0]:
            tok_id = (a_basefname, unicode(itok_id))
            tweet_ids.add(TOKID2TWEETID[tok_id])
        assert len(tweet_ids) == 1, \
            "Multiple tweet ids found for one " \
            "markable: {:s} (tokes: {:s})".format(
                repr(tweet_ids), repr(imtuple[0]))
        for t_id in tweet_ids:
            tweet_id = t_id
        tweet_ids.clear()
        tweet_stat = TWEETID2MSTAT[tweet_id][a_mname]
        # update counters of markables if needed
        if a_update_cnt:
            tweet_stat[MRKBLS_IDX].add(imtuple[-1]["id"])
        # update agreement counters (we do not consider exact match here)
        if a_cmp & PROPORTIONAL_OVERLAP:
            unseen_toks = set([w for w in imtuple[0]
                               if w not in added_toks])
            tweet_stat[a_idx] += len(unseen_toks & a_word_ids2)
        elif a_cmp & BINARY_OVERLAP:
            unseen_toks = set(imtuple[0])
            DBL_ANNO[tweet_id][a_mname][min(a_anno_id, 1)] += \
                len(unseen_toks & added_toks)

            if unseen_toks & a_word_ids2:
                tweet_stat[a_idx] += len(unseen_toks)
        else:
            raise RuntimeError("Comparison scheme is not supported.")
        added_toks.update(unseen_toks)
        tweet_stat[m_idx] += len(unseen_toks)


def _compute_cat_stat(a_tid2cnt_kappa):
    """Compute mean and variance per each tweet category.

    Args:
    a_tid2cnt_kappa (dict):
      tweet statistcs on tweets' markables (their count and kappa)

    Returns:
    4-tuple(dict, dict, defaultdict, defaultdict)"
      mapping from topic to index, mapping from category to index,
      topic-specific correlation, category-specific correlation

    """
    topics = set()
    types = set()
    for itopic, itype in TWEETID2CAT.itervalues():
        topics.add(itopic)
        types.add(itype)
    n_topics = len(topics)
    n_types = len(types)
    n_tweets = len(TWEETID2CAT)
    # expectation of categories
    # initialize mapping from topic and type to row indices
    topic2idx = {t: i for i, t in enumerate(topics)}
    type2idx = {t: i for t, i in zip(types, xrange(n_topics,
                                                   n_topics + n_types))}
    # initialize matrices of samples
    x = np.zeros((n_topics + n_types, n_tweets))
    y_cnt = defaultdict(lambda:
                        np.zeros((n_topics + n_types, n_tweets)))
    y_kappa = defaultdict(lambda:
                          np.zeros((n_topics + n_types, n_tweets)))
    # populate sample matrices
    itopic = itype = ""
    topic_id = type_id = 0
    for i, (t_id, tstat) in enumerate(a_tid2cnt_kappa.iteritems()):
        itopic, itype = TWEETID2CAT[t_id]
        topic_id, type_id = topic2idx[itopic], type2idx[itype]
        x[topic_id][i] = x[type_id][i] = 1.
        for mname, (icnt, ikappa) in tstat.iteritems():
            y_cnt[mname][:, i] = icnt
            y_kappa[mname][:, i] = ikappa
    rho_cnt = defaultdict(lambda:
                          np.zeros((n_topics + n_types, n_tweets)))
    rho_kappa = defaultdict(lambda:
                            np.zeros((n_topics + n_types, n_tweets)))
    # compute correlation coefficients
    for mname, ystat in y_cnt.iteritems():
        rho_cnt[mname] = np.corrcoef(x, ystat)
    for mname, ystat in y_kappa.iteritems():
        rho_kappa[mname] = np.corrcoef(x, ystat)
    return (topic2idx, type2idx, rho_cnt, rho_kappa)


def _compute_mean_var(a_stat, a_total):
    """Compute probability, mean, and variance of the given variable.

    Args:
    a_stat: Counter
      statistics on values that the given variable might take on
    a_total: float
      total number of items in population

    Returns:
    (tuple(float, float)):
      mean and var

    """
    if a_total <= 0:
        a_total = 1
    mean = var = iprob = 0.
    stat = [(val, float(cnt) / a_total) for val, cnt in
            a_stat.iteritems()]
    for val, iprob in stat:
        mean += val * iprob
    # to prevent catastrophic cancellation
    for val, iprob in stat:
        var += iprob * ((val - mean)**2)
    return (mean, var)


def _cov_cor(a_EXY, a_mu_x, a_var_x, a_mu_y, a_var_y):
    """Compute char/class covariance/correlation.

    Args:
    a_EXY: float
      joint expectation
    a_mu_x: float
      mean of the covariate
    a_var_x: float
      variance of the covariate
    a_mu_y: float
      mean of the dependent variable
    a_var_y: float
      variance of the dependent variable

    Returns:
      2-tuple: covariances, correlation coefficient

    """
    cov = a_EXY - a_mu_x * a_mu_y
    cor = cov / sqrt(a_var_x * a_var_y)
    return (cov, cor)


##################################################################
# Main
def main():
    """Find character n-grams characteristic to specific sentiment classes.

    Returns:
    0 on success, non-0 otherwise

    """
    argparser = argparse.ArgumentParser(description="""Script for computing
correlation coefficients of sentiments and emotional expressions on the one
hand and topic and formal categories on the other hand.""")
    argparser.add_argument("src_corpus",
                           help="source corpus file")
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

    # process raw corpus (populate TWEETID2CAT)
    read_src_corpus(args.src_corpus)

    # process basedata (populate TOKID2TWEETID)
    read_basedata(args.basedata_dir, args.src_dir, args.pattern)
    # compute the number of tokens pertaining to a tweet
    global TWEETTOK_CNT
    TWEETTOK_CNT = Counter(TOKID2TWEETID.itervalues())

    # check if comparison scheme was specified
    cmp_scheme = args.binary_overlap | args.proportional_overlap
    if cmp_scheme == 0:
        cmp_scheme = BINARY_OVERLAP
    # check existence and readability of directories
    dir1 = args.directory1
    dir2 = args.directory2

    assert os.path.isdir(dir1) and os.access(dir1, os.X_OK), \
        "Directory '{:s}' does not exist or cannot be accessed.".format(dir1)
    assert os.path.isdir(dir2) and os.access(dir2, os.X_OK), \
        "Directory '{:s}' does not exist or cannot be accessed.".format(dir2)
    compute_agr_stat(args.basedata_dir, dir1, dir2,
                     a_ptrn=args.pattern,
                     a_cmp=cmp_scheme)
    # do some sanity check and compute per-tweet statistics
    for t_id, tstats in TWEETID2MSTAT.iteritems():
        for mname, (m1, a1, m2, a2, marks) in tstats.iteritems():
            assert m1 <= a1, \
                "Number of matching annotations exceeds the " \
                "number of annotated tokens (1): tweet {:s} " \
                "markables {:s}". format(t_id, repr(marks))
            assert m2 <= a2, \
                "Number of matching annotations exceeds the " \
                "number of annotated tokens (2): tweet {:s} " \
                "markables {:s}". format(t_id, repr(marks))
            TWEETID2CNT_KAPPA[t_id][mname] = \
                (len(marks), _compute_kappa(m1, a1, m2, a2,
                                            max(DBL_ANNO[t_id][mname]) +
                                            TWEETTOK_CNT[t_id],
                                            cmp_scheme))
    # compute mean and variance per each category
    topic2idx, type2idx, rho_cnt, rho_kappa = \
        _compute_cat_stat(TWEETID2CNT_KAPPA)
    n = len(topic2idx) + len(type2idx)
    icnt = ikappa = 0.
    cnt_stat = kappa_stat = None
    for mname in REL_MARKABLES:
        print("{:s}".format(mname))
        cnt_stat, kappa_stat = rho_cnt[mname], rho_kappa[mname]
        print("{:20s}{:>15s}{:>25s}".format("Topic", "$\\rho_{cnt}$",
                                            "$\\rho_{\\kappa}$"))
        for itopic, idx in topic2idx.iteritems():
            icnt, ikappa = cnt_stat[idx][n], kappa_stat[idx][n]
            assert icnt == cnt_stat[idx][-1]
            assert ikappa == kappa_stat[idx][-1]
            print("{:20s}{:15.4f}{:25.4f}".format(itopic, icnt,
                                                  ikappa))
        print()
        print("{:20s}{:>15s}{:>25s}".format("Category", "$\\rho_{cnt}$",
                                            "$\\rho_{\\kappa}$"))
        for itype, idx in type2idx.iteritems():
            icnt, ikappa = cnt_stat[idx][n], kappa_stat[idx][n]
            assert icnt == cnt_stat[idx][-1]
            assert ikappa == kappa_stat[idx][-1]
            print("{:20s}{:15.4f}{:25.4f}".format(itype, icnt,
                                                  ikappa))
        print()

if __name__ == "__main__":
    main()
