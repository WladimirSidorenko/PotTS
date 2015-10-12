#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
DESCRIPTION:
============
Script for merging data from an annotated MMAX corpus with automatically
obtained CONLL DG trees obtained for the same raw input, that was used for the
MMAX failes.  The output is in plain CONLL format with the annotations from
the MMAX corpus represented as DG features.

USAGE:
======
merge_conll_mmax.py [OPTIONS] conll_file token_file word_file

EXAMPLE:
========
(envoke from the top directory of this archive)
./scripts/merge_conll_mmax.py scripts/examples/3.federal_election_addition.conll \
corpus/source/3.federal_election_addition.xml \
corpus/basedata/3.federal_election_addition.words.xml \
corpus/annotator-2/markables/3.federal_election_addition_*.xml

LICENSE:
========
Open domain.

"""

##################################################################
# Libraries
import argparse
import os
import re
import sys
import string
import xml.etree.ElementTree as ET

from conll import CONLL, FEAT_NAME_SEP
from align import nw_align
from alt_fio import AltFileInput, AltFileOutput

##################################################################
# Constants
DEFAULT_LANG = "utf-8"
MS_PREFIX    = re.compile("\s*[$]")
# suffix of file with MMAX words
WFNAME_SFX   = re.compile("[.]words[.]xml$", re.IGNORECASE)
WSPAN_PREFIX = "word_"
WSPAN_PREFIX_RE = re.compile(WSPAN_PREFIX)
# regexp matching separate pieces of a span
COMMA_SEP = ','
# regexp matching spans encompassing multiple words
WMULTISPAN  = re.compile("{:s}(\d+)..+{:s}(\d+)".format(WSPAN_PREFIX, \
                                                            WSPAN_PREFIX), \
                             re.IGNORECASE)
# regexp matching span encompassing single word
WSPAN = re.compile("{:s}(\d+)\Z".format(WSPAN_PREFIX), re.IGNORECASE)
# regexp matching beginning of markable id's
MARK_ID_PRFX = re.compile("markable_(\d+)\Z", re.IGNORECASE)
# regexp representing all punctuation characters
PUNCT_RE = re.compile("[{:s}]".format(string.punctuation))
OFILE_EXTENSION = ".xml"
LINE_SEP = re.compile("\s*\n+\s*")
SPAN_SEP = ".."
REFID_SEP = ':'
EMPTY_REF = "empty"
SPAN_PRFX = "word_"
ID_PRFX  = re.compile("^markable_", re.IGNORECASE)
# referential attributes, i.e. those which establish links to other elements
REF_ATTRS = ["sentiment_ref", "emo_expression_ref"]
EOL = "EOL"
LINE_SEP = '\n'
FIELD_SEP = '\t'
FIELD_SEP_RE = re.compile(FIELD_SEP)
HSH_TAG_RE = re.compile('^#+')
ESC_CHAR = ''

##################################################################
# Methods
def _cmp_words(w1, w2):
    """Function for comparing words.

    @param w1 - first word
    @param w2 - second word

    @return int representing penalty score for case when words are not equal
    """
    # strip off hash tags at the beginning of the words
    w1 = HSH_TAG_RE.sub("", w1).lower()
    w2 = HSH_TAG_RE.sub("", w2).lower()
    if w1 == w2:
        return 2
    else:
        return -1

def read_conll(istream):
    """Read file with CONLL trees for single tweets and return a dict.

    @param istream - input stream from which CONLL data should be read

    @return dictionary with tweet id's as keys and CONLL objects as values

    """
    # auxiliary variables
    t_id = None
    tweet2conll = {}
    metafields = []
    txt_lines = []
    # iterate over lines in input stream
    for line in istream:
        if line:
            if line[0] == ESC_CHAR:
                # update `t_id` and output CONLL tree if necessary
                metafields = FIELD_SEP_RE.split(line)
                if metafields[1] == "id":
                    __update_conlldic__(tweet2conll, t_id, txt_lines)
                    t_id = metafields[2]
                txt_lines.append(line)
            # if a non-empty line appears but no tweet is known, raise an
            # exception
            elif t_id == None:
                raise Exception("""Non-empty line {:d} could not be
assigned to any tweet.""".format(istream.fnr))
            # remember lines corresponding to tweets
            else:
                txt_lines.append(line)
        elif t_id != None:
            txt_lines.append(line)
    # add any lines that are left, to dictionary
    __update_conlldic__(tweet2conll, t_id, txt_lines)
    # return dictionary
    return tweet2conll

def __update_conlldic__(idic, ikey, ilines):
    """Insert CONLL object generated from ilines in dictionary. """
    if ilines:
        idic[ikey] = CONLL(LINE_SEP.join(ilines))
        del ilines[:]

def merge_conll_mmax_doc(conlldic, tkndoc, wrddoc, markable_paths):
    """Apply MMAX markables to CONLL data and output enriched CONLL string.

    @param conlldic - dictionary with CONLL data for single tweets
    @param tkndoc   - path to file with original tokenization
    @param wrddoc   - path to MMAX words file
    @param markable_paths - list of files containing markables

    """
    # initialize auxiliary variables
    conll_o = tkn_w = tkn_lw = mmax_w = mmax_lw = None
    mmax_w_id = i = 0
    conll_words = []
    mmax_words  = []
    conll_mmax_words = []       # aligned CONLL and MMAX annotations
    # create a dict for storing w_id => markable
    word2mark = dict()
    # populate dict of correspondences between words and markables
    markables2dict(markable_paths, word2mark)
    # establish an iterator over words in `wdoc`
    wrd_iter = wrddoc.iter("word")
    t_id = None
    # iterate over tweets in `tkndoc`
    for t in tkndoc.iter("tweet"):
        # find CONLL trees corresponding to given tweet
        t_id = t.get("id")
        conll_o = conlldic[t_id]
        # get all CONLL words along with their sentence and word indices and
        # lowercase and strip these words
        conll_words = [(w.strip().lower(), s_id, w_id) for w, s_id, w_id in \
                           conll_o.get_words()]
        # set list of MMAX words to empty list
        mmax_words  = []
        # iterate over tweet's words, find their corresponding word ids in
        # wrddoc, and populate a list of 2-tuples in which the first element
        for tkn_w in t.text.strip().splitlines():
            tkn_lw = tkn_w.strip().lower()
            # retrieve corresponding MMAX word
            mmax_w = wrd_iter.next()
            # keep retrieving MMAX words until we hit first word which is not
            # end-of-line marker
            try:
                while mmax_w.text == EOL:
                    mmax_w = wrd_iter.next()
            except StopIteration:
                raise Exception("Number of token words is greater than number of MMAX words.")
            mmax_lw = mmax_w.text.strip().lower()
            # check that MMAX word corresponds to given tweet word
            if tkn_lw != mmax_lw:
                raise Exception(u"Token ('{:s}') and MMAX word ('{:s}') do not match".format( \
                        tkn_lw.encode("utf-8"), mmax_lw.encode("utf-8")))
            # add 2-tuple of tkn_lw and MMAX word id to the list of MMAX words
            mmax_words.append((tkn_lw, mmax_w.get("id")))
        # Align CONLL and MMAX word lists.  Result will be a list of three
        # elements tuples, in which first element will be sentence id of CONLL
        # word, second element will be word's id in that sentence, and the
        # third element will be a list of MMAX word ids corresponding to given
        # CONLL word
        # print >> sys.stderr, "conll_words = ", repr(conll_words)
        # print >> sys.stderr, "mmax_words = ", repr(mmax_words)
        conll_mmax_words = merge_conll_mmax(conll_words, mmax_words)
        # print >> sys.stderr, "conll_mmax_words = ", repr(conll_mmax_words)
        # iterate over merged list and add annotation from MMAXX to
        # corresponding CONLL words
        for s_id, w_id, mmax_w_id_list in conll_mmax_words:
            # iterate over all MMAX words and add their id's
            for mmax_w_id in mmax_w_id_list:
                conll_o[s_id][w_id].add_features(word2mark.get(mmax_w_id, {}))
        # output enriched CONLL object
        print unicode(conll_o).encode("utf-8")

def merge_conll_mmax(conll_wlist, mmax_wlist):
    """Align elements from conll_wlist and mmax_wlist.

    conll_wlist's are lists of 3-tuples in which first element is conll word,
    second and third elements are sentence and word id of this word,
    respectively. mmax_list is a list of two tuples, in which first element is
    MMAX word and second element is MMAX id of this word.  This method
    determines to which CONLL word given MMAX corresponds and returns an
    augmented CONLL list whose tuples contain one additional 4-th element which
    is itself a list of MMAX word id's corresponding to given CONLL word.

    @param conll_wlist - list of 3-tuples with CONLL word, its sentence and word id
    @param mmax_wlist  - list of 2-tuples with MMAX word and its MMAX id

    @return list of three elements tuples, in which first element will be
        sentence id of CONLL word, second element will be word's id in that
        sentence, and the third element will be a list of MMAX word ids
        corresponding to given CONLL word

    """
    conll_words = [e[0] for e in conll_wlist]
    mmax_words  = [e[0] for e in mmax_wlist]
    s_id = w_id = 0
    mw_id_list = []
    # aligned_words will be a list of length len(conll_words) in which every
    # element will also be a list of the indices of mmax_words which correspond
    # to CONLL word at given position
    aligned_words = nw_align(conll_words, mmax_words, substitute = _cmp_words)
    # join CONLL and MMAX indices
    ret = []
    for i, mwlist in enumerate(aligned_words):
        # append to `ret` a three-tuple with sentence and word indices of
        # CONLL word
        s_id, w_id = conll_wlist[i][1:]
        mw_id_list = [mmax_wlist[mw_id][1] for mw_id in mwlist]
        ret.append((s_id, w_id, mw_id_list))
    return ret

def markables2dict(markable_paths, word2mark_dict):
    """Read and parse markable documents populating word2mark dict.

    @param markable_paths - list of files with information about markables
    @param word2mark_dict - dictionary which should be populated with
                            information about markables

    """
    # initialize auxiliary variables
    mdoc = mspan = attrs = None
    # parse markable documents and store their information in `markable_dict`
    for mp in markable_paths:
        # print >> sys.stderr, "Looking for markables in file", mp
        mdoc = ET.parse(mp)
        # due to presence of a namespace, we can't explicitly specify tags over
        # which we should iterate, so we look at their `span` attribute
        for mark in mdoc.iter():
            mspan = mark.get("span")
            if not mspan:
                continue
            # convert span string to a list
            mspan = parse_span(mspan)
            # get attributes from markables and adjust them appropriately
            selfattr, attrs = __adjust_attrs__(mark.attrib)
            # populate dictionary of word id's with their corresponding
            # markables.  If a markable encompasses a range of words, the full
            # list of markables' attributes will only be preserved for the
            # first word in the range.  Since attributes pertain to a markable
            # rather than to a span, repeating them for all words in span will
            # rather be redundant.
            w_id = mspan[0]
            # check that attributes in `attrs` do not intersect with those
            # already present in `word2mark_dict`
            if (word2mark_dict.setdefault(w_id, dict([])).viewkeys() & attrs.viewkeys()):
                print >> sys.stderr, "word2mark_dict.setdefault(w_id, dict([])).viewkeys():", \
                    repr(word2mark_dict.setdefault(w_id, dict([])).viewkeys())
                print >> sys.stderr, "attrs.viewkeys():", repr(attrs.viewkeys())
            assert(not (word2mark_dict.setdefault(w_id, dict([])).viewkeys() & attrs.viewkeys()))
            word2mark_dict[w_id].update(attrs)
            # remove all attributes from attrs except for markable's name
            for w_id in mspan[1:]:
                # add reference to the markable to rest of attributes in the
                # span
                word2mark_dict.setdefault(w_id, dict([])).update(selfattr)

def __adjust_attrs__(attrdict):
    """Adjust attributes obtained from markables."""
    # all information from span has already been processed, so remove it
    attrdict.pop("span", None)
    # take id from attributes
    _id_ = attrdict.pop("id")
    # if `sentiment_ref` or another `ref` attribute is specified, replace
    # `_id_` with the `ref` value unless it is "empty"
    ref = None
    for ref_attr in REF_ATTRS:
        ref = attrdict.pop(ref_attr, None)
        if ref and ref != EMPTY_REF:
            _id_ = __get_refid__(ref)
            break
    # get markable's name
    mtype = __adjust_attr_key__(attrdict.pop("mmax_level"))
    # ok, now what is left are true attributes pertaining to given markable.
    # This markable, however, can coincide with another markable of the same
    # type and with the same set of attributes.  So, we need to make
    # attribute's name unambiguous by prepending them with the markable type
    # and its id
    attrdict = dict([(mtype + FEAT_NAME_SEP + _id_ + FEAT_NAME_SEP + __adjust_attr_key__(k), v) \
                    for k, v in attrdict.iteritems()])
    # construct separate attribute for the markable itself
    selfattr = dict([])
    selfattr[mtype + FEAT_NAME_SEP + _id_ + FEAT_NAME_SEP + mtype] = "True"
    # add separate attribute for the markable itself to the list of attributes
    attrdict.update(selfattr)
    return [selfattr, attrdict]

def __get_refid__(ref):
    """Get markable id from reference attribute."""
    return ref.split(REFID_SEP)[1]

def __adjust_attr_key__(ikey):
    """Capitalize key's components and remove any punctuation from it."""
    return PUNCT_RE.sub("", ikey.title())

def parse_span(ispan, a_int_fmt = False):
    """Generate and return a list of all word ids encompassed by ispan."""
    ret = []
    # split span on commas
    spans = ispan.split(COMMA_SEP)
    for s in spans:
        if WSPAN.match(s):
            if a_int_fmt:
                ret.append(int(WSPAN_PREFIX_RE.sub("", s)))
            else:
                ret.append(s)
        else:
            mobj = WMULTISPAN.match(s)
            if mobj:
                start, end = int(mobj.group(1)), int(mobj.group(2)) + 1
                if a_int_fmt:
                    ret += [w_id for w_id in xrange(start, end)]
                else:
                    ret += [(WSPAN_PREFIX + str(w_id)) for w_id in xrange(start, end)]
            else:
                raise ValueError("Unrecognized span format: {:s}".format(ispan))
    return ret

def apply_annotation(wlist, wrd_iter, markable_hash):
    """Convert each word in wlist to a tuple and add annotation to it."""
    retlist = wfeatures = []
    w2 = ""
    w_id = None
    for w1 in wlist:
        w2 = wrd_iter.next()
        while w2.text == "EOL":
            w2 = wrd_iter.next()
        # print >> sys.stderr, "w1:", repr(w1)
        # print >> sys.stderr, "w2 text:", repr(w2.text)
        assert(w1 == w2.text)
        w_id = w2.get("id")
        if w_id in markable_hash:
            wfeatures = [f for f in markable_hash[w_id]]
        else:
            wfeatures = []
        retlist.append(tuple([w1] + wfeatures))
    return retlist

##################################################################
# Main
if __name__ == "__main__":
    # Arguments
    argparser = argparse.ArgumentParser(description="""Utility for merging DG CONLL data with
annotation from MMAX corpus.""")
    argparser.add_argument("-c", "--esc-char", help = """escape character which should
precede lines with meta-information""", nargs = 1, type = str, default = ESC_CHAR)
    argparser.add_argument("-e", "--encoding", help="input/output encoding", \
                           default = DEFAULT_LANG)
    argparser.add_argument("conll_file", help = "file with DG trees in CONLL format")
    argparser.add_argument("token_file", help = "file with original tokenization")
    argparser.add_argument("word_file", help = "file with MMAX words")
    argparser.add_argument("annotation_files", help = "files with MMAX markables", nargs = '*')
    args = argparser.parse_args()

    # variables
    ESC_CHAR = args.esc_char
    foutput = AltFileOutput(encoding = args.encoding)
    finput  = AltFileInput(args.conll_file, print_func = foutput.fprint)

    # skip files with no annotation
    if not args.annotation_files:
        sys.exit(0)
    # read and parse CONLL file
    conlldic = read_conll(finput)
    # read and parse tokenization file
    tkndoc      = ET.parse(args.token_file)
    # read and parse MMAX word file
    wrddoc      = ET.parse(args.word_file)
    # merge annotation with CONLL data
    merge_conll_mmax_doc(conlldic, tkndoc, wrddoc, args.annotation_files)
