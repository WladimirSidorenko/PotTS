#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
DESCRIPTION:
============
This module provides a convenient interface for handling CONLL data.

CONLL data are represented in the form of individual lines with tab-separated
fields.  This module provides several classes which parse such lines either
incrementally, one by one, or all at once, and store their information in their
internal data structures.

Constants:
EOS         - end of sentence marker
EOL         - end of line marker
FIELDSEP    - separator of fields for description of a single word
ESC_CHAR    - character that stands at the beginning if lines representin meta-information
EMPTY_FIELD - separator of fields for description of a single word

Classes:
CONLL()         - class storing CONLL information as al list of individual sentences
CONLLSentence() - class storing information pertaining to a single CONLL sentence
CONLLWord()     - class storing information about a single CONLL word
ECONLLWord()    - an extension of CONLLWord with additional fields

LICENSE (modified MIT):
=======================
Copyright (c) 2014-2015, ANONYMOUS
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the authors nor the names of its contributors may be
used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

##################################################################
# Loaded Modules
import os
import re
import sys
from collections import defaultdict

##################################################################
# Constants
EOS      = u'\n'
EOL      = u'\n'
ESC_CHAR = ''
EOS_TAG_RE = re.compile(re.escape("<sentence />"))
FIELDSEP = u'\t'
EMPTY_FIELD = u'_'

FEAT_SEP = u'|'
FEAT_VALUE_SEP = u'='
FEAT_VALUE_SEP_RE = re.compile(FEAT_VALUE_SEP)
FEAT_NAME_SEP  = u"::"
FEAT_NAME_SEP_RE = re.compile(re.escape(FEAT_NAME_SEP))

##################################################################
# Classes
class CONLL:
    """
    Class for storing and manipulating CONLL parse forrest information.

    An instance of this class comprises information about one or multiple
    parsed sentences in CONLL format.

    This class provides following instance variables:
    self.metainfo - list of lines representing meta-information
    self.sentences - list of all sentences gathered in tree forest
    self.s_id      - list index of last parsed sentence

    This class provides following public methods:
    __init__()      - class constructor (can accept)
    self.add_line() - parse specified single line and incrementally add
                      it to the data of current tree or append a new tree to the
                      forrest
    self.is_empty() - return true if any sentences are stored
    self.clear() - drop all stored information
    self.get_words() - return list of words with their sentence and word indices
    __str__()       - return string representation of current forrest
    __getitem__()   - return sentence from forrest
    __setitem__()   - set sentence in forrest
    """

    def __init__(self, istring = ''):
        """Initialize instance variables and parse input string if specified.

        @param istring - input string(s) with CONLL data (optional)

        """
        self.metainfo = []
        self.sentences = []
        self.s_id      = -1
        self.__eos_seen__ = True
        for iline in istring.splitlines():
            self.add_line(iline)

    def add_line(self, iline = u''):
        """Parse line and add it as CONLL word to either current or new
        sentence.

        @param iline - input line(s) to parse
        """
        iline = iline.strip()
        if not iline or EOS_TAG_RE.match(iline):
            # if we see a line which appears to be an end of sentence, we
            # simply set corresponding flag
            self.__eos_seen__ = True
        elif iline and iline[0] == ESC_CHAR:
            # metainfo will pertain to the whole forrest
            self.metainfo.append(iline)
        elif self.__eos_seen__:
            # otherwise, if end of sentence has been seen before and the line
            # appears to be non-empty, increase the counter of sentences and
            # append next sentence to the list
            self._add_sentence(CONLLWord(iline))
            self.__eos_seen__ = False
        else:
            # otherwise, parse line as a CONLL word and compare its index to
            # the index of last parsed CONLL sentence. If the index of the new
            # word is less than the index of the last word, that means that a
            # new sentence has started.
            w = CONLLWord(iline)
            if self.s_id == -1 or int(w.idx) < int(self.sentences[self.s_id].words[-1].idx):
                self._add_sentence(w)
            else:
                self.sentences[self.s_id].push_word(w)

    def is_empty(self):
        """
        Check whether any sentences are stored.
        """
        return self.s_id == -1

    def clear(self):
        """
        Remove all stored information.
        """
        del self.metainfo[:]
        del self.sentences[:]
        self.s_id  = -1
        self.__eos_seen__ = False

    def get_words(self):
        """
        Return list of all words wird indices from all sentences.

        Return a list of all words from all sentences in consecutive order as
        tuples with three elements (word, sentence_idx, word_idx) where the
        first element is a word, the next element is its index in the list of
        sentences, and the third element is word's index within the sentence.

        """
        retlist = []
        for s_id in xrange(self.s_id + 1):
            retlist += [(w.form, s_id, w_id) for w, w_id in \
                            self.sentences[s_id].get_words()]
        return retlist

    def __str__(self):
        """Return string representation of current object."""
        ostring = u'\n'.join([unicode(s) for s in self.metainfo])
        if self.metainfo:
            ostring += u'\n'
        ostring += u'\n'.join([unicode(s) for s in self.sentences])
        return ostring

    def __getitem__(self, i):
        """
        Return reference to `i`-th sentence in forrest.

        @param i - integer index of sentence in forrest

        @return `i`-th CONLL sentence in forrest. IndexError is raised if `i`
        is outside of forrest boundaries.

        """
        return self.sentences[i]

    def __setitem__(self, i, value):
        """
        Set `i`-th sentence in forrest to specified value.

        @param i - integer index of sentence in forrest
        @param value - CONLL sentence to which i-th sentence should be set

        @return new value of `i`-th sentence. IndexError is raised if `i`
        is outside of forrest boundaries.

        """
        self.sentences[i] = value
        return self.sentences[i]

    def __iter__(self):
        """Return iterator object over sentences."""
        for s in self.sentences:
            yield s

    def _add_sentence(self, iword):
        """Add new sentence populating it with iword."""
        self.s_id += 1
        self.sentences.append(CONLLSentence(iword))


class CONLLSentence:
    """
    Class for storing and manipulating single CONLL sentence.

    An instance of this class comprises information about a single sentence in
    CONLL format.

    This class provides following instance variables:
    self.words - list of all words belonging to given sentence
    self.w_id  - index of last word in self.words
    self.children  - index of last word in self.words

    This class provides following public methods:
    __init__()   - class constructor
    self.clear() - remove all words and reset counters
    self.is_empty() - check if any words are present in sentence
    self.push_word() - add given CONLLWord to sentence's list of words
    self.get_words() - return list of words with their indices
    __str__() - return string representation of sentence
    __unicode__() - return UNICODE representation of sentence
    __iter__() - return an iterator object over words
    __getitem__() - return word from sentence
    __setitem__() - set word in sentence
    __reversed__() - retun a reverse iterator over words
    __len__() - return the number of words in sentence

    """

    def __init__(self, iword = ""):
        """Initialize instance variables and parse iline if specified."""
        self.w_id  = -1
        self.words = []
        self.children = defaultdict(list)
        if iword:
            self.push_word(iword)

    def clear(self):
        """Remove all words and reset counters."""
        self.w_id  = -1
        self.children.clear()
        del self.words[:]

    def is_empty(self):
        """Check if any words are present in sentence."""
        return self.w_id  == -1

    def push_word(self, iword):
        """Parse iline storing its information in instance variables."""
        self.w_id += 1
        self.words.append(iword)
        self.children[iword.phead].append(self.words[self.w_id])

    def get_words(self):
        """
        Return list of all words with their indices.

        Return a list of all words in this sentence in consecutive order as
        tuples with two elements where the first element is the word itself and
        second element is its index within the sentence.
        """
        return zip(self.words, xrange(self.w_id + 1))

    def __str__(self):
        """Return string representation of this object."""
        ostring = u''
        ostring = EOL.join([unicode(w) for w in self.words]) + EOS
        return ostring

    def __unicode__(self):
        """Return string representation of this object."""
        return self.__str__()

    def __iter__(self):
        """Return iterator object over words."""
        for w in self.words:
            yield w

    def __reversed__(self):
        """Return iterator object over words."""
        for w in self.words[::-1]:
            yield w

    def __getitem__(self, i):
        """
        Return reference to `i`-th word in sentence.

        @param i - integer index of word in sentence

        @return value of `i`-th word in sentence. IndexError is raised if `i`
        is outside of sentence boundaries.

        """
        return self.words[i]

    def __setitem__(self, i, value):
        """Set `i`-th word in sentence to specified value.

        @param i - integer index of sentence in forrest
        @param value - CONLL word to which i-th instance should be set

        @return new value of `i`-th word. IndexError is raised if `i` is
        outside of sentence boundaries.

        """
        self.words[i] = value
        return self.words[i]

    def __len__(self):
        """Return the number of words in sentence."""
        return len(self.words)

class CONLLWord(object):

    """Class for storing and manipulating information about a single word.

    An instance of this class comprises information about one word of CONLL
    tree.

    This class provides following static variables:
    key2field - mapping from attribute name to its position in attribute list
    REQFIELDS   - number of fields which has to be specified for a word

    This class provides following instance variables:
    self.fields - list of all word's attributes as they are defined in fields
    self.features - dictionary of features
    self.pfeatures - dictionary of pfeatures

    This class provides following public methods:
    __init__()      - class constructor
    self.parse_line() - parse specified CONLL line and populate instance
                      variables correspondingly
    add_features()  - update dictionary of features from another dictionary
    get()           - safe method for accessing missing attributes
    __getattr__()   - this method returns `self.field`s item if the name of
                      attribute is found in `key2field`
    __getitem__()  - this method allows access to CONLLWord field using
                     the standard dictionary like syntax, e.g. iword["token]
    __setitem__()   - this method allows to set values of CONLLWord fields by
                      using the dictionary like syntax, e.g. iword["token] = "sky"
    __str__()       - return string representation of current forrest

    """

    key2field = {'idx': 0, 'form': 1, 'lemma': 2, 'plemma': 3, 'pos': 4, \
                     'ppos': 5, 'feat': 6, 'pfeat': 7, 'head': 8, 'phead': 9, \
                     'deprel': 10, 'pdeprel': 11, 'fillpred': 12, 'pred': 13}
    REQFIELDS = len(key2field)

    def __init__(self, iline = None):
        """Initialize instance variables and parse iline if specified."""
        self.fields = []
        self.features = {}
        self.pfeatures = {}
        if iline:
            self.parse_line(iline)

    def parse_line(self, iline):
        """Parse iline storing its information in instance variables."""
        self.fields = iline.split(FIELDSEP)
        nfields = len(self.fields)
        # check that proper number of fields is provided
        if nfields != self.REQFIELDS:
            raise Exception( \
                "Incorrect line format ({:d} fields expected instead of {:d}):\n'{:s}'".format( \
                    self.REQFIELDS, nfields, iline))
        # convert features and pfeatures to dicts
        feat_i = CONLLWord.key2field["feat"]
        self.features = self.fields[feat_i] = self.__str2dict__(self.fields[feat_i])
        feat_i = CONLLWord.key2field["pfeat"]
        self.pfeatures = self.fields[feat_i] = self.__str2dict__(self.fields[feat_i])

    def add_features(self, newfeatures = {}):
        """Update dictionary of features with new features from `newfeatures'."""
        self.features.update(newfeatures)
        self.pfeatures.update(newfeatures)

    def get(self, ikey, idefault = None):
        """Return value of ikey field or idefault if the field is not present."""
        try:
            return self.__getattr__(ikey)
        except AttributeError:
            return idefault

    def __getattr__(self, name):
        """Return self.field's item if this item's name is present in key2field.

        This method looks for passed name in `key2field` dict and returns
        corresponding item of `self.fields` or raises an AttributeException
        if no such item was found.

        @param name - name of the field to be retrieved

        """
        if name in self.key2field:
            return self.fields[self.key2field[name]]
        else:
            raise AttributeError, name

    def __getitem__(self, name):
        """Return self.field's item if this item's name is present in key2field.

        This method uses the self.__getattr__() method but converts the
        AttributeException to IndexError in case when lookup was not
        successful.

        @param name - name of the field to be retrieved

        """
        try:
            return self.__getattr__(name)
        except AttributeError:
            raise IndexError, name

    def __setitem__(self, name, value):
        """Set the value of given item `name' to `value'.

        @param name - name of the attribute to be set
        @param value - new value of the attribute

        """
        if name in self.key2field:
            self.fields[self.key2field[name]] = value
        else:
            raise IndexError, name

    def __str__(self):
        """Return string representation of this object."""
        retStr = u''
        # convert features and pfeatures to strings
        feat_i = CONLLWord.key2field["feat"]
        feat_str = self.__dict2str__(self.fields[feat_i])
        pfeat_i = CONLLWord.key2field["pfeat"]
        pfeat_str = self.__dict2str__(self.fields[pfeat_i])
        # get minimum and maximum indices
        min_i = min(feat_i, pfeat_i)
        max_i = max(feat_i, pfeat_i)
        # construct return string (we can't change feature dictionary in place
        # (because next call to __str__() would be invalid), so slicing is
        # needed)
        retStr += FIELDSEP.join(self.fields[:min_i])
        retStr += FIELDSEP + self.__dict2str__(self.fields[min_i])
        if min_i + 1 < max_i:
            retStr += FIELDSEP + FIELDSEP.join(self.fields[min_i + 1:max_i])
        retStr += FIELDSEP + self.__dict2str__(self.fields[max_i]) + FIELDSEP
        retStr += FIELDSEP.join(self.fields[max_i + 1:])
        return retStr

    def __str2dict__(self, istring):
        """Convert string of features to a dictionary."""
        retDict = {}
        if istring == EMPTY_FIELD:
            return retDict
        for feat in istring.split(FEAT_SEP):
            # feature format changed in MATE
            if FEAT_VALUE_SEP_RE.search(feat):
                retDict.update((feat.split(FEAT_VALUE_SEP),))
            else:
                retDict.update([self._new2old(feat)])
        return retDict

    def __dict2str__(self, idict):
        """Convert dictionary of features to a string."""
        fList = []
        if not idict:
            return EMPTY_FIELD
        for fname, fvalue in idict.iteritems():
            fList.append(fname + FEAT_VALUE_SEP + fvalue)
        return FEAT_SEP.join(fList)

    def _new2old(self, ifeat):
        """Translate new representation of features to the old one
        @param ifeat - feature value

        @return  2-tuple of key value pair
        """
        ifeat = ifeat.lower()
        if ifeat in set(["nom", "gen", "dat", "acc"]):
            return ("case", ifeat)
        elif ifeat in set(["fem", "masc", "neut"]):
            return ("gender", ifeat)
        elif ifeat in set(["sg", "pl"]):
            return ("num", ifeat)
        elif ifeat in set(["1", "2", "3"]):
            return ("pers", ifeat)
        elif ifeat in set(["ind", "imp", "subj"]):
            return ("mood", ifeat)
        elif ifeat in set(["pres", "past"]):
            return ("tense", ifeat)
        return (ifeat, "True")

class ECONLLWord(CONLLWord):
    """
    Extended version of CONLLWord (additional fields are added).

    This class is an extension of CONLLWord class.  It extends the latter by
    adding a few more attributes to the class variable `key2field` and updating
    the class constant REQFIELDS correspondingly.

    This class extends:
    key2field - dictionary holding a mapping from field name to its index
    REQFIELDS - number of fields allowed in CONLL line

    This class overwrites:
    __init__() - class initializer
    __getattr__() - return ECONLLWord attribute
    __setitem__() - set ECONLLWord attribute

    """
    e_key2field = {'sentiment': 14}

    def __init__(self, iline = None):
        """Call parent's initializer and remember additional fields."""
        # call parent's intialize with empty line
        super(ECONLLWord, self).__init__(None)
        # process the first 13 fields of non-empty line in the usual way like
        # for any other CONLL words
        fields = []
        if iline:
            fields = iline.split(FIELDSEP)
            super(ECONLLWord, self).parse_line(FIELDSEP.join(fields[:CONLLWord.REQFIELDS]))
            # add additional features, if any
            if fields[CONLLWord.REQFIELDS:]:
                self.fields += fields[CONLLWord.REQFIELDS:]
            else:
                for _ in self.e_key2field:
                    self.fields.append(EMPTY_FIELD)

    def __getattr__(self, name):
        """Return self.field's item if this item's name is present in key2field.

        This method looks for passed name in `key2field` dict and returns
        corresponding item of `self.fields` or raises an AttributeException
        if no such item was found.

        @param name - name of the field to be retrieved

        """
        if name in CONLLWord.key2field:
            return super(ECONLLWord, self).__getitem__(name)
        elif name in self.e_key2field:
            return self.fields[self.e_key2field[name]]
        else:
            raise AttributeError, name

    def __setitem__(self, name, value):
        """Set the value of given item `name' to `value'.

        @param name - name of the attribute to be set
        @param value - new value of the attribute

        """
        if name in CONLLWord.key2field:
            return super(ECONLLWord, self).__setitem__(name, value)
        elif name in self.e_key2field:
            self.fields[self.e_key2field[name]] = value
        else:
            raise IndexError, name
