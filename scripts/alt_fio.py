#!/usr/bin/env python2.7
# -*- coding: utf-8; mode: python; -*-

"""
DESCRIPTION:
============
This module is a light-weight alternative to standard fileinput library.

Classes:
AltFileInput - class for interactive reading of input files
AltFileOutput - class for interactive reading of input files

LICENSE (modified MIT):
=======================
Copyright (c) 2014-2015, Uladzimir Sidarenka <sidarenk at uni dash potsdam dot de>

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
import sys
from fileinput import *

##################################################################
# Interface
__all__ = ['AltFileInput', 'AltFileOutput']

##################################################################
# Constants
DEFAULT_LANG = "utf-8"
DEFAULT_INPUT  = sys.stdin
DEFAULT_OUTPUT = sys.stdout

##################################################################
# Class AltFileInput
class AltFileInput:
    """
    Class for reading and appropriate decoding of input strings.

    Public methods:
    __init__() -- class constructor
    next() -- yield next line of the input file

    Instance items:
    encoding - encoding of the input file stream
    errors - tolerance level for character decoding
    skip - function indictaing which lines should be skipped from processing and
           output without changes
    print_func - default output function
    print_dest - default output stream
    files - list of input files
    fcnt - counter of read files
    current_file - input file currently being read
    filename - name of the current file as a string
    fnr - line counter for current input file
    nr - number of lines totally read for all files
    line - current input line
    """

    def __init__(self, *ifiles, **kwargs):
        """
        Create an instance of AltFileInput.

        @param ifiles - list of input files
        @param kwargs - keyword arguments for setting instance variables

        """
        # set up input encoding - use environment variable
        # SOCMEDIA_LANG or 'utf-8' by default
        self.encoding = kwargs.get('encoding', DEFAULT_LANG)
        # specify how to handle characters, which couldn't be decoded
        self.errors   = kwargs.get('errors', 'strict')
        # if skip_line was specified, establish an ad hoc function, which will
        # return true if its arg is equal
        if 'skip' in kwargs:
            self.skip = kwargs['skip']
        else:
            self.skip = lambda line: False
        # associate a print function with current fileinput, so that any input
        # lines, which should be skipped, could be sent to it
        if 'print_func' in kwargs:
            self.print_func = kwargs['print_func']
        else:
            # otherwise, standard print function will be used, however we
            # provide for a possibility, to specify the print destination via
            # 'print_dest' kwarg, so that even standard print function could be
            # easily re-directed
            if 'print_dest' in kwargs:
                self.print_dest = kwargs['print_dest']
            else:
                self.print_dest = DEFAULT_OUTPUT
            self.print_func = self._print_func
        #allow ifiles to appear both as list and as a
        # kw argument
        if not ifiles:
            ifiles = kwargs.get('ifiles', [DEFAULT_INPUT])
        # setting up instance variables
        self.files = ifiles     # list of input files
        self.fcnt  = -1         # counter for files
        self.current_file = None # file currently being read
        self.filename = None     # name of the file as a string
        self.fnr = 0             # current record number in the current file
        self.nr = 0              # number of records processed since
                                 # the beginning of the execution
        self.line = ''           # last line read-in
        # going to the 1-st file
        self._next_file_()

    def next(self):
        """Yield next line or stop iteration if input exhausted."""
        self.line = self.current_file.readline()
        # print repr(self.line)
        if not self.line:
            self._next_file_()
            return self.next()
        self.fnr +=1
        self.nr  +=1
        self.line = self.line.decode(encoding = self.encoding, \
                                         errors = self.errors).rstrip()
        # If the line read should be skipped, print this line and read the next
        # one.
        if self.skip(self.line):
            self.print_func(self.line)
            return self.next()
        else:
            return self.line

    def __iter__(self):
        """Standard method for iterator protocol."""
        return self

    def __stop__(self):
        """Unconditionally raise StopIteration() error."""
        raise StopIteration

    def _next_file_(self):
        """Switch to new file if possible and update counters."""
        # close any existing opened files
        if self.current_file:
            self.current_file.close()
        # increment counter
        self.fcnt += 1
        # didn't calculate len() in __init__ for the case that
        # self.files changes somewhere in the middle
        if self.fcnt < len(self.files):
            # reset counters
            self.current_file = self._open(self.files[self.fcnt])
            self.filename = self.current_file.name
            self.fnr = 0
            self.line = ''
        else:
            # if we have exhausted the list of available files, all subsequent
            # calls to self.next will promptly redirect to another functon
            # which will unconditionally raise a StopIterantion error
            self.next = self.__stop__
            self.next()

    def _open(self, ifile):
        """
        Determine type of ifile argument and open it appropriately.

        @param ifile - name or descriptor of input file

        @return descriptor of the opened file
        """
        # Duck-Typing in real world - no matter what the object's name is, as
        # far as it provides the necessary method
        if hasattr(ifile, 'readline'):
            # file is already open
            return ifile
        elif isinstance(ifile, str) or \
                isinstance(ifile, buffer):
            if ifile == '-':
                return DEFAULT_INPUT
            # open it otherwise
            return open(ifile, 'r')
        else:
            raise TypeError('Wrong type of argument')

    def _print_func(self, oline = ""):
        """Private function for outputting oline to particular file stream."""
        print >> self.dest, oline


##################################################################
# Class AltFileOutput
class AltFileOutput:

    """
    Class for outputing strings in appropriate encoding.

    Public Methods:
    __init__ - class constructir
    fprint - print function

    Instance Items:
    encoding - output encoding for Unicode strings
    flush - binary flag indictaing whether flush operations should be applied
    ofile - output file
    """

    def __init__(self, encoding = DEFAULT_LANG, ofile = DEFAULT_OUTPUT, \
                     flush = False):
        """
        Create an instance of AltFileOutput.

        @param encoding - output encoding for Unicode strings
        @param ofile - output file
        @param flush - binary flag indictaing whether flush operations should be applied
        """
        self.encoding = encoding
        self.flush    = flush
        self.ofile    = ofile

    def fprint(self, *ostrings):
        """
        Encode ostrings and print them, flushing the output if necessary.

        @param *ostrings - Unicode strings that should be printed

        If you don't want to redirect fprint's output, you will have to re-set
        self.ofile first. Unfortunately, it's not possible to use argument
        syntax like this: *ostrings, ofile = DEFAULT_OUTPUT
        """
        for ostring in ostrings:
            if isinstance(ostring, unicode):
                # print >> sys.stderr, "Unicode instance detected", repr(ostring)
                ostring = ostring.encode(self.encoding)
            print >> self.ofile, ostring,
        print >> self.ofile
        if self.flush:
            self.ofile.flush()
