Description
===========

[![The MIT License](https://img.shields.io/dub/l/vibe-d.svg)](http://opensource.org/licenses/MIT)

This directory contains the data of the Potsdam Twitter Sentiment
Corpus (ISLRN 714-621-985-491-3).  To open the files of this
corpus, you need to download and launch
[MMAX2](http://mmax2.sourceforge.net/)&mdash;a freely distributed
annotation tool&mdash;and then select one of the *.mmax projects from the
directories `corpus/annotator-1/` or `corpus/annotator-2/`.

Folder Structure
----------------

The folders of this project are structured as follows:

* `corpus/` &ndash; directory containing corpus files;
  * `annotator1/` &ndash; directory containing MMAX projects for the first
    annotator;
    * `markables/` &ndash; directory containing annotation files for the
       first annotator;
  * `annotator2/` &ndash; directory containing MMAX projects for the second
    annotator;
    * `markables/` &ndash; directory containing annotation files for the
       second annotator;
  * `basedata/` and `source/` &ndash; original corpus tokenization;
  * `custom/`, `scheme/`, and `style/` &ndash; auxiliary MMAX2 data;

* `docs/` &ndash; directory containing annotation guidelines and other
  accompanying documents;

* `scripts/` &ndash; directory containing scripts that were used to process
  corpus data;
  * `examples/` &ndash; directory containing examples of input files for
    the scripts;
  * `align.py` &ndash; auxiliary module used for annotation alignment;
  * `alt_fio.py` &ndash; auxiliary module for AWK-like input/output operations;
  * `conll.py` &ndash; auxiliary module for handling CONLL sentences;
  * `measure_corpus_agreement.py` &ndash; script for measuring corpus
    agreement;
  * `merge_conll_mmax.py` &ndash; script for aligning annotation from the
    corpus with the automatically processed CONLL data;

You can see the examples of invocations in the script files or by just
typing `--help` to see their usage.

Note
----

<span style="color:red">I strongly recommend using the [annotation of annotator-2](https://github.com/WladimirSidorenko/PotTS/tree/eexpression-revision/corpus/annotator-2) on the branch `eexpression-revision` (run `git checkout eexpression-revision` after cloning this project).</span>
