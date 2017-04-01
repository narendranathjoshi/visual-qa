for f in *.selected.qtxt.gz
do
    sh ~/dev/stanford-corenlp-full-2014-08-27/mycorenlp.sh $f > "out_"$f
    python parse.py < "out_"$f > "parsed_"$f
done