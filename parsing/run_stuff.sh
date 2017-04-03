for f in "$@"
do
    sh /home/ubuntu/matt/stanford-corenlp-full-2014-08-27/mycorenlp.sh $f > "out_"$f
    python $(dirname "$0")/parse.py < "out_"$f > "parsed_"$f
done

