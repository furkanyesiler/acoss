set -e -x

cd wheelhouse
for whl in acoss-*.whl; do
    curl https://bashupload.com/"$whl" --data-binary @"$whl"
done

cd ../dist
for sdist in acoss-*.tar.gz; do
    curl https://bashupload.com/"$sdist" --data-binary @"$sdist"
done
