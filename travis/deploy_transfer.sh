set -e -x

cd wheelhouse
for whl in acoss-*.whl; do
    curl --upload-file "$whl"  https://transfer.sh/"$whl" -w "\n"
done
cd ../dist
for sdist in acoss-*.tar.gz; do
    curl --upload-file "$sdist"  https://transfer.sh/"$sdist" -w "\n"
done